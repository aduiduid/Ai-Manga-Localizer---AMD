"""Tests for the vision-OCR adapter that talks to ``llama-server``.

``LlamaVisionOCR`` is meant to be the same reader as ``OllamaVisionOCR`` over a
different transport, so the interesting parts are exactly the two things that
differ: the page is converted to PNG (llama.cpp rejects WebP) and the request is
OpenAI-shaped.  Everything else -- prompts, schemas, clamping, role filters -- is
inherited, which is what keeps an A/B against the Ollama path honest.
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from manga_localizer.llama_client import LlamaChatClient
from manga_localizer.ocr import LlamaVisionOCR, OllamaVisionOCR


def _reply(content, finish_reason="stop"):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ]
    }


class _Recorder:
    """Collects the payloads a client sends and replays canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads: list[dict] = []

    def __call__(self, url, payload=None, api_key="", timeout=180):
        self.payloads.append({"url": url, "payload": payload, "timeout": timeout})
        return self.replies.pop(0)


def _client(monkeypatch, recorder, *, max_output_tokens=65_536):
    monkeypatch.setattr("manga_localizer.llama_client._json_request", recorder)
    return LlamaChatClient(
        "http://127.0.0.1:8082", "DeepSeek-9B", max_output_tokens=max_output_tokens
    )


def test_a_webp_page_is_converted_before_it_reaches_llama_cpp(tmp_path):
    source = tmp_path / "page.webp"
    Image.new("RGB", (16, 16), "white").save(source, format="WEBP")
    service = LlamaVisionOCR("http://127.0.0.1:8082", "DeepSeek-9B")
    payload = service._image_bytes(source)
    # llama.cpp answers 400 "Failed to load image or audio file" for WebP, and
    # every page this app stores is WebP, so this conversion is page 1 or bust.
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert OllamaVisionOCR("http://127.0.0.1:11434", "m")._image_bytes(source) == (
        source.read_bytes()
    )


def test_a_png_page_is_passed_through_untouched(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (16, 16), "white").save(source, format="PNG")
    service = LlamaVisionOCR("http://127.0.0.1:8082", "DeepSeek-9B")
    assert service._image_bytes(source) == source.read_bytes()


def test_the_request_is_openai_shaped_and_carries_the_schema(tmp_path, monkeypatch):
    schema = {"type": "object", "properties": {"regions": {"type": "array"}}}
    recorder = _Recorder([_reply(json.dumps({"regions": []}))])
    service = LlamaVisionOCR("http://127.0.0.1:8082", "DeepSeek-9B")
    service.client = _client(monkeypatch, recorder)
    assert service._chat([b"\x89PNG\r\n\x1a\n"], "读气泡", schema) == {"regions": []}

    sent = recorder.payloads[0]
    # Ollama's native /api/chat would 404 here; only the OpenAI route exists.
    assert sent["url"] == "http://127.0.0.1:8082/v1/chat/completions"
    assert sent["payload"]["model"] == "DeepSeek-9B"
    parts = sent["payload"]["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[-1] == {"type": "text", "text": "读气泡"}
    # Ollama's ``format`` field is meaningless to llama.cpp: the grammar comes
    # from response_format, so a missing one silently unconstrains the model.
    assert sent["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": schema},
    }
    assert "format" not in sent["payload"]


def test_a_truncated_page_reply_grows_the_budget(monkeypatch):
    schema = {"type": "object"}
    recorder = _Recorder(
        [
            _reply('{"regions":', finish_reason="length"),
            _reply('{"regions": []}', finish_reason="stop"),
        ]
    )
    service = LlamaVisionOCR("http://127.0.0.1:8082", "DeepSeek-9B")
    service.client = _client(monkeypatch, recorder)
    assert service._chat([b"\x89PNG\r\n\x1a\n"], "prompt", schema) == {"regions": []}
    budgets = [entry["payload"]["max_tokens"] for entry in recorder.payloads]
    # A dense page can legitimately need more JSON than the first attempt allows,
    # and giving up there would report a page as unreadable.
    assert budgets == [8_192, 16_384]


def test_the_ocr_adapter_feeds_the_shared_prompts(tmp_path, monkeypatch):
    """``analyze`` must run the inherited Ollama prompt, not a new one."""
    source = tmp_path / "page.webp"
    Image.new("RGB", (400, 600), "white").save(source, format="WEBP")
    recorder = _Recorder(
        [
            _reply(
                json.dumps(
                    {
                        "regions": [
                            {
                                "bbox": [40, 120, 160, 300],
                                "text": "こんにちは",
                                "score": 0.9,
                                "is_sfx": False,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        ]
    )
    service = LlamaVisionOCR("http://127.0.0.1:8082", "DeepSeek-9B")
    service.client = _client(monkeypatch, recorder)
    page = service.analyze(source, 2)

    assert page.units[0].bbox == [40, 120, 160, 300]
    assert page.units[0].id == "p002u01"
    prompt = recorder.payloads[0]["payload"]["messages"][0]["content"][-1]["text"]
    assert "Detect every" in prompt
    # The inherited schema is what makes the two transports comparable.
    schema = recorder.payloads[0]["payload"]["response_format"]["json_schema"]["schema"]
    assert "regions" in schema["required"]


def test_an_unknown_source_language_is_rejected():
    with pytest.raises(ValueError, match="Unknown source language"):
        LlamaVisionOCR("http://127.0.0.1:8082", "m", source_language="fr")


def test_a_missing_model_name_is_rejected():
    with pytest.raises(ValueError, match="vision model is required"):
        LlamaVisionOCR("http://127.0.0.1:8082", "   ")

def test_the_detection_schema_bounds_the_confidence_score(tmp_path, monkeypatch):
    """llama.cpp compiles the schema into a grammar, so an unbounded number drifts.

    Every box on a page it had read perfectly came back as ``score: 0.0``,
    which silently turned off the score gates downstream (``prefer_semantic_ocr``
    needs 0.72 and ``_local_crop_is_suspicious`` rejects <= 0.05).  The range has
    to be stated in the schema for the constrained decoder to honour it.
    """
    source = tmp_path / "page.webp"
    Image.new("RGB", (400, 600), "white").save(source, format="WEBP")
    recorder = _Recorder([_reply(json.dumps({"regions": []}))])
    service = LlamaVisionOCR("http://127.0.0.1:8082", "DeepSeek-9B")
    service.client = _client(monkeypatch, recorder)
    service.analyze(source, 1)

    schema = recorder.payloads[0]["payload"]["response_format"]["json_schema"]["schema"]
    score = schema["properties"]["regions"]["items"]["properties"]["score"]
    assert score["type"] == "number"
    assert score["minimum"] == 0
    assert score["maximum"] == 1
