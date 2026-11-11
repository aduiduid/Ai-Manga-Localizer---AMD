"""Tests for the OpenAI-compatible client that talks to ``llama-server``.

Every test stubs ``_json_request``: the point is the request shape (which fields
llama.cpp needs) and the retry rules (a truncated reply must grow its budget,
not come back as an empty string).
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from manga_localizer import llama_client
from manga_localizer.llama_client import LlamaChatClient, png_bytes
from manga_localizer.translator import DegenerateReply, InferenceConnectionError


def _reply(content, finish_reason="stop"):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
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


def _client(monkeypatch, recorder):
    monkeypatch.setattr("manga_localizer.llama_client._json_request", recorder)
    return LlamaChatClient("http://127.0.0.1:8080", "DeepSeek-9B")


def test_chat_posts_to_the_openai_route_with_the_budget(monkeypatch):
    recorder = _Recorder([_reply("  你好  ")])
    client = _client(monkeypatch, recorder)
    assert client.chat("翻译这句话", max_tokens=256) == "你好"
    sent = recorder.payloads[0]
    # /v1/chat/completions is the only route that carries the OpenAI shape; the
    # Ollama suffix would 404 here.
    assert sent["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert sent["payload"]["model"] == "DeepSeek-9B"
    assert sent["payload"]["max_tokens"] == 256
    assert sent["payload"]["stream"] is False


def test_chat_json_sends_a_json_schema_response_format(monkeypatch):
    schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    recorder = _Recorder([_reply(json.dumps({"text": "你好"}))])
    client = _client(monkeypatch, recorder)
    assert client.chat_json("prompt", schema, max_tokens=200) == {"text": "你好"}
    payload = recorder.payloads[0]["payload"]
    # Ollama's ``format`` field is ignored by llama.cpp: only response_format
    # binds the grammar, so this assertion is the whole structured-output path.
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": schema},
    }
    assert "format" not in payload


def test_a_truncated_reply_grows_the_budget_instead_of_failing(monkeypatch):
    schema = {"type": "object"}
    recorder = _Recorder(
        [
            _reply('{"a":', finish_reason="length"),
            _reply('{"a": 1}', finish_reason="stop"),
        ]
    )
    client = _client(monkeypatch, recorder)
    assert client.chat_json("prompt", schema, max_tokens=100) == {"a": 1}
    budgets = [entry["payload"]["max_tokens"] for entry in recorder.payloads]
    assert budgets == [100, 200]


def test_an_endless_truncation_reports_the_ceiling_it_hit(monkeypatch):
    schema = {"type": "object"}
    client = LlamaChatClient(
        "http://127.0.0.1:8080", "m", max_output_tokens=400
    )
    recorder = _Recorder([_reply('{"a":', finish_reason="length")] * 8)
    monkeypatch.setattr("manga_localizer.llama_client._json_request", recorder)
    with pytest.raises(InferenceConnectionError) as excinfo:
        client.chat_json("prompt", schema, max_tokens=100)
    assert "400" in str(excinfo.value)


def test_an_empty_content_is_reported_rather_than_returned(monkeypatch):
    # llama.cpp leaves content null when the whole budget went to the reasoning
    # channel; an empty string there would be silently translated into "skip".
    recorder = _Recorder([_reply(None)])
    client = _client(monkeypatch, recorder)
    assert client.chat("prompt") == ""


def test_a_server_error_becomes_an_inference_error(monkeypatch):
    monkeypatch.setattr(
        "manga_localizer.llama_client._json_request",
        lambda *args, **kwargs: {"error": {"message": "model not found"}},
    )
    client = LlamaChatClient("http://127.0.0.1:8080", "m")
    with pytest.raises(InferenceConnectionError) as excinfo:
        client.chat("prompt")
    assert "model not found" in str(excinfo.value)


def test_vision_sends_image_parts_before_the_instruction(monkeypatch):
    schema = {"type": "object"}
    recorder = _Recorder([_reply(json.dumps({"boxes": []}))])
    client = _client(monkeypatch, recorder)
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 24
    assert client.chat_vision([png], "读气泡", schema) == {"boxes": []}
    parts = recorder.payloads[0]["payload"]["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[-1] == {"type": "text", "text": "读气泡"}


def test_png_bytes_converts_webp_because_the_server_rejects_it(tmp_path):
    path = tmp_path / "page.webp"
    Image.new("RGB", (8, 8), "white").save(path, format="WEBP")
    converted = png_bytes(path)
    # llama.cpp answers 400 "Failed to load image or audio file" for WebP, so
    # the conversion is not an optimisation.
    assert converted[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(converted)).format == "PNG"


def test_png_bytes_passes_png_through(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(path, format="PNG")
    assert png_bytes(path) == path.read_bytes()


def test_a_vision_retry_penalises_repeats_like_the_ollama_adapter(monkeypatch):
    # A dense page makes the model emit the same glyph with ever-growing
    # coordinates until the budget runs out, so the bigger budget alone never
    # recovers; the daemon adapter learned this and the llama.cpp one must match
    # it or the densest page fails on the migrated path only.
    schema = {"type": "object"}
    recorder = _Recorder(
        [
            _reply('{"boxes": ', finish_reason="length"),
            _reply(json.dumps({"boxes": []})),
        ]
    )
    client = _client(monkeypatch, recorder)
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 24
    assert client.chat_vision([png], "读气泡", schema, max_tokens=100) == {"boxes": []}
    first, second = (sent["payload"] for sent in recorder.payloads)
    assert "repeat_penalty" not in first
    assert second["repeat_penalty"] == 1.3
    assert second["repeat_last_n"] == 256


def test_a_vision_timeout_grows_with_the_escalated_budget(monkeypatch):
    # The wait has to outlive the budget it is waiting for, otherwise the socket
    # dies exactly when a slow generation reaches its ceiling and the escalation
    # never happens.  The caller's floor still wins for the first attempt.
    schema = {"type": "object"}
    recorder = _Recorder(
        [
            _reply('{"boxes": ', finish_reason="length"),
            _reply(json.dumps({"boxes": []})),
        ]
    )
    client = _client(monkeypatch, recorder)
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 24
    client.chat_vision([png], "读气泡", schema, max_tokens=32_768, timeout=600)
    first, second = (sent["timeout"] for sent in recorder.payloads)
    assert first == 600
    assert second == 1_200

def test_a_translation_request_asks_for_a_repeat_penalty(monkeypatch):
    recorder = _Recorder([_reply("译文")])
    client = _client(monkeypatch, recorder)
    assert client.chat("提示", max_tokens=96) == "译文"
    payload = recorder.payloads[0]["payload"]
    # llama-server defaults to 1.00, i.e. repetition is unpunished: that is how
    # one job answered "@@@..." for 87 minutes with nothing to interrupt it.
    assert payload["repeat_penalty"] == llama_client.REPEAT_PENALTY
    assert payload["repeat_last_n"] == llama_client.REPEAT_LAST_N


def test_a_collapsed_reply_restarts_the_server_once_and_retries(monkeypatch):
    recorder = _Recorder([_reply("@" * 96), _reply("译文")])
    restarts = []

    def restart(base_url):
        restarts.append(base_url)
        return "http://127.0.0.1:9090"

    monkeypatch.setattr(llama_client, "restart_llama_server", restart)
    client = _client(monkeypatch, recorder)
    assert client.chat("提示", max_tokens=96) == "译文"
    assert restarts == ["http://127.0.0.1:8080"]
    assert client.base_url == "http://127.0.0.1:9090"
    assert len(recorder.payloads) == 2


def test_a_second_collapse_is_reported_rather_than_retried(monkeypatch):
    recorder = _Recorder([_reply("@" * 96), _reply("@" * 96)])
    monkeypatch.setattr(
        llama_client, "restart_llama_server", lambda base_url: "http://127.0.0.1:9090"
    )
    client = _client(monkeypatch, recorder)
    with pytest.raises(DegenerateReply):
        client.chat("提示", max_tokens=96)
    assert len(recorder.payloads) == 2


def test_a_collapse_from_a_server_we_do_not_own_is_reported(monkeypatch):
    recorder = _Recorder([_reply("@" * 96)])
    monkeypatch.setattr(llama_client, "restart_llama_server", lambda base_url: None)
    client = _client(monkeypatch, recorder)
    with pytest.raises(DegenerateReply):
        client.chat("提示", max_tokens=96)
    assert len(recorder.payloads) == 1


def test_the_translation_wait_is_capped(monkeypatch):
    recorder = _Recorder([_reply("译文")])
    client = _client(monkeypatch, recorder)
    client.chat("提示", max_tokens=4_096, timeout=9_999)
    assert recorder.payloads[0]["timeout"] == llama_client.MAX_REQUEST_WAIT_S
