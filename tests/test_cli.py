import logging
import io
import json

from manga_localizer.cli import (
    _font_status,
    _llama_options,
    configure_cli_encoding,
    configure_dependency_logging,
    existing_ui_url,
    torch_pair_compatible,
)
from manga_localizer.config import UserSettings


def test_existing_local_workspace_is_reused(monkeypatch):
    payload = io.BytesIO(json.dumps({"version": "0.4.5", "local_only": True}).encode())
    monkeypatch.setattr(
        "manga_localizer.cli.urlopen", lambda *_args, **_kwargs: payload
    )
    assert existing_ui_url("127.0.0.1", 8765) == "http://127.0.0.1:8765"


def test_unrelated_service_is_not_treated_as_the_workspace(monkeypatch):
    payload = io.BytesIO(json.dumps({"service": "something-else"}).encode())
    monkeypatch.setattr(
        "manga_localizer.cli.urlopen", lambda *_args, **_kwargs: payload
    )
    assert existing_ui_url("127.0.0.1", 8765) == ""


class ReconfigurableStream:
    def __init__(self):
        self.calls = []

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


def test_cli_streams_are_normalized_to_utf8():
    first = ReconfigurableStream()
    second = ReconfigurableStream()
    configure_cli_encoding(first, second)
    assert first.calls == [{"encoding": "utf-8", "errors": "backslashreplace"}]
    assert second.calls == first.calls


def test_optional_http_probes_do_not_fill_the_user_console():
    configure_dependency_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_doctor_font_status_is_safe_before_assets_are_downloaded(tmp_path):
    def missing(_paths):
        raise FileNotFoundError("no system CJK font")

    assert _font_status(missing, tmp_path).startswith("missing;")


def test_doctor_requires_the_pinned_torch_runtime_pair():
    assert torch_pair_compatible("2.8.0+cu129", "0.23.0+cu129")
    assert torch_pair_compatible("2.8.0+cpu", "0.23.0+cpu")
    assert not torch_pair_compatible("2.9.0+cu129", "0.23.0+cu129")
    assert not torch_pair_compatible("2.8.0+cu129", "0.1.6")


def test_cli_llama_flags_fall_back_to_the_saved_settings():
    # A flag overrides one run; everything else has to match what the UI saved,
    # or `--backend llama` would quietly use the default ports and model tags.
    settings = UserSettings(
        llama_model="saved:q6",
        llama_main_port=9100,
        llama_vision_port=9001,
    )
    values = _llama_options(
        settings,
        llama_model="flag:q8",
        llama_vision_port=0,
        llama_context_tokens=0,
        llama_ollama_root="",
    )
    assert values["llama_model"] == "flag:q8"
    assert values["llama_vision_port"] == 9001
    assert values["llama_main_port"] == 9100
    assert values["llama_context_tokens"] == settings.llama_context_tokens
    assert values["llama_ollama_root"] == settings.llama_ollama_root


def test_cli_llama_options_cover_every_pipeline_field():
    from manga_localizer.pipeline import PipelineRequest

    values = _llama_options(UserSettings())
    fields = {
        name for name in PipelineRequest.__dataclass_fields__ if name.startswith("llama_")
    }
    assert set(values) == fields
