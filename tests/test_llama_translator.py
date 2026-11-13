"""The translator that starts its own server when a pass calls for it."""

from __future__ import annotations

from pathlib import Path

import pytest

from manga_localizer.llama_server import LlamaSlot
from manga_localizer.llama_translator import LlamaTranslator
from manga_localizer.translator import InferenceConnectionError


class _FakeManager:
    """Records the lifecycle calls a translator makes on its server."""

    def __init__(self):
        self.stopped: list[str] = []
        self.relaunched: list[str] = []

    def stop(self, role: str) -> None:
        self.stopped.append(role)

    def relaunch(self, role: str) -> LlamaSlot:
        self.relaunched.append(role)
        return LlamaSlot(
            role=role,
            alias="Local-7B",
            base_url="http://127.0.0.1:8081",
            port=8081,
            context=8192,
            weights=Path("Local-7B.gguf"),
        )


def _slot() -> LlamaSlot:
    return LlamaSlot(
        role="editor",
        alias="DeepSeek-9B",
        base_url="http://127.0.0.1:8080",
        port=8080,
        context=65536,
        weights=Path("DeepSeek-9B.gguf"),
    )


def test_a_translator_whose_server_was_freed_starts_it_again():
    manager = _FakeManager()
    translator = LlamaTranslator(
        None, "简体中文", None, manager=manager, role="review", label="Local-7B",
        evict=True,
    )
    assert translator._stopped is True

    client = translator._client()

    assert manager.relaunched == ["review"]
    assert client.base_url == "http://127.0.0.1:8081"
    assert translator.model == "Local-7B"
    assert translator._stopped is False


def test_an_evicting_pass_stops_its_server_when_it_is_done():
    manager = _FakeManager()
    translator = LlamaTranslator(
        _slot(), "简体中文", None, manager=manager, role="editor", evict=True
    )
    translator._client()
    assert manager.stopped == []

    translator.unload()

    # 14,586 MB for the 9B beside the 7.5B candidate against the 15,439 MiB the
    # card had free: the pass that used the model gives the VRAM back first.
    assert manager.stopped == ["editor"]
    assert translator._stopped is True

    translator._client()

    assert manager.relaunched == ["editor"]


def test_a_candidate_that_shares_the_card_is_not_stopped():
    manager = _FakeManager()
    translator = LlamaTranslator(
        _slot(), "简体中文", None, manager=manager, role="editor"
    )

    translator.unload()

    # The 1.8B candidate fits beside the 9B, so unloading stays what it always
    # was: a bookkeeping flag while ``hold()`` owns the process.
    assert manager.stopped == []
    assert translator._stopped is False


def test_an_evicting_translator_that_was_never_used_stops_nothing():
    manager = _FakeManager()
    translator = LlamaTranslator(
        None, "简体中文", None, manager=manager, role="review", label="Local-7B",
        evict=True,
    )

    translator.unload()

    assert manager.stopped == []


def test_a_stopped_translator_without_a_manager_says_why():
    translator = LlamaTranslator(
        None, "简体中文", None, role="review", label="Local-7B", evict=True
    )

    with pytest.raises(InferenceConnectionError) as excinfo:
        translator._client()

    assert "重启参数" in str(excinfo.value)
