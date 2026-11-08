"""The translator that speaks to ``llama-server`` instead of Ollama.

It subclasses ``OllamaTranslator`` and overrides only the transport: the review,
judge and retranslate passes are pure prompt logic that call ``self._generate``
and ``self._generate_json``, so keeping them in one place means a fix to the
prompting applies to both backends.  What changes here is *how* a message
travels -- OpenAI-compatible ``/v1/chat/completions``, ``response_format`` for
structured output, and ``finish_reason`` instead of ``done_reason``.
"""

from __future__ import annotations

from .llama_client import LlamaChatClient
from .llama_server import LlamaSlot
from .translator import (
    InferenceConnectionError,
    JapaneseNameDictionary,
    OllamaTranslator,
)


class LlamaTranslator(OllamaTranslator):
    """Translation passes served by a running ``llama-server`` instance."""

    def __init__(
        self,
        slot: LlamaSlot,
        target_language: str = "简体中文",
        name_dictionary: JapaneseNameDictionary | None = None,
        api_key: str = "",
    ):
        # The base class only stores these; no Ollama daemon is contacted on
        # this path, and the slot's alias stands in for the Ollama tag.
        super().__init__(
            slot.base_url, slot.alias, target_language, name_dictionary
        )
        self.slot = slot
        self.client = LlamaChatClient(slot.base_url, slot.alias, api_key=api_key)

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        self._loaded = True
        return self.client.chat(prompt, max_tokens=max_new_tokens)

    def _generate_json(
        self,
        prompt: str,
        schema: dict,
        max_new_tokens: int = 4096,
        think: bool = False,
    ) -> dict:
        # ``think`` is accepted and ignored: the server was started with
        # ``-rea off`` because llama.cpp's reasoning channel otherwise swallows
        # the entire budget and returns an empty ``content``.  Keeping the
        # parameter makes the override signature-compatible with the caller in
        # ``OllamaTranslator.judge_candidates``.
        self._loaded = True
        return self.client.chat_json(prompt, schema, max_tokens=max_new_tokens)

    def unload(self) -> None:
        """Nothing to release per translator: the process owns the model.

        ``LlamaServerManager.hold()`` brackets the whole run and stops the
        servers when the last holder leaves, which is stronger than evicting one
        model through an API call: it gives the VRAM back to the driver.
        """
        self._loaded = False


def require_review_slot(slot: LlamaSlot | None) -> LlamaSlot:
    """Fail loudly when the review pass was requested without its server."""
    if slot is None:
        raise InferenceConnectionError(
            "质量模式需要 llama.cpp 的第二个实例（审校候选），但它没有启动。"
            "请在设置里检查审校模型与审校端口。"
        )
    return slot


__all__ = ["LlamaTranslator", "require_review_slot"]
