"""The translator that speaks to ``llama-server`` instead of Ollama.

It subclasses ``OllamaTranslator`` and overrides only the transport: the review,
judge and retranslate passes are pure prompt logic that call ``self._generate``
and ``self._generate_json``, so keeping them in one place means a fix to the
prompting applies to both backends.  What changes here is *how* a message
travels -- OpenAI-compatible ``/v1/chat/completions``, ``response_format`` for
structured output, and ``finish_reason`` instead of ``done_reason``.

The server a pass needs is not always running when the pass starts.  When the
editor's weights and the candidate's cannot share the card, each pass releases
its server in ``unload`` and the next one starts it again from the recipe the
manager recorded.
"""

from __future__ import annotations

from .llama_client import LlamaChatClient
from .llama_server import LlamaServerManager, LlamaSlot
from .translator import (
    InferenceConnectionError,
    JapaneseNameDictionary,
    OllamaTranslator,
)


class LlamaTranslator(OllamaTranslator):
    """Translation passes served by a running ``llama-server`` instance."""

    def __init__(
        self,
        slot: LlamaSlot | None,
        target_language: str = "简体中文",
        name_dictionary: JapaneseNameDictionary | None = None,
        api_key: str = "",
        manager: LlamaServerManager | None = None,
        role: str = "",
        label: str = "",
        evict: bool = False,
    ):
        # The base class only stores these; no Ollama daemon is contacted on
        # this path, and the slot's alias stands in for the Ollama tag.
        super().__init__(
            slot.base_url if slot is not None else "",
            slot.alias if slot is not None else label,
            target_language,
            name_dictionary,
        )
        self.slot = slot
        self.role = role or (slot.role if slot is not None else "")
        self.evict = evict
        self._manager = manager
        self._api_key = api_key
        self._stopped = slot is None
        self.client = (
            LlamaChatClient(slot.base_url, slot.alias, api_key=api_key)
            if slot is not None
            else None
        )

    def _client(self) -> LlamaChatClient:
        """The client for this pass, starting the server again if it was freed."""
        if self._stopped:
            if self._manager is None:
                raise InferenceConnectionError(
                    f"{self.role or 'llama.cpp'} 的推理服务已经停止，且没有可用的重启参数"
                )
            slot = self._manager.relaunch(self.role)
            self.slot = slot
            self._stopped = False
            self.base_url = slot.base_url
            self.model = slot.alias
            self.client = LlamaChatClient(
                slot.base_url, slot.alias, api_key=self._api_key
            )
        if self.client is None:  # pragma: no cover - only without a server at all
            raise InferenceConnectionError(
                f"{self.role or 'llama.cpp'} 没有可用的推理服务"
            )
        return self.client

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        self._loaded = True
        return self._client().chat(prompt, max_tokens=max_new_tokens)

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
        return self._client().chat_json(prompt, schema, max_tokens=max_new_tokens)

    def unload(self) -> None:
        """Release the model this pass is finished with.

        Usually there is nothing to release: ``LlamaServerManager.hold()``
        brackets the whole run and stops the servers when the last holder
        leaves, which is stronger than evicting one model through an API call
        because it gives the VRAM back to the driver.  When the two weight sets
        cannot share the card the pass has to be stricter than that -- the
        editor's 9B and the 7.5B candidate measured 14,586 MB together against
        the 15,439 MiB the card had free -- so the server is stopped here and
        ``_client`` starts it again for the next pass.
        """
        self._loaded = False
        if not self.evict or self._manager is None or self.slot is None:
            return
        self._manager.stop(self.role)
        self._stopped = True
        self.client = None


def require_review_slot(slot: LlamaSlot | None) -> LlamaSlot:
    """Fail loudly when the review pass was requested without its server."""
    if slot is None:
        raise InferenceConnectionError(
            "质量模式需要 llama.cpp 的第二个实例（审校候选），但它没有启动。"
            "请在设置里检查审校模型与审校端口。"
        )
    return slot


__all__ = ["LlamaTranslator", "require_review_slot"]
