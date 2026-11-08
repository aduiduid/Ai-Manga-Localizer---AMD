"""A minimal OpenAI-compatible client for ``llama-server``.

``translator._json_request`` already speaks HTTP with the standard library and
turns every failure into ``InferenceConnectionError``, so this module only adds
the OpenAI request/response shape on top of it.  No SDK: the app ships as a
single virtualenv that must not grow a second HTTP stack, and the four endpoints
it needs are small enough to spell out.

Two llama.cpp details are load-bearing and easy to lose in a rewrite:

* ``content`` can be empty when the model spends the whole budget on its
  reasoning channel, so a truncation is detected from ``finish_reason`` and
  retried with a bigger budget rather than reported as an empty reply.
* Structured output uses ``response_format.json_schema`` (not Ollama's
  ``format``), and the served model must have been started with the matching
  grammar flag or the field is silently ignored.
"""

from __future__ import annotations

import json
from typing import Iterable

from .llama_server import llama_timeout
from .translator import InferenceConnectionError, _endpoint, _json_request

# Matches OLLAMA_MAX_OUTPUT_TOKENS so a migrated setting keeps its meaning.
DEFAULT_MAX_OUTPUT_TOKENS = 65_536
DEFAULT_CONTEXT_TOKENS = 65_536


class LlamaChatClient:
    """Chat, structured-output and vision calls against one server."""

    def __init__(
        self,
        base_url: str,
        model: str = "",
        *,
        api_key: str = "",
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        self.loaded = False

    # -- transport ---------------------------------------------------------
    def _url(self) -> str:
        return _endpoint(self.base_url, "/v1/chat/completions")

    def _post(self, payload: dict, timeout: int) -> dict:
        self.loaded = True
        reply = _json_request(self._url(), payload, api_key=self.api_key, timeout=timeout)
        if "error" in reply and not reply.get("choices"):
            detail = reply.get("error")
            message = detail.get("message") if isinstance(detail, dict) else str(detail)
            raise InferenceConnectionError(f"llama-server 返回错误: {message}")
        return reply

    @staticmethod
    def _content(reply: dict) -> tuple[str, str]:
        """The reply text and its ``finish_reason``."""
        try:
            choice = (reply.get("choices") or [{}])[0]
            message = choice.get("message") or {}
        except (AttributeError, IndexError, TypeError) as exc:
            raise InferenceConnectionError("llama-server 返回格式缺少 choices") from exc
        text = message.get("content")
        if not isinstance(text, str):
            # llama.cpp puts the chain of thought in ``reasoning_content`` and
            # leaves ``content`` null when the budget ran out there.
            text = ""
        return text, str(choice.get("finish_reason") or "")

    # -- chat --------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int = 1_600,
        temperature: float = 0.2,
        timeout: int | None = None,
    ) -> str:
        budget = min(max_tokens, self.max_output_tokens)
        payload = {
            "model": self.model or "default",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": temperature,
            "max_tokens": budget,
        }
        reply = self._post(payload, timeout or llama_timeout(budget))
        text, _ = self._content(reply)
        return text.strip()

    def chat_json(
        self,
        prompt: str,
        schema: dict,
        *,
        max_tokens: int = 4_096,
        temperature: float = 0.0,
        timeout: int | None = None,
    ) -> dict:
        """A reply validated against ``schema``, growing the budget on truncation.

        A structured reply is as long as the data it carries, and the judge and
        retranslate passes put several candidates in one request, so a single
        fixed ceiling truncates exactly the passes that matter most.
        """
        budgets = [min(max_tokens, self.max_output_tokens)]
        while budgets[-1] < self.max_output_tokens:
            budgets.append(min(self.max_output_tokens, budgets[-1] * 2))
        last = ""
        for budget in budgets:
            payload = {
                "model": self.model or "default",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": temperature,
                "max_tokens": budget,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "schema": schema},
                },
            }
            reply = self._post(payload, timeout or llama_timeout(budget))
            text, reason = self._content(reply)
            last = text
            if reason == "length":
                continue
            break
        else:
            raise InferenceConnectionError(
                f"llama-server 结构化输出在 {budgets[-1]} token 上限下仍被截断"
            )
        try:
            return json.loads(last)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InferenceConnectionError("llama-server 未返回有效的结构化 JSON") from exc

    def chat_vision(
        self,
        images: Iterable[bytes],
        prompt: str,
        schema: dict,
        *,
        max_tokens: int = 8_192,
        temperature: float = 0.0,
        timeout: int | None = None,
    ) -> dict:
        """A structured reply from page images.

        The images must already be PNG or JPEG: llama.cpp answers 400
        ``Failed to load image or audio file`` for WebP, which is why the caller
        converts before it gets here.
        """
        parts: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(blob)},
            }
            for blob in images
        ]
        parts.append({"type": "text", "text": prompt})
        budgets = [min(max_tokens, self.max_output_tokens)]
        while budgets[-1] < self.max_output_tokens:
            budgets.append(min(self.max_output_tokens, budgets[-1] * 2))
        last = ""
        for index, budget in enumerate(budgets):
            payload = {
                "model": self.model or "default",
                "messages": [{"role": "user", "content": parts}],
                "stream": False,
                "temperature": temperature,
                "max_tokens": budget,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "schema": schema},
                },
            }
            if index:
                # A dense page can make the model emit the same glyph with
                # ever-growing coordinates until the budget runs out, so a bigger
                # budget alone never recovers.  Ollama's OCR adapter learned the
                # same lesson; the first attempt keeps the upstream sampling.
                payload["repeat_penalty"] = 1.3
                payload["repeat_last_n"] = 256
            # The wait escalates with the budget: llama.cpp's vision calls run
            # longer than the daemon's on the same page, and a flat timeout
            # aborts exactly the generation that needed the bigger budget.
            reply = self._post(payload, max(timeout or 0, llama_timeout(budget)))
            text, reason = self._content(reply)
            last = text
            if reason == "length":
                continue
            break
        else:
            raise InferenceConnectionError(
                f"llama-server 视觉输出在 {budgets[-1]} token 上限下仍被截断"
            )
        try:
            return json.loads(last)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InferenceConnectionError("llama-server 未返回有效的结构化 JSON") from exc


def _data_url(blob: bytes) -> str:
    import base64

    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif blob[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}"


def png_bytes(image_path) -> bytes:
    """``image_path`` as PNG.

    llama.cpp's image loader does not accept WebP, which every page in this app
    can be, so the conversion lives next to the request that needs it.
    """
    from io import BytesIO

    from PIL import Image

    with Image.open(image_path) as image:
        if image.format in {"PNG", "JPEG"}:
            return image_path.read_bytes()
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


__all__ = [
    "DEFAULT_CONTEXT_TOKENS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "LlamaChatClient",
    "png_bytes",
]
