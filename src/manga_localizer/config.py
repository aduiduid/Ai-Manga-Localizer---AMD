from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# 默认本地模型：DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic（i1-Q6_K 量化 + f16 mmproj），
# 以 `ollama create` 从两个 GGUF 合成。标签必须带 `:latest`，因为 ensure_ollama_model
# 用严格字符串匹配 /api/tags 返回的名字。
DEFAULT_OLLAMA_MODEL = "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K:latest"

# Ollama takes the working context and the generation ceiling as independent
# options: ``num_ctx`` bounds prompt + generated tokens, ``num_predict`` bounds
# the reply alone.  Both are 64K because a structured JSON reply travels with a
# long prompt attached, and the judge/retranslate passes used to stop at 2048
# output tokens -- a ceiling a real batch carrying the source, the draft and two
# alternatives per line can exceed, which aborted the whole book after all the
# expensive work was already done.  A ceiling costs nothing when the reply ends
# at EOS, so these only need to be generous, not precise.  The model advertises
# 262K; staying at 64K keeps the KV cache affordable on one local GPU.
OLLAMA_CONTEXT_TOKENS = 65_536
OLLAMA_MAX_OUTPUT_TOKENS = 65_536

# A request timeout has to cover the whole generation, so it scales with the
# budget instead of staying at the 180 s that was right for a 2K reply.  At the
# ~110 tok/s this 9B reaches on one Vulkan GPU, 64K output would need ~10 min;
# 1200 s leaves headroom for a cold model load and a slower prefill.  Without
# this, raising the ceiling would only replace "truncated at 2048" with
# "timed out at 180 s" on exactly the long replies the raised ceiling allows.
OLLAMA_REQUEST_TIMEOUT_S = 1_200


def ollama_timeout(token_budget: int) -> int:
    """Seconds to wait for a reply of ``token_budget`` output tokens.

    Scales the historical 180 s at 2K tokens up to OLLAMA_REQUEST_TIMEOUT_S for
    a full-ceiling reply, so a small batch keeps its old responsiveness while a
    large one is not cut off mid-generation.
    """
    scaled = OLLAMA_REQUEST_TIMEOUT_S * min(
        1.0, max(2_048, token_budget) / OLLAMA_MAX_OUTPUT_TOKENS
    )
    return max(180, int(scaled))


@dataclass(frozen=True)
class AppPaths:
    root: Path
    models: Path
    cache: Path
    jobs: Path
    settings: Path

    @classmethod
    def from_env(cls) -> "AppPaths":
        # No .resolve(): paddle's C++ inference and torch.jit.load cannot open
        # non-ASCII paths, and resolve() would turn an ASCII junction
        # (MLS_HOME=E:\mls\data) back into the real Chinese directory.
        root = Path(
            os.environ.get("MLS_HOME", Path.home() / ".manga-localizer-studio")
        ).expanduser()
        return cls(
            root=root,
            models=root / "models",
            cache=root / "cache",
            jobs=root / "jobs",
            settings=root / "settings.json",
        )

    def ensure(self) -> "AppPaths":
        for path in (self.root, self.models, self.cache, self.jobs):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class UserSettings:
    target_language: str = "简体中文"
    source_language: str = "auto"
    story_context: bool = True
    context_pages: int = 6
    preserve_sfx: bool = False
    quality_profile: str = "quality"
    output_format: str = "webp"
    prefer_modelscope: bool = True
    device: str = "auto"
    ocr_backend: str = "hybrid"
    inference_backend: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_ocr_model: str = DEFAULT_OLLAMA_MODEL
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""

    @classmethod
    def load(cls, path: Path) -> "UserSettings":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def configure_model_caches(paths: AppPaths) -> None:
    """Keep every downloaded artifact inside the application data directory."""
    paths.ensure()
    os.environ.setdefault("MODELSCOPE_CACHE", str(paths.models / "modelscope-cache"))
    os.environ.setdefault("HF_HOME", str(paths.models / "huggingface"))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(paths.models / "paddlex"))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
