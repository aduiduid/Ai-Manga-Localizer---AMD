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


# llama.cpp serves one model per process, so the app runs two: the 9B editor on
# the first port and the 1.8B review candidate on the second.  Both are the
# documented ports rather than llama-server's own default of 9931, because the
# UI shows them and the user may have a browser bookmark to one.
DEFAULT_LLAMA_MAIN_PORT = 8080
DEFAULT_LLAMA_REVIEW_PORT = 8081
# Vision OCR gets its own port because it runs before translation starts: the
# page-read pass must not queue behind (or pre-empt) the 9B translation slot,
# and a server serves one request at a time.
DEFAULT_LLAMA_VISION_PORT = 8082

# The tags below name Ollama manifests, not Ollama daemons: the manifest is the
# only index that maps a readable name to the content-addressed GGUF blobs, and
# llama-server loads those blobs directly.  Migrating therefore needs no copy
# and no conversion, and switching back to Ollama keeps working.
DEFAULT_LLAMA_MODEL = DEFAULT_OLLAMA_MODEL
DEFAULT_LLAMA_REVIEW_MODEL = "hy-mt2-1.8b:q8"

# The single-model backend names one specific file instead of taking a
# setting: the point of it is "run this one alone, one pass, least VRAM".
LLAMA_SINGLE_MODEL = "HY-MT2-7B-Q6_K"

# llama-server's ``-c`` is per slot; one slot per server keeps the whole budget
# for the single request a translation pass makes at a time.
LLAMA_CONTEXT_TOKENS = 65_536


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

    @property
    def batches(self) -> Path:
        """Where batch (queue) records live, one JSON per batch.

        A property rather than a field because AppPaths is built positionally by
        older callers, and a batch is just another record store next to jobs.
        """
        return self.root / "batches"

    @property
    def llama_models(self) -> Path:
        """Where llama.cpp model files and its runtime directory are staged.

        Property for the same reason as ``batches``: positional construction.
        The directory holds hard links into the Ollama blob store plus a private
        copy of ``llama-server.exe`` and the GPU backend it needs, so the app
        never has to modify the Ollama installation it borrows from.
        """
        return self.root / "llama-models"

    def ensure(self) -> "AppPaths":
        for path in (
            self.root,
            self.models,
            self.cache,
            self.jobs,
            self.batches,
            self.llama_models,
        ):
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
    # 0 keeps every library on its own default (all logical cores for each
    # backend, which is what this app always did before the setting existed).
    cpu_threads: int = 0
    # A full book takes tens of minutes, so a run that ends while the user is in
    # another window should announce itself.  Two separate switches because the
    # two events need different sounds: success is an optional nicety, whereas a
    # failure is what the user actually waits for.
    notify_on_complete: bool = True
    notify_on_failure: bool = True
    # Percent of the cue's own peak level, whose ceiling is set in the cue table
    # in app.js.  The table is sized so the loudest cue still peaks well under
    # full scale with its notes overlapping, which is what lets 100 be twice the
    # level the chime used before it was adjustable; the default (70) sits a
    # little above that old fixed level so a quiet laptop speaker has headroom.
    notify_volume: int = 70
    ocr_backend: str = "hybrid"
    inference_backend: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_ocr_model: str = DEFAULT_OLLAMA_MODEL
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    # llama.cpp runs the same GGUF blobs Ollama already downloaded, so these are
    # Ollama *manifest tags* used only to locate the files; nothing talks to the
    # Ollama daemon on this path.  The ports are per server because the bundled
    # build has no router mode.
    llama_model: str = DEFAULT_LLAMA_MODEL
    llama_review_model: str = DEFAULT_LLAMA_REVIEW_MODEL
    llama_context_tokens: int = LLAMA_CONTEXT_TOKENS
    llama_main_port: int = DEFAULT_LLAMA_MAIN_PORT
    llama_review_port: int = DEFAULT_LLAMA_REVIEW_PORT
    llama_vision_port: int = DEFAULT_LLAMA_VISION_PORT
    # Empty means "look for the llama-server.exe that ships inside Ollama"; a
    # path lets a user point at a standalone llama.cpp build instead.
    llama_server_path: str = ""
    # Where the Ollama blobs live.  Empty means "<data root>/../ollama-models",
    # which is where start.bat puts them.
    llama_ollama_root: str = ""

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
