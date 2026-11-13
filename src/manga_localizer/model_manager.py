from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from .config import (
    LLAMA_SINGLE_MODEL,
    AppPaths,
    UserSettings,
    configure_model_caches,
    cleanup_profile_for,
    translate_quality,
)


ProgressCallback = Callable[[int, str], None]

LAMA_WEIGHTS_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/"
    "v0.1.0/big-lama.pt"
)
LAMA_WEIGHTS_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


class ModelDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaRuntime:
    """The Ollama service actually serving this configuration.

    ``available`` maps each pulled tag to its size in bytes, or is ``None``
    when the service could not be reached — that keeps the model page honest
    instead of reporting a stale "ready".
    """

    translation_model: str = ""
    ocr_model: str = ""
    hy_mt2_tag: str = ""
    base_url: str = "http://127.0.0.1:11434"
    available: dict[str, int] | None = None

    @classmethod
    def from_settings(
        cls, settings: "UserSettings", tags: dict[str, int] | None = None
    ) -> "OllamaRuntime":
        return cls(
            translation_model=(settings.ollama_model or "").strip(),
            ocr_model=(settings.ollama_ocr_model or "").strip(),
            # Mirrors pipeline.py: the quality candidate pass is served from
            # Ollama only when this opt-in tag is set.
            hy_mt2_tag=os.environ.get("MLS_HY_MT2_OLLAMA", "").strip(),
            base_url=settings.ollama_base_url,
            available=tags,
        )


def ollama_tags(base_url: str, timeout: float = 3.0) -> dict[str, int] | None:
    """Pulled tags and their sizes; ``None`` when Ollama is not answering."""
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return {
        item["name"]: int(item.get("size") or 0)
        for item in payload.get("models", [])
        if item.get("name")
    }


def _human_size(size: int) -> str:
    if size <= 0:
        return ""
    if size >= 1024**3:
        return f"~{size / 1024**3:.1f} GB"
    return f"~{size / 1024**2:.0f} MB"


def ollama_specs(runtime: OllamaRuntime, ocr_backend: str, quality_profile: str):
    """One row per Ollama tag this configuration actually runs."""
    wanted: list[tuple[str, str]] = []
    if runtime.translation_model:
        wanted.append((runtime.translation_model, "translation"))
    if (
        translate_quality(quality_profile)
        and runtime.hy_mt2_tag
        and runtime.hy_mt2_tag != runtime.translation_model
    ):
        wanted.append((runtime.hy_mt2_tag, "review"))
    if ocr_backend in {"hybrid", "ollama"} and runtime.ocr_model:
        if all(runtime.ocr_model != tag for tag, _ in wanted):
            wanted.append((runtime.ocr_model, "ocr"))
    rows = []
    for tag, role in wanted:
        size = (runtime.available or {}).get(tag)
        rows.append(
            {
                "id": f"ollama:{tag}",
                "name": tag,
                "kind": "ollama",
                "role": {
                    "translation": "分阶段语义审校的主翻译模型",
                    "review": "独立候选译文（质量模式第二遍）",
                    "ocr": "整页视觉语义 OCR",
                }[role],
                "role_key": f"models.role.{role}",
                "provider": "Ollama",
                "repository": tag,
                "size": _human_size(size or 0),
                "license": "Upstream",
                "required": True,
                "ready": size is not None,
                "path": f"{runtime.base_url} · {tag}",
                "status_code": (
                    "offline"
                    if runtime.available is None
                    else ("pulled" if size is not None else "missing")
                ),
            }
        )
    return rows


def llama_specs(
    available: dict[str, "OllamaModelFiles"] | None,
    settings: "UserSettings",
) -> list[dict]:
    """One row per GGUF llama-server loads, mirrored from the Ollama manifests.

    The blob is the same file Ollama downloaded, so "ready" means the manifest
    resolved to a blob on disk -- there is no separate download step and no
    daemon to ask.  ``available`` is ``None`` when the manifest folder itself
    could not be read, which is reported as offline rather than missing.
    """
    if settings.inference_backend == "llama_7b":
        # The single-model backend runs one file and opens no candidate pass.
        wanted: list[tuple[str, str]] = [(LLAMA_SINGLE_MODEL, "translation")]
    else:
        wanted = [((settings.llama_model or "").strip(), "translation")]
        review = (settings.llama_review_model or "").strip()
        if translate_quality(settings.quality_profile) and review and review != wanted[0][0]:
            wanted.append((review, "review"))
    # ``hybrid``/``ollama`` read pages through Ollama but the model itself may
    # already be one of these blobs, so listing it is still honest.  ``llama``
    # means llama.cpp reads the pages directly, which is exactly this file.
    if settings.ocr_backend in {"hybrid", "ollama", "llama"}:
        ocr = (
            (settings.ollama_ocr_model or "").strip()
            if settings.ocr_backend != "llama"
            else (settings.ollama_ocr_model or settings.llama_model or "").strip()
        )
        if ocr and all(ocr != tag for tag, _ in wanted):
            wanted.append((ocr, "ocr"))
    rows = []
    for tag, role in wanted:
        if not tag:
            continue
        files = (available or {}).get(tag)
        size = files.weights.stat().st_size if files else 0
        rows.append(
            {
                "id": f"llama:{tag}",
                "name": tag,
                "kind": "llama",
                "role": {
                    "translation": "llama.cpp 主翻译模型",
                    "review": "独立候选译文（质量模式第二遍）",
                    "ocr": (
                        "llama.cpp 视觉 OCR（整页读字）"
                        if settings.ocr_backend == "llama"
                        else "视觉 OCR 候选（仍由 Ollama 提供）"
                    ),
                }[role],
                "role_key": f"models.role.{role}",
                "provider": "llama.cpp",
                "repository": tag,
                "size": _human_size(size),
                "license": "Upstream",
                "required": True,
                "ready": files is not None,
                "path": str(files.weights) if files else "",
                "status_code": (
                    "offline"
                    if available is None
                    else ("pulled" if files is not None else "missing")
                ),
            }
        )
    return rows


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    role: str
    provider: str
    repository: str
    size: str
    license: str
    required: bool = True


MODEL_REGISTRY = (
    ModelSpec(
        id="paddleocr",
        name="PaddleOCR",
        role="文字检测与行级识别",
        provider="ModelScope via PaddleX",
        repository="PP-OCRv5_mobile_det + PP-OCRv5_server_rec",
        size="~93 MB",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="manga-ocr",
        name="Manga OCR",
        role="日文漫画识别",
        provider="Hugging Face fallback",
        repository="kha-white/manga-ocr-base",
        size="~850 MB",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="lama",
        name="Big-LaMa",
        role="文字区域背景补全",
        provider="Pinned upstream fallback",
        repository=LAMA_WEIGHTS_URL,
        size="~196 MB",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="hy-mt2",
        name="Hy-MT2 1.8B",
        role="连贯翻译",
        provider="ModelScope",
        repository="Tencent-Hunyuan/Hy-MT2-1.8B",
        size="~4.1 GB",
        license="Apache-2.0",
    ),
)


class ModelManager:
    def __init__(self, paths: AppPaths):
        self.paths = paths.ensure()
        configure_model_caches(paths)

    def model_dir(self, model_id: str) -> Path:
        return self.paths.models / model_id

    def _marker(self, model_id: str) -> Path:
        return self.model_dir(model_id) / ".ready.json"

    def is_ready(self, model_id: str) -> bool:
        marker = self._marker(model_id)
        if not marker.exists():
            return False
        if model_id == "lama" and not self.lama_weights_path().exists():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return payload.get("model_id") == model_id
        except (json.JSONDecodeError, OSError):
            return False

    def status(
        self,
        inference_backend: str = "builtin",
        quality_profile: str = "quality",
        ocr_backend: str = "builtin",
        runtime: OllamaRuntime | None = None,
    ) -> list[dict]:
        local = [
            {
                **asdict(spec),
                "kind": "local",
                "required": (
                    (
                        spec.id != "hy-mt2"
                        or inference_backend == "builtin"
                        or (
                            inference_backend == "ollama"
                            and translate_quality(quality_profile)
                            # Served from Ollama instead of the torch weights.
                            and not (runtime and runtime.hy_mt2_tag)
                        )
                    )
                    and (
                        spec.id != "lama"
                        or cleanup_profile_for(quality_profile) == "quality"
                    )
                    and (
                        # llama.cpp vision OCR still uses the Paddle/MangaOCR
                        # detector for geometry, so those weights stay required.
                        spec.id not in {"paddleocr", "manga-ocr"}
                        or ocr_backend in {"builtin", "hybrid", "llama"}
                    )
                ),
                "ready": self.is_ready(spec.id),
                "path": str(self.model_dir(spec.id)),
            }
            for spec in MODEL_REGISTRY
        ]
        # Rows above describe weights on disk.  The models that actually run
        # this configuration live in Ollama, so they are listed separately
        # instead of pretending Hy-MT2-on-disk is the active translator.
        if inference_backend != "ollama" or runtime is None:
            return local
        return local + ollama_specs(runtime, ocr_backend, quality_profile)

    def lama_weights_path(self) -> Path:
        return self.model_dir("lama") / "big-lama.pt"

    def mark_ready(self, model_id: str, metadata: dict | None = None) -> None:
        directory = self.model_dir(model_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"model_id": model_id, **(metadata or {})}
        self._marker(model_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _require(self, module: str, extra: str = "ml") -> None:
        if importlib.util.find_spec(module) is None:
            raise ModelDependencyError(
                f"Missing optional dependency '{module}'. Run scripts/bootstrap with "
                f"the {extra} profile, or install 'manga-localizer-studio[{extra}]'."
            )

    def download(self, model_id: str, progress: ProgressCallback | None = None) -> Path:
        callback = progress or (lambda _value, _message: None)
        spec = next((item for item in MODEL_REGISTRY if item.id == model_id), None)
        if spec is None:
            raise KeyError(f"Unknown model: {model_id}")
        target = self.model_dir(model_id)
        target.mkdir(parents=True, exist_ok=True)
        callback(5, f"Preparing {spec.name}")

        if model_id == "lama":
            # ModelScope has damo/cv_fft_inpainting_lama, but its legacy CV
            # pipeline pulls a substantially larger dependency graph and is not
            # currently compatible with the supported Python 3.12 environment.
            # Keep a transparent, pinned fallback rather than silently fetching
            # an unversioned binary.
            weights = self.lama_weights_path()
            partial = weights.with_suffix(".pt.part")
            digest = hashlib.sha256()
            request = Request(
                LAMA_WEIGHTS_URL,
                headers={"User-Agent": "Manga-Localizer-Studio/0.3"},
            )
            try:
                with (
                    urlopen(request, timeout=180) as response,
                    partial.open("wb") as handle,
                ):
                    total_bytes = int(response.headers.get("Content-Length", "0"))
                    received = 0
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        if total_bytes:
                            callback(
                                min(90, 10 + round(received / total_bytes * 80)),
                                "Downloading Big-LaMa weights",
                            )
                if digest.hexdigest() != LAMA_WEIGHTS_SHA256:
                    raise RuntimeError("Big-LaMa weight checksum mismatch")
                partial.replace(weights)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        elif model_id == "hy-mt2":
            self._require("modelscope")
            from modelscope import snapshot_download

            snapshot_download(spec.repository, local_dir=str(target))
        elif model_id == "manga-ocr":
            self._require("manga_ocr")
            from manga_ocr import MangaOcr

            callback(
                15, "Hugging Face fallback: manga-ocr has no ModelScope equivalent"
            )
            MangaOcr(force_cpu=True)
        elif model_id == "paddleocr":
            self._require("paddleocr")
            from paddleocr import PaddleOCR

            callback(15, "Downloading PaddleOCR models through ModelScope/PaddleX")
            PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device="cpu",
                enable_mkldnn=False,
            )
        callback(95, f"Verifying {spec.name}")
        self.mark_ready(
            model_id, {"repository": spec.repository, "provider": spec.provider}
        )
        callback(100, f"{spec.name} is ready")
        return target

    def download_all(self, progress: ProgressCallback | None = None) -> None:
        callback = progress or (lambda _value, _message: None)
        total = len(MODEL_REGISTRY)
        for index, spec in enumerate(MODEL_REGISTRY):
            if self.is_ready(spec.id):
                callback(round((index + 1) / total * 100), f"{spec.name} already ready")
                continue
            self.download(
                spec.id,
                lambda value, message, base=index: callback(
                    round((base + value / 100) / total * 100), message
                ),
            )
