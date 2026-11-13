from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .config import (
    DEFAULT_LLAMA_MAIN_PORT,
    DEFAULT_LLAMA_MODEL,
    DEFAULT_LLAMA_REVIEW_MODEL,
    DEFAULT_LLAMA_REVIEW_PORT,
    DEFAULT_LLAMA_VISION_PORT,
    LLAMA_CONTEXT_TOKENS,
    AppPaths,
    UserSettings,
)
from .hardware import apply_cpu_threads
from .name_dictionary import JapaneseNameDictionary
from .ocr import (
    DEFAULT_OLLAMA_VISION_MODEL,
    EXPLICIT_SKIP_REASONS,
    UNRESOLVED_SKIP_REASON,
    HybridMangaOCR,
    LlamaVisionOCR,
    PageOCR,
    OllamaVisionOCR,
    PaddleMangaOCR,
    TextUnit,
    classify_non_story_text,
    likely_sfx_text,
    list_images,
    semantic_sfx_classification,
)
from .llama_server import manager_from_settings, open_slots, open_vision_slot
from .llama_translator import LlamaTranslator
from .renderer import ArtworkPreservingRenderer
from .model_manager import ModelManager
from .translator import (
    DEFAULT_LOCAL_TRANSLATION_MODEL,
    HyMTTranslator,
    LocalQualityTranslator,
    OllamaTranslator,
    OpenAICompatibleTranslator,
    PromptTranslator,
    clean_translations,
    ensure_ollama_model,
)


ProgressCallback = Callable[[str, int, int, str], None]
# 8: recognition payloads now keep Paddle's own text, so every cached page from
# version 7 holds MangaOCR's Japanese hallucination for English sources.
# 9: link blocks are now preserved instead of translated, so a cached credit
# page still carries the un-skippable unit that failed the whole book.
# 10: a text-less detector box only becomes a line when MangaOCR reads it the
# same way from shifted crops, so a cached illustration page still holds the
# invented "２０１８年０"/"その" that was stamped onto the drawing.
# 11: a vertical bubble is re-read together with an uncovered outer column, so a
# cached page still carries the truncated "それなのにわざわざ私に" whose third
# column stayed in Japanese under the Chinese text.
# 12: the tight-crop audit may no longer demote a unit the full page read as
# dialogue, so a cached page still holds the Latin aside ("SUPONGO...") that the
# crop pass labelled artwork_text and left untranslated.
# 13: a page read as Latin now keeps recognized strings below the Japanese
# confidence floor, so a cached page still misses the "'SUP" bubble that Paddle
# read correctly and the 0.72 floor threw away.
OCR_CACHE_VERSION = 13

# Backends whose result depends on a vision model, so a cached page is only
# reusable when the model name matches.  ``llama`` belongs here for the same
# reason as ``ollama``: a different 9B build reads the same page differently.
VISION_OCR_BACKENDS = frozenset({"hybrid", "ollama", "llama"})


def unsafe_semantic_missing(page: PageOCR, preserve_sfx: bool) -> list[dict]:
    """Return full-page findings that cannot safely remain outside exact geometry."""
    unsafe = []
    for item in page.semantic_missing:
        # Regional recovery re-ran the exact detector on an enlarged crop of
        # every claim and still owned no geometry for this one, so there is
        # nothing to erase and no box to place a translation in.  The OCR
        # backend marks those ``unconfirmed``; failing the book over them is
        # what turned a stray screentone read ("効果か:") into a dead run.
        if item.get("unconfirmed"):
            continue
        text = str(item.get("text", "")).strip()
        japanese_chars = re.findall(r"[ぁ-ゖァ-ヺ\u3400-\u9fff]", text)
        single_kana_fragment = len(japanese_chars) == 1 and bool(
            re.fullmatch(r"[ぁ-ゖァ-ヺ]", japanese_chars[0])
        )
        short_vocalization = len(japanese_chars) <= 3 and bool(
            re.search(r"[っッゃゅょャュョ♡♥]", text)
        )
        preservable = (
            single_kana_fragment
            or short_vocalization
            or semantic_sfx_classification(
                text,
                float(item.get("score", 0.0)),
                bool(item.get("is_sfx", False)),
            )
        )
        if not preserve_sfx or not preservable:
            unsafe.append(item)
    return unsafe


def merge_glossaries(
    imported: object, explicit: dict[str, str] | None
) -> dict[str, str]:
    """Merge a transcript glossary with explicit request overrides."""
    imported = imported or {}
    if not isinstance(imported, dict) or not all(
        isinstance(source, str) and isinstance(target, str)
        for source, target in imported.items()
    ):
        raise ValueError("Reviewed transcript glossary must map strings to strings")
    return {**imported, **(explicit or {})}


@dataclass
class PipelineRequest:
    source: Path
    output: Path
    target_language: str = "简体中文"
    source_language: str = "auto"
    story_context: bool = True
    context_pages: int = 6
    preserve_sfx: bool = False
    quality_profile: str = "quality"
    output_format: str = "webp"
    device: str = "auto"
    cpu_threads: int = 0
    glossary: dict[str, str] | None = None
    inference_backend: str = "ollama"
    ocr_backend: str = "hybrid"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_LOCAL_TRANSLATION_MODEL
    ollama_ocr_model: str = DEFAULT_OLLAMA_VISION_MODEL
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    online_api_key: str = ""
    reviewed_transcript: Path | None = None
    # llama.cpp settings.  The model names are Ollama manifest tags, because the
    # manifest is what maps a readable name to the content-addressed GGUF blob
    # that llama-server loads; nothing here talks to the Ollama daemon.
    llama_model: str = DEFAULT_LLAMA_MODEL
    llama_review_model: str = DEFAULT_LLAMA_REVIEW_MODEL
    llama_context_tokens: int = LLAMA_CONTEXT_TOKENS
    llama_main_port: int = DEFAULT_LLAMA_MAIN_PORT
    llama_review_port: int = DEFAULT_LLAMA_REVIEW_PORT
    llama_vision_port: int = DEFAULT_LLAMA_VISION_PORT
    llama_server_path: str = ""
    llama_ollama_root: str = ""


class LocalizerPipeline:
    def __init__(self, paths: AppPaths):
        self.paths = paths.ensure()

    @staticmethod
    def _emit(
        callback: ProgressCallback, phase: str, current: int, total: int, message: str
    ):
        callback(phase, current, total, message)

    @staticmethod
    def _write_transcript(
        path: Path,
        source: Path,
        pages: list[PageOCR],
        glossary: dict[str, str] | None = None,
        target_language: str = "",
    ) -> None:
        payload = {
            "source": str(source),
            # Recorded so a later run can tell whether these lines are still
            # wanted: resuming a draft written for another target language
            # would leave half a book in the wrong one.
            "target_language": target_language,
            "pages": [page.to_dict() for page in pages],
        }
        if glossary:
            payload["glossary"] = glossary
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _resume_translations(
        work: Path, pages: list[PageOCR], target_language: str = ""
    ) -> int:
        """Reuse the lines a previous run already wrote into the draft.

        A long book that dies late -- a crashed server, a cancellation --
        leaves its OCR cache and its draft on disk, and the translation it
        already paid for is worth more than the run.  Geometry still comes
        from this run's OCR pages; only the text is carried over, and only
        when the draft agrees on the page file and the unit id.  The
        translator re-translates anything that no longer satisfies its
        contract, so a stale or partial draft costs a wasted read at most.
        """
        path = work / "translation-draft.json"
        if not path.is_file():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        declared = str(payload.get("target_language") or "")
        if declared and target_language and declared != target_language:
            return 0
        rows = payload.get("pages")
        if not isinstance(rows, list):
            return 0
        saved: dict[str, dict[str, str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("file") or "")
            if not name:
                continue
            saved[name] = {
                str(unit.get("id") or ""): str(unit.get("zh") or "")
                for unit in row.get("units") or []
                if isinstance(unit, dict)
            }
        if not saved:
            return 0
        reused = 0
        for page in pages:
            lines = saved.get(page.file)
            if not lines:
                continue
            for unit in page.units:
                text = lines.get(unit.id) or ""
                if text and not unit.zh:
                    unit.zh = text
                    reused += 1
        return reused

    @staticmethod
    def _load_cached_ocr(
        work: Path,
        image_path: Path,
        index: int,
        ocr_backend: str = "builtin",
        quality_profile: str = "quality",
        ocr_model: str = "",
        source_language: str = "auto",
    ) -> PageOCR | None:
        cache_path = work / f"{image_path.stem}.ocr.json"
        if not cache_path.is_file():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            page = page_from_dict(payload)
            with Image.open(image_path) as image:
                expected_size = image.size
            source_stat = image_path.stat()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if page.page != index or page.file != image_path.name:
            return None
        if (page.width, page.height) != expected_size:
            return None
        metadata = payload.get("_cache")
        if metadata:
            if int(metadata.get("version", -1)) != OCR_CACHE_VERSION:
                return None
            if metadata.get("ocr_backend") != ocr_backend:
                return None
            if metadata.get("quality_profile") != quality_profile:
                return None
            # The declared input language changes which recognizer owns each box,
            # so a page cached under another declaration is not reusable.
            if metadata.get("source_language", "auto") != source_language:
                return None
            if (
                ocr_backend in VISION_OCR_BACKENDS
                and metadata.get("ocr_model") != ocr_model
            ):
                return None
            if int(metadata.get("source_size", -1)) != source_stat.st_size:
                return None
            if int(metadata.get("source_mtime_ns", -1)) != source_stat.st_mtime_ns:
                return None
        else:
            # Legacy caches predate detector-preserving quality OCR and can
            # silently omit valid Japanese bubbles.  They are never safe to
            # reuse after the cache contract changed.
            return None
        # Text-only classification improvements are safe to apply to an OCR
        # cache without repeating model inference.  This keeps expensive OCR
        # reusable while preventing legacy mixed-kana sound effects from being
        # sent to the translator as dialogue.
        for unit in page.units:
            unit.is_sfx = semantic_sfx_classification(
                unit.ja, unit.score, unit.is_sfx or likely_sfx_text(unit.ja)
            )
        return page

    def run(
        self, request: PipelineRequest, callback: ProgressCallback | None = None
    ) -> dict:
        emit = callback or (lambda _phase, _current, _total, _message: None)
        # Every recognizer in this run sizes its own thread pool, so the budget
        # has to be in place before the first model is constructed.
        apply_cpu_threads(request.cpu_threads)
        source = request.source.expanduser().resolve()
        output = request.output.expanduser().resolve()
        if source == output:
            raise ValueError("Output directory must not be the source directory.")
        files = list_images(source)
        if not files:
            raise ValueError("No supported images were found in the source folder.")
        output.mkdir(parents=True, exist_ok=True)
        work = output / ".manga-localizer-work"
        work.mkdir(parents=True, exist_ok=True)

        pages: list[PageOCR] = []
        glossary = dict(request.glossary or {})
        total = len(files)
        if request.reviewed_transcript:
            transcript_path = request.reviewed_transcript.expanduser().resolve()
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            # The transcript is the portable source of truth for a reviewed
            # book. Explicit request terms still win when callers intentionally
            # override one spelling for a new render.
            glossary = merge_glossaries(payload.get("glossary"), glossary)
            pages = [page_from_dict(item) for item in payload["pages"]]
            self._validate_reviewed_pages(files, pages)
            self._emit(
                emit, "ocr", total, total, f"已导入审校稿 {transcript_path.name}"
            )
        else:
            cached_pages = [
                self._load_cached_ocr(
                    work,
                    image_path,
                    index,
                    request.ocr_backend,
                    request.quality_profile,
                    (
                        request.ollama_ocr_model
                        if request.ocr_backend in VISION_OCR_BACKENDS
                        else ""
                    ),
                    request.source_language,
                )
                for index, image_path in enumerate(files, start=1)
            ]
            title_refinements = [
                bool(
                    request.ocr_backend in {"builtin", "hybrid", "llama"}
                    and page is not None
                    and PaddleMangaOCR.is_cover_title_candidate(page)
                )
                for page in cached_pages
            ]
            ocr = None
            if any(page is None for page in cached_pages) or any(title_refinements):
                if request.ocr_backend == "builtin":
                    ocr = PaddleMangaOCR(
                        self.paths,
                        request.device,
                        request.quality_profile,
                        request.source_language,
                    )
                elif request.ocr_backend == "hybrid":
                    self._emit(emit, "ocr", 0, total, "检查本地视觉语义模型")
                    ensure_ollama_model(
                        request.ollama_base_url, request.ollama_ocr_model
                    )
                    ocr = HybridMangaOCR(
                        self.paths,
                        request.ollama_base_url,
                        request.ollama_ocr_model,
                        request.device,
                        request.quality_profile,
                        request.source_language,
                    )
                elif request.ocr_backend == "llama":
                    # Same geometry+semantics split as hybrid; only the vision
                    # transport changes, so an A/B against hybrid differs in
                    # exactly one variable.
                    ocr = HybridMangaOCR(
                        self.paths,
                        "",
                        source_language=request.source_language,
                        device=request.device,
                        profile=request.quality_profile,
                        semantics=self._llama_vision(request, emit),
                    )
                elif request.ocr_backend == "ollama":
                    ocr = OllamaVisionOCR(
                        request.ollama_base_url,
                        request.ollama_ocr_model,
                        source_language=request.source_language,
                    )
                else:
                    raise ValueError(f"Unknown OCR backend: {request.ocr_backend}")
            for index, (image_path, cached_page) in enumerate(
                zip(files, cached_pages), start=1
            ):
                cache_updated = False
                if cached_page is not None:
                    page = cached_page
                    if title_refinements[index - 1]:
                        page = ocr.refine_cover_title(image_path, page)
                        cache_updated = page is not cached_page
                    self._emit(
                        emit, "ocr", index - 1, total, f"复用 OCR {image_path.name}"
                    )
                else:
                    self._emit(emit, "ocr", index - 1, total, f"识别 {image_path.name}")
                    page = ocr.analyze(image_path, index)
                # Every OCR backend shares the same non-story contract.  This
                # also upgrades reusable OCR caches without rerunning models.
                page = classify_non_story_text(page)
                unsafe_missing = unsafe_semantic_missing(page, request.preserve_sfx)
                if unsafe_missing:
                    texts = " / ".join(
                        str(item.get("text", "")).strip() or "未识别文字"
                        for item in unsafe_missing[:4]
                    )
                    raise RuntimeError(
                        f"第 {index} 页发现未被精确检测框覆盖的文字：{texts}。"
                        "质量模式拒绝猜测擦除区域，请更换 OCR 模型或重试。"
                    )
                pages.append(page)
                if cached_page is None or cache_updated:
                    page_payload = page.to_dict()
                    source_stat = image_path.stat()
                    page_payload["_cache"] = {
                        "version": OCR_CACHE_VERSION,
                        "ocr_backend": request.ocr_backend,
                        "quality_profile": request.quality_profile,
                        "ocr_model": (
                            request.ollama_ocr_model
                            if request.ocr_backend in VISION_OCR_BACKENDS
                            else ""
                        ),
                        "source_size": source_stat.st_size,
                        "source_mtime_ns": source_stat.st_mtime_ns,
                        "source_language": request.source_language,
                    }
                    (work / f"{image_path.stem}.ocr.json").write_text(
                        json.dumps(page_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                self._emit(emit, "ocr", index, total, f"已识别 {image_path.name}")

        resumed = self._resume_translations(work, pages, request.target_language)
        if resumed:
            # The draft of an interrupted run is worth more than the run.
            self._emit(
                emit,
                "translate",
                0,
                total,
                f"续用上次未完成的草稿：{resumed} 处译文",
            )

        needs_translation = any(
            not unit.skip
            and not (request.preserve_sfx and unit.is_sfx)
            and not PromptTranslator._valid_translation(unit, unit.zh)
            for page in pages
            for unit in page.units
        )
        self._emit(
            emit,
            "translate",
            0,
            total,
            "加载翻译后端" if needs_translation else "审校稿已包含译文",
        )
        if not needs_translation:
            translator = None
        elif request.inference_backend == "builtin":
            translator = HyMTTranslator(
                self.paths.models / "hy-mt2",
                request.target_language,
                request.device,
                JapaneseNameDictionary(self.paths.cache),
            )
        elif request.inference_backend == "ollama":
            if not request.ollama_model.strip():
                raise ValueError("Ollama model name is required")
            self._emit(emit, "translate", 0, total, "检查本地解除限制翻译模型")
            downloaded = ensure_ollama_model(
                request.ollama_base_url, request.ollama_model
            )
            if downloaded:
                self._emit(emit, "translate", 0, total, "本地翻译模型下载完成")
            editor = OllamaTranslator(
                request.ollama_base_url,
                request.ollama_model,
                request.target_language,
                JapaneseNameDictionary(self.paths.cache),
            )
            if request.quality_profile == "quality":
                model_manager = ModelManager(self.paths)
                if not model_manager.is_ready("hy-mt2"):
                    self._emit(emit, "translate", 0, total, "下载 Hy-MT2 独立翻译候选")
                    model_manager.download("hy-mt2")
                # Serving the same Hy-MT2 weights through Ollama keeps the
                # candidate pass on the GPU; torch on this CPU spent 172 s of a
                # 642 s run here.  Opt-in via MLS_HY_MT2_OLLAMA=<ollama tag>.
                hy_mt2_tag = os.environ.get("MLS_HY_MT2_OLLAMA", "").strip()
                if hy_mt2_tag:
                    candidate: PromptTranslator = OllamaTranslator(
                        request.ollama_base_url,
                        hy_mt2_tag,
                        request.target_language,
                        JapaneseNameDictionary(self.paths.cache),
                    )
                else:
                    candidate = HyMTTranslator(
                        self.paths.models / "hy-mt2",
                        request.target_language,
                        request.device,
                        JapaneseNameDictionary(self.paths.cache),
                    )
                self._emit(emit, "translate", 0, total, "启用 9B 分阶段语义审校")
                translator = LocalQualityTranslator(editor, candidate)
            else:
                translator = editor
        elif request.inference_backend == "online":
            if not request.online_model.strip():
                raise ValueError("Online model name is required")
            translator = OpenAICompatibleTranslator(
                request.online_base_url,
                request.online_model,
                request.online_api_key,
                request.target_language,
                JapaneseNameDictionary(self.paths.cache),
            )
        elif request.inference_backend == "llama":
            translator = self._llama_translator(request, emit, total)
        else:
            raise ValueError(f"Unknown inference backend: {request.inference_backend}")
        if translator is not None:
            try:
                translator.translate_pages(
                    pages,
                    context_pages=max(0, min(12, request.context_pages)),
                    story_context=request.story_context,
                    preserve_sfx=request.preserve_sfx,
                    glossary=glossary,
                    progress=lambda current, count: self._emit(
                        emit,
                        "translate",
                        current,
                        count,
                        f"翻译质量进度 {current}/{count}",
                    ),
                )
            finally:
                self._write_transcript(
                    work / "translation-draft.json",
                    source,
                    pages,
                    translator.resolved_glossary,
                    request.target_language,
                )
        else:
            self._write_transcript(
                work / "translation-draft.json",
                source,
                pages,
                glossary,
                request.target_language,
            )
        if translator is not None:
            summarise = getattr(translator, "collapsed_summary", None)
            collapsed_chunks, collapsed_units = summarise() if summarise else (0, 0)
            if collapsed_chunks:
                # Staying quiet here reads as "the translator failed on these
                # pages".  It is the server that stopped producing words, and the
                # pages kept their source text, so the job says which it was.
                self._emit(
                    emit,
                    "translate",
                    total,
                    total,
                    f"推理服务输出退化，{collapsed_chunks} 个批次"
                    f"（{collapsed_units} 处）已跳过并保留原文",
                )
        # Every backend, staged role and resumed draft can leave the prompt's
        # ``[拟声词]`` label inside the answer; strip it once here instead of
        # teaching the renderer about translation instructions.
        relabelled = clean_translations(pages)
        if relabelled:
            self._emit(
                emit,
                "translate",
                total,
                total,
                f"{relabelled} 处译文里的说明标签已剥离",
            )
        quality = completion_summary(pages, request.preserve_sfx)
        fallback_ids = preserve_unresolved_units(pages, quality)
        if fallback_ids:
            self._emit(
                emit,
                "translate",
                total,
                total,
                f"{len(fallback_ids)} 处未能翻译，已保留原文"
                f"：{', '.join(fallback_ids[:6])}",
            )
            quality = completion_summary(pages, request.preserve_sfx)
            quality["fallback_preserved_units"] = len(fallback_ids)
            quality["fallback_preserved_ids"] = fallback_ids
        self._emit(emit, "translate", total, total, "连贯翻译完成")
        self._write_transcript(
            work / "transcript.json",
            source,
            pages,
            translator.resolved_glossary
            if translator is not None
            else glossary,
        )

        if request.quality_profile == "quality":
            manager = ModelManager(self.paths)
            if not manager.lama_weights_path().exists():
                self._emit(emit, "render", 0, total, "首次使用：下载 LaMa 补画模型")
                manager.download(
                    "lama",
                    lambda value, message: self._emit(
                        emit, "render", 0, total, f"{message} {value}%"
                    ),
                )
        manifest = []
        if request.output_format not in {"webp", "png"}:
            raise ValueError(f"Unknown lossless output format: {request.output_format}")
        renderer = ArtworkPreservingRenderer(
            cleanup_profile=request.quality_profile,
            paths=self.paths,
            device=request.device,
        )
        suffix = ".webp" if request.output_format == "webp" else ".png"
        for index, page in enumerate(pages, start=1):
            self._emit(emit, "render", index - 1, total, f"替换第 {index} 页文字")
            source_path = source / page.file
            output_path = output / f"{source_path.stem}{suffix}"
            manifest.append(
                renderer.render_page(
                    source_path, page, output_path, request.preserve_sfx
                )
            )
            self._emit(emit, "render", index, total, f"已生成 {output_path.name}")

        result = {
            "source": str(source),
            "output": str(output),
            "pages": total,
            "images": manifest,
            "transcript": str(work / "transcript.json"),
            "quality": quality,
        }
        (output / "translation_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._emit(emit, "complete", total, total, "本地化完成")
        return result

    def _llama_translator(
        self,
        request: PipelineRequest,
        emit: ProgressCallback,
        total: int,
    ) -> PromptTranslator:
        """Build the translator backed by local ``llama-server`` processes.

        The servers are started here rather than by the caller because the
        pipeline is the only place that knows whether the translation stage is
        needed at all: a folder whose transcript is already reviewed must not
        pay for a 7 GB model load.
        """
        manager = manager_from_settings(self.paths, request)
        notes = getattr(manager, "port_notes", lambda: {})()
        if notes:
            self._emit(
                emit, "translate", 0, total, "；".join(notes.values())
            )
        self._emit(emit, "translate", 0, total, "启动 llama.cpp 推理服务")
        slots = open_slots(
            manager,
            request,
            emit=lambda message: self._emit(emit, "translate", 0, total, message),
        )
        review_slot = slots.get("review")
        # A missing candidate slot with a registered recipe means the pair does
        # not fit on the card: both translators then release their server at the
        # end of their pass and start it again for the next one, so the editor
        # and the candidate are never resident together.
        staged = review_slot is None and manager.can_relaunch("review")
        names = JapaneseNameDictionary(self.paths.cache)
        editor = LlamaTranslator(
            slots["editor"],
            request.target_language,
            names,
            manager=manager,
            role="editor",
            label=request.llama_model,
            evict=staged,
        )
        if request.quality_profile != "quality":
            return editor
        if staged:
            self._emit(
                emit, "translate", 0, total, "主模型与审校模型交替使用显存"
            )
            return LocalQualityTranslator(
                editor,
                LlamaTranslator(
                    None,
                    request.target_language,
                    names,
                    manager=manager,
                    role="review",
                    label=request.llama_review_model,
                    evict=True,
                ),
            )
        if review_slot is None or review_slot is slots["editor"]:
            # Without a separate 1.8B instance the quality passes would all run
            # on the 9B, which is exactly what the Ollama path avoided; the
            # candidate is optional so a missing review model degrades to a
            # single-editor run instead of failing the book.
            self._emit(
                emit, "translate", 0, total, "未配置审校模型，仅使用主模型翻译"
            )
            return editor
        candidate = LlamaTranslator(
            review_slot,
            request.target_language,
            names,
            manager=manager,
            role="review",
            label=request.llama_review_model,
        )
        # The editor is not always the 9B: the message names what actually runs.
        self._emit(
            emit,
            "translate",
            0,
            total,
            f"启用分阶段语义审校：{request.llama_model} + {request.llama_review_model}",
        )
        return LocalQualityTranslator(editor, candidate)

    def _llama_vision(
        self,
        request: PipelineRequest,
        emit: ProgressCallback,
    ) -> LlamaVisionOCR:
        """The vision-OCR adapter backed by a local ``llama-server``.

        Started lazily from the OCR stage because a book that reuses its whole
        OCR cache never needs the server at all, and loading a 7 GB model for
        nothing is the one cost a cached page must not pay.
        """
        manager = manager_from_settings(self.paths, request)
        notes = getattr(manager, "port_notes", lambda: {})()
        if notes:
            self._emit(emit, "ocr", 0, 0, "；".join(notes.values()))
        slot = open_vision_slot(
            manager,
            request,
            emit=lambda message: self._emit(emit, "ocr", 0, 0, message),
        )
        return LlamaVisionOCR(
            slot.base_url,
            slot.alias,
            source_language=request.source_language,
            context_tokens=slot.context,
        )

    @staticmethod
    def _validate_reviewed_pages(files: list[Path], pages: list[PageOCR]) -> None:
        if len(files) != len(pages):
            raise ValueError(
                f"Reviewed transcript has {len(pages)} pages, source folder has {len(files)}"
            )
        for index, (source, page) in enumerate(zip(files, pages), start=1):
            if source.name != page.file:
                raise ValueError(
                    f"Reviewed transcript page {index} expects {page.file}, got {source.name}"
                )
            with Image.open(source) as image:
                if image.size != (page.width, page.height):
                    raise ValueError(
                        f"Reviewed transcript page {index} dimensions do not match source"
                    )
        ambiguous = [
            unit.id
            for page in pages
            for unit in page.units
            if unit.skip and unit.skip_reason not in EXPLICIT_SKIP_REASONS
        ]
        if ambiguous:
            sample = ", ".join(ambiguous[:8])
            raise ValueError(
                f"Reviewed transcript has {len(ambiguous)} ambiguous skipped units"
                f" ({sample}). Legacy skip=true is unresolved; use one of: "
                f"{', '.join(sorted(EXPLICIT_SKIP_REASONS))}."
            )


def request_from_settings(
    source: Path, output: Path, settings: UserSettings
) -> PipelineRequest:
    return PipelineRequest(
        source=source,
        output=output,
        target_language=settings.target_language,
        source_language=settings.source_language,
        story_context=settings.story_context,
        context_pages=settings.context_pages,
        preserve_sfx=settings.preserve_sfx,
        quality_profile=settings.quality_profile,
        output_format=settings.output_format,
        device=settings.device,
        cpu_threads=settings.cpu_threads,
        inference_backend=settings.inference_backend,
        ocr_backend=settings.ocr_backend,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        ollama_ocr_model=settings.ollama_ocr_model,
        online_base_url=settings.online_base_url,
        online_model=settings.online_model,
        llama_model=settings.llama_model,
        llama_review_model=settings.llama_review_model,
        llama_context_tokens=settings.llama_context_tokens,
        llama_main_port=settings.llama_main_port,
        llama_review_port=settings.llama_review_port,
        llama_vision_port=settings.llama_vision_port,
        llama_server_path=settings.llama_server_path,
        llama_ollama_root=settings.llama_ollama_root,
    )


def page_from_dict(payload: dict) -> PageOCR:
    return PageOCR(
        page=payload["page"],
        file=payload["file"],
        width=payload["width"],
        height=payload["height"],
        units=[
            TextUnit(
                id=unit["id"],
                bbox=unit["bbox"],
                crop_bbox=unit.get("crop_bbox", unit["bbox"]),
                ja=unit.get("ja", ""),
                score=float(unit.get("score", max(unit.get("paddle_scores") or [0.0]))),
                is_sfx=bool(unit.get("is_sfx", False)),
                zh=unit.get("zh", ""),
                skip=bool(unit.get("skip", False)),
                skip_reason=(
                    str(unit.get("skip_reason", "")).strip()
                    or (UNRESOLVED_SKIP_REASON if bool(unit.get("skip", False)) else "")
                ),
                erase_boxes=unit.get("erase_boxes", []),
                translation_attempts=list(unit.get("translation_attempts", [])),
                special=str(unit.get("special", "")).strip(),
            )
            for unit in payload["units"]
        ],
        semantic_missing=list(payload.get("semantic_missing", [])),
    )


def preserve_unresolved_units(pages: list[PageOCR], summary: dict) -> list[str]:
    """Keep the printed source for units that exhausted the translation ladder.

    A heavily merged English block can defeat even the eight-attempt retry
    ladder: the model never produces a candidate that satisfies the audit
    contract.  The upstream contract failed the whole book over a handful of
    print-sized lines, which threw away half an hour of finished work, so those
    units stay as printed and every id is reported back to the caller.
    """
    unresolved = sorted(set(summary.get("unresolved_ids", [])))
    if not unresolved:
        return []
    ids = set(unresolved)
    for page in pages:
        for unit in page.units:
            if unit.id in ids:
                unit.zh = ""
                unit.skip = True
                unit.skip_reason = "preserve"
                unit.special = "unresolved_fallback"
    return unresolved


def completion_summary(pages: list[PageOCR], preserve_sfx: bool = False) -> dict:
    """Classify every OCR unit; unresolved work must remain visible to callers."""
    translated = explicit_skips = preserved_sfx_count = 0
    unresolved_ids: list[str] = []
    invalid_translation_ids: list[str] = []
    for page in pages:
        for unit in page.units:
            if unit.skip:
                if unit.skip_reason in EXPLICIT_SKIP_REASONS:
                    explicit_skips += 1
                else:
                    unresolved_ids.append(unit.id)
            elif preserve_sfx and unit.is_sfx:
                preserved_sfx_count += 1
            elif unit.zh.strip():
                if PromptTranslator._valid_translation(unit, unit.zh):
                    translated += 1
                else:
                    invalid_translation_ids.append(unit.id)
                    unresolved_ids.append(unit.id)
            else:
                unresolved_ids.append(unit.id)
    return {
        "total_units": translated
        + explicit_skips
        + preserved_sfx_count
        + len(unresolved_ids),
        "translated_units": translated,
        "explicitly_skipped_units": explicit_skips,
        "preserved_sfx_units": preserved_sfx_count,
        "unresolved_units": len(unresolved_ids),
        "unresolved_ids": unresolved_ids,
        "invalid_translation_units": len(invalid_translation_ids),
        "invalid_translation_ids": invalid_translation_ids,
    }
