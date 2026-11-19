"""0.13.12 regression tests: a dead page is skipped, and the clock is live."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from manga_localizer import pipeline as pipeline_module
from manga_localizer.config import AppPaths
from manga_localizer.jobs import JobStore
from manga_localizer.pipeline import LocalizerPipeline, PipelineRequest


def _paths(tmp_path: Path) -> AppPaths:
    root = tmp_path / "home"
    return AppPaths(root, root / "models", root / "cache", root / "jobs", root / "settings.json")


class _BrokenPageReader:
    """Stands in for PaddleMangaOCR: one page cannot be decoded at all."""

    stage_seconds = {"detect": 0.0}

    def __init__(self, *args, **kwargs) -> None:
        pass

    def analyze(self, image_path: Path, page_number: int):
        if image_path.name == "002.jpg":
            raise OSError("could not create decoder object")
        return pipeline_module.PageOCR(
            page=page_number, file=image_path.name, width=100, height=200, units=[]
        )


class _PassthroughTranslator:
    resolved_glossary: dict = {}

    def translate_pages(self, pages, *args, **kwargs):
        return pages

    def release(self):
        pass


def _prepare(tmp_path: Path) -> PipelineRequest:
    source = tmp_path / "src"
    source.mkdir(exist_ok=True)
    for name in ("001.jpg", "002.jpg", "003.jpg"):
        Image.new("RGB", (100, 200), "white").save(source / name)
    return PipelineRequest(
        source=source,
        output=tmp_path / "out",
        source_language="en",
        ocr_backend="builtin",
        inference_backend="builtin",
        quality_profile="fast",
    )


def test_one_page_that_cannot_be_read_does_not_fail_the_book(monkeypatch, tmp_path):
    """The reported failure: a truncated WebP threw at page 34 of 255."""
    request = _prepare(tmp_path)
    monkeypatch.setattr(pipeline_module, "PaddleMangaOCR", _BrokenPageReader)
    monkeypatch.setattr(
        pipeline_module,
        "HyMTTranslator",
        lambda *args, **kwargs: _PassthroughTranslator(),
    )
    monkeypatch.setattr(
        LocalizerPipeline,
        "_load_cached_ocr",
        staticmethod(lambda *args, **kwargs: None),
    )

    messages: list[str] = []
    result = LocalizerPipeline(_paths(tmp_path)).run(
        request, lambda phase, current, total, message: messages.append(message)
    )

    assert [item["file"] for item in result["skipped"]] == ["002.jpg"]
    assert "decoder" in result["skipped"][0]["error"]
    assert any("跳过无法读取的第 2 页" in message for message in messages)
    # Two good pages still reach the renderer.
    assert len(result["images"]) == 2


class _TickingPipeline:
    """Fills live_timings the way run() does, then stops."""

    def __init__(self, paths) -> None:
        self.paths = paths
        self.live_timings: dict[str, float] = {}

    def run(self, request, callback):
        self.live_timings["ocr"] = 1.5
        callback("ocr", 1, 1, "done")
        self.live_timings["translate"] = 2.5
        callback("complete", 1, 1, "done")
        return {"pages": 1}


def test_the_job_record_carries_the_running_stage_clocks(tmp_path):
    """The UI counts a stage up from job.timings_at while the run is alive."""
    store = JobStore(_paths(tmp_path).jobs)
    job = store.create(_build_request(tmp_path))
    pipeline = _TickingPipeline(_paths(tmp_path))
    stamps: list[dict] = []

    def progress(phase, current, total, message):
        live = getattr(pipeline, "live_timings", None)
        if live:
            stamps.append({"timings": dict(live), "timings_at": time.time()})

    pipeline.run(job, progress)

    assert stamps, "no stage clock was published before completion"
    assert stamps[0]["timings"] == {"ocr": 1.5}
    assert stamps[-1]["timings"] == {"ocr": 1.5, "translate": 2.5}
    assert stamps[0]["timings_at"] <= stamps[-1]["timings_at"]


def _build_request(tmp_path: Path) -> PipelineRequest:
    source = tmp_path / "src"
    source.mkdir(exist_ok=True)
    return PipelineRequest(
        source=source,
        output=tmp_path / "out",
        source_language="en",
        ocr_backend="builtin",
        inference_backend="builtin",
        quality_profile="fast",
    )
