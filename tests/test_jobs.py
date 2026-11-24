"""What a job record keeps, so a run can be replayed without changing it."""

from __future__ import annotations

from pathlib import Path

from manga_localizer.config import AppPaths
from manga_localizer.jobs import JobStore
from manga_localizer.pipeline import PipelineRequest


def _paths(tmp_path: Path) -> AppPaths:
    root = tmp_path / "home"
    return AppPaths(root, root / "models", root / "cache", root / "jobs", root / "settings.json")


def test_a_replayed_job_keeps_its_source_language(tmp_path: Path) -> None:
    """The record is what a re-run is rebuilt from.

    Leaving source_language out made every replay fall back to "auto", which
    does not match the language the OCR cache was written for, so the whole
    book was re-OCR'd before a single line was translated.
    """
    store = JobStore(_paths(tmp_path).jobs)
    source = tmp_path / "src"
    source.mkdir()
    request = PipelineRequest(
        source=source,
        output=tmp_path / "out",
        source_language="en",
        ocr_backend="builtin",
        inference_backend="builtin",
        quality_profile="fast",
    )

    job = store.create(request)
    stored = store.get(job["id"])

    assert stored is not None
    assert stored["request"]["source_language"] == "en"
    assert stored["request"]["source_language"] != "auto"
