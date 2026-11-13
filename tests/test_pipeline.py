import json

import pytest
from PIL import Image

from manga_localizer.config import AppPaths, LLAMA_SINGLE_MODEL
from manga_localizer.llama_server import LlamaSlot
from manga_localizer.llama_translator import LlamaTranslator
from manga_localizer.pipeline import (
    OCR_CACHE_VERSION,
    LocalizerPipeline,
    PipelineRequest,
    completion_summary,
    merge_glossaries,
    page_from_dict,
    preserve_unresolved_units,
    unsafe_semantic_missing,
)


def test_unresolved_units_are_preserved_instead_of_failing_the_book():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [
                {"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "FINALLY", "zh": ""},
                {"id": "p001u02", "bbox": [40, 2, 70, 40], "ja": "THANKS", "zh": "谢谢"},
            ],
        }
    )
    summary = completion_summary([page])
    assert summary["unresolved_ids"] == ["p001u01"]

    assert preserve_unresolved_units([page], {"unresolved_ids": []}) == []
    assert preserve_unresolved_units([page], summary) == ["p001u01"]

    assert page.units[0].zh == ""
    assert page.units[0].skip is True
    assert page.units[0].skip_reason == "preserve"
    assert page.units[0].special == "unresolved_fallback"
    followed = completion_summary([page])
    assert followed["unresolved_units"] == 0
    assert followed["explicitly_skipped_units"] == 1
    assert followed["translated_units"] == 1


def test_reviewed_transcript_glossary_survives_with_explicit_overrides():
    assert merge_glossaries(
        {"サナ": "佐奈", "ガク": "岳"}, {"ガク": "学"}
    ) == {"サナ": "佐奈", "ガク": "学"}
    with pytest.raises(ValueError, match="map strings to strings"):
        merge_glossaries(["サナ", "佐奈"], None)


def test_legacy_skip_is_imported_as_unresolved_instead_of_completed():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [
                {
                    "id": "p001u01",
                    "bbox": [1, 2, 30, 40],
                    "ja": "秘密基地",
                    "zh": "秘密基地",
                    "paddle_scores": [0.9],
                },
                {
                    "id": "p001u02",
                    "bbox": [40, 2, 70, 40],
                    "ja": "ノイズ",
                    "zh": "噪点",
                    "skip": True,
                },
            ],
        }
    )
    assert page.units[0].score == 0.9
    assert page.units[0].crop_bbox == page.units[0].bbox
    assert page.units[1].zh == "噪点"
    assert page.units[1].skip is True
    assert page.units[1].skip_reason == "unresolved"
    assert completion_summary([page])["unresolved_ids"] == ["p001u02"]


def test_explicit_skip_reason_is_counted_as_reviewed():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [
                {
                    "id": "p001u01",
                    "bbox": [1, 2, 30, 40],
                    "ja": "ノイズ",
                    "skip": True,
                    "skip_reason": "noise",
                }
            ],
        }
    )
    summary = completion_summary([page])
    assert summary["explicitly_skipped_units"] == 1
    assert summary["unresolved_units"] == 0


def test_reviewed_transcript_rejects_ambiguous_legacy_skip(tmp_path):
    source = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(source)
    page = page_from_dict(
        {
            "page": 1,
            "file": source.name,
            "width": 100,
            "height": 200,
            "units": [
                {
                    "id": "p001u01",
                    "bbox": [1, 2, 30, 40],
                    "ja": "台詞",
                    "skip": True,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="ambiguous skipped units"):
        LocalizerPipeline._validate_reviewed_pages([source], [page])


def test_japanese_left_in_translation_is_unresolved():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [
                {
                    "id": "p001u01",
                    "bbox": [1, 2, 30, 40],
                    "ja": "ニュプ",
                    "zh": "ニュプ",
                }
            ],
        }
    )
    summary = completion_summary([page])
    assert summary["unresolved_ids"] == ["p001u01"]
    assert summary["invalid_translation_ids"] == ["p001u01"]


def test_valid_cached_ocr_page_is_reused(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [{"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "秘密"}],
        "_cache": {
            "version": OCR_CACHE_VERSION,
            "ocr_backend": "builtin",
            "quality_profile": "quality",
            "source_size": image_path.stat().st_size,
            "source_mtime_ns": image_path.stat().st_mtime_ns,
        },
    }
    (work / "001.ocr.json").write_text(
        json.dumps(cached, ensure_ascii=False), encoding="utf-8"
    )
    page = LocalizerPipeline._load_cached_ocr(work, image_path, 1)
    assert page is not None
    assert page.units[0].ja == "秘密"


def test_cached_mixed_kana_sound_effect_gets_current_classification(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [
            {
                "id": "p001u01",
                "bbox": [1, 2, 30, 40],
                "ja": "そのカキノカキ",
                "is_sfx": False,
            }
        ],
        "_cache": {
            "version": OCR_CACHE_VERSION,
            "ocr_backend": "builtin",
            "quality_profile": "quality",
            "source_size": image_path.stat().st_size,
            "source_mtime_ns": image_path.stat().st_mtime_ns,
        },
    }
    (work / "001.ocr.json").write_text(
        json.dumps(cached, ensure_ascii=False), encoding="utf-8"
    )
    page = LocalizerPipeline._load_cached_ocr(work, image_path, 1)
    assert page is not None
    assert page.units[0].is_sfx is True


def test_stale_cached_ocr_page_is_rejected(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 2,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
    }
    (work / "001.ocr.json").write_text(json.dumps(cached), encoding="utf-8")
    assert LocalizerPipeline._load_cached_ocr(work, image_path, 1) is None


def test_legacy_cache_is_not_reused_for_a_different_ocr_profile(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
    }
    (work / "001.ocr.json").write_text(json.dumps(cached), encoding="utf-8")
    assert (
        LocalizerPipeline._load_cached_ocr(work, image_path, 1, "ollama", "quality")
        is None
    )


def test_fingerprinted_ocr_cache_matches_the_source_and_backend(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    source_stat = image_path.stat()
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
        "_cache": {
            "version": OCR_CACHE_VERSION,
            "ocr_backend": "builtin",
            "quality_profile": "quality",
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
        },
    }
    cache_path = work / "001.ocr.json"
    cache_path.write_text(json.dumps(cached), encoding="utf-8")
    assert LocalizerPipeline._load_cached_ocr(work, image_path, 1) is not None

    cached["_cache"]["source_size"] += 1
    cache_path.write_text(json.dumps(cached), encoding="utf-8")
    assert LocalizerPipeline._load_cached_ocr(work, image_path, 1) is None


def test_old_fingerprinted_ocr_cache_version_is_rejected(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    source_stat = image_path.stat()
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
        "_cache": {
            "version": OCR_CACHE_VERSION - 1,
            "ocr_backend": "builtin",
            "quality_profile": "quality",
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
        },
    }
    (work / "001.ocr.json").write_text(json.dumps(cached), encoding="utf-8")
    assert LocalizerPipeline._load_cached_ocr(work, image_path, 1) is None


def test_hybrid_cache_is_bound_to_the_exact_vision_model(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
        "_cache": {
            "version": OCR_CACHE_VERSION,
            "ocr_backend": "hybrid",
            "quality_profile": "quality",
            "ocr_model": "vision-a",
            "source_size": image_path.stat().st_size,
            "source_mtime_ns": image_path.stat().st_mtime_ns,
        },
    }
    (work / "001.ocr.json").write_text(json.dumps(cached), encoding="utf-8")
    assert (
        LocalizerPipeline._load_cached_ocr(
            work, image_path, 1, "hybrid", "quality", "vision-a"
        )
        is not None
    )
    assert (
        LocalizerPipeline._load_cached_ocr(
            work, image_path, 1, "hybrid", "quality", "vision-b"
        )
        is None
    )


def test_unboxed_dialogue_fails_but_preserved_sfx_is_safe():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [],
            "semantic_missing": [
                {"text": "台詞", "is_sfx": False},
                {"text": "ドン", "is_sfx": True},
                {"text": "と", "is_sfx": False, "score": 0.9},
                {"text": "き…", "is_sfx": False, "score": 0.9},
                {"text": "あっ♡", "is_sfx": False, "score": 0.9},
                {"text": "私", "is_sfx": False, "score": 0.9},
                {"text": "その", "is_sfx": False, "score": 0.9},
            ],
        }
    )
    assert [item["text"] for item in unsafe_semantic_missing(page, True)] == [
        "台詞",
        "私",
        "その",
    ]
    assert [item["text"] for item in unsafe_semantic_missing(page, False)] == [
        "台詞",
        "ドン",
        "と",
        "き…",
        "あっ♡",
        "私",
        "その",
    ]


def test_a_draft_from_an_interrupted_run_is_reused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "translation-draft.json").write_text(
        json.dumps(
            {
                "source": "src",
                "target_language": "简体中文",
                "pages": [
                    {
                        "page": 1,
                        "file": "001.jpg",
                        "units": [
                            {"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "FINALLY", "zh": "终于"},
                            {"id": "p001u02", "bbox": [40, 2, 70, 40], "ja": "THANKS", "zh": "谢谢"},
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [
                {"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "FINALLY", "zh": ""},
                {"id": "p001u02", "bbox": [40, 2, 70, 40], "ja": "THANKS", "zh": ""},
            ],
        }
    )
    # The geometry of this run wins; only the text is carried over.
    assert LocalizerPipeline._resume_translations(work, [page], "简体中文") == 2
    assert [unit.zh for unit in page.units] == ["终于", "谢谢"]


def test_a_draft_for_another_target_language_is_ignored(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "translation-draft.json").write_text(
        json.dumps(
            {
                "target_language": "繁體中文",
                "pages": [
                    {
                        "page": 1,
                        "file": "001.jpg",
                        "units": [{"id": "p001u01", "zh": "終於"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [{"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "FINALLY", "zh": ""}],
        }
    )
    # Half a book in the wrong language is worse than re-translating it.
    assert LocalizerPipeline._resume_translations(work, [page], "简体中文") == 0
    assert page.units[0].zh == ""


def _pipeline_paths(tmp_path):
    root = tmp_path / "home"
    return AppPaths(
        root,
        root / "models",
        root / "cache",
        root / "jobs",
        root / "settings.json",
    )


def test_the_single_model_backend_pins_one_file_and_opens_no_candidate(monkeypatch, tmp_path):
    from manga_localizer import pipeline as pipeline_module

    slot = LlamaSlot(
        role="editor",
        alias=LLAMA_SINGLE_MODEL,
        base_url="http://127.0.0.1:8080",
        port=8080,
        context=65536,
        weights=tmp_path / "HY-MT2-7B-Q6_K.gguf",
    )
    asked: dict[str, str] = {}
    calls: list[tuple[str, str]] = []

    class FakeManager:
        def port_notes(self):
            return {}

        def stop(self, role):
            calls.append(("stop", role))

        def can_relaunch(self, role):
            return False

    def fake_manager_from_settings(paths, settings):
        return FakeManager()

    def fake_open_slots(manager, settings, emit=None):
        asked["model"] = settings.llama_model
        asked["review"] = settings.llama_review_model
        return {"editor": slot, "review": None}

    monkeypatch.setattr(pipeline_module, "manager_from_settings", fake_manager_from_settings)
    monkeypatch.setattr(pipeline_module, "open_slots", fake_open_slots)

    request = PipelineRequest(
        source=tmp_path / "src",
        output=tmp_path / "out",
        quality_profile="quality",
        inference_backend="llama_7b",
        ocr_backend="llama",
        llama_model="DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K:latest",
        llama_review_model="hy-mt2-1.8b:q8",
    )
    messages: list[str] = []
    translator = LocalizerPipeline(_pipeline_paths(tmp_path))._llama_translator(
        request, lambda phase, current, total, message: messages.append(message), 5
    )
    # The one pass is the editor itself: no candidate, no auditor, no cross-check.
    assert type(translator) is LlamaTranslator
    assert translator.evict is False
    # The recipe pins the single file and clears the review model.
    assert asked == {"model": LLAMA_SINGLE_MODEL, "review": ""}
    assert f"单模型模式，仅运行 {LLAMA_SINGLE_MODEL}" in messages
    # A vision server from the OCR stage is a second model on the card.
    assert ("stop", "vision") in calls


def test_the_single_model_backend_keeps_a_vision_server_it_never_started(monkeypatch, tmp_path):
    from manga_localizer import pipeline as pipeline_module

    slot = LlamaSlot(
        role="editor",
        alias=LLAMA_SINGLE_MODEL,
        base_url="http://127.0.0.1:8080",
        port=8080,
        context=65536,
        weights=tmp_path / "HY-MT2-7B-Q6_K.gguf",
    )
    calls: list[tuple[str, str]] = []

    class FakeManager:
        def port_notes(self):
            return {}

        def stop(self, role):
            calls.append(("stop", role))

        def can_relaunch(self, role):
            return False

    monkeypatch.setattr(
        pipeline_module, "manager_from_settings", lambda paths, settings: FakeManager()
    )
    monkeypatch.setattr(
        pipeline_module,
        "open_slots",
        lambda manager, settings, emit=None: {"editor": slot, "review": None},
    )
    request = PipelineRequest(
        source=tmp_path / "src",
        output=tmp_path / "out",
        quality_profile="quality",
        inference_backend="llama_7b",
        ocr_backend="builtin",
    )
    translator = LocalizerPipeline(_pipeline_paths(tmp_path))._llama_translator(
        request, lambda *args: None, 5
    )
    assert type(translator) is LlamaTranslator
    assert calls == []
