import json
import time
from pathlib import Path

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
    preserve_implausible_translations,
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
                "ocr_version": OCR_CACHE_VERSION,
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


def test_a_draft_read_differently_is_not_reused(tmp_path):
    """The ids of a page are renumbered by every OCR run.

    Page 47 of the reported book was written twice by two jobs, the second run
    read one line more, and every translation after it moved down one unit:
    "POKE" was given the line belonging to "Ready y-yeAAAH!".  Matching on the
    unit id alone cannot see that; matching on the source line can.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "translation-draft.json").write_text(
        json.dumps(
            {
                "target_language": "简体中文",
                "ocr_version": OCR_CACHE_VERSION,
                "pages": [
                    {
                        "page": 1,
                        "file": "001.jpg",
                        "units": [
                            {"id": "p001u01", "ja": "POKE", "zh": "准备好了！"},
                            # Same id, a different reading of the page.
                            {"id": "p001u02", "ja": "Ready y-yeAAAH!", "zh": "不行。"},
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
                {"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "POKE", "zh": ""},
                {"id": "p001u02", "bbox": [40, 2, 70, 40], "ja": "NO", "zh": ""},
            ],
        }
    )
    assert LocalizerPipeline._resume_translations(work, [page], "简体中文") == 1
    assert [unit.zh for unit in page.units] == ["准备好了！", ""]


def test_a_translation_far_too_short_for_its_source_is_dropped():
    """An answer of two characters cannot render a 39-letter line.

    The reported page handed "抽送。" to a line reading
    "ABSOLTELYNOWATO OET MORE,IMLET WIW NOOOC O AAV".  That answer came from a
    previous shifted draft, but a model answering a garbled line it cannot read
    produces the same shape, so the guard has to catch it either way.
    """
    page = page_from_dict(
        {
            "page": 47,
            "file": "047.jpg",
            "width": 800,
            "height": 566,
            "units": [
                {
                    "id": "p047u16",
                    "bbox": [1, 2, 300, 40],
                    "ja": "ABSOLTELYNOWATO OET MORE,IMLET WIW NOOOC O AAV",
                    "zh": "抽送。",
                },
                {
                    "id": "p047u17",
                    "bbox": [1, 40, 300, 80],
                    "ja": "POKE",
                    "zh": "戳。",
                },
                {
                    "id": "p047u18",
                    "bbox": [1, 80, 300, 120],
                    "ja": "C-CAN YOU P-PLEASE STOP TALONS",
                    "zh": "噗！",
                },
            ],
        }
    )
    assert preserve_implausible_translations([page]) == ["p047u16", "p047u18"]
    assert [unit.skip for unit in page.units] == [True, False, True]
    assert page.units[0].skip_reason == "implausible_translation"
    # The short source keeps its short answer: "POKE" is one word.
    assert page.units[1].zh == "戳。"


def test_the_draft_for_another_target_language_is_ignored(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "translation-draft.json").write_text(
        json.dumps(
            {
                "target_language": "繁體中文",
                "ocr_version": OCR_CACHE_VERSION,
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


@pytest.mark.parametrize("stamp", [None, OCR_CACHE_VERSION - 1, "banana"])
def test_a_draft_from_another_ocr_version_is_thrown_away(tmp_path, stamp):
    """A draft carries the reading it was paired with, all the way down.

    The reported page 15 was reused from a run whose OCR numbered one more unit
    at the top: every id stayed the same but moved one line, so the per-line
    source check compared "NO!" against "WAIT, WHAT'S THAT SMELL?" and saw a
    mismatch it could not act on -- the old pairing had already been stored as
    a matching (ja, zh) pair.  A version stamp rejects the whole draft instead.
    """
    work = tmp_path / "work"
    work.mkdir()
    payload = {
        "target_language": "简体中文",
        "pages": [
            {
                "page": 1,
                "file": "001.jpg",
                "units": [{"id": "p001u01", "ja": "NO!", "zh": "等等，那是什么味道？"}],
            }
        ],
    }
    if stamp is not None:
        payload["ocr_version"] = stamp
    (work / "translation-draft.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [{"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "NO!", "zh": ""}],
        }
    )
    assert LocalizerPipeline._resume_translations(work, [page], "简体中文") == 0
    assert page.units[0].zh == ""


def test_a_draft_that_cannot_be_resumed_says_why(tmp_path, caplog):
    """A gate that drops the draft used to return 0 without a word.

    The caller then rewrote the draft out of the OCR cache, whose ``zh`` is
    always empty -- so a run that could not resume also erased what it failed
    to resume, and the log never said which gate had refused it.  The reported
    book lost 4708 finished lines that way.
    """
    work = tmp_path / "work"
    work.mkdir()
    draft = work / "translation-draft.json"
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [{"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "NO!", "zh": ""}],
        }
    )
    draft.write_text(
        json.dumps(
            {
                "target_language": "简体中文",
                "ocr_version": OCR_CACHE_VERSION - 1,
                "pages": [
                    {
                        "page": 1,
                        "file": "001.jpg",
                        "units": [{"id": "p001u01", "ja": "NO!", "zh": "不！"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="manga_localizer.pipeline"):
        assert LocalizerPipeline._resume_translations(work, [page], "简体中文") == 0
    assert "版本不符" in caplog.text
    assert str(draft) in caplog.text
    assert page.units[0].zh == ""

    caplog.clear()
    draft.write_text(
        json.dumps(
            {
                "target_language": "繁體中文",
                "ocr_version": OCR_CACHE_VERSION,
                "pages": [
                    {
                        "page": 1,
                        "file": "001.jpg",
                        "units": [{"id": "p001u01", "ja": "NO!", "zh": "不！"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="manga_localizer.pipeline"):
        assert LocalizerPipeline._resume_translations(work, [page], "简体中文") == 0
    assert "目标语言不符" in caplog.text
    assert page.units[0].zh == ""


def test_writing_the_draft_is_atomic(tmp_path, monkeypatch):
    """A run killed mid-write leaves the old draft, not half a new one."""
    work = tmp_path / "work"
    work.mkdir()
    target = work / "translation-draft.json"
    original = {"target_language": "简体中文", "pages": []}
    target.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [{"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "NO!", "zh": "不！"}],
        }
    )

    def refuse(self, other):
        raise OSError("the drive went away")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError):
        LocalizerPipeline._write_transcript(
            target, tmp_path / "src", [page], None, "简体中文"
        )
    assert json.loads(target.read_text(encoding="utf-8")) == original


def test_the_previous_draft_survives_a_new_write(tmp_path):
    """The draft is the only copy of text the book already paid for."""
    work = tmp_path / "work"
    work.mkdir()
    target = work / "translation-draft.json"
    filled = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [{"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "NO!", "zh": "不！"}],
        }
    )
    LocalizerPipeline._write_transcript(
        target, tmp_path / "src", [filled], None, "简体中文"
    )
    empty = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [{"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "NO!", "zh": ""}],
        }
    )
    LocalizerPipeline._write_transcript(
        target, tmp_path / "src", [empty], None, "简体中文", keep_previous=True
    )
    backup = json.loads(
        target.with_suffix(target.suffix + ".bak").read_text(encoding="utf-8")
    )
    assert backup["pages"][0]["units"][0]["zh"] == "不！"
    assert json.loads(target.read_text(encoding="utf-8"))["pages"][0]["units"][0]["zh"] == ""


def _pipeline_paths(tmp_path):
    root = tmp_path / "home"
    return AppPaths(
        root,
        root / "models",
        root / "cache",
        root / "jobs",
        root / "settings.json",
    )


def test_the_hybrid_profile_pays_for_quality_pixels_but_not_for_review(monkeypatch, tmp_path):
    """Fast translation and quality rendering are independent choices."""
    from manga_localizer import pipeline as pipeline_module
    from manga_localizer.config import (
        cleanup_profile_for,
        ocr_profile_for,
        translate_quality,
    )

    # One dropdown, three answers: review, page reading, pixel repair.
    assert translate_quality("quality") is True
    assert translate_quality("fast") is False
    assert translate_quality("fast_translate_quality_render") is False
    assert ocr_profile_for("fast") == "fast"
    assert ocr_profile_for("fast_translate_quality_render") == "quality"
    assert cleanup_profile_for("fast") == "artwork"
    assert cleanup_profile_for("quality") == "quality"
    assert cleanup_profile_for("fast_translate_quality_render") == "quality"

    # The renderer only knows "quality" and "artwork"; the hybrid profile lands
    # on the LaMa path without renderer.py learning a third name.
    renderer = pipeline_module.ArtworkPreservingRenderer(
        cleanup_profile=cleanup_profile_for("fast_translate_quality_render"),
        paths=_pipeline_paths(tmp_path),
    )
    assert renderer.cleanup_profile == "quality"

    slot = LlamaSlot(
        role="editor",
        alias=LLAMA_SINGLE_MODEL,
        base_url="http://127.0.0.1:8080",
        port=8080,
        context=65536,
        weights=tmp_path / "model.gguf",
    )
    asked: dict[str, str] = {}

    class FakeManager:
        def port_notes(self):
            return {}

        def stop(self, role):
            pass

        def can_relaunch(self, role):
            return False

    monkeypatch.setattr(
        pipeline_module, "manager_from_settings", lambda paths, settings: FakeManager()
    )

    def fake_open_slots(manager, settings, emit=None):
        asked["model"] = settings.llama_model
        asked["review"] = settings.llama_review_model
        return {"editor": slot, "review": None}

    monkeypatch.setattr(pipeline_module, "open_slots", fake_open_slots)
    request = PipelineRequest(
        source=tmp_path / "src",
        output=tmp_path / "out",
        quality_profile="fast_translate_quality_render",
        inference_backend="llama",
        ocr_backend="builtin",
        llama_model="model",
        llama_review_model="review-model",
    )
    translator = LocalizerPipeline(_pipeline_paths(tmp_path))._llama_translator(
        request, lambda *args: None, 5
    )
    # The hybrid profile keeps one pass: the candidate never runs.
    assert type(translator) is LlamaTranslator
    assert asked == {"model": "model", "review": "review-model"}


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


def test_the_translate_clock_is_live_while_the_stage_runs(monkeypatch, tmp_path):
    """The fourth step read nothing for the whole stage.

    The stage total is only written once ``translate_pages`` returns, so a
    run that spends an hour there looked frozen the whole time.  The
    progress callback has to publish a running clock while it still can.
    """
    from manga_localizer import pipeline as pipeline_module

    source = tmp_path / "src"
    source.mkdir()
    Image.new("RGB", (200, 400), "white").save(source / "001.jpg")

    seen: dict = {}

    class TickingTranslator:
        resolved_glossary: dict = {}

        def translate_pages(self, pages, *args, **kwargs):
            progress = kwargs["progress"]
            # Spend real time, then report a page: the clock the UI reads
            # has to be non-zero at that moment, not only at stage end.
            time.sleep(0.2)
            progress(1, len(pages))
            seen.update(getattr(pipeline, "live_timings", None) or {})
            return pages

        def release(self):
            pass

    class Reader:
        stage_seconds: dict = {}

        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, image_path, page_number):
            # A dialogue-sized box: a page-filling one is classified as
            # artwork and skipped, and then no translate stage ever runs.
            unit = pipeline_module.TextUnit(
                id=f"p{page_number:03d}u01",
                bbox=[10, 10, 60, 30],
                crop_bbox=[10, 10, 60, 30],
                ja="\u30c6\u30b9\u30c8",
                score=0.95,
            )
            return pipeline_module.PageOCR(
                page=page_number,
                file=image_path.name,
                width=200,
                height=400,
                units=[unit],
            )

    monkeypatch.setattr(pipeline_module, "PaddleMangaOCR", Reader)
    monkeypatch.setattr(pipeline_module, "HyMTTranslator", lambda *a, **k: TickingTranslator())
    monkeypatch.setattr(
        LocalizerPipeline,
        "_load_cached_ocr",
        staticmethod(lambda *a, **k: None),
    )

    pipeline = LocalizerPipeline(_pipeline_paths(tmp_path))
    request = PipelineRequest(
        source=source,
        output=tmp_path / "out",
        source_language="en",
        ocr_backend="builtin",
        inference_backend="builtin",
        quality_profile="fast",
    )
    pipeline.run(request)

    assert "translate" in seen, "the stage clock never ticked"
    assert seen["translate"] > 0
