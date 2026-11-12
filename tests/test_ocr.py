import io
import json
import warnings
from pathlib import Path
from urllib.error import HTTPError

import pytest
from PIL import Image, ImageDraw

import numpy as np

from manga_localizer.ocr import (
    HybridMangaOCR,
    JP_RE,
    LATIN_EFFECT_SCORE_FLOOR,
    OllamaVisionOCR,
    PageOCR,
    PaddleMangaOCR,
    SOURCE_LANGUAGES,
    TextUnit,
    _best_recognized_text,
    _connects,
    _is_latin_region,
    _light_on_dark_regions,
    _outlined_light_text_regions_near,
    _suppress_optional_ccache_warning,
    classify_non_story_text,
    duplicate_tiny_fragment,
    deduplicate_nested_region_groups,
    is_latin_page,
    is_source_text,
    ink_tile_hints,
    language_label,
    language_label_zh,
    likely_sfx_text,
    link_block_text,
    list_images,
    malformed_tiny_ocr,
    manga_force_cpu,
    merge_region_candidates,
    merge_semantic_missing,
    overlapping_sfx_echo,
    page_reads_as_latin,
    page_spanning_display_artwork,
    prefer_semantic_ocr,
    prune_nested_duplicate_units,
    prune_detached_orthogonal_outliers,
    rescue_reading_is_stable,
    semantic_sfx_classification,
    scores_are_informative,
    tiny_low_confidence_nontext,
)
from manga_localizer.pipeline import unsafe_semantic_missing


def test_publication_metadata_and_misread_edge_folio_are_preserved():
    metadata = TextUnit(
        "p077u01",
        [575, 2122, 2012, 2932],
        [518, 2098, 2069, 2956],
        "初版発行日：2026/08/16 発行者・サークル 漫画・イラスト "
        "Twitter(X) ロゴデザイン・協力 印刷",
        0.92,
        erase_boxes=[
            [846, 2122, 1752, 2212],
            [975, 2219, 1623, 2309],
            [899, 2316, 1695, 2406],
            [575, 2410, 2012, 2503],
        ],
    )
    folio = TextUnit(
        "p077u02",
        [27, 3212, 69, 3286],
        [17, 3202, 79, 3296],
        "フ８",
        0.88,
    )
    page = PageOCR(77, "0077.webp", 2560, 3581, [metadata, folio])

    classify_non_story_text(page)

    assert (metadata.skip_reason, metadata.special) == (
        "preserve",
        "publication_metadata",
    )
    assert (folio.skip_reason, folio.special) == ("decorative", "page_folio")


def test_story_dialogue_with_one_metadata_word_is_not_skipped():
    unit = TextUnit(
        "p001u01",
        [100, 100, 300, 500],
        [90, 90, 310, 510],
        "漫画を読もう",
        0.9,
    )
    page = PageOCR(1, "001.webp", 1000, 1400, [unit])

    classify_non_story_text(page)

    assert unit.skip is False


def test_cover_display_artwork_is_preserved_instead_of_translated():
    """A cover art plate is not a balloon.

    Paddle groups the three display lines of a cover title into one region
    covering a third of the page, and MangaOCR answers the stylized Latin art
    with an invented date ("２０１８年０９月１０日").  Built-in OCR has no
    semantic pass, so nothing marked the unit: the cleaner erased every art
    line and "2018 年 09 月 10 日" was typeset across the cover.
    """
    cover = TextUnit(
        "p001u01",
        [110, 15, 800, 403],
        [82, 3, 800, 415],
        "２０１８年０９月１０日",
        0.9892,
        erase_boxes=[
            [590, 325, 790, 403],
            [247, 164, 800, 352],
            [110, 15, 579, 203],
        ],
    )
    page = PageOCR(1, "00000001.webp", 800, 1132, [cover])

    classify_non_story_text(page)

    assert cover.skip is True
    assert cover.skip_reason == "preserve"
    assert cover.special == "artwork_text"
    assert cover.zh == ""


def test_inner_title_plate_display_artwork_is_preserved():
    """A smaller inner title plate is the same misread, not a caption.

    The credits page repeats the cover lettering at roughly 13% of the page and
    MangaOCR invents a longer date for it, so the cover-only threshold missed
    it and the letters were replaced by "这是 1009 年 9 月 30 日".
    """
    plate = TextUnit(
        "p003u01",
        [210, 43, 671, 307],
        [200, 33, 681, 317],
        "これは１００９年９月３０：００１",
        0.93,
    )
    page = PageOCR(3, "00000003.webp", 800, 1132, [plate])

    assert page_spanning_display_artwork(page, plate) is True

    classify_non_story_text(page)

    assert plate.skip is True
    assert plate.skip_reason == "preserve"
    assert plate.special == "artwork_text"
    assert plate.zh == ""


def test_ordinary_large_dialogue_region_is_not_treated_as_cover_art():
    """The rule must not swallow wide dialogue or inner-page display text."""
    line = TextUnit(
        "p004u08",
        [12, 108, 788, 172],
        [6, 100, 794, 180],
        "HEEEEY CARROT TOP, HOW'S IT HANGING?",
        0.94,
    )
    caption = TextUnit(
        "p003u02",
        [138, 360, 660, 479],
        [130, 350, 670, 490],
        "HELLO! Thank you very much for buying my book.",
        0.93,
    )
    # A page-filling SFX belongs to oversized_decorative_sfx, not to this rule.
    thunder = TextUnit(
        "p009u01",
        [40, 30, 760, 900],
        [30, 20, 770, 910],
        "ドドドド",
        0.95,
        is_sfx=True,
    )
    page = PageOCR(4, "00000004.webp", 800, 1132, [line, caption, thunder])

    for unit in page.units:
        assert page_spanning_display_artwork(page, unit) is False

    # Small type over a large block is a text page, not display lettering:
    # dense credit lists and long captions reach 10-17% of the page but their
    # glyphs claim only 0.02-0.2% each, a magnitude below any display face.
    dense_credits = TextUnit(
        "p073u02",
        [80, 400, 720, 588],
        [70, 390, 730, 598],
        (
            "This project was our first big debut into the MLP fandom, and over "
            "the last two years it grew far beyond anything I could have "
            "imagined when I first started drawing the opening page of it. "
            "Thank you for reading all the way through."
        ),
        0.9,
    )
    credits_page = PageOCR(73, "00000073.webp", 800, 1132, [dense_credits])
    assert page_spanning_display_artwork(credits_page, dense_credits) is False

    classify_non_story_text(page)

    assert line.skip is False
    assert caption.skip is False


def test_merged_cover_title_stays_translatable():
    """An explicitly merged cover title is translated in place, not preserved."""
    merged = TextUnit(
        "p001u01",
        [300, 590, 520, 1320],
        [290, 580, 530, 1330],
        "僕に勇気があったなら",
        0.9,
        special="cover_title",
    )
    page = PageOCR(1, "000.webp", 1000, 1400, [merged])

    assert page_spanning_display_artwork(page, merged) is False

    classify_non_story_text(page)

    assert merged.skip is False


def test_orthogonal_scene_text_does_not_join_vertical_dialogue():
    vertical_dialogue = {"box": [15, 1986, 188, 2693]}
    horizontal_garment_logo = {"box": [380, 1602, 1855, 2299]}
    assert _connects(vertical_dialogue, horizontal_garment_logo) is False


def test_detached_orthogonal_outlier_is_removed_from_legacy_group():
    boxes = [
        [151, 1997, 357, 2556],
        [15, 1986, 188, 2693],
        [380, 1602, 1855, 2299],
    ]
    assert prune_detached_orthogonal_outliers(boxes) == boxes[:2]


def test_tiny_zero_confidence_fragment_is_preserved_as_nontext():
    unit = TextUnit(
        "p001u01",
        [1099, 2439, 1186, 2468],
        [1089, 2429, 1196, 2478],
        "『インターネットの",
        0.0,
    )
    page = PageOCR(1, "001.png", 2126, 3661, [unit])
    assert tiny_low_confidence_nontext(page, unit) is True

    dialogue = TextUnit(
        "p001u02", [100, 100, 220, 180], [90, 90, 230, 190], "ん？", 0.0
    )
    assert tiny_low_confidence_nontext(page, dialogue) is False
    tiny_narration = TextUnit(
        "p001u03",
        [1546, 99, 1612, 125],
        [1536, 89, 1622, 135],
        "そして最近では、",
        0.0,
    )
    assert tiny_low_confidence_nontext(page, tiny_narration) is False


def test_low_confidence_tiny_repeated_kanji_is_malformed_ocr():
    malformed = TextUnit(
        "p001u01",
        [1766, 1442, 1806, 1571],
        [1756, 1432, 1816, 1581],
        "目目まで、",
        0.76,
    )
    normal = TextUnit(
        "p001u02", [1766, 1442, 1806, 1571], [1756, 1432, 1816, 1581], "目まで、", 0.76
    )
    page = PageOCR(1, "001.png", 2126, 3661, [malformed, normal])
    assert malformed_tiny_ocr(page, malformed) is True
    assert malformed_tiny_ocr(page, normal) is False


def test_tiny_ruby_or_ocr_fragment_repeated_in_larger_unit_is_preserved():
    small = TextUnit(
        "p001u01",
        [1712, 661, 1758, 729],
        [1700, 650, 1770, 740],
        "彼氏\nかれし",
        0.92,
    )
    large = TextUnit(
        "p001u02",
        [1602, 0, 2095, 672],
        [1580, 0, 2110, 690],
        "でももう\n流石に彼氏とか\nいんだろ？",
        0.95,
    )
    page = PageOCR(1, "001.png", 2126, 3661, [small, large])
    assert duplicate_tiny_fragment(page, small) is True
    assert duplicate_tiny_fragment(page, large) is False

    repeated_dialogue = TextUnit(
        "p001u03",
        [1098, 1869, 1266, 2147],
        [1088, 1859, 1276, 2157],
        "彼氏",
        0.69,
    )
    page.units.append(repeated_dialogue)
    assert duplicate_tiny_fragment(page, repeated_dialogue) is False


def test_short_katakana_echo_inside_confirmed_non_dialogue_region_is_duplicate():
    effect = TextUnit(
        "p002u03",
        [809, 1423, 1631, 2199],
        [790, 1400, 1650, 2220],
        "カチカチ",
        0.96,
        is_sfx=False,
        skip=True,
        skip_reason="preserve",
        special="artwork_text",
    )
    echo = TextUnit(
        "p002u06",
        [1579, 1972, 1739, 2200],
        [1569, 1962, 1749, 2210],
        "カナ、",
        0.85,
    )
    unrelated_name = TextUnit(
        "p002u07",
        [100, 2500, 260, 2750],
        [90, 2490, 270, 2760],
        "サナ、",
        0.9,
    )
    page = PageOCR(2, "002.png", 2126, 3661, [effect, echo, unrelated_name])

    assert overlapping_sfx_echo(page, echo) is True
    assert overlapping_sfx_echo(page, unrelated_name) is False


def test_horizontal_vlm_echo_yields_to_detailed_vertical_detector_boxes():
    echo = TextUnit(
        "p001u01",
        [739, 1617, 1290, 2063],
        [717, 1604, 1312, 2076],
        "ああミナッ\nイクイク…",
        0.95,
        erase_boxes=[[739, 1617, 1290, 2063]],
    )
    dialogue = TextUnit(
        "p001u02",
        [27, 1495, 293, 2303],
        [16, 1471, 304, 2327],
        "あぁミナッ\nイクイクイク…",
        0.92,
        erase_boxes=[[27, 1503, 164, 2303], [145, 1495, 293, 2124]],
    )
    page = PageOCR(1, "001.png", 2126, 3661, [echo, dialogue])
    assert duplicate_tiny_fragment(page, echo) is True
    assert duplicate_tiny_fragment(page, dialogue) is False


def test_only_the_optional_ccache_notice_is_suppressed():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _suppress_optional_ccache_warning():
            warnings.warn(
                "No ccache found. Compilation cache is optional.", UserWarning
            )
            warnings.warn("actionable inference warning", RuntimeWarning)
    assert [str(item.message) for item in caught] == ["actionable inference warning"]


def test_images_are_naturally_sorted(tmp_path):
    for name in ("10.png", "2.jpg", "1.webp", "notes.txt"):
        path = tmp_path / name
        if path.suffix == ".txt":
            path.write_text("ignore", encoding="utf-8")
        else:
            Image.new("RGB", (8, 8), "white").save(path)
    assert [item.name for item in list_images(tmp_path)] == [
        "1.webp",
        "2.jpg",
        "10.png",
    ]


def test_second_pass_missing_drops_duplicate_vlm_hallucination():
    page = PageOCR(
        6,
        "006.png",
        1000,
        2000,
        [
            TextUnit(
                "p006u01",
                [700, 50, 800, 120],
                [690, 40, 810, 130],
                "そして最近では、",
                0.9,
                erase_boxes=[[700, 50, 800, 120]],
            )
        ],
    )
    missing = [
        {
            "text": "そして最近では、",
            "bbox": [10, 400, 200, 500],
            "score": 0.9,
            "is_sfx": False,
        }
    ]
    assert OllamaVisionOCR.filter_covered_missing(page, missing) == []


def test_second_pass_missing_drops_fragments_already_inside_exact_ocr():
    page = PageOCR(
        2,
        "002.png",
        1000,
        2000,
        [
            TextUnit(
                "p002u01",
                [20, 1000, 220, 1500],
                [10, 990, 230, 1510],
                "たくさん\nたくさんあって",
                0.9,
                erase_boxes=[[20, 1000, 220, 1500]],
            )
        ],
    )
    missing = [
        {"text": "たくさん", "bbox": [500, 500, 600, 600], "score": 0.95},
        {"text": "あって", "bbox": [500, 600, 600, 700], "score": 0.95},
    ]
    assert OllamaVisionOCR.filter_covered_missing(page, missing) == []


def test_second_pass_missing_drops_geometry_already_owned_by_detector():
    page = PageOCR(
        2,
        "002.png",
        1000,
        2000,
        [
            TextUnit(
                "p002u01",
                [100, 800, 300, 1200],
                [90, 790, 310, 1210],
                "違う文字",
                0.9,
                erase_boxes=[[100, 800, 300, 1200]],
            )
        ],
    )
    missing = [
        {
            "text": "別の候補",
            "bbox": [120, 420, 280, 580],
            "score": 0.9,
            "is_sfx": False,
        }
    ]
    assert OllamaVisionOCR.filter_covered_missing(page, missing) == []


def test_large_decorative_sfx_contains_short_missing_fragment():
    unit = TextUnit(
        "p002u03",
        [300, 200, 900, 900],
        [290, 190, 910, 910],
        "カチカチ",
        0.9,
        is_sfx=True,
        erase_boxes=[[700, 500, 850, 800], [500, 400, 650, 650]],
    )
    page = PageOCR(2, "002.png", 1000, 1000, [unit])
    missing = [
        {"text": "ボ", "bbox": [350, 250, 430, 350], "score": 0.95, "is_sfx": True}
    ]

    assert OllamaVisionOCR.filter_covered_missing(page, missing) == []


def test_punctuated_short_hiragana_is_dialogue_even_without_detector_score():
    assert semantic_sfx_classification("いや、", 0.0, True) is False
    assert semantic_sfx_classification("しかし、", 0.0, True) is False
    assert semantic_sfx_classification("ん？", 0.85, True) is False
    assert semantic_sfx_classification("ここは", 0.95, True) is False
    assert semantic_sfx_classification("いいですよ．．．", 0.96, True) is False
    assert semantic_sfx_classification("サナは", 0.82, True) is False
    assert semantic_sfx_classification("いつまでも", 0.86, True) is False
    assert semantic_sfx_classification("その", 0.0, True) is False
    assert semantic_sfx_classification("いい", 0.87, True) is False
    assert semantic_sfx_classification("カナ、", 0.94, True) is False
    assert semantic_sfx_classification("彼氏：", 0.68, True) is False
    assert semantic_sfx_classification("し", 0.2, False) is True


def test_missing_recovery_accepts_only_enlarged_detector_owned_geometry(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(source)

    class Detector:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "dt_polys": [[[100, 100], [300, 100], [300, 300], [100, 300]]],
                        "dt_scores": [0.91],
                    }
                }
            ]

    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.reader = lambda _crop: "見落とし"
    page = PageOCR(4, source.name, 1000, 1000, [])
    recovered = service.recover_missing(
        source,
        page,
        [
            {
                "text": "見落とし",
                "bbox": [400, 400, 600, 600],
                "score": 0.9,
                "is_sfx": False,
            }
        ],
    )
    assert len(recovered.units) == 1
    assert recovered.units[0].ja == "見落とし"
    assert recovered.units[0].erase_boxes == [recovered.units[0].bbox]


def test_missing_outlined_sfx_uses_pixel_geometry_and_semantic_spelling(tmp_path):
    source = tmp_path / "outlined-sfx.png"
    image = Image.new("RGB", (1000, 1200), (175, 175, 175))
    draw = ImageDraw.Draw(image)
    for left, top, width, height in (
        (380, 360, 95, 180),
        (490, 410, 90, 140),
        (610, 455, 60, 95),
    ):
        draw.rectangle(
            (left - 10, top - 10, left + width + 10, top + height + 10),
            fill="black",
        )
        draw.rectangle((left, top, left + width, top + height), fill="white")
    image.save(source)

    class Detector:
        def predict(self, _image):
            return []

    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.reader = lambda _crop: "レキット"
    page = PageOCR(10, source.name, 1000, 1200, [])

    recovered = service.recover_missing(
        source,
        page,
        [
            {
                "text": "ムチッ",
                "bbox": [350, 275, 520, 417],
                "score": 0.82,
                "is_sfx": True,
            }
        ],
    )

    assert len(recovered.units) == 1
    assert recovered.units[0].ja == "ムチッ"
    assert len(recovered.units[0].erase_boxes) == 3


def test_missing_outlined_sfx_repairs_existing_low_confidence_read(tmp_path):
    source = tmp_path / "outlined-existing.png"
    image = Image.new("RGB", (1000, 1200), (175, 175, 175))
    draw = ImageDraw.Draw(image)
    for left, top, width, height in (
        (380, 360, 95, 180),
        (490, 410, 90, 140),
        (610, 455, 60, 95),
    ):
        draw.rectangle(
            (left - 10, top - 10, left + width + 10, top + height + 10),
            fill="black",
        )
        draw.rectangle((left, top, left + width, top + height), fill="white")
    image.save(source)

    existing = TextUnit(
        "p010u01",
        [350, 330, 700, 580],
        [340, 320, 710, 590],
        "しかし、",
        0.0,
        erase_boxes=[[350, 330, 700, 580]],
    )
    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = type("Detector", (), {"predict": lambda self, image: []})()
    service.reader = lambda _crop: "unused"

    recovered = service.recover_missing(
        source,
        PageOCR(10, source.name, 1000, 1200, [existing]),
        [
            {
                "text": "ムチッ",
                "bbox": [350, 275, 520, 417],
                "score": 0.82,
                "is_sfx": True,
            }
        ],
    )

    assert len(recovered.units) == 1
    assert recovered.units[0].ja == "ムチッ"
    assert recovered.units[0].is_sfx is True
    assert recovered.units[0].skip is False


def test_missing_sfx_can_enrich_existing_detector_geometry_without_vlm_mask(tmp_path):
    source = tmp_path / "existing-sfx.png"
    Image.new("RGB", (1000, 1000), "white").save(source)
    existing = TextUnit(
        "p002u03",
        [300, 300, 760, 760],
        [290, 290, 770, 770],
        "カチカチ",
        0.9,
        is_sfx=False,
        skip=True,
        skip_reason="preserve",
        erase_boxes=[[520, 500, 620, 600], [640, 610, 700, 680]],
        special="artwork_text",
    )
    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = type("Detector", (), {"predict": lambda self, image: []})()
    service.reader = lambda _crop: "unused"

    recovered = service.recover_missing(
        source,
        PageOCR(2, source.name, 1000, 1000, [existing]),
        [{"text": "ボォ", "bbox": [320, 320, 420, 420], "score": 0.9, "is_sfx": True}],
    )

    assert recovered.units[0].ja == "カチカチ\nボォ"
    assert recovered.units[0].erase_boxes == [
        [520, 500, 620, 600],
        [640, 610, 700, 680],
    ]
    assert recovered.units[0].skip is False
    assert recovered.units[0].special == ""


def test_hybrid_ocr_does_not_silently_accept_an_empty_detector_page(tmp_path):
    source = tmp_path / "empty-detector.png"
    Image.new("RGB", (1000, 2000), "white").save(source)
    recovered_unit = TextUnit(
        "p003u01",
        [100, 200, 300, 600],
        [90, 190, 310, 610],
        "見落とし",
        0.9,
        erase_boxes=[[100, 200, 300, 600]],
    )

    class Geometry:
        def analyze(self, _path, page_number):
            return PageOCR(page_number, source.name, 1000, 2000, [])

        def recover_missing(self, _path, page, missing):
            assert missing[0]["text"] == "見落とし"
            page.units = [recovered_unit]
            return page

    class Semantics:
        def analyze(self, _path, page_number):
            return PageOCR(page_number, source.name, 1000, 2000, [recovered_unit])

        def refine(self, _path, page):
            return page, []

        def filter_covered_missing(self, page, missing):
            return OllamaVisionOCR.filter_covered_missing(page, missing)

    service = HybridMangaOCR.__new__(HybridMangaOCR)
    service.geometry = Geometry()
    service.semantics = Semantics()
    service.last_missing = []
    page = service.analyze(source, 3)
    assert [unit.ja for unit in page.units] == ["見落とし"]
    assert page.semantic_missing == []


def test_hybrid_ocr_keeps_missing_until_exact_geometry_proves_coverage(tmp_path):
    source = tmp_path / "persistent-missing.png"
    Image.new("RGB", (1000, 2000), "white").save(source)
    page = PageOCR(
        3,
        source.name,
        1000,
        2000,
        [TextUnit("p003u01", [50, 50, 200, 300], [40, 40, 210, 310], "既存", 0.9)],
    )
    missing = [
        {
            "text": "見落とし",
            "bbox": [700, 700, 900, 900],
            "score": 0.9,
            "is_sfx": False,
        }
    ]

    class Geometry:
        def analyze(self, _path, _page_number):
            return page

        def recover_missing(self, _path, current, _missing):
            return current

    class Semantics:
        calls = 0

        def refine(self, _path, current):
            self.calls += 1
            return current, (missing if self.calls == 1 else [])

        def filter_covered_missing(self, current, unresolved):
            return OllamaVisionOCR.filter_covered_missing(current, unresolved)

        def refine_local_crops(self, _path, current):
            return current

    service = HybridMangaOCR.__new__(HybridMangaOCR)
    service.geometry = Geometry()
    service.semantics = Semantics()
    service.profile = "quality"
    service.last_missing = []

    result = service.analyze(source, 3)

    assert [item["text"] for item in result.semantic_missing] == ["見落とし"]
    # Recovery could not confirm it, so it is recorded rather than acted on.
    assert result.semantic_missing[0]["unconfirmed"] is True


def test_garbled_reread_of_known_line_is_not_a_missing_unit():
    """The recall pass re-reads owned lines with a mangled character.

    ``何か理由があると思ってちゃんとなんて話を聞いてあげたのよ`` is a garbled
    re-read of a line MangaOCR already recognised, and reporting it aborted a
    whole book.  Short effects and genuinely new lines must stay reported.
    """
    from manga_localizer.ocr import _rereads_known_text

    known = {
        "何か理由があると思ってちゃんと話を聞いてあげたのよ",
        "それなのにわざわざ私に",
        "げっ、マジかよ",
        "えっどういう彼女にはもう前みたいな力は無いの",
    }
    bigrams: set[str] = set()
    for value in known:
        bigrams.update(value[i : i + 2] for i in range(len(value) - 1))

    assert _rereads_known_text(
        "何か理由があると思ってちゃんとなんて話を聞いてあげたのよ", known, bigrams
    )
    assert _rereads_known_text(
        "それなのにわさざわざ私にしてくるん化けてくるすもの", known, bigrams
    )
    assert _rereads_known_text("前彼女にはもう力はみたいなの", known, bigrams)
    # ``マジかよ`` is a re-read of the known ``げっ、マジかよ``.
    assert _rereads_known_text("マジかよ", known, bigrams)
    # Two-character effects share nothing and stay exempt.
    assert _rereads_known_text("がー！", known, bigrams) is False
    # A genuinely new line must keep failing the book.
    assert _rereads_known_text("見落とし", known, bigrams) is False


def test_hybrid_ocr_marks_unconfirmed_claims_instead_of_failing_the_book(tmp_path):
    """A VLM reads screentone as effects; exact recovery is the arbiter.

    ``ただだ`` was reported on a blank patch of hatching.  Recovery re-ran the
    detector on an enlarged crop and found no geometry, which is positive
    evidence there is nothing to erase -- failing the book over it is wrong.
    The claim is kept as ``unconfirmed`` so the transcript still records it,
    and ``unsafe_semantic_missing`` ignores it.
    """
    source = tmp_path / "screentone.png"
    Image.new("RGB", (1000, 2000), "white").save(source)
    page = PageOCR(
        3,
        source.name,
        1000,
        2000,
        [TextUnit("p003u01", [50, 50, 200, 300], [40, 40, 210, 310], "既存", 0.9)],
    )
    missing = [
        {"text": "ただだ", "bbox": [600, 700, 700, 750], "score": 0.98, "is_sfx": True},
        {"text": "見落とし", "bbox": [700, 900, 900, 980], "score": 0.9, "is_sfx": False},
    ]

    class Geometry:
        def analyze(self, _path, _page_number):
            return page

        def recover_missing(self, _path, current, _missing):
            return current

    class Semantics:
        calls = 0

        def refine(self, _path, current):
            self.calls += 1
            return current, (missing if self.calls == 1 else [])

        def find_missing(self, _path, _current):
            return []

        def filter_covered_missing(self, current, unresolved):
            return OllamaVisionOCR.filter_covered_missing(current, unresolved)

        def refine_local_crops(self, _path, current):
            return current

    service = HybridMangaOCR.__new__(HybridMangaOCR)
    service.geometry = Geometry()
    service.semantics = Semantics()
    service.profile = "quality"
    service.last_missing = []

    result = service.analyze(source, 3)

    assert [item["text"] for item in result.semantic_missing] == ["ただだ", "見落とし"]
    assert all(item["unconfirmed"] for item in result.semantic_missing)
    assert unsafe_semantic_missing(result, True) == []


def test_a_declared_latin_source_skips_recall_that_can_never_be_used(tmp_path):
    """English books have no recovery path, so asking for proposals is waste.

    ``PaddleMangaOCR.recover_missing`` returns the page untouched for a declared
    ``en`` source because the write-back reading that confirms a proposal is
    Japanese-only.  A recall proposal still costs 13-21% of the page, so the
    hybrid pass must not request one.  ``auto`` keeps it: a Latin page under
    ``auto`` still routes through recovery.
    """
    source = tmp_path / "page.png"
    Image.new("RGB", (1000, 2000), "white").save(source)
    page = PageOCR(
        3,
        source.name,
        1000,
        2000,
        [TextUnit("p003u01", [50, 50, 200, 300], [40, 40, 210, 310], "HELLO", 0.9)],
    )

    class Geometry:
        def analyze(self, _path, _page_number):
            return page

        def recover_missing(self, _path, current, _missing):
            return current

    class Semantics:
        def __init__(self):
            self.find_missing_calls = 0

        def refine(self, _path, current):
            return current, []

        def find_missing(self, _path, _current):
            self.find_missing_calls += 1
            return []

        def filter_covered_missing(self, current, unresolved):
            return OllamaVisionOCR.filter_covered_missing(current, unresolved)

        def refine_local_crops(self, _path, current):
            return current

    english = Semantics()
    service = HybridMangaOCR.__new__(HybridMangaOCR)
    service.geometry = Geometry()
    service.semantics = english
    service.profile = "quality"
    service.source_language = "en"
    service.last_missing = []
    service.analyze(source, 3)
    assert english.find_missing_calls == 0

    undeclared = Semantics()
    auto = HybridMangaOCR.__new__(HybridMangaOCR)
    auto.geometry = Geometry()
    auto.semantics = undeclared
    auto.profile = "quality"
    auto.source_language = "auto"
    auto.last_missing = []
    auto.analyze(source, 3)
    assert undeclared.find_missing_calls == 1


def test_manga_ocr_gpu_choice_is_independent_from_paddle():
    assert manga_force_cpu("auto", cuda_available=True) is False
    assert manga_force_cpu("gpu:0", cuda_available=True) is False
    assert manga_force_cpu("cpu", cuda_available=True) is True
    assert manga_force_cpu("auto", cuda_available=False) is True


def test_repeated_mixed_kana_ocr_noise_is_preserved_as_sfx():
    assert likely_sfx_text("カチ") is True
    assert likely_sfx_text("そのカキノカキ") is True
    assert likely_sfx_text("このゲーム") is False
    assert likely_sfx_text("サナはいつも") is False


def test_latin_sound_words_are_effects_but_latin_dialogue_is_not():
    """English pages spell effects as words; without this every GRRRRR became
    a phonetic Chinese transliteration instead of a sound."""
    assert likely_sfx_text("Grrrrr") is True
    assert likely_sfx_text("poof") is True
    assert likely_sfx_text("AUGH!") is True
    assert likely_sfx_text("STUPID MASK") is False
    assert likely_sfx_text("Banana") is False


def test_repeated_latin_sound_words_are_one_effect():
    """English pages double their effects ("sniff snif"); reading the pair as a
    sentence turned it into the half-transliterated "嗅嗅，尼夫"."""
    assert likely_sfx_text("sniff snif") is True
    assert likely_sfx_text("poof poof") is True
    assert likely_sfx_text("you sniff") is False


def test_windows_cpu_detector_disables_mkldnn_regression():
    source = (
        Path(__file__).parents[1] / "src" / "manga_localizer" / "ocr.py"
    ).read_text(encoding="utf-8")
    assert "enable_mkldnn=False" in source


def test_detection_only_payload_becomes_tight_regions():
    payload = {
        "dt_polys": [
            [[10, 20], [70, 18], [72, 90], [12, 92]],
            [[0, 0], [5, 0], [5, 5], [0, 5]],
        ],
        "dt_scores": [0.91, 0.99],
    }
    assert PaddleMangaOCR._regions(payload) == [
        {
            "box": [10, 18, 72, 92],
            "text": "",
            "score": 0.91,
        }
    ]


def test_semantic_missing_merge_keeps_equal_sfx_at_distinct_locations():
    first = {"text": "ムチッ", "bbox": [480, 140, 640, 220], "score": 0.78}
    repeated = {"text": "ムチッ", "bbox": [485, 145, 635, 215], "score": 0.82}
    second = {"text": "ムチッ", "bbox": [230, 340, 340, 420], "score": 0.76}

    merged = merge_semantic_missing([first], [repeated, second])

    assert len(merged) == 2
    assert merged[0]["score"] == 0.82


def test_rotation_alone_does_not_skip_dialogue_as_sfx():
    payload = {
        "dt_polys": [[[10, 20], [70, 5], [90, 75], [30, 90]]],
        "dt_scores": [0.8],
    }
    assert "sfx_hint" not in PaddleMangaOCR._regions(payload)[0]


def test_light_on_dark_title_becomes_generic_ocr_candidate():
    image = Image.new("RGB", (1000, 1400), "black")
    draw = ImageDraw.Draw(image)
    for index in range(7):
        left = 100 + index * 115
        draw.rectangle((left, 1120, left + 68, 1240), fill="white")
        draw.rectangle((left + 20, 1085, left + 48, 1270), fill="white")
    regions = _light_on_dark_regions(image)
    assert len(regions) == 1
    assert regions[0]["reverse"] is True
    assert regions[0]["box"][0] <= 100
    assert regions[0]["box"][2] >= 858


def test_light_background_does_not_trigger_reverse_title_fallback():
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    for index in range(7):
        left = 100 + index * 115
        draw.rectangle((left, 1120, left + 68, 1240), fill="black")
    assert _light_on_dark_regions(image) == []


def test_outlined_light_sfx_recovery_uses_source_pixel_components():
    image = Image.new("RGB", (1000, 1200), (175, 175, 175))
    draw = ImageDraw.Draw(image)
    for left, top, width, height in (
        (380, 360, 95, 180),
        (490, 410, 90, 140),
        (610, 455, 60, 95),
    ):
        draw.rectangle(
            (left - 10, top - 10, left + width + 10, top + height + 10),
            fill="black",
        )
        draw.rectangle((left, top, left + width, top + height), fill="white")

    regions = _outlined_light_text_regions_near(image, [350, 330, 520, 500], 0.82)

    assert len(regions) == 3
    assert all(region["outlined_fallback"] is True for region in regions)
    assert min(region["box"][0] for region in regions) < 380
    assert max(region["box"][2] for region in regions) > 670

    single = _outlined_light_text_regions_near(
        image, [350, 330, 430, 430], 0.82, minimum_components=1
    )
    assert len(single) >= 1


def test_sparse_full_page_display_text_is_reread_as_one_cover_title(tmp_path):
    source = tmp_path / "cover.png"
    Image.new("RGB", (1000, 1400), "white").save(source)
    units = [
        TextUnit("p001u01", [580, 160, 720, 330], [570, 150, 730, 340], "さて", 0.9),
        TextUnit("p001u02", [560, 380, 720, 560], [550, 370, 730, 570], "にし", 0.9),
        TextUnit("p001u03", [300, 590, 520, 930], [290, 580, 530, 940], "勇気", 0.9),
        TextUnit(
            "p001u04",
            [320, 950, 500, 1320],
            [310, 940, 510, 1330],
            "があったなら",
            0.9,
        ),
    ]
    page = PageOCR(1, source.name, 1000, 1400, units)
    assert PaddleMangaOCR.is_cover_title_candidate(page) is True

    service = object.__new__(PaddleMangaOCR)
    service.reader = lambda _crop: "僕に勇気があったなら"
    refined = service.refine_cover_title(source, page)
    assert len(refined.units) == 1
    assert refined.units[0].ja == "僕に勇気があったなら"
    assert refined.units[0].special == "cover_title"
    assert len(refined.units[0].erase_boxes) == 4


def test_recognition_candidates_can_add_but_never_delete_detector_regions():
    primary = [
        {"box": [10, 10, 80, 120], "text": "", "score": 0.9},
        {"box": [200, 10, 280, 120], "text": "", "score": 0.8},
    ]
    secondary = [
        {"box": [12, 12, 78, 118], "text": "日本", "score": 0.7},
        {"box": [400, 10, 480, 120], "text": "追加", "score": 0.7},
    ]
    merged = merge_region_candidates(primary, secondary)
    assert [item["box"] for item in merged] == [
        [10, 10, 80, 120],
        [200, 10, 280, 120],
        [400, 10, 480, 120],
    ]


def test_single_glyph_spread_over_artwork_loses_its_text():
    """A 1-glyph "box" covering 160x137 px of drawing is a mis-read, not a line."""
    payload = {
        "rec_boxes": [[339, 966, 499, 1103], [275, 919, 357, 940]],
        "rec_texts": ["X", "UHM..I-I MEAN."],
        "rec_scores": [0.896, 0.798],
    }
    regions = PaddleMangaOCR._regions(payload)
    assert [item["text"] for item in regions] == ["", "UHM..I-I MEAN."]


def test_english_region_keeps_paddle_text_instead_of_mangaocr_kana():
    """MangaOCR hallucinates kana for English bubbles; Paddle reads them fine."""
    group = [{"box": [10, 10, 300, 120], "text": "STUPID MASK TODAY?", "score": 0.94}]
    text, score = _best_recognized_text(group)
    assert _is_latin_region(text, score) is True
    # Without this guard the Japanese-only reader replaced the line with
    # made-up kana that still passed the JP_RE filter.
    assert _is_latin_region("いや、そういうことですが、", 0.94) is False
    assert _is_latin_region("STUPID MASK", 0.40) is False
    assert _is_latin_region("２０１５年１０月１４日", 0.95) is False


def test_latin_page_drops_text_less_artwork_boxes():
    """A Latin page keeps only boxes that carry a string.

    Text-less detector boxes are artwork on an English page; keeping them
    inflated unit bboxes and bridged two balloons into one unit.
    """
    latin = [
        {"box": [10, 10, 300, 40], "text": "ASK YOUR MOTHER", "score": 0.94},
        {"box": [10, 400, 300, 600], "text": "", "score": 0.90},
    ]
    assert is_latin_page(latin) is True
    assert is_latin_page([{"box": [0, 0, 9, 9], "text": "それは", "score": 0.9}]) is False


def test_rescue_reading_stability_tells_lettering_from_artwork():
    """Measured separation on a cover title, a wooden sign and thirteen figures.

    Real lettering agreed at 0.75-1.00, every invention at 0.00-0.40.
    """
    assert rescue_reading_is_stable(
        [
            "調教室地域２４号四番以外立人禁止",
            "調教室地隊２４号駅都外立人禁止",
            "調教室地域２４号県有り外立人禁止",
            "調教室地域２４号関係者以外立人禁止",
        ]
    )
    assert not rescue_reading_is_stable(
        ["２０１８年０", "Ｐｏｏｄ", "２０１８年１０月２０日", "２０１８年０"]
    )
    # MangaOCR returned nothing for one crop; that is not a stable reading.
    assert not rescue_reading_is_stable(["その", "", "その"])


def _textless_box_service(tmp_path, readings):
    source = tmp_path / "artwork.png"
    Image.new("RGB", (800, 1129), "white").save(source)

    class Detector:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "dt_polys": [
                            [[372, 521], [664, 521], [664, 715], [372, 715]]
                        ],
                        "dt_scores": [0.655],
                    }
                }
            ]

    pending = iter(readings)
    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.recognition_detector = None
    service.reader = lambda _crop: next(pending)
    return service, source


def test_textless_artwork_reading_is_dropped_when_crops_disagree(tmp_path):
    """A ribbon swirl was translated as "２０１８年０" and stamped on the drawing.

    Paddle found the box but read nothing, and MangaOCR invents kana for any
    texture; when shifted crops disagree there was never a string there.
    """
    service, source = _textless_box_service(
        tmp_path, ["２０１８年０", "Ｐｏｏｄ", "２０１８年１０月２０日", "その"]
    )
    assert service.analyze(source, 1).units == []


def test_textless_display_lettering_is_rescued_when_every_crop_agrees(tmp_path):
    """A wooden sign defeats both Paddle passes yet reads the same every time."""
    service, source = _textless_box_service(
        tmp_path,
        [
            "調教室地域２４号四番以外立人禁止",
            "調教室地隊２４号駅都外立人禁止",
            "調教室地域２４号県有り外立人禁止",
            "調教室地域２４号関係者以外立人禁止",
        ],
    )
    page = service.analyze(source, 1)
    assert [unit.ja for unit in page.units] == ["調教室地域２４号四番以外立人禁止"]


def test_vertical_bubble_recovers_the_outer_column_its_detector_missed(tmp_path):
    """A three-column bubble arrived with only its two inner columns boxed.

    Paddle scored the thin left column below its detection floor once the page
    was downscaled, so "化けてくるの" stayed in Japanese on the rendered page.
    The column is now found from the page's own ink and folded into the reread,
    while the group's own boxes and readings stay untouched.
    """
    # Three vertical columns of dark glyphs on a light balloon.
    source = tmp_path / "bubble.png"
    image = Image.new("RGB", (800, 1143), "white")
    draw = ImageDraw.Draw(image)
    for left in (455, 486, 517):
        for top in range(700, 795, 18):
            draw.rectangle((left, top, left + 22, top + 12), fill=(20, 20, 20))
    image.save(source)

    class Detector:
        def predict(self, _image):
            # Only the two inner columns were boxed; the thin outer one was not.
            return [
                {
                    "res": {
                        "rec_boxes": [[486, 700, 517, 795], [517, 700, 548, 795]],
                        "rec_texts": ["それに", "わざわざ"],
                        "rec_scores": [0.72, 0.85],
                    }
                }
            ]

    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.recognition_detector = None
    readings = iter(
        [
            "それなのにわざわざ",  # production crop of the boxed columns
            "それなのにわざわざ私に化けてくるんですもの",  # widened reread
            # reproducibility crops of the widened box
            "それなのにわざわざ私に化けてくるんですもの",
            "それなのにわざわざ私に化けてくるんですもの",
            "それなのにわざわざ私に化けてくるんですもの",
        ]
    )
    service.reader = lambda _crop: next(readings)

    page = service.analyze(source, 1)
    assert [unit.ja for unit in page.units] == [
        "それなのにわざわざ私に化けてくるんですもの"
    ]
    # The outer column must be erased too, or its Japanese shows under the text.
    erase = page.units[0].erase_boxes
    assert min(box[0] for box in erase) <= 460


def test_widened_reading_is_rejected_when_it_loses_the_original_text(tmp_path):
    """A worse reread must leave the unit exactly as it was."""
    source = tmp_path / "bubble.png"
    image = Image.new("RGB", (800, 1143), "white")
    draw = ImageDraw.Draw(image)
    for left in (455, 486, 517):
        for top in range(700, 795, 18):
            draw.rectangle((left, top, left + 22, top + 12), fill=(20, 20, 20))
    image.save(source)

    class Detector:
        def predict(self, _image):
            # Only the two inner columns were boxed; the thin outer one was not.
            return [
                {
                    "res": {
                        "rec_boxes": [[486, 700, 517, 795], [517, 700, 548, 795]],
                        "rec_texts": ["それに", "わざわざ"],
                        "rec_scores": [0.72, 0.85],
                    }
                }
            ]

    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.recognition_detector = None
    readings = iter(["それなのにわざわざ私に", "ぐらい", "ぐらい", "ぐらい", "ぐらい"])
    service.reader = lambda _crop: next(readings)

    page = service.analyze(source, 1)
    assert [unit.ja for unit in page.units] == ["それなのにわざわざ私に"]


def test_declared_source_language_overrides_latin_detection():
    """Declaring the input script settles what ``auto`` has to infer.

    ``auto`` still asks the page; ``en`` forces the Latin path and ``ja`` forces
    the Japanese one, so a single English sign in a Japanese book cannot drag a
    kana line onto the Latin path and an all-kana English page still gets OCRed.
    """
    japanese = [{"box": [0, 0, 90, 90], "text": "それは", "score": 0.9}]
    latin = [{"box": [0, 0, 300, 40], "text": "ASK YOUR MOTHER", "score": 0.94}]

    assert page_reads_as_latin(japanese) is False
    assert page_reads_as_latin(latin) is True
    assert page_reads_as_latin(japanese, "en") is True
    assert page_reads_as_latin(latin, "ja") is False
    assert page_reads_as_latin(japanese, "auto") is False

    assert is_source_text("それは", "auto") is True
    assert is_source_text("ASK YOUR MOTHER", "auto") is False
    assert is_source_text("ASK YOUR MOTHER", "en") is True
    assert is_source_text("それは", "en") is False
    assert is_source_text("それは", "ja") is True

    assert language_label("en") == "English"
    assert language_label("ja") == "Japanese"
    assert language_label("auto") == "auto"
    assert language_label_zh("en") == "英语"
    assert language_label_zh("ja") == "日语"


def test_declared_english_book_never_calls_the_japanese_recognizer(tmp_path):
    """MangaOCR only reads Japanese; on a declared English book it can only
    invent kana, so the recognizer must not be consulted at all."""
    source = tmp_path / "page.png"
    Image.new("RGB", (800, 1129), "white").save(source)

    class Detector:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "rec_boxes": [[40, 60, 760, 110], [40, 300, 700, 420]],
                        "rec_texts": ["ASK YOUR MOTHER ABOUT THE STARS", ""],
                        "rec_scores": [0.94, 0.62],
                    }
                }
            ]

    def refuse(_crop):
        raise AssertionError("the Japanese recognizer must not run on an English book")

    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.recognition_detector = None
    service.reader = refuse
    service.source_language = "en"

    page = service.analyze(source, 1)
    assert [unit.ja for unit in page.units] == ["ASK YOUR MOTHER ABOUT THE STARS"]


def test_declared_japanese_book_treats_latin_reading_as_not_latin():
    """A stray Latin sign must not reroute a kana-only line."""
    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.source_language = "ja"
    assert service._reads_latin("ASK YOUR MOTHER", 0.95) is False
    assert service._reads_latin("それは", 0.95) is False

    service.source_language = "en"
    assert service._reads_latin("ASK YOUR MOTHER", 0.10) is True
    assert service._reads_latin("", 0.95) is False

    # Instances rebuilt through __new__ predate the setting.
    del service.source_language
    assert service._declared_language() == "auto"
    assert service._reads_latin("ASK YOUR MOTHER", 0.95) is True


def test_unknown_source_language_is_rejected():
    """A typo in the setting must fail loudly, not silently fall back to auto."""
    assert "klingon" not in SOURCE_LANGUAGES
    assert SOURCE_LANGUAGES == ("auto", "en", "ja")


def test_link_blocks_are_preserved_instead_of_translated():
    """A credit page's link list cannot be translated: the model can only echo
    it or replace it with a summary, and both are rejected, so the unit stayed
    unresolved and failed the whole 74-page book."""
    assert (
        link_block_text(
            "https://derpibooru.org/profiles/Alcor https://www.patreon.com/alcorart "
            "https://discordapp.com/invite/zdKRakh https://e621.net/post/index/1/alcor90"
        )
        is True
    )
    assert link_block_text("Twitter: @AlcorArt / www.alcorart.com") is True
    assert link_block_text("THANK YOU FOR READING MY COMIC!") is False
    assert link_block_text("I have 3.5 hours left.") is False
    assert link_block_text("") is False


def test_english_only_book_survives_the_japanese_character_filter():
    """A page of pure English must not be dropped for lacking JP_RE matches."""
    assert JP_RE.search("DO YOU REALLY HAVE TO WEAR THAT STUPID MASK TODAY?") is None
    text, score = _best_recognized_text(
        [
            {"box": [10, 10, 300, 120], "text": "DO YOU", "score": 0.88},
            {"box": [10, 130, 300, 240], "text": "HAVE TO WEAR", "score": 0.91},
        ]
    )
    assert (text, score) == ("DO YOU HAVE TO WEAR", 0.91)
    assert _is_latin_region(text, score) is True


def test_semantic_ocr_cannot_replace_a_complete_line_with_its_prefix():
    assert (
        prefer_semantic_ocr("私っ妊娠とか．．．無理だって！！", "私っ妊娠とか…", 0.92)
        is False
    )
    assert prefer_semantic_ocr(
        "行ったとか聞いてたけど", "東京に行ったとか聞いてたけど", 0.9
    )


def test_full_page_semantics_cannot_swap_neighbouring_region_text():
    assert (
        prefer_semantic_ocr("ってかサナ．．．", "おっぱいデカくね！？", 0.99) is False
    )


def test_missing_recovery_drops_nested_duplicate_groups():
    broad = [
        {"box": [440, 560, 610, 1040], "score": 0.9},
        {"box": [590, 580, 850, 1030], "score": 0.9},
    ]
    nested = [{"box": [450, 575, 845, 725], "score": 0.9}]

    accepted = deduplicate_nested_region_groups([nested, broad])

    assert len(accepted) == 1
    assert accepted[0][1] == [440, 560, 850, 1040]


def test_recovered_effect_units_drop_nested_duplicate_fragments():
    broad = TextUnit(
        "p014u01",
        [440, 560, 850, 1040],
        [440, 560, 850, 1040],
        "バクンッ\nバクンッ\nバクンッ",
        0.9,
        is_sfx=True,
    )
    nested = TextUnit(
        "p014u02",
        [450, 575, 845, 725],
        [450, 575, 845, 725],
        "バクンッ",
        0.9,
        is_sfx=True,
    )
    page = PageOCR(14, "014.png", 1000, 1200, [nested, broad])

    result = prune_nested_duplicate_units(page)

    assert result.units == [broad]
    assert broad.id == "p014u01"


def test_a_different_sound_inside_a_larger_effect_is_not_a_duplicate():
    """Geometry alone must not delete a translation.

    ``びっ`` sits inside the same patch of screentone as ``バアン`` on the
    Japanese test page; dropping it as a nested duplicate lost a sound the
    reader expects for no reason other than overlap.
    """
    broad = TextUnit(
        "p001u01",
        [228, 616, 268, 689],
        [228, 616, 268, 689],
        "バアン",
        0.9,
        is_sfx=True,
    )
    nested = TextUnit(
        "p001u02",
        [231, 660, 268, 689],
        [231, 660, 268, 689],
        "びっ",
        0.9,
        is_sfx=True,
    )
    page = PageOCR(1, "001.png", 1000, 1200, [nested, broad])

    result = prune_nested_duplicate_units(page)

    assert [unit.id for unit in result.units] == ["p001u01", "p001u02"]


def test_semantic_sfx_classification_preserves_uncertain_fragments():
    assert semantic_sfx_classification("い！！", 0.62, False) is True
    assert semantic_sfx_classification("し", 0.0, False) is True
    assert semantic_sfx_classification("いい", 0.0, False) is False


def test_semantic_sfx_classification_recovers_confident_narration():
    assert semantic_sfx_classification("ついに、", 0.92, True) is False


def test_semantic_refiner_applies_fragment_gate_when_vlm_omits_a_region(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(source)
    page = PageOCR(
        1,
        source.name,
        400,
        600,
        [
            TextUnit("p001u01", [10, 10, 50, 50], [5, 5, 55, 55], "い！！", 0.62),
            TextUnit("p001u02", [80, 10, 180, 60], [75, 5, 185, 65], "ついに、", 0.92),
        ],
    )
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision-local")
    service._chat = lambda _images, _prompt, _schema: {
        "regions": [
            {
                "id": "p001u02",
                "text": "ついに、",
                "score": 0.92,
                "is_sfx": True,
            }
        ],
        "missing": [],
    }
    refined, _missing = service.refine(source, page)
    assert refined.units[0].is_sfx is True
    assert refined.units[1].is_sfx is False


def test_ollama_vision_ocr_uses_local_image_endpoint(monkeypatch, tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(source)
    response_payload = {
        "message": {
            "content": json.dumps(
                {
                    "regions": [
                        {
                            "bbox": [0.1, 0.2, 0.4, 0.5],
                            "text": "こんにちは",
                            "score": 0.9,
                            "is_sfx": False,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        }
    }
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return io.BytesIO(
            json.dumps(response_payload, ensure_ascii=False).encode("utf-8")
        )

    monkeypatch.setattr("manga_localizer.ocr.urlopen", fake_urlopen)
    page = OllamaVisionOCR("http://127.0.0.1:11434", "vision-local").analyze(source, 2)
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["body"]["model"] == "vision-local"
    assert captured["body"]["messages"][0]["images"]
    assert page.units[0].bbox == [40, 120, 160, 300]
    assert page.units[0].id == "p002u01"


def test_ollama_vision_ocr_retries_transient_server_error(monkeypatch, tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(source)
    response_payload = {
        "message": {
            "content": json.dumps(
                {"regions": []},
                ensure_ascii=False,
            )
        }
    }
    calls = []

    def flaky_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        if len(calls) == 1:
            raise HTTPError(
                request.full_url,
                500,
                "Internal Server Error",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"model runner unexpectedly stopped"}'),
            )
        return io.BytesIO(json.dumps(response_payload).encode("utf-8"))

    monkeypatch.setattr("manga_localizer.ocr.urlopen", flaky_urlopen)
    monkeypatch.setattr("manga_localizer.ocr.time.sleep", lambda _delay: None)

    page = OllamaVisionOCR("http://127.0.0.1:11434", "vision-local").analyze(
        source, 2
    )

    assert page.units == []
    assert len(calls) == 2


def test_ollama_vision_ocr_surfaces_non_retryable_error_detail(monkeypatch, tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(source)
    calls = []

    def rejected_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        raise HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"model does not support images"}'),
        )

    monkeypatch.setattr("manga_localizer.ocr.urlopen", rejected_urlopen)

    with pytest.raises(RuntimeError, match="model does not support images"):
        OllamaVisionOCR("http://127.0.0.1:11434", "text-only").analyze(source, 2)

    assert len(calls) == 1


def test_ollama_vision_ocr_retries_truncated_structured_output(
    monkeypatch, tmp_path
):
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(source)
    calls = []

    def truncated_then_complete(request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        calls.append(
            (body["options"]["num_predict"], body["options"].get("repeat_penalty"))
        )
        if len(calls) == 1:
            payload = {
                "done_reason": "length",
                "message": {"content": '{"regions":['},
            }
        else:
            payload = {"done_reason": "stop", "message": {"content": '{"regions":[]}'}}
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("manga_localizer.ocr.urlopen", truncated_then_complete)
    monkeypatch.setattr("manga_localizer.ocr.time.sleep", lambda _delay: None)

    page = OllamaVisionOCR("http://127.0.0.1:11434", "vision-local").analyze(
        source, 2
    )

    assert page.units == []
    # A dense page can degenerate into repeating one glyph with ever-growing
    # coordinates until the budget runs out, so the retry must also penalise
    # repeats; raising the budget alone never terminates.
    assert calls == [(8192, None), (16_384, 1.3)]


def test_ollama_vision_ocr_rejects_repeated_truncation(monkeypatch, tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(source)
    budgets = []
    contexts = []

    def always_truncated(request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        budgets.append(body["options"]["num_predict"])
        contexts.append(body["options"]["num_ctx"])
        payload = {
            "done_reason": "length",
            "message": {"content": '{"regions":['},
        }
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("manga_localizer.ocr.urlopen", always_truncated)
    monkeypatch.setattr("manga_localizer.ocr.time.sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="8192, 16384, 32768, 65536"):
        OllamaVisionOCR("http://127.0.0.1:11434", "vision-local").analyze(source, 2)

    # Truncation walks the whole budget plan, and every attempt asks for the
    # shared 64K working context so a dense page is never limited by it.
    assert budgets == [8192, 16_384, 32_768, 65_536]
    assert contexts == [65_536, 65_536, 65_536, 65_536]


def test_tight_crop_audit_does_not_overwrite_confident_sfx(monkeypatch, tmp_path):
    source = tmp_path / "crop-audit.png"
    Image.new("RGB", (600, 800), "white").save(source)
    page = PageOCR(
        1,
        source.name,
        600,
        800,
        [
            TextUnit(
                "p001u01",
                [20, 20, 45, 85],
                [20, 20, 45, 85],
                "お客様から何でしょうか",
                0.0,
                False,
                "错误译文",
            ),
            TextUnit(
                "p001u02",
                [320, 250, 560, 720],
                [320, 250, 560, 720],
                "カチャ",
                0.86,
                True,
                "",
            ),
            TextUnit(
                "p001u03",
                [100, 300, 260, 520],
                [100, 300, 260, 520],
                "１０月２９日",
                0.0,
                False,
                "10月29日",
            ),
        ],
    )
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": "p001u01",
                    "text": "",
                    "role": "nontext",
                    "confidence": 0.74,
                },
                {
                    "id": "p001u02",
                    "text": "僕は、秘密基地が好きだ。",
                    "role": "dialogue",
                    "confidence": 0.92,
                },
                {
                    "id": "p001u03",
                    "text": "ピロッ♪",
                    "role": "sfx",
                    "confidence": 0.9,
                },
            ]
        },
    )
    result = service.refine_local_crops(source, page)
    assert result.units[0].is_sfx is False
    assert result.units[1].ja == "カチャ"
    assert result.units[1].is_sfx is True
    assert result.units[2].is_sfx is True


def test_missing_only_audit_returns_unboxed_japanese(monkeypatch, tmp_path):
    source = tmp_path / "missing-only.png"
    Image.new("RGB", (600, 800), "white").save(source)
    page = PageOCR(
        1,
        source.name,
        600,
        800,
        [TextUnit("p001u01", [20, 20, 200, 300], [10, 10, 210, 310], "既存", 0.9)],
    )
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "missing": [
                {
                    "text": "ムチッ",
                    "bbox": [500, 400, 700, 600],
                    "score": 0.82,
                    "is_sfx": True,
                },
                {
                    "text": "not Japanese",
                    "bbox": [100, 100, 200, 200],
                    "score": 0.99,
                    "is_sfx": False,
                },
            ]
        },
    )

    result = service.find_missing(source, page)

    assert [item["text"] for item in result] == ["ムチッ"]


def test_tight_crop_audit_preserves_artwork_text(monkeypatch, tmp_path):
    source = tmp_path / "shirt-logo.png"
    Image.new("RGB", (600, 800), "white").save(source)
    unit = TextUnit(
        "p001u01",
        [120, 200, 480, 500],
        [120, 200, 480, 500],
        "HAREKITA",
        0.95,
        False,
        "错误译文",
    )
    page = PageOCR(1, source.name, 600, 800, [unit])
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": unit.id,
                    "text": "HAREKITA",
                    "role": "artwork",
                    "effect_kind": "none",
                    "confidence": 0.97,
                }
            ]
        },
    )

    result = service.refine_local_crops(source, page)

    assert result.units[0].skip is True
    assert result.units[0].skip_reason == "preserve"
    assert result.units[0].special == "artwork_text"
    assert result.units[0].zh == ""


def test_a_page_level_dialogue_verdict_survives_the_tight_crop_audit(
    monkeypatch, tmp_path
):
    """A stylised Latin aside is dialogue on the page, artwork in a crop."""

    source = tmp_path / "spanish-aside.png"
    Image.new("RGB", (600, 800), "white").save(source)
    unit = TextUnit(
        "p001u04",
        [240, 511, 319, 532],
        [236, 507, 323, 536],
        "SUPONGO...",
        0.93,
        False,
        "",
    )
    page = PageOCR(1, source.name, 600, 800, [unit])
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "regions": [
                {
                    "id": unit.id,
                    "text": "SUPONGO...",
                    "score": 0.93,
                    "role": "dialogue",
                }
            ],
            "missing": [],
        },
    )

    refined, _missing = service.refine(source, page)
    assert refined.units[0].special == "page_dialogue"
    assert refined.units[0].skip is False

    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": unit.id,
                    "text": "SUPONGO...",
                    "role": "artwork",
                    "effect_kind": "none",
                    "confidence": 0.95,
                }
            ]
        },
    )

    result = service.refine_local_crops(source, refined)

    assert result.units[0].skip is False, "the crop must not protect the aside"
    assert result.units[0].special == "", "the marker is not stored"


def test_the_page_dialogue_marker_never_overwrites_a_cover_title(
    monkeypatch, tmp_path
):
    """The marker is a guard for the crop audit, not a replacement label."""

    source = tmp_path / "cover-page.png"
    Image.new("RGB", (1200, 2000), "white").save(source)
    unit = TextUnit(
        "p001u01",
        [360, 280, 820, 1760],
        [340, 260, 840, 1780],
        "\u50d5\u306b\\n\u52c7\u6c17\u304c\u3042\u3063\u305f\u306a\u3089",
        0.96,
        False,
        "",
        special="cover_title",
    )
    page = PageOCR(1, source.name, 1200, 2000, [unit])
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "regions": [
                {
                    "id": unit.id,
                    "text": "\u50d5\u306b\u52c7\u6c17\u304c\u3042\u3063\u305f\u306a\u3089",
                    "score": 0.96,
                    "role": "narration",
                }
            ],
            "missing": [],
        },
    )

    refined, _missing = service.refine(source, page)

    assert refined.units[0].special == "cover_title"


def test_tight_crop_audit_accepts_latin_logo_over_suspicious_japanese_ocr(
    monkeypatch, tmp_path
):
    source = tmp_path / "shirt-logo-misread.png"
    Image.new("RGB", (1200, 1000), "white").save(source)
    unit = TextUnit(
        "p021u04",
        [300, 200, 900, 500],
        [280, 180, 920, 520],
        "いろいろな",
        0.9,
        zh="各种事情",
        erase_boxes=[[300, 200, 900, 500]],
    )
    page = PageOCR(21, source.name, 1200, 1000, [unit])
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": unit.id,
                    "text": "HAREKITA",
                    "role": "artwork",
                    "effect_kind": "none",
                    "confidence": 0.98,
                }
            ]
        },
    )

    result = service.refine_local_crops(source, page)

    assert result.units[0].skip is True
    assert result.units[0].special == "artwork_text"
    assert result.units[0].zh == ""


def test_crop_audit_prunes_detached_horizontal_logo_before_dialogue_review(
    monkeypatch, tmp_path
):
    source = tmp_path / "mixed-dialogue-logo.png"
    Image.new("RGB", (2000, 3000), "white").save(source)
    unit = TextUnit(
        "p021u06",
        [15, 1500, 1850, 2700],
        [0, 1450, 1900, 2750],
        "おっぱい\nデカくね!?",
        0.9,
        skip=True,
        skip_reason="preserve",
        erase_boxes=[
            [150, 2000, 350, 2550],
            [15, 1980, 190, 2700],
            [400, 1500, 1850, 2300],
        ],
        special="artwork_text",
    )
    page = PageOCR(21, source.name, 2000, 3000, [unit])
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": unit.id,
                    "text": "おっぱいデカくね!?",
                    "role": "dialogue",
                    "effect_kind": "none",
                    "confidence": 0.98,
                }
            ]
        },
    )

    result = service.refine_local_crops(source, page)

    assert result.units[0].erase_boxes == [
        [150, 2000, 350, 2550],
        [15, 1980, 190, 2700],
    ]
    assert result.units[0].skip is False
    assert result.units[0].special == "semantic_dialogue"


def test_cover_title_cannot_be_reclassified_as_artwork(monkeypatch, tmp_path):
    source = tmp_path / "cover-title.png"
    Image.new("RGB", (1200, 2000), "white").save(source)
    unit = TextUnit(
        "p001u01",
        [360, 280, 820, 1760],
        [340, 260, 840, 1780],
        "僕に\n勇気があったなら",
        0.96,
        False,
        "",
        special="cover_title",
    )
    page = PageOCR(1, source.name, 1200, 2000, [unit])
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": unit.id,
                    "text": "僕に勇気があったなら",
                    "role": "artwork",
                    "effect_kind": "none",
                    "confidence": 0.98,
                }
            ]
        },
    )

    result = service.refine_local_crops(source, page)

    assert result.units[0].skip is False
    assert result.units[0].is_sfx is False
    assert result.units[0].special == "cover_title"


def test_japanese_grammar_overrides_wrong_vlm_sfx_role(monkeypatch, tmp_path):
    source = tmp_path / "wrong-role.png"
    Image.new("RGB", (600, 800), "white").save(source)
    unit = TextUnit(
        "p015u06",
        [0, 200, 440, 700],
        [0, 200, 440, 700],
        "だ．．．俺だって！",
        0.9,
    )
    page = PageOCR(15, source.name, 2000, 3000, [unit])
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": unit.id,
                    "text": "だ…誰だ！？",
                    "role": "sfx",
                    "effect_kind": "none",
                    "confidence": 0.98,
                }
            ]
        },
    )

    result = service.refine_local_crops(source, page)

    assert result.units[0].ja == "だ…誰だ！？"
    assert result.units[0].is_sfx is False
    assert result.units[0].special == "semantic_dialogue"


def test_exact_crop_recovers_missed_sfx_from_auto_duplicate_skip(monkeypatch, tmp_path):
    source = tmp_path / "missed-sfx.png"
    Image.new("RGB", (600, 800), "white").save(source)
    unit = TextUnit(
        "p015u09",
        [200, 200, 260, 230],
        [200, 200, 260, 230],
        "いや、",
        0.9,
        skip=True,
        skip_reason="duplicate",
        special="ocr_duplicate",
    )
    page = PageOCR(15, source.name, 600, 800, [unit])
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": unit.id,
                    "text": "ゴッ",
                    "role": "sfx",
                    "effect_kind": "impact",
                    "confidence": 0.85,
                }
            ]
        },
    )

    result = service.refine_local_crops(source, page)

    assert result.units[0].ja == "ゴッ"
    assert result.units[0].is_sfx is True
    assert result.units[0].skip is False
    assert result.units[0].skip_reason == ""
    assert result.units[0].special == "sfx:impact"


def test_a_decoder_that_never_varies_its_scores_keeps_its_proposals():
    """llama.cpp answered the same -0.0127 for every region, twice in a row.

    The recall gate exists to throw away low-confidence inventions, but on that
    transport it threw away the entire pass -- including the two real lines the
    decoder had just proposed.  A flat or saturated number carries no signal, so
    the reading itself has to be the filter.
    """
    assert scores_are_informative([0.9933, 0.709, 0.95, 0.7722]) is True
    assert scores_are_informative([-0.0126953125, -0.0126953125]) is False
    assert scores_are_informative([1.0, 1.0, 1.0, 1.0]) is False
    assert scores_are_informative([]) is False
    assert scores_are_informative([0.9, 0.9, 0.9]) is False


def test_ink_tiles_propose_only_uncovered_areas_with_ink():
    """A deterministic pass has to skip the boxes the detector already owns."""
    covered = TextUnit(
        "p001u01",
        [0, 0, 333, 285],
        [0, 0, 333, 285],
        "あ",
        0.9,
    )
    page = PageOCR(1, "001.png", 999, 1140, [covered])
    gray = np.full((1140, 999), 255, dtype=np.uint8)
    gray[300:560, 350:660] = 30  # ink inside the second tile
    gray[0:280, 0:330] = 30  # ink the detector already covers

    hints = ink_tile_hints(page, gray, columns=3, rows=4)

    assert len(hints) == 1
    assert hints[0]["is_sfx"] is True
    assert hints[0]["bbox"] == [333, 250, 667, 500]


def test_ink_tiles_stay_quiet_on_an_empty_page():
    page = PageOCR(
        1,
        "001.png",
        999,
        1140,
        [TextUnit("p001u01", [0, 0, 100, 100], [0, 0, 100, 100], "あ", 0.9)],
    )
    gray = np.full((1140, 999), 255, dtype=np.uint8)

    assert ink_tile_hints(page, gray) == []

def test_the_japanese_floor_drops_the_sup_bubble_paddle_read_correctly():
    """Paddle read "'Sup" and the confidence floor threw the box away.

    The bubble's recognized box is 43x27 px and its string scored 0.67, under
    the 0.72 floor, so the page reached the renderer without the bubble at all.
    """
    payload = {
        "rec_boxes": [[385, 679, 428, 706]],
        "rec_texts": ["'Sup"],
        "rec_scores": [0.6686],
    }

    assert PaddleMangaOCR._regions(payload) == []
    kept = PaddleMangaOCR._regions(payload, LATIN_EFFECT_SCORE_FLOOR)

    assert [(item["text"], item["box"]) for item in kept] == [
        ("'Sup", [385, 679, 428, 706])
    ]


def test_the_latin_floor_still_drops_a_tiny_misreading():
    """Relaxing the floor must not let one-glyph noise through.

    A 15x7 px box scored 0.61 on an English page; it stays under the 650 px^2
    floor that only a confident string of two characters or more may cross.
    """
    payload = {
        "rec_boxes": [[146, 78, 161, 85], [700, 1109, 794, 1122]],
        "rec_texts": ["5", "MY DICKYOU CUmE."],
        "rec_scores": [0.61, 0.6315],
    }

    kept = PaddleMangaOCR._regions(payload, LATIN_EFFECT_SCORE_FLOOR)

    assert [item["text"] for item in kept] == ["MY DICKYOU CUmE."]


def test_a_declared_latin_page_keeps_a_low_confidence_bubble(tmp_path):
    """The floor is chosen per page, not per region.

    The detector payload carries no text; the recognizer's box for the bubble
    scored 0.67 and only the lower floor lets it in.
    """
    source = tmp_path / "page.png"
    Image.new("RGB", (800, 1129), "white").save(source)

    class Detector:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "dt_polys": [
                            [[40, 60], [760, 60], [760, 110], [40, 110]],
                        ],
                        "dt_scores": [0.9],
                    }
                }
            ]

    class Recognizer:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "rec_boxes": [
                            [40, 60, 760, 110],
                            [385, 679, 428, 706],
                        ],
                        "rec_texts": ["ASK YOUR MOTHER ABOUT THE STARS", "'SUP"],
                        "rec_scores": [0.94, 0.6686],
                    }
                }
            ]

    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.recognition_detector = Recognizer()
    service.reader = lambda _crop: "それなのに"
    service.profile = "quality"
    service.source_language = "en"

    page = service.analyze(source, 1)

    assert [unit.ja for unit in page.units] == [
        "ASK YOUR MOTHER ABOUT THE STARS",
        "'SUP",
    ]


def test_a_declared_japanese_page_does_not_relax_its_floor(tmp_path):
    """A Japanese book keeps the floor that was tuned on its own pages.

    The Latin bubble beside a kana caption is dropped there: on a Japanese page
    a low-score Latin string is a misreading until the page says otherwise.
    """
    source = tmp_path / "page.png"
    Image.new("RGB", (800, 1129), "white").save(source)

    class Detector:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "dt_polys": [
                            [[40, 60], [760, 60], [760, 110], [40, 110]],
                        ],
                        "dt_scores": [0.9],
                    }
                }
            ]

    class Recognizer:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "rec_boxes": [
                            [40, 60, 760, 110],
                            [385, 679, 428, 706],
                        ],
                        "rec_texts": ["それはなのに", "'SUP"],
                        "rec_scores": [0.94, 0.6686],
                    }
                }
            ]

    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.recognition_detector = Recognizer()
    service.reader = lambda _crop: "それなのに"
    service.profile = "quality"
    service.source_language = "ja"

    page = service.analyze(source, 1)

    assert [unit.ja for unit in page.units] == ["それなのに"]
