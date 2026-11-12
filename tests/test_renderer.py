import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_localizer.config import AppPaths
from manga_localizer.ocr import PageOCR, TextUnit
from manga_localizer.renderer import (
    CLOSING_PUNCTUATION,
    MIN_READABLE_FONT,
    ArtworkPreservingRenderer,
    TextStyle,
    _fit_readable,
    _font,
    _lines_fit,
    _text_padding,
    _tokens,
    _wrap,
    ensure_font,
    find_bold_font,
)


def test_a_latin_source_is_never_typeset_vertically():
    """A nearly square Latin box must not become right-to-left columns.

    ``"'Sup"`` in a 43x27 box met the ``height >= width * 0.62`` rule
    meant for stacks of Japanese columns, so the three-character
    translation was drawn one character per column starting at the
    right edge: it read backwards (\u300c\uff1f\u6837\u548b\u300d).
    """
    rgb = np.full((70, 70, 3), 240, dtype=np.uint8)
    box = [8, 16, 51, 43]
    latin = TextUnit("p001u10", box, list(box), "'Sup", 0.93)
    latin.zh = "\u548b\u6837\uff1f"
    assert ArtworkPreservingRenderer._analyze_style(rgb, latin).vertical is False

    japanese = TextUnit("p001u01", box, list(box), "\u305d\u308c\u306a\u306e", 0.93)
    japanese.zh = "\u7136\u800c"
    assert ArtworkPreservingRenderer._analyze_style(rgb, japanese).vertical is True


def _measured_font():
    """A real CJK font, or a skip: the wrapping tests measure glyph widths."""
    try:
        return find_bold_font()
    except FileNotFoundError:
        pytest.skip("no CJK font installed to measure glyph widths with")


def _measuring_renderer(tmp_path):
    font = _measured_font()
    root = tmp_path / "home"
    paths = AppPaths(
        root, root / "models", root / "cache", root / "jobs", root / "settings.json"
    )
    return ArtworkPreservingRenderer(
        paths=paths,
        font_path=font,
        bold_font_path=font,
        cleanup_profile="quality",
    )


class BoundedTestRenderer(ArtworkPreservingRenderer):
    def __init__(self):
        self.font_path = None

    def _draw(self, image, text, box, style):
        del text, style
        x0, y0, x1, y1 = box
        ImageDraw.Draw(image).rectangle((x0 + 8, y0 + 8, x1 - 8, y1 - 8), fill="black")


def test_renderer_preserves_pixels_outside_text_box(tmp_path):
    source = tmp_path / "page.jpg"
    base = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(base)
    draw.rectangle((8, 8, 311, 231), outline="black", width=5)
    draw.text((120, 95), "TEST", fill="black")
    base.save(source, quality=100, subsampling=0)
    page = PageOCR(
        1,
        source.name,
        320,
        240,
        [
            TextUnit(
                "p001u01",
                [90, 70, 235, 170],
                [90, 70, 235, 170],
                "テスト",
                0.9,
                False,
                "测试",
            )
        ],
    )
    output = tmp_path / "page.png"
    BoundedTestRenderer().render_page(source, page, output)
    before = np.asarray(Image.open(source).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    mask = np.ones(before.shape[:2], dtype=bool)
    mask[70:170, 90:235] = False
    assert output.exists()
    assert before.shape == after.shape
    assert np.array_equal(before[mask], after[mask])


def test_grouped_text_erases_only_tight_detector_boxes(tmp_path):
    class EraseOnlyRenderer(ArtworkPreservingRenderer):
        def __init__(self):
            self.font_path = None

        def _draw(self, *_args, **_kwargs):
            return None

    source = tmp_path / "grouped.png"
    base = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(base)
    draw.rectangle((20, 75, 220, 105), fill="black")
    draw.text((45, 35), "TOP", fill="black")
    draw.text((145, 125), "BOTTOM", fill="black")
    base.save(source)
    page = PageOCR(
        1,
        source.name,
        240,
        180,
        [
            TextUnit(
                "p001u01",
                [35, 25, 215, 155],
                [35, 25, 215, 155],
                "テスト",
                0.9,
                False,
                "测试",
                erase_boxes=[[35, 25, 95, 65], [135, 115, 215, 155]],
            )
        ],
    )
    output = tmp_path / "grouped-output.png"
    EraseOnlyRenderer().render_page(source, page, output)
    before = np.asarray(Image.open(source).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    assert np.array_equal(before[75:106, 20:221], after[75:106, 20:221])


def test_quality_lama_mask_never_uses_group_union_as_erase_geometry(tmp_path):
    class MaskPaintingInpainter:
        def __call__(self, crop, mask):
            result = crop.copy()
            result[mask > 0] = (255, 0, 255)
            return result

    source = tmp_path / "grouped-outlined.png"
    image = np.full((220, 320, 3), 180, dtype=np.uint8)
    # Two outlined text columns surround artwork that belongs to the panel.
    cv2.rectangle(image, (35, 35), (85, 185), (255, 255, 255), 12)
    cv2.rectangle(image, (45, 45), (75, 175), (0, 0, 0), -1)
    cv2.rectangle(image, (235, 35), (285, 185), (255, 255, 255), 12)
    cv2.rectangle(image, (245, 45), (275, 175), (0, 0, 0), -1)
    cv2.rectangle(image, (130, 60), (190, 160), (15, 15, 15), -1)
    Image.fromarray(image).save(source)
    page = PageOCR(
        1,
        source.name,
        320,
        220,
        [
            TextUnit(
                "p001u01",
                [20, 20, 300, 200],
                [20, 20, 300, 200],
                "日本語題名",
                1.0,
                False,
                "中文标题",
                erase_boxes=[[25, 25, 95, 195], [225, 25, 295, 195]],
            )
        ],
    )
    renderer = ArtworkPreservingRenderer(
        cleanup_profile="quality",
        inpainter=MaskPaintingInpainter(),
    )
    renderer._draw = lambda *_args, **_kwargs: None
    output = tmp_path / "grouped-outlined-output.png"
    renderer.render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.array_equal(rendered[60:161, 130:191], image[60:161, 130:191])


def test_cover_title_uses_a_wider_per_box_outline_mask():
    ordinary_style = TextStyle(
        220, 0.6, "black", "white", 0.06, True, 100, True, True, 15, False
    )
    display_style = TextStyle(
        220, 0.6, "black", "white", 0.06, True, 228, True, True, 15, True
    )
    ordinary = TextUnit("ordinary", [0, 0, 100, 200], [0, 0, 100, 200], "題", 1.0)
    cover = TextUnit(
        "cover", [0, 0, 100, 200], [0, 0, 100, 200], "題", 1.0, special="cover_title"
    )
    assert ArtworkPreservingRenderer._mask_dilation(ordinary, ordinary_style) == 15
    assert ArtworkPreservingRenderer._mask_dilation(ordinary, display_style) == 51
    assert ArtworkPreservingRenderer._mask_dilation(cover, display_style) == 51


def test_large_outlined_cleanup_fills_only_detector_owned_boxes(tmp_path):
    class MaskPaintingInpainter:
        def __call__(self, crop, mask):
            result = crop.copy()
            result[mask > 0] = (255, 0, 255)
            return result

    source = tmp_path / "large-outlined.png"
    image = np.full((300, 600, 3), 80, dtype=np.uint8)
    Image.fromarray(image).save(source)
    unit = TextUnit(
        "display",
        [20, 20, 580, 280],
        [20, 20, 580, 280],
        "大きい文字",
        1.0,
        False,
        "大字",
        erase_boxes=[[40, 40, 180, 260], [420, 40, 560, 260]],
    )
    page = PageOCR(1, source.name, 600, 300, [unit])
    renderer = ArtworkPreservingRenderer(
        cleanup_profile="quality", inpainter=MaskPaintingInpainter()
    )
    renderer._draw = lambda *_args, **_kwargs: None
    renderer._analyze_style = lambda *_args: TextStyle(
        80, 0.0, "white", "black", 0.066, True, 160, True, True, 40, False
    )
    output = tmp_path / "large-outlined-output.png"
    renderer.render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.all(rendered[40:260, 40:180] == (255, 0, 255))
    assert np.all(rendered[40:260, 420:560] == (255, 0, 255))
    assert np.array_equal(rendered[40:260, 260:340], image[40:260, 260:340])


def test_textured_outlined_cleanup_preserves_background_outside_glyphs(tmp_path):
    class MaskRecordingInpainter:
        def __init__(self):
            self.mask = None

        def __call__(self, crop, mask):
            self.mask = mask.copy()
            result = crop.copy()
            result[mask > 0] = (255, 0, 255)
            return result

    source = tmp_path / "outlined-halo.png"
    gradient = np.linspace(145, 205, 260, dtype=np.uint8)[None, :, None]
    image = np.repeat(np.repeat(gradient, 300, axis=0), 3, axis=2)
    # Three separated glyph cores with contrasting outlines.  The background
    # is deliberately textured so this follows the former solid-box branch.
    for y in (65, 135, 205):
        cv2.rectangle(image, (95, y - 18), (165, y + 18), (255, 255, 255), 8)
        cv2.rectangle(image, (108, y - 10), (152, y + 10), (0, 0, 0), -1)
    Image.fromarray(image).save(source)
    unit = TextUnit(
        "outlined",
        [50, 30, 210, 270],
        [40, 20, 220, 280],
        "日本語",
        1.0,
        False,
        "中文",
        erase_boxes=[[80, 35, 180, 245]],
    )
    page = PageOCR(1, source.name, 260, 300, [unit])
    inpainter = MaskRecordingInpainter()
    renderer = ArtworkPreservingRenderer(
        cleanup_profile="quality", inpainter=inpainter
    )
    renderer._draw = lambda *_args, **_kwargs: None
    renderer._analyze_style = lambda *_args: TextStyle(
        180, 0.2, "black", "white", 0.066, True, 80, True, True, 24, False
    )
    output = tmp_path / "outlined-halo-output.png"
    renderer.render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert inpainter.mask is not None
    # The mask follows glyph shapes rather than filling the detector rectangle
    # plus a type-size halo.  Most nearby screentone remains byte-identical.
    assert np.count_nonzero(inpainter.mask) < 14_000
    changed = np.any(rendered != image, axis=2)
    assert changed.sum() < 14_000
    assert np.array_equal(rendered[35:245, 45:75], image[35:245, 45:75])
    assert np.array_equal(rendered[35:245, 185:215], image[35:245, 185:215])


def test_renderer_preserves_detached_garment_logo_from_legacy_group(tmp_path):
    class MaskPaintingInpainter:
        def __call__(self, crop, mask):
            result = crop.copy()
            result[mask > 0] = (255, 0, 255)
            return result

    source = tmp_path / "garment-logo.png"
    image = np.full((800, 600, 3), 180, dtype=np.uint8)
    for x in (35, 85):
        for y in (390, 470, 550):
            cv2.rectangle(image, (x - 8, y - 18), (x + 28, y + 18), (255, 255, 255), 6)
            cv2.rectangle(image, (x, y - 10), (x + 20, y + 10), (0, 0, 0), -1)
    # Detached artwork in the legacy semantic group must not become erase
    # authority merely because it is text-like.
    cv2.rectangle(image, (220, 260), (500, 360), (25, 25, 25), -1)
    Image.fromarray(image).save(source)
    unit = TextUnit(
        "mixed",
        [10, 200, 580, 700],
        [0, 180, 590, 720],
        "おっぱいデカくね!?",
        0.94,
        False,
        "胸是不是变大了！？",
        erase_boxes=[
            [20, 330, 80, 680],
            [70, 340, 130, 620],
            [150, 200, 580, 430],
        ],
    )
    page = PageOCR(1, source.name, 600, 800, [unit])
    renderer = ArtworkPreservingRenderer(
        cleanup_profile="quality", inpainter=MaskPaintingInpainter()
    )
    renderer._draw = lambda *_args, **_kwargs: None
    renderer._analyze_style = lambda *_args: TextStyle(
        180, 0.2, "black", "white", 0.066, True, 80, True, True, 24, True
    )
    output = tmp_path / "garment-logo-output.png"
    renderer.render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.any(np.all(rendered[360:590, 20:80] == (255, 0, 255), axis=2))
    assert np.any(np.all(rendered[360:590, 70:130] == (255, 0, 255), axis=2))
    assert np.array_equal(rendered[200:430, 180:560], image[200:430, 180:560])


def test_grouped_column_width_is_an_upper_bound_for_font_size():
    rgb = np.full((700, 500, 3), 210, dtype=np.uint8)
    for x in (70, 300):
        cv2.rectangle(rgb, (x, 80), (x + 120, 620), (255, 255, 255), 18)
        cv2.rectangle(rgb, (x + 18, 100), (x + 102, 600), (0, 0, 0), -1)
    unit = TextUnit(
        "group",
        [40, 50, 450, 650],
        [40, 50, 450, 650],
        "日本語の縦書き",
        1.0,
        False,
        "中文竖排文字",
        erase_boxes=[[50, 60, 210, 640], [280, 60, 440, 640]],
    )
    style = ArtworkPreservingRenderer._analyze_style(rgb, unit)
    assert style.font_size <= 122


def test_cleanup_preserves_artwork_connected_to_detector_edge():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[48:53, :] = 0
    image[25:40, 40:60] = 0
    cleaned, _, _ = ArtworkPreservingRenderer._erase(image.copy(), [0, 0, 100, 100])
    assert np.array_equal(cleaned[48:53, :], image[48:53, :])
    assert np.all(cleaned[28:37, 43:57] == 255)


def test_quality_cleanup_removes_boundary_connected_source_glyphs():
    image = np.full((60, 60, 3), 255, dtype=np.uint8)
    image[0:45, 28:33] = 0
    cleaned, _, _ = ArtworkPreservingRenderer._erase_complete(
        image.copy(), [0, 0, 60, 60]
    )
    assert np.all(cleaned[0:45, 28:33] == 255)


def test_typesetting_is_clipped_to_declared_box(tmp_path):
    class OverdrawRenderer(ArtworkPreservingRenderer):
        def __init__(self):
            self.font_path = None

        def _draw(self, image, *_args):
            ImageDraw.Draw(image).rectangle(
                (-10, -10, image.width + 10, image.height + 10), fill="black"
            )

    source = tmp_path / "clip.png"
    Image.new("RGB", (100, 100), "white").save(source)
    page = PageOCR(
        1,
        source.name,
        100,
        100,
        [
            TextUnit(
                "p001u01",
                [20, 20, 80, 80],
                [20, 20, 80, 80],
                "日本",
                1.0,
                False,
                "中国",
            )
        ],
    )
    output = tmp_path / "clip-output.png"
    OverdrawRenderer().render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.all(rendered[:20] == 255)
    assert np.all(rendered[80:] == 255)
    assert np.all(rendered[:, :20] == 255)
    assert np.all(rendered[:, 80:] == 255)


def test_managed_font_download_is_validated_and_atomic(monkeypatch, tmp_path):
    root = tmp_path / "home"
    paths = AppPaths(
        root, root / "models", root / "cache", root / "jobs", root / "settings.json"
    )
    payload = b"OTTO" + (b"0" * 1_000_000)
    monkeypatch.setattr(
        "manga_localizer.renderer.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    font = ensure_font(paths, force_managed=True)
    assert font.exists()
    assert font.read_bytes()[:4] == b"OTTO"
    assert not font.with_suffix(".otf.part").exists()


def test_display_style_is_detected_without_title_specific_hint():
    height, width = 1400, 600
    gradient = np.linspace(170, 245, height, dtype=np.uint8)[:, None]
    image = np.repeat(gradient, width, axis=1)
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    # Synthetic thick black glyph cores with white outline on textured artwork.
    for y in (180, 420, 660, 900):
        cv2.rectangle(rgb, (205, y), (395, y + 150), (255, 255, 255), 28)
        cv2.rectangle(rgb, (225, y + 20), (375, y + 130), (0, 0, 0), -1)
    unit = TextUnit(
        "title",
        [50, 80, 550, 1280],
        [50, 80, 550, 1280],
        "日本語題名",
        1.0,
        False,
        "中文标题",
    )
    style = ArtworkPreservingRenderer._analyze_style(rgb, unit)
    assert style.display is True
    assert style.outlined is True
    assert style.bold is True
    assert style.stroke_fill == "white"
    assert style.font_size > 92


def test_uniform_white_dialogue_uses_bold_without_fake_outline():
    rgb = np.full((400, 300, 3), 255, dtype=np.uint8)
    rgb[80:320, 120:180] = 0
    unit = TextUnit(
        "dialogue", [40, 30, 260, 370], [40, 30, 260, 370], "日本語", 1.0, False, "对白"
    )
    style = ArtworkPreservingRenderer._analyze_style(rgb, unit)
    assert style.bold is True
    assert style.outlined is False
    assert style.stroke_ratio == 0


def test_quality_plain_balloon_cleans_furigana_outside_main_bbox(tmp_path):
    class EraseOnlyQuality(ArtworkPreservingRenderer):
        def __init__(self):
            self.font_path = None
            self.cleanup_profile = "quality"

        def _draw(self, *_args, **_kwargs):
            return None

    source = tmp_path / "furigana.png"
    image = Image.new("RGB", (220, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((75, 55, 125, 125), fill="black")
    # Small annotation lies outside bbox but inside the reviewed crop.
    draw.rectangle((148, 65, 158, 95), fill="black")
    image.save(source)
    page = PageOCR(
        1,
        source.name,
        220,
        180,
        [
            TextUnit(
                "dialogue",
                [60, 40, 140, 140],
                [40, 30, 175, 150],
                "日本語",
                1.0,
                False,
                "对白",
            )
        ],
    )
    output = tmp_path / "furigana-output.png"
    EraseOnlyQuality().render_page(source, page, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert np.all(rendered[65:96, 148:159] == 255)


def test_fast_quality_profile_maps_to_the_artwork_cleanup(tmp_path):
    # The web UI and the API accept quality_profile="fast"; the renderer used to
    # reject that name and failed at the render step, after OCR and translation
    # had already run.  "fast" means no LaMa inpainting, i.e. artwork cleanup.
    renderer = ArtworkPreservingRenderer(cleanup_profile="fast")
    assert renderer.cleanup_profile == "artwork"
    assert ArtworkPreservingRenderer(cleanup_profile="quality").cleanup_profile == "quality"


def test_fast_profile_renders_without_loading_lama(tmp_path):
    def explode(*_args, **_kwargs):
        raise AssertionError("fast profile must not run LaMa inpainting")

    source = tmp_path / "fast.png"
    Image.new("RGB", (120, 90), "white").save(source)
    page = PageOCR(
        1,
        source.name,
        120,
        90,
        [TextUnit("p001u01", [10, 10, 110, 80], [10, 10, 110, 80], "日本語", 1.0, False, "中文")],
    )
    renderer = ArtworkPreservingRenderer(cleanup_profile="fast", inpainter=explode)
    renderer._draw = lambda *_args, **_kwargs: None
    output = tmp_path / "fast-out.png"
    renderer.render_page(source, page, output)
    assert output.exists()


def _balloon_page():
    """A 120x40 white ellipse on a pink page carrying a two-line text strip."""
    image = np.full((400, 400, 3), (232, 130, 180), dtype=np.uint8)
    cv2.ellipse(image, (200, 200), (60, 20), 0, 0, 360, (255, 255, 255), -1)
    for y in (190, 203):
        for x in range(160, 240, 20):
            cv2.rectangle(image, (x, y), (x + 12, y + 6), (10, 10, 10), -1)
    return image


def test_growing_follows_a_curved_balloon_instead_of_stalling():
    # Crawling one pixel at a time and re-testing each edge against the span
    # grown so far stalled on the curve: the 80x20 strip stopped near 95x32
    # inside the 120x40 balloon, and 12px CJK still did not fit.
    image = _balloon_page()
    grown = ArtworkPreservingRenderer._uniform_balloon_box(
        image, [160, 190, 240, 210]
    )
    assert grown[0] <= 141 and grown[2] >= 259
    assert grown[1] <= 181 and grown[3] >= 219
    assert grown[2] - grown[0] >= 118
    assert grown[3] - grown[1] >= 38


def test_typesetting_inside_a_curved_balloon_reaches_the_readable_floor(tmp_path):
    renderer = _measuring_renderer(tmp_path)
    image = _balloon_page()
    box, fitting = _typeset(image, renderer, [160, 190, 240, 210], "嘿，姑娘们，Twilight 在楼上等着呢。")
    assert box[0] < 160 and box[2] > 240
    assert fitting is not None
    assert fitting[1] >= MIN_READABLE_FONT


def _mid_tone_fixture():
    """Dark lettering on a coloured balloon, with dark art in the same window."""
    image = np.full((120, 160, 3), (200, 150, 180), dtype=np.uint8)
    for x in range(40, 120, 16):
        cv2.rectangle(image, (x, 40), (x + 8, 80), (15, 15, 15), -1)
    # A lock of hair, outside the lettering's ring but inside the window and
    # small enough to pass the component guard.
    cv2.rectangle(image, (10, 20), (30, 60), (15, 15, 15), -1)
    return image


def test_erase_keeps_dark_artwork_outside_the_lettering_ring():
    image = _mid_tone_fixture()
    window = [0, 0, 160, 120]
    # The focus box has to be mostly ground: a box filled with glyphs flips the
    # polarity vote and nothing is erased at all.
    focus = [30, 10, 140, 105]
    stale, _, _ = ArtworkPreservingRenderer._erase(
        image.copy(), window, focus=focus
    )
    assert _dark_pixels(stale, (10, 20, 31, 61)) == 0, "the hair was in reach"
    cleaned, _, _ = ArtworkPreservingRenderer._erase(
        image.copy(), window, focus=focus, bands=[[36, 36, 124, 84]]
    )
    assert _dark_pixels(cleaned, (10, 20, 31, 61)) > 700, "the hair must survive"
    assert _dark_pixels(cleaned, (40, 40, 120, 81)) == 0, "the glyphs must not"


def test_the_cleanup_bands_never_reach_far_past_the_lettering(tmp_path):
    renderer = _measuring_renderer(tmp_path)
    source = tmp_path / "window.png"
    image = Image.new("RGB", (300, 200), (232, 130, 180))
    draw = ImageDraw.Draw(image)
    draw.ellipse((80, 40, 220, 100), fill=(238, 150, 90))
    draw.rectangle((90, 60, 210, 70), fill=(10, 10, 10))
    draw.rectangle((100, 150, 200, 180), fill=(20, 20, 20))
    image.save(source)
    unit = TextUnit(
        "p001u01", [90, 60, 210, 70], [88, 58, 212, 72], "日本語", 1.0, False, "中文对白"
    )
    page = PageOCR(1, source.name, 300, 200, [unit])
    style = renderer._analyze_style(np.asarray(image).copy(), unit)
    seen = []

    def record(rgb, box, focus=None, bands=None):
        seen.append((list(box), [list(band) for band in bands or []]))
        return rgb, 0.0, 0.0

    renderer._erase = record
    renderer._draw = lambda *_args, **_kwargs: None
    renderer.render_page(source, page, tmp_path / "window-out.png")
    assert seen, "the quality profile cleans a flat balloon"
    window, bands = seen[0]
    across = max(8, round(style.font_size * 0.35))
    along = max(24, round(style.font_size * 1.2))
    # The searched rectangle stays generous on purpose: the paper branch and
    # the glyph polarity are read from it, and a crop holding nothing but
    # lettering makes both statistics meaningless.  The bands are what keeps
    # the promise, so the band is what has to stay near the lettering.
    assert bands and bands[0][3] <= 70 + across
    assert bands[0][2] <= 210 + along
    assert bands[0][3] < 150, "the hat 80px below must stay out of the band"
    assert window[3] < 200, "the search stays on the page"



def test_unknown_cleanup_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown cleanup profile"):
        ArtworkPreservingRenderer(cleanup_profile="turbo")


def test_cleanup_fills_erased_glyphs_with_the_local_page_colour():
    # Cream paper, not white: a solid #FFFFFF patch where the source text used
    # to be is more visible than the text it replaced.  The fill has to come
    # from the surrounding paper.
    paper = np.array([250, 243, 231], dtype=np.uint8)
    image = np.full((120, 240, 3), 255, dtype=np.uint8)
    image[:] = paper
    for x in range(70, 170, 20):
        cv2.rectangle(image, (x, 55), (x + 10, 80), (30, 30, 30), -1)
    cleaned, _, _ = ArtworkPreservingRenderer._erase(image.copy(), [40, 40, 200, 95])
    filled = cleaned[58:78, 80:160]
    assert np.all(np.abs(filled.astype(int) - paper.astype(int)) <= 2)


def test_cleanup_still_uses_white_when_the_page_is_white():
    image = np.full((60, 60, 3), 255, dtype=np.uint8)
    image[25:35, 20:40] = 0
    cleaned, _, _ = ArtworkPreservingRenderer._erase(image.copy(), [10, 15, 50, 45])
    assert np.all(cleaned[27:33, 23:37] >= 250)


def _polarity_fixture():
    """Dark lettering on a pink sign, with bright artwork beside it.

    Only the sign is mid-tone, so polling the padded erase region let the white
    panel on the right outvote six glyphs and the mask picked the empty side.
    """
    image = np.full((120, 160, 3), 45, dtype=np.uint8)
    image[:, 120:] = 255
    image[40:80, 40:120] = (200, 150, 180)
    for x in range(48, 112, 16):
        cv2.rectangle(image, (x, 52), (x + 8, 68), (15, 15, 15), -1)
    return image


def _dark_pixels(image, box=(48, 52, 112, 69)):
    x0, y0, x1, y1 = box
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return int((gray[y0:y1, x0:x1] < 120).sum())


def test_glyph_polarity_votes_only_inside_the_detector_box():
    image = _polarity_fixture()
    level, glyphs_are_dark = ArtworkPreservingRenderer._glyph_polarity(
        image, [40, 40, 120, 80]
    )
    assert glyphs_are_dark is True
    assert level == pytest.approx(168.0)
    assert _dark_pixels(image) == 612


def test_erase_uses_the_focus_box_to_pick_which_side_is_text():
    image = _polarity_fixture()
    # Without the focus the padded region votes and the glyphs survive: this is
    # the reported "the original text is still under the translation" bug.
    stale, _, _ = ArtworkPreservingRenderer._erase(image.copy(), [0, 0, 160, 120])
    assert _dark_pixels(stale) == 612
    cleaned, _, _ = ArtworkPreservingRenderer._erase(
        image.copy(), [0, 0, 160, 120], focus=[40, 40, 120, 80]
    )
    assert _dark_pixels(cleaned) == 0


def _plaque_page():
    """A 159x82 white plaque on a pink page with a 79x21 band of lettering."""
    image = np.full((400, 400, 3), (232, 130, 180), dtype=np.uint8)
    cv2.rectangle(image, (150, 180), (309, 262), (255, 255, 255), -1)
    cv2.rectangle(image, (150, 180), (309, 262), (60, 40, 50), 2)
    for x in range(160, 300, 18):
        cv2.rectangle(image, (x, 218), (x + 12, 232), (10, 10, 10), -1)
    return image


def _typeset(rgb, renderer, bbox, text):
    unit = TextUnit(
        "p001u01", bbox, [bbox[0] - 10, bbox[1] - 8, bbox[2] + 10, bbox[3] + 8],
        "ＳＵＰＯＮＧＯ", 1.0, True, text,
    )
    style = renderer._analyze_style(rgb, unit)
    box = renderer._typeset_box(rgb, unit, style)
    pad_x, pad_y = _text_padding(box)
    width = max(12, (box[2] - box[0]) - 2 * pad_x)
    height = max(12, (box[3] - box[1]) - 2 * pad_y)
    font_path = renderer.bold_font_path if style.bold else renderer.font_path
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    fitting = _fit_readable(font_path, draw, text, width, height, style.font_size)
    return box, fitting


def test_typesetting_keeps_a_roomy_balloon_at_its_declared_box(tmp_path):
    renderer = _measuring_renderer(tmp_path)
    rgb = np.full((400, 400, 3), 230, dtype=np.uint8)
    cv2.rectangle(rgb, (60, 60), (340, 340), (255, 255, 255), -1)
    box, fitting = _typeset(rgb, renderer, [80, 120, 320, 220], "这是一段比较长的对白文字")
    assert box == [80, 120, 320, 220]
    assert fitting is not None


def test_typesetting_grows_a_short_strip_into_its_plaque(tmp_path):
    renderer = _measuring_renderer(tmp_path)
    rgb = _plaque_page()
    # The detector box is one 21px tall line inside an 82px tall plaque: at the
    # declared height even 8px CJK does not fit, so the box has to grow.
    box, fitting = _typeset(rgb, renderer, [160, 211, 239, 232], "我想。这件事你得自己决定")
    assert box != [160, 211, 239, 232]
    # The plaque itself is the box now: the strip grows up to the plaque's top
    # inner edge and down to its bottom one.  The pixel crawl used to stop at
    # the strip's own bottom, because growing upward made the left and right
    # edge lines cross the dark border on every further step.
    assert box[1] <= 182 and box[3] >= 260
    assert fitting is not None
    assert fitting[1] >= MIN_READABLE_FONT


def test_typesetting_overflows_a_bubble_too_small_for_readable_cjk(tmp_path):
    renderer = _measuring_renderer(tmp_path)
    rgb = np.full((240, 240, 3), (30, 90, 120), dtype=np.uint8)
    cv2.rectangle(rgb, (90, 100), (114, 114), (255, 255, 255), -1)
    box, fitting = _typeset(rgb, renderer, [90, 100, 114, 114], "我想……")
    # A 24x14 bubble cannot hold 12px CJK.  Spilling past the border is the
    # chosen trade: an overlapping glyph is legible, an eight-pixel one is not.
    assert box[0] < 90 and box[1] < 100 and box[2] > 114 and box[3] > 114
    assert fitting is not None
    assert MIN_READABLE_FONT <= fitting[1] <= 14


def test_trailing_punctuation_hangs_instead_of_starting_a_line():
    font = _font(_measured_font(), 12)
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    assert draw.textbbox((0, 0), "我们，", font=font)[2] == 36
    assert draw.textbbox((0, 0), "我们", font=font)[2] == 24
    # The marks may hang past the inner width, so two glyphs plus a comma fit a
    # 24px balloon instead of the comma dropping to a line of its own.
    assert _lines_fit(draw, ["我们，"], font, 24) is True
    # A one-glyph line has no body to hang from: "不！！" at 39px is 46px wide
    # and one size smaller always fits, so it must not be accepted here.
    assert _lines_fit(draw, ["不！！"], font, 12) is False


def test_wrapping_keeps_latin_words_and_their_punctuation_together():
    font = _font(_measured_font(), 12)
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    assert _tokens("来吧，Twilight，") == ["来", "吧，", "Twilight，"]
    assert _tokens("SUPONGO...") == ["SUPONGO..."]
    # Only closing marks glue onto the unit before them; an opening bracket must
    # keep starting its own unit or "「那句话」" would wrap around it.
    assert "」" in CLOSING_PUNCTUATION
    assert "「" not in CLOSING_PUNCTUATION
    # 51px inner width: "Twilight，" is 63px, which used to be shredded into
    # "Twiligh" + "t" and push a lone comma onto a third line.
    assert _wrap(draw, "来吧，Twilight，", font, 51) == ["来吧，", "Twilight，"]

def test_two_boxes_in_one_shared_balloon_do_not_claim_the_same_space():
    # Two balloons that touch are a single region for the colour search, so
    # both boxes would grow to the whole merged shape and the two translations
    # would be painted on top of each other.  ``avoid`` splits the shared space
    # along the midline between the two boxes.
    image = np.full((260, 300, 3), (232, 130, 180), dtype=np.uint8)
    cv2.ellipse(image, (150, 110), (110, 70), 0, 0, 360, (255, 255, 255), -1)
    upper = [110, 80, 190, 92]
    lower = [110, 128, 190, 140]
    both = ArtworkPreservingRenderer._uniform_balloon_box(image, upper)
    assert both[3] > lower[1], "the merged region does reach the lower box"
    first = ArtworkPreservingRenderer._uniform_balloon_box(
        image, upper, avoid=[lower]
    )
    second = ArtworkPreservingRenderer._uniform_balloon_box(
        image, lower, avoid=[upper]
    )
    assert first[3] <= second[1] or second[3] <= first[1], "the boxes must not overlap"
    assert first[3] > upper[3] and second[1] < lower[1], "both keep room to grow"

def test_an_artwork_strip_window_still_erases_its_letters():
    # The artwork profile cleans one detector line at a time, so its window is
    # barely wider than that line: 100x30 here, where each letter is 252px.
    # A 6% area cap is 180px, smaller than a single letter, so every glyph was
    # dismissed as "too big to be text" and the English stayed on the page
    # under the translation ("ARRIVED", "LOOKS ADORABLE", "ARTNER").
    image = Image.new("RGB", (200, 120), (245, 205, 230))
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 20, 190, 100), fill=(238, 212, 233))
    for x in (40, 62, 84, 106):
        draw.rectangle((x, 42, x + 11, 62), fill=(14, 14, 14))
    strip = [34, 38, 134, 68]
    rgb = np.asarray(image).copy()
    source_ink = _dark_pixels(rgb, (34, 38, 134, 68))
    assert source_ink > 900

    tight, _, _ = ArtworkPreservingRenderer._erase(
        rgb.copy(), strip, area_ratio=0.06
    )
    assert _dark_pixels(tight, (34, 38, 134, 68)) == source_ink, "6% keeps every glyph"

    cleaned, _, _ = ArtworkPreservingRenderer._erase(
        rgb.copy(), strip, area_ratio=0.12
    )
    assert _dark_pixels(cleaned, (34, 38, 134, 68)) == 0, "12% fits a whole letter"


def test_the_grown_box_is_always_integral():
    # A proportional growth limit is a float, and the box goes straight into
    # PIL's crop/paste: page 38 of Pink World died with "'float' object cannot
    # be interpreted as an integer" once that limit won over the flat
    # allowance.  Odd widths are what expose it.
    image = np.full((200, 240, 3), (232, 130, 180), dtype=np.uint8)
    cv2.ellipse(image, (120, 100), (100, 70), 0, 0, 360, (255, 255, 255), -1)
    for box in ([100, 90, 155, 103], [101, 91, 154, 102], [61, 41, 79, 49]):
        grown = ArtworkPreservingRenderer._uniform_balloon_box(image, box)
        assert all(type(value) is int for value in grown), (box, grown)

