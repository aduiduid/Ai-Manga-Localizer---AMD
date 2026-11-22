"""Golden colour tests: source crop -> real render -> compare drawn pixels.

Written as a separate module on purpose.  The rest of test_renderer asserts on
the TextStyle object, and that is precisely why the 0.13.15 brightness rule got
through review: the object still said ink=(84,40,117) while the renderer painted
black, because the decision moved to a different branch than the assertion
checked.  These tests never look at the style; they compare what is drawn with
what the source art contains.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_localizer.config import AppPaths
from manga_localizer.ocr import PageOCR, TextUnit, glyph_ink
from manga_localizer.renderer import ArtworkPreservingRenderer, find_bold_font

FIXTURES = Path(__file__).parent / "fixtures" / "colour"
#: A drawn glyph must stay this far from neutral, matching the source's spread.
SATURATION_FLOOR = 40
#: How far the drawn hue may drift, in degrees.
HUE_TOLERANCE = 30.0


def _cases() -> list[dict]:
    manifest = FIXTURES / "manifest.json"
    if not manifest.is_file():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))


CASES = _cases()


def _glyph_pixels(rgb: np.ndarray, box, percentile: float = 10) -> np.ndarray:
    """The stroke interior: the darkest tenth of the box.

    Tighter than the obvious fifth on purpose -- the drawn glyphs carry a stroke
    and anti-aliasing that lighten their own mean, and the source has to be
    sampled the same way for the comparison to mean anything.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(int)
    level = crop @ np.array([0.299, 0.587, 0.114])
    return crop[level <= np.percentile(level, percentile)]


def _hue(colour: np.ndarray) -> float:
    """Hue in degrees, or NaN when the colour is too neutral to have one."""
    import colorsys

    r, g, b = (float(v) / 255.0 for v in colour)
    return colorsys.rgb_to_hls(r, g, b)[0] * 360.0


def _hue_gap(first: float, second: float) -> float:
    """Distance between two hues on the colour circle, 0-180."""
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


def _render(fixture: dict, tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Render one fixture crop and return (source, rendered) page arrays."""
    image = Image.open(FIXTURES / fixture["file"]).convert("RGB")
    source = tmp_path / "crop.png"
    image.save(source)

    x0, y0, x1, y1 = fixture["bbox"]
    unit = TextUnit("p1u1", [x0, y0, x1, y1], [x0, y0, x1, y1], "source", 0.95)
    unit.zh = "译文"
    page = PageOCR(1, source.name, image.width, image.height, [unit])

    try:
        font = find_bold_font()
    except FileNotFoundError:
        pytest.skip("no CJK font installed to render with")

    class FlatErase:
        """Stands in for the erase network: fill masked pixels with the paper colour."""

        def __call__(self, rgb, mask):
            filled = rgb.copy()
            keep = mask == 0
            colour = (
                np.median(rgb[keep].reshape(-1, 3), axis=0)
                if keep.any()
                else np.array([255, 255, 255])
            )
            filled[mask > 0] = colour.astype(filled.dtype)
            return filled

    renderer = ArtworkPreservingRenderer(
        paths=AppPaths(
            tmp_path, tmp_path / "models", tmp_path / "cache",
            tmp_path / "jobs", tmp_path / "settings.json",
        ),
        font_path=font,
        bold_font_path=font,
        cleanup_profile="quality",
        preserve_text_color=True,
        inpainter=FlatErase(),
    )
    out = tmp_path / "out.png"
    renderer.render_page(source, page, out, False)
    return np.asarray(Image.open(source).convert("RGB")), np.asarray(
        Image.open(out).convert("RGB")
    )


@pytest.mark.skipif(not CASES, reason="colour fixtures are not present")
@pytest.mark.parametrize("fixture", CASES, ids=[c["unit"] for c in CASES])
def test_drawn_glyphs_keep_the_source_colour(fixture, tmp_path):
    """The letters must keep their colour through an actual render.

    Ten crops of real art, all coloured lettering on bright paper -- the exact
    combination the deleted brightness rule repainted black.  Comparing drawn
    pixels against source pixels is what makes this test able to fail: the round
    that shipped that rule had every object-level assertion green.
    """
    source, rendered = _render(fixture, tmp_path)

    box = fixture["bbox"]
    src = _glyph_pixels(source, box)
    # The translation sits in the same region; take its stroke interior too.
    drawn = _glyph_pixels(rendered, box)
    assert src.size and drawn.size, f"{fixture['unit']}: empty glyph mask"

    src_colour = src.mean(axis=0)
    drawn_colour = drawn.mean(axis=0)

    # The failure this guards against is a colourless translation, so the first
    # claim is that the drawn glyphs still carry the source's channel spread.
    drawn_saturation = int(drawn_colour.max() - drawn_colour.min())
    assert drawn_saturation >= SATURATION_FLOOR, (
        f"{fixture['unit']} ({fixture['hue_want']}): drawn glyphs "
        f"{tuple(int(v) for v in drawn_colour)} are neutral "
        f"(spread {drawn_saturation}); the colour was dropped"
    )

    # And it must be the *same* colour: a red line painted violet is still wrong.
    gap = _hue_gap(_hue(src_colour), _hue(drawn_colour))
    assert gap <= HUE_TOLERANCE, (
        f"{fixture['unit']} ({fixture['hue_want']}): source glyphs "
        f"{tuple(int(v) for v in src_colour)} but drawn glyphs "
        f"{tuple(int(v) for v in drawn_colour)}, hue differs by {gap:.0f}deg"
    )


@pytest.mark.skipif(not CASES, reason="colour fixtures are not present")
@pytest.mark.parametrize("fixture", CASES, ids=[c["unit"] for c in CASES])
def test_the_fixture_really_is_coloured(fixture, tmp_path):
    """A guard on the fixtures themselves: a neutral crop would prove nothing."""
    image = np.asarray(Image.open(FIXTURES / fixture["file"]).convert("RGB"))
    ink = glyph_ink(image, fixture["bbox"])
    assert ink is not None, f"{fixture['unit']}: fixture has no ink reading"
    saturation = int(max(ink) - min(ink))
    assert saturation >= 40, (
        f"{fixture['unit']} is a {fixture['hue_want']} case but measures "
        f"saturation {saturation}; a neutral fixture cannot test colour loss"
    )
