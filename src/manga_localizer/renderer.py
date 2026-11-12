from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import AppPaths
from .inpainting import LaMaInpainter
from .model_manager import ModelManager
from .ocr import (
    JP_RE,
    PageOCR,
    TextUnit,
    prune_detached_orthogonal_outliers,
)


FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
)
BOLD_FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf"
)
SYSTEM_FONT_CANDIDATES = (
    "C:/Windows/Fonts/SourceHanSansCN-Medium.otf",
    "C:/Windows/Fonts/NotoSansSC-VF.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
SYSTEM_BOLD_FONT_CANDIDATES = (
    "C:/Windows/Fonts/SourceHanSansCN-Bold.otf",
    "C:/Windows/Fonts/NotoSansSC-Bold.ttf",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


@dataclass(frozen=True)
class TextStyle:
    mean: float
    white_ratio: float
    fill: str
    stroke_fill: str
    stroke_ratio: float
    bold: bool
    font_size: int
    vertical: bool
    outlined: bool
    background_std: float
    display: bool


def managed_font_path(paths: AppPaths | None = None) -> Path:
    root = (paths or AppPaths.from_env()).root
    return root / "fonts" / "NotoSansCJKsc-Regular.otf"


def managed_bold_font_path(paths: AppPaths | None = None) -> Path:
    root = (paths or AppPaths.from_env()).root
    return root / "fonts" / "NotoSansCJKsc-Bold.otf"


def find_font(paths: AppPaths | None = None) -> Path:
    candidates = (
        os.environ.get("MLS_FONT"),
        managed_font_path(paths),
        *SYSTEM_FONT_CANDIDATES,
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No CJK font found. Set MLS_FONT to a Simplified Chinese TTF/OTF/TTC file."
    )


def find_bold_font(paths: AppPaths | None = None) -> Path:
    candidates = (
        os.environ.get("MLS_BOLD_FONT"),
        managed_bold_font_path(paths),
        *SYSTEM_BOLD_FONT_CANDIDATES,
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return find_font(paths)


def _download_font(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".otf.part")
    request = Request(url, headers={"User-Agent": "Manga-Localizer-Studio/0.3"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if partial.stat().st_size < 1_000_000 or partial.read_bytes()[:4] != b"OTTO":
            raise RuntimeError("Downloaded CJK font failed size/header validation")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def ensure_font(paths: AppPaths | None = None, force_managed: bool = False) -> Path:
    """Return a CJK font, downloading a pinned OFL font when none is available."""
    target = managed_font_path(paths)
    if target.exists():
        return target
    if not force_managed:
        try:
            return find_font(paths)
        except FileNotFoundError:
            pass
    return _download_font(FONT_URL, target)


def ensure_bold_font(
    paths: AppPaths | None = None, force_managed: bool = False
) -> Path:
    target = managed_bold_font_path(paths)
    if target.exists():
        return target
    if not force_managed:
        return find_bold_font(paths)
    return _download_font(BOLD_FONT_URL, target)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _clean(text: str) -> str:
    return "".join(text.replace("．．．", "……").replace("．", "。").split())


def _fit_vertical(
    font_path: Path,
    text: str,
    width: int,
    height: int,
    preferred_size: int | None = None,
):
    start = min(height, max(20, preferred_size or min(160, width, height)))
    for size in range(start, 11, -2):
        capacity = max(1, height // round(size * 1.08))
        columns = math.ceil(len(text) / capacity)
        if columns * round(size * 1.12) <= width:
            return _font(font_path, size), size, capacity
    # Small round bubbles fit barely one glyph column, so the fixed 20px
    # fallback spilled its columns past the bbox edge and the leftmost text
    # was clipped.  Keep shrinking until the columns fit.
    size = 12
    capacity = max(1, height // round(size * 1.08))
    while size > 7 and math.ceil(len(text) / capacity) * round(size * 1.12) > width:
        size -= 1
        capacity = max(1, height // round(size * 1.08))
    return _font(font_path, size), size, capacity


def _tokens(text: str) -> list[str]:
    """Split into wrap units: CJK glyphs alone, Latin words kept whole.

    Wrapping character by character is right for Chinese but mangles the Latin
    names a translation keeps ("Twilight" became "T" + "wilight" across two
    lines in a small balloon).  A run of ASCII letters, digits and the
    punctuation that belongs inside a word stays one unit, so the only break
    points offered are the ones a reader expects.

    Closing punctuation is glued onto the unit before it rather than left as a
    unit of its own: a comma or a "……" that can start a line always ends up as a
    stray mark on a line of its own in a narrow balloon.
    """
    tokens: list[str] = []
    word = ""
    for char in text:
        if char.isascii() and (char.isalnum() or char in "'’-&./+"):
            word += char
            continue
        if word:
            tokens.append(word)
            word = ""
        if char in CLOSING_PUNCTUATION and tokens:
            tokens[-1] += char
            continue
        tokens.append(char)
    if word:
        tokens.append(word)
    return tokens


# Punctuation that may never open a line.  A comma or full stop that wraps onto
# its own line reads as a stray mark ("来吧，/Twilight/，" in a 46x33 bubble), so
# these are appended to the line before them even when that costs a pixel of
# width.
CLOSING_PUNCTUATION = "，。、；：？！）」』】》,.;:?!)]}…—～~"

# How far a single word may hang past the inner width before it is broken up.
# The inner width already excludes the box inset, so a word a couple of pixels
# wider still fits inside the balloon; breaking it there traded a legible line
# for a shredded one ("Twilight" became "Twiligh" + "t，" and pushed a lone
# comma onto a third line).  Centering splits the overhang over both sides.
WORD_OVERFLOW_SLACK = 8


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, current = [], ""

    def measure(candidate: str) -> int:
        return draw.textbbox((0, 0), candidate, font=font)[2]

    for token in _tokens(text):
        if current and measure(current + token) > width:
            lines.append(current)
            current = ""
        if measure(token) <= width or not current:
            current += token
            continue
        # A word that misses the inner width by less than the box's own inset is
        # kept whole: the inset is unused space, and splitting "Twilight" into
        # "Twiligh" + "t，" cost a whole extra line in a 46x33 bubble.  Only a
        # word genuinely wider than the balloon falls back to per-glyph breaks.
        if measure(token) <= width + WORD_OVERFLOW_SLACK:
            current += token
            continue
        # A single word wider than the balloon: breaking inside it is the only
        # way to keep it on the page, so fall back to one glyph at a time.
        for char in token:
            if current and measure(current + char) > width:
                lines.append(current)
                current = ""
            current += char
    if current:
        lines.append(current)
    return _attach_orphan_punctuation(lines)


def _attach_orphan_punctuation(lines: list[str]) -> list[str]:
    """Fold a line that is nothing but closing punctuation into the line above.

    The per-token rule above keeps punctuation with the glyph in front of it,
    but a run of two marks ("！！", "……") can still land alone when the line
    before it is a single glyph wide.  A bare "！！" on its own line reads as a
    smudge, so it goes back onto the line above and the fit check decides
    whether the size still works.
    """
    merged: list[str] = []
    for line in lines:
        if merged and not line.strip(CLOSING_PUNCTUATION):
            merged[-1] += line
        else:
            merged.append(line)
    return merged


def _lines_fit(draw: ImageDraw.ImageDraw, lines: list[str], font, width: int) -> bool:
    """Whether every wrapped line stays inside ``width``.

    Trailing punctuation is allowed to hang into the margin, which is the
    typographic rule that keeps "，"/"！" attached to the word before it instead
    of starting a line of its own.  A line whose only glyph is the one the
    punctuation hangs off is exempt from that allowance: there one size smaller
    always fits, and letting "不！！" hang 22px past a 39px balloon looked far
    worse than printing it a size down.
    """
    for line in lines:
        if draw.textbbox((0, 0), line, font=font)[2] <= width + WORD_OVERFLOW_SLACK:
            continue
        body = line.rstrip(CLOSING_PUNCTUATION)
        if len(body) > 1 and draw.textbbox((0, 0), body, font=font)[2] <= width:
            continue
        return False
    return True


# Line pitch as a multiple of the font size.  CJK glyphs already carry their
# own breathing room inside the em box, so 1.24 wasted a line of height that a
# short balloon did not have and pushed the type a size or two down.  Measured
# across the sample, 1.14 keeps the lines clearly apart while losing one fewer
# size than 1.18 on the tightest bubbles.
LINE_HEIGHT_RATIO = 1.14

# Inset used by both ``_typeset_box`` and ``_draw``: it must be the same number
# in both places or the box is judged against room the drawing never uses.  A
# 4px floor stole a quarter of the height from the 15px strips that need the
# room most, and a 3px floor still cost three of the 71 dialogue units a size.
# 2px measured best across the sample (6 units under 12px, mean 15.7px) while
# still keeping the stroke off the balloon border.
TEXT_PADDING_RATIO_X = 0.06
TEXT_PADDING_RATIO_Y = 0.04
TEXT_PADDING_MIN = 2


def _text_padding(box: list[int]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    return (
        max(TEXT_PADDING_MIN, round((x1 - x0) * TEXT_PADDING_RATIO_X)),
        max(TEXT_PADDING_MIN, round((y1 - y0) * TEXT_PADDING_RATIO_Y)),
    )


def _fit_horizontal(
    font_path: Path,
    draw,
    text: str,
    width: int,
    height: int,
    preferred_size: int | None = None,
):
    start = min(height, max(20, preferred_size or min(160, height)))
    for size in range(start, 7, -2):
        font = _font(font_path, size)
        lines = _wrap(draw, text, font, width)
        line_height = round(size * LINE_HEIGHT_RATIO)
        if len(lines) * line_height <= height and _lines_fit(draw, lines, font, width):
            return font, size, lines, line_height
    # A long English line translated into Chinese needs far more glyphs than
    # the source occupied, so stopping at 20px spilled the text out of the
    # balloon.  Keep shrinking instead of overflowing the region.
    font = _font(font_path, 8)
    return font, 8, _wrap(draw, text, font, width), round(8 * LINE_HEIGHT_RATIO)


# Below this size a CJK glyph is a grey smear at manga page resolution: the
# strokes of 想 and 集 merge into one blob.  Shrinking further is worse than
# growing the typesetting region, so ``_typeset_box`` widens the box first and
# only falls back to the 8px overflow guard when even that is impossible.
MIN_READABLE_FONT = 12


def _fit_readable(
    font_path: Path,
    draw,
    text: str,
    width: int,
    height: int,
    preferred_size: int | None = None,
    floor: int = MIN_READABLE_FONT,
):
    """Largest font size that fits, or ``None`` when even ``floor`` does not."""
    start = min(height, max(20, preferred_size or min(160, height)))
    for size in range(start, floor - 1, -1):
        font = _font(font_path, size)
        lines = _wrap(draw, text, font, width)
        line_height = round(size * LINE_HEIGHT_RATIO)
        if len(lines) * line_height <= height and _lines_fit(draw, lines, font, width):
            return font, size, lines, line_height
    return None


# ``quality_profile`` ("quality"/"fast") is the user-facing setting; the
# renderer only distinguishes how aggressively it repairs the erased pixels.
# The fast profile means "no LaMa inpainting, plain fill", which is exactly the
# artwork cleanup.  Accepting the user-facing names here keeps the two
# vocabularies from drifting apart: a caller passing "fast" used to reach the
# render step and fail there with "Unknown cleanup profile: fast" after OCR and
# translation had already finished.
CLEANUP_PROFILE_ALIASES = {"fast": "artwork", "artwork": "artwork", "quality": "quality"}


class ArtworkPreservingRenderer:
    def __init__(
        self,
        font_path: Path | None = None,
        bold_font_path: Path | None = None,
        cleanup_profile: str = "artwork",
        paths: AppPaths | None = None,
        device: str = "auto",
        inpainter=None,
    ):
        self.paths = (paths or AppPaths.from_env()).ensure()
        self.font_path = font_path or ensure_font(self.paths)
        self.bold_font_path = bold_font_path or find_bold_font(self.paths)
        resolved_profile = CLEANUP_PROFILE_ALIASES.get(cleanup_profile)
        if resolved_profile is None:
            raise ValueError(f"Unknown cleanup profile: {cleanup_profile}")
        self.cleanup_profile = resolved_profile
        self.device = device
        self._inpainter = inpainter

    @staticmethod
    def _appearance(rgb: np.ndarray, box: list[int]) -> tuple[float, float]:
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1]
        if not crop.size:
            return 255.0, 1.0
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return float(gray.mean()), float((gray > 225).mean())

    @staticmethod
    def _analyze_style(rgb: np.ndarray, unit: TextUnit) -> TextStyle:
        x0, y0, x1, y1 = unit.bbox
        crop = rgb[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        mean = float(gray.mean()) if gray.size else 255.0
        white_ratio = float((gray > 225).mean()) if gray.size else 1.0
        background = gray[gray > 135]
        background_std = float(background.std()) if background.size else 0.0
        # Japanese manga dialogue is conventionally vertical.  A union of
        # several vertical columns is often almost square, so aspect ratio
        # alone must only classify clearly wide strips as horizontal.
        # A union of wide stacked lines (signs, captions, cover blurbs) is the
        # exception: treating it as one vertical column inflates the type until
        # the columns overflow the sign.  Per-column detector boxes are tall
        # and narrow, so their shape separates the two layouts.
        source_boxes = unit.erase_boxes or [unit.bbox]
        stacked_lines = len(source_boxes) >= 2 and sum(
            1
            for box in source_boxes
            if (box[2] - box[0]) > (box[3] - box[1])
        ) >= len(source_boxes) / 2
        # A Latin source is horizontal by definition.  The height rule
        # below exists for stacks of Japanese columns, and on a nearly
        # square Latin box it turned "'Sup" into three one-character
        # columns drawn from the right, so the translation read
        # backwards.
        vertical = unit.special == "cover_title" or (
            JP_RE.search(unit.ja) is not None
            and height >= width * 0.62
            and not stacked_lines
        )

        dark = (gray < 105).astype(np.uint8)
        if dark.any():
            near = cv2.dilate(dark, np.ones((21, 21), np.uint8), iterations=1) > 0
            core = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1) > 0
            ring = near & ~core
            halo_ratio = float((gray[ring] > 225).mean()) if ring.any() else 0.0
        else:
            halo_ratio = 0.0
        outlined = unit.special == "cover_title" or (
            background_std >= 12 and float(dark.mean()) >= 0.01 and halo_ratio >= 0.38
        )
        page_area = max(1, rgb.shape[0] * rgb.shape[1])
        display = unit.special == "cover_title" or (
            outlined
            and width * height / page_area >= 0.12
            and len(_clean(unit.zh)) <= 24
        )

        erase_boxes = unit.erase_boxes or [unit.bbox]
        if display:
            count = max(1, len(_clean(unit.zh)))
            estimated = min(width * 0.23, height / (count * 1.45))
        elif not unit.erase_boxes:
            source_count = max(1, len(_clean(unit.ja)), len(_clean(unit.zh)))
            estimated = math.sqrt(width * height / source_count) * 0.58
        elif vertical:
            line_widths = [max(1, box[2] - box[0]) for box in erase_boxes]
            # Detector rectangles include ruby, punctuation and sometimes two
            # touching glyph columns.  Their full width is therefore an upper
            # bound, not the source em-size.  Keeping roughly three quarters
            # of an outlined column (and two thirds of a plain one) matches
            # the visible main-glyph footprint without letting grouped OCR
            # regions inflate the translated typography.
            box_ratio = 0.76 if outlined else 0.68
            estimated = float(np.median(line_widths)) * box_ratio
            if len(erase_boxes) == 1 and line_widths[0] >= width * 0.8:
                estimated = min(
                    width * 0.72, height / max(1, len(_clean(unit.zh))) * 0.82
                )
        else:
            line_heights = [max(1, box[3] - box[1]) for box in erase_boxes]
            estimated = float(np.median(line_heights)) * (0.78 if outlined else 0.7)
            if stacked_lines:
                # Detector boxes on a sign wrap loosely around the glyphs, so
                # their median height is an upper bound, not the em-size.  Using
                # it fills the whole sign and pushes the last line off the
                # plaque.  Budget the available height across the source lines
                # instead.
                estimated = min(
                    estimated, height / max(1, len(erase_boxes)) * 0.62
                )
        font_size = max(20, min(320, round(estimated)))

        # Text colour is the polarity of the glyph pixels against the local
        # background, not overall page brightness.  Cyan signboards and wood
        # plaques carry black lettering while their mean luminance sits in the
        # mid-tone range, so a fixed mean/white-ratio threshold renders white
        # text on top of black source glyphs (and seeds an empty erase mask).
        background_level = float(np.median(gray)) if gray.size else 255.0
        darker_ratio = float((gray < background_level - 45).mean()) if gray.size else 0.0
        brighter_ratio = (
            float((gray > background_level + 45).mean()) if gray.size else 0.0
        )
        if halo_ratio >= 0.38 and float(dark.mean()) >= 0.01:
            # Outlined lettering is a dark core inside a bright halo, so the
            # halo *is* the stroke.  Counting bright ring pixels against dark
            # cores would flip the polarity on thick outlines.
            fill, stroke_fill = "black", "white"
        elif brighter_ratio > darker_ratio:
            fill, stroke_fill = "white", "black"
        else:
            fill, stroke_fill = "black", "white"
        bold = outlined or (float(dark.mean()) >= 0.025 and font_size >= 32)
        stroke_ratio = 0.066 if outlined else (0.045 if fill == "white" else 0.0)
        return TextStyle(
            mean=mean,
            white_ratio=white_ratio,
            fill=fill,
            stroke_fill=stroke_fill,
            stroke_ratio=stroke_ratio,
            bold=bold,
            font_size=font_size,
            vertical=vertical,
            outlined=outlined,
            background_std=background_std,
            display=display,
        )

    def _get_inpainter(self):
        if self._inpainter is None:
            weights = ModelManager(self.paths).lama_weights_path()
            self._inpainter = LaMaInpainter(weights, self.device)
        return self._inpainter

    @staticmethod
    def _text_mask(crop: np.ndarray, style: TextStyle, dilation: int) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Fixed thresholds assume a white balloon.  Black lettering on a cyan
        # sign or a wood plaque sits at mid/dark luminance, where a constant
        # cutoff selects the entire background and every component is rejected
        # as edge-connected, leaving the source text in place.  Anchor the
        # cutoff to the local median instead.
        level = float(np.median(gray)) if gray.size else 255.0
        if style.fill == "white":
            seed = (gray > min(215, level + 55)).astype(np.uint8)
        else:
            seed = (gray < max(15, min(150, level - 45))).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
        selected = np.zeros_like(seed, dtype=np.uint8)
        height, width = seed.shape
        # Furigana and punctuation are much smaller than the main glyphs but
        # belong to the same replacement region.  Keep their components while
        # still rejecting isolated screentone speckles.
        min_area = max(4, round(style.font_size * style.font_size * 0.0005))
        max_area = max(80, round(width * height * 0.45))
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            touches_edge = (
                x <= 1
                or y <= 1
                or x + component_width >= width - 1
                or y + component_height >= height - 1
            )
            aspect = max(component_width, component_height) / max(
                1, min(component_width, component_height)
            )
            # A tight detector box often cuts through its own glyphs, so the
            # stroke cluster reaches the padded crop border.  Rejecting every
            # edge-connected component there drops the whole source text and
            # leaves it visible under the new lettering.  Only large edge blobs
            # are artwork; the caller still clips the mask to the text box.
            if (touches_edge and area > width * height * 0.25) or area < min_area:
                continue
            if area > max_area or aspect > 12:
                continue
            selected[labels == label] = 255
        if dilation > 1 and selected.any():
            selected = cv2.dilate(
                selected,
                np.ones((dilation, dilation), np.uint8),
                iterations=1,
            )
        return selected

    @staticmethod
    def _mask_dilation(unit: TextUnit, style: TextStyle) -> int:
        if style.outlined and (
            unit.special == "cover_title" or style.display or style.font_size >= 120
        ):
            value = min(51, max(23, round(style.font_size * 0.22)))
        elif style.outlined:
            value = min(23, max(7, round(style.font_size * 0.14)))
        else:
            value = min(13, max(5, round(style.font_size * 0.08)))
        return value + 1 if value % 2 == 0 else value

    def _erase_with_lama(
        self, rgb: np.ndarray, unit: TextUnit, style: TextStyle
    ) -> np.ndarray:
        page_height, page_width = rgb.shape[:2]
        x0, y0, x1, y1 = unit.crop_bbox or unit.bbox
        cleanup_margin = (
            min(64, max(12, round(style.font_size * 0.4))) if style.outlined else 0
        )
        x0, y0 = max(0, x0 - cleanup_margin), max(0, y0 - cleanup_margin)
        x1, y1 = (
            min(page_width, x1 + cleanup_margin),
            min(page_height, y1 + cleanup_margin),
        )
        context = min(256, max(64, round(style.font_size * 1.5)))
        rx0, ry0 = max(0, x0 - context), max(0, y0 - context)
        rx1, ry1 = min(page_width, x1 + context), min(page_height, y1 + context)
        crop = rgb[ry0:ry1, rx0:rx1].copy()
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        # OCR may group several columns into one semantic unit.  The union is
        # a typesetting region only; it is never valid erase geometry because
        # it can contain faces, line art, furniture, or an entire panel.  LaMa
        # receives only detector-owned source rectangles and builds a pixel
        # mask independently inside each one.
        source_boxes = unit.erase_boxes or [[x0, y0, x1, y1]]
        # Display titles need enough reach to include their proportionally
        # thick white outline. Each detector-owned box remains independent, so
        # the wider title halo cannot turn the semantic group union into a mask.
        dilation = self._mask_dilation(unit, style)
        # Prefer a glyph-shaped mask even on textured artwork.  Filling every
        # outlined detector rectangle makes LaMa synthesize far more of the
        # panel than the source text occupied, which softens screentone and
        # line art.  A tight detector rectangle remains a last-resort fallback
        # only for exceptionally large non-display lettering whose glyph core
        # can merge with the artwork threshold.
        solid_source_boxes = (
            style.outlined and style.font_size >= 120 and not style.display
        )
        for source_box in source_boxes:
            sx0 = max(x0, source_box[0]) - rx0
            sy0 = max(y0, source_box[1]) - ry0
            sx1 = min(x1, source_box[2]) - rx0
            sy1 = min(y1, source_box[3]) - ry0
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            pad_x = min(32, max(8, round((sx1 - sx0) * 0.08)))
            pad_y = min(32, max(8, round((sy1 - sy0) * 0.06)))
            bx0, by0 = max(0, sx0 - pad_x), max(0, sy0 - pad_y)
            bx1, by1 = min(crop.shape[1], sx1 + pad_x), min(crop.shape[0], sy1 + pad_y)
            if solid_source_boxes:
                mask[sy0:sy1, sx0:sx1] = 255
                continue
            local = self._text_mask(crop[by0:by1, bx0:bx1], style, dilation)
            mask[by0:by1, bx0:bx1] = np.maximum(mask[by0:by1, bx0:bx1], local)
        allowed = np.zeros_like(mask)
        allowed[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = 255
        mask = cv2.bitwise_and(mask, allowed)
        if not mask.any():
            return rgb
        crop = self._get_inpainter()(crop, mask)
        rgb[ry0:ry1, rx0:rx1] = crop
        return rgb

    @staticmethod
    def _interior_component_mask(
        binary: np.ndarray, dilation: int, max_area_ratio: float = 0.18
    ) -> np.ndarray:
        """Select text-like components without touching surrounding line art.

        Manga detector rectangles often cross panel art. Artwork normally enters
        through a rectangle edge, while glyph components sit inside it. Keeping
        edge-connected and very large components prevents the cleanup pass from
        turning characters, furniture, and panel borders into white blocks.

        ``max_area_ratio`` is measured against the searched region, and that
        region is now the lettering's own ring rather than a 96 px square, so a
        tighter ratio is still a generous absolute ceiling in which every glyph
        fits while a lock of hair or a hat brim does not.
        """
        height, width = binary.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        mask = np.zeros_like(binary, dtype=np.uint8)
        max_area = max(64, round(width * height * max_area_ratio))
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            touches_edge = (
                x <= 1
                or y <= 1
                or x + component_width >= width - 1
                or y + component_height >= height - 1
            )
            if touches_edge or area < 4 or area > max_area:
                continue
            mask[labels == label] = 255
        if dilation > 1 and mask.any():
            mask = cv2.dilate(
                mask, np.ones((dilation, dilation), np.uint8), iterations=1
            )
        return mask

    @staticmethod
    def _background_color(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Median colour of the pixels the text mask does not claim.

        Erased glyphs must be filled with the local page colour, not pure
        white.  Cream, grey and coloured panels are common, and a solid white
        patch is far more noticeable than the source text it replaced.  The
        median of the untouched pixels is the background by construction: a
        dense text block still leaves the surrounding paper in the majority.
        """
        keep = mask == 0
        if int(keep.sum()) >= 16:
            sample = crop[keep]
        else:
            # The mask claims nearly the whole crop (tight detector box around
            # one large glyph).  Fall back to the lighter half, which is the
            # paper side of any plain glyph/ground pair.
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            sample = crop[gray >= float(np.median(gray))]
        if not sample.size:
            return np.array([255, 255, 255], dtype=np.uint8)
        median = np.median(sample.reshape(-1, 3), axis=0)
        return np.clip(median, 0, 255).astype(np.uint8)

    @staticmethod
    def _glyph_polarity(
        rgb: np.ndarray, focus: list[int]
    ) -> tuple[float, bool]:
        """Decide which side of the local median the source glyphs are on.

        ``focus`` is the detector box, and only it may vote on polarity.  The
        erase region is deliberately padded well beyond the text so stray
        furigana and balloon borders are reachable, but that padding also
        contains panel art, mint backgrounds and highlights.  Counting the
        padded area let the artwork outvote six glyphs: a pink sign with black
        Spanish lettering was classified as white-on-dark, the mask selected
        empty pixels, and the original text survived under the translation
        (the reported "text is not erased" bug).

        Returns ``(level, glyphs_are_dark)`` where ``level`` is the median
        luminance the two sides are measured against.
        """
        fx0, fy0, fx1, fy1 = focus
        crop = rgb[fy0:fy1, fx0:fx1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if not gray.size:
            return 255.0, True
        level = float(np.median(gray))
        darker = float((gray < level - 45).mean())
        brighter = float((gray > level + 45).mean())
        if brighter > darker:
            return level, False
        return level, True

    @staticmethod
    def _erase(
        rgb: np.ndarray,
        box: list[int],
        focus: list[int] | None = None,
        bands: list[list[int]] | None = None,
        area_ratio: float = 0.06,
    ) -> tuple[np.ndarray, float, float]:
        """Erase the source glyphs inside ``box``.

        ``focus`` decides which side of the local median the glyphs are on, and
        ``bands`` is the lettering's own ring: the mask is clipped to it, so a
        balloon border, a lock of hair or a hat brim that falls inside the
        searched rectangle is left alone.  The mask used to be free to claim
        anything in the rectangle, which is how those came back inpainted.

        ``area_ratio`` caps one component as a share of ``box``, so it has to
        match how large ``box`` is.  The quality path searches a generous
        window around the unit, where 6% leaves room for a whole glyph.  The
        artwork path calls this once per detector line on a window barely wider
        than the line itself, and there 6% of the area is smaller than a single
        letter: the letter was dropped as "too big to be text" and stayed on the
        page underneath the translation.
        """
        # Integral by construction.  numpy accepts floats in a slice and
        # silently truncates them, which hides the mistake until the mask and
        # the write-back disagree by one pixel.
        x0, y0, x1, y1 = (int(round(value)) for value in box)
        crop = rgb[y0:y1, x0:x1].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean()) if gray.size else 255.0
        white_ratio = float((gray > 225).mean()) if gray.size else 1.0
        allowed = None
        if bands:
            height, width = gray.shape
            allowed = np.zeros(gray.shape, dtype=np.uint8)
            for band_x0, band_y0, band_x1, band_y1 in bands:
                top = max(0, min(height, band_y0 - y0))
                bottom = max(0, min(height, band_y1 - y0))
                left = max(0, min(width, band_x0 - x0))
                right = max(0, min(width, band_x1 - x0))
                allowed[top:bottom, left:right] = 1

        def clip(mask: np.ndarray) -> np.ndarray:
            if allowed is None:
                return mask
            return np.where(allowed > 0, mask, 0).astype(np.uint8)

        if white_ratio > 0.45 or (mean > 175 and white_ratio > 0.35):
            mask = clip(
                ArtworkPreservingRenderer._interior_component_mask(
                    (gray < 150).astype(np.uint8), 5
                )
            )
            if mask.any():
                crop[mask > 0] = ArtworkPreservingRenderer._background_color(
                    crop, mask
                )
        else:
            # Mid-tone grounds (cyan signboards, colour panels) carry dark
            # lettering, so "mean < 145" must not imply light glyphs.  The
            # padding is reachable by the mask but must never vote on polarity:
            # the local median is measured on the detector box only, so the
            # surrounding artwork cannot flip which side is text.
            focus = focus or box
            level, glyphs_are_dark = ArtworkPreservingRenderer._glyph_polarity(
                rgb, focus
            )
            if glyphs_are_dark:
                seeds = (gray < level - 45).astype(np.uint8)
            else:
                seeds = (gray > level + 45).astype(np.uint8)
            mask = clip(
                ArtworkPreservingRenderer._interior_component_mask(
                    seeds, 7, max_area_ratio=area_ratio
                )
            )
            if mask.any():
                crop = cv2.inpaint(crop, mask, 6, cv2.INPAINT_TELEA)
        rgb[y0:y1, x0:x1] = crop
        return rgb, mean, white_ratio

    @staticmethod
    def _erase_complete(
        rgb: np.ndarray, box: list[int]
    ) -> tuple[np.ndarray, float, float]:
        """Remove all text-colored pixels inside an OCR-confirmed rectangle.

        This intentionally prefers complete source-text removal over conserving
        pixels *inside* the detected text region. Changes remain bounded to that
        region; surrounding artwork is never sampled, resized, or overwritten.
        """
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean()) if gray.size else 255.0
        white_ratio = float((gray > 225).mean()) if gray.size else 1.0
        if white_ratio > 0.58:
            mask = cv2.dilate(
                (gray < 150).astype(np.uint8) * 255,
                np.ones((7, 7), np.uint8),
                iterations=1,
            )
            crop[mask > 0] = 255
        elif mean < 145:
            mask = cv2.dilate(
                ((gray > 210) | (gray < 35)).astype(np.uint8) * 255,
                np.ones((13, 13), np.uint8),
                iterations=1,
            )
            crop = cv2.inpaint(crop, mask, 7, cv2.INPAINT_TELEA)
        else:
            mask = cv2.dilate(
                (gray < 75).astype(np.uint8) * 255,
                np.ones((31, 31), np.uint8),
                iterations=1,
            )
            crop = cv2.inpaint(crop, mask, 7, cv2.INPAINT_TELEA)
        rgb[y0:y1, x0:x1] = crop
        return rgb, mean, white_ratio

    @staticmethod
    def _uniform_balloon_box(
        rgb: np.ndarray,
        box: list[int],
        tolerance: float = 22.0,
        width_limit: float = 1.7,
        height_limit: float = 2.4,
        avoid: list[list[int]] | None = None,
    ) -> list[int]:
        """Grow ``box`` to the flat balloon or sign it sits on.

        A detector box around short lettering can be much shorter than the
        balloon it belongs to ("SUPONGO..." occupied 79x21 inside a 79x42
        plaque).  Typesetting inside the detector box alone forces the font
        down to 9px, where CJK strokes merge into a smear, so the box has to
        reach the balloon's real inside.

        Growing one pixel at a time and re-testing each edge against the span
        grown so far only works on a rectangle.  On a balloon the box gets
        taller first, the taller edge line then crosses the widening curve, and
        growth stops while the box is still inside the bubble (a 78x22 box grew
        to 83x28 against a 120x40 balloon, still too small for 12px) -- so the
        text spilled onto the panel outside.  Instead the page is split into
        regions that match the local background colour, and the balloon is the
        region the box's rim touches.  That follows any outline, curved or not,
        and a line of lettering cutting the inside in two simply contributes two
        regions.

        Two balloons that touch are one region, so ``avoid`` -- the other boxes
        on the page -- splits the shared space along the midline between this
        box and its nearest neighbour.  Without it both boxes grow to the whole
        merged region and the two translations are painted on top of each other.
        """
        page_height, page_width = rgb.shape[:2]
        x0, y0, x1, y1 = box
        sample = rgb[y0:y1, x0:x1].reshape(-1, 3)
        if not sample.size:
            return list(box)
        background = np.median(sample, axis=0)
        # A proportional limit alone is useless on very short strips: 2.4x of a
        # 15px box is still 36px, which no 12px CJK line fits into.  The flat
        # allowance keeps those boxes growing until the balloon border blocks
        # them, which is the only stop that actually matters.
        max_width = max((x1 - x0) * width_limit, (x1 - x0) + 64)
        max_height = max((y1 - y0) * height_limit, (y1 - y0) + 48)
        matches = (
            np.abs(rgb.astype(np.int16) - background).max(axis=2) <= tolerance
        ).astype(np.uint8)
        _, labels = cv2.connectedComponents(matches, 8)
        touched: set[int] = set()
        for ring in (1, 2, 3):
            for row in (y0 - ring, y1 + ring - 1):
                if 0 <= row < page_height:
                    touched.update(int(value) for value in np.unique(labels[row, x0:x1]))
            for column in (x0 - ring, x1 + ring - 1):
                if 0 <= column < page_width:
                    touched.update(
                        int(value) for value in np.unique(labels[y0:y1, column])
                    )
        touched.discard(0)
        if not touched:
            return list(box)
        rows, columns = np.nonzero(np.isin(labels, sorted(touched)))
        grow_x = max_width - (x1 - x0)
        grow_y = max_height - (y1 - y0)
        grown = [
            max(int(columns.min()), x0 - grow_x),
            max(int(rows.min()), y0 - grow_y),
            min(int(columns.max()) + 1, x1 + grow_x),
            min(int(rows.max()) + 1, y1 + grow_y),
        ]
        for other in avoid or []:
            if (
                grown[2] <= other[0]
                or other[2] <= grown[0]
                or grown[3] <= other[1]
                or other[3] <= grown[1]
            ):
                continue
            apart_x = (other[0] + other[2]) / 2.0 - (grown[0] + grown[2]) / 2.0
            apart_y = (other[1] + other[3]) / 2.0 - (grown[1] + grown[3]) / 2.0
            if abs(apart_y) >= abs(apart_x):
                if apart_y >= 0:
                    grown[3] = min(grown[3], (y1 + other[1]) // 2)
                else:
                    grown[1] = max(grown[1], (other[3] + y0) // 2)
            elif apart_x >= 0:
                grown[2] = min(grown[2], (x1 + other[0]) // 2)
            else:
                grown[0] = max(grown[0], (other[2] + x0) // 2)
        if grown[2] <= grown[0] or grown[3] <= grown[1]:
            return list(box)
        return [
            min(grown[0], x0),
            min(grown[1], y0),
            max(grown[2], x1),
            max(grown[3], y1),
        ]

    def _typeset_box(
        self,
        rgb: np.ndarray,
        unit: TextUnit,
        style: TextStyle,
        avoid: list[list[int]] | None = None,
    ) -> list[int]:
        """The rectangle the translation is typeset into.

        Normally the detector box.  A box too short to hold readable CJK grows
        into the balloon it belongs to, because a 9px smear is unreadable while
        slightly larger type inside the same balloon is not.  When even the
        grown box cannot reach ``MIN_READABLE_FONT`` (a genuinely tiny bubble)
        the box that renders the largest type wins, so the text at least stops
        getting smaller than it has to.
        """
        x0, y0, x1, y1 = unit.bbox
        text = _clean(unit.zh)
        declared = [x0, y0, x1, y1]
        if not text or style.vertical:
            return declared
        font_path = getattr(
            self, "bold_font_path" if style.bold else "font_path", None
        ) or getattr(self, "font_path", None)
        if font_path is None:
            # Without a font there is nothing to measure with, so the detector
            # box is the only honest answer (tests override ``_draw`` and leave
            # the font paths unset).
            return declared
        scratch = ImageDraw.Draw(Image.new("L", (8, 8)))

        def largest(box: list[int]) -> int:
            """Largest font size that fits ``box``, 0 when nothing fits."""
            bx0, by0, bx1, by1 = box
            pad_x, pad_y = _text_padding(box)
            width = max(12, (bx1 - bx0) - 2 * pad_x)
            height = max(12, (by1 - by0) - 2 * pad_y)
            fitting = _fit_readable(
                font_path, scratch, text, width, height, style.font_size, floor=8
            )
            return fitting[1] if fitting else 0

        declared_size = largest(declared)
        if declared_size >= MIN_READABLE_FONT:
            return declared
        grown = self._uniform_balloon_box(rgb, declared, avoid=avoid)
        best = grown if largest(grown) > declared_size else declared
        if largest(best) >= MIN_READABLE_FONT:
            return best
        # The balloon itself is too small for readable CJK (a 46x33 bubble
        # holding "来吧，Twilight，").  Shrinking to 8px is what produced the
        # unreadable smear this fixes, so the last resort is to let the type
        # overflow the balloon: grow symmetrically around it, staying on the
        # page, until the floor fits.  Overlapping artwork is a far smaller
        # defect than text nobody can read.
        return self._overflow_box(rgb, best, largest)

    def _overflow_box(
        self, rgb: np.ndarray, box: list[int], largest, limit: float = 2.2
    ) -> list[int]:
        """Widen ``box`` symmetrically until readable CJK fits.

        The last resort for a balloon too small to hold ``MIN_READABLE_FONT``.
        The type is then allowed to spill past the balloon, because an
        overlapping glyph is legible while an eight-pixel one is not.  Growth is
        bounded by the page and by ``limit`` times the original size, and the
        box that yields the largest type wins, so this never returns something
        smaller than the balloon-only attempt.
        """
        page_height, page_width = rgb.shape[:2]
        x0, y0, x1, y1 = box
        best = list(box)
        ceiling = largest(best)
        grow = 2
        while ceiling < MIN_READABLE_FONT:
            if (x1 - x0) + 2 * grow > (x1 - x0) * limit and (y1 - y0) + 2 * grow > (
                y1 - y0
            ) * limit:
                break
            candidate = [
                max(0, x0 - grow),
                max(0, y0 - grow),
                min(page_width, x1 + grow),
                min(page_height, y1 + grow),
            ]
            if candidate == best:
                # Pinned against the page edge on every side: no more room.
                break
            size = largest(candidate)
            if size > ceiling:
                ceiling = size
                best = candidate
            grow += 2
        return best

    def _draw(self, image: Image.Image, text: str, box: list[int], style: TextStyle):
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box
        pad_x, pad_y = _text_padding(box)
        left, top, right, bottom = x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y
        width, height = max(12, right - left), max(12, bottom - top)
        text = _clean(text)
        if not text:
            return
        font_path = self.bold_font_path if style.bold else self.font_path
        if style.vertical:
            font, size, capacity = _fit_vertical(
                font_path, text, width, height, style.font_size
            )
            # Split into balanced columns instead of filling each one to
            # capacity: greedy filling leaves the last column holding only
            # trailing punctuation ("！！"), which reads as a stray mark.
            columns_count = max(1, math.ceil(len(text) / capacity))
            chunk = math.ceil(len(text) / columns_count)
            chunks = [
                text[index : index + chunk]
                for index in range(0, len(text), chunk)
            ]
            step_x = round(size * (1.12 if style.display else 1.1))
            step_y = round(size * (1.18 if style.display else 1.07))
            start_x = right - max(0, (width - len(chunks) * step_x) / 2) - size
            stroke = round(size * style.stroke_ratio)
            for column, chunk in enumerate(chunks):
                y = top + max(0, (height - len(chunk) * step_y) / 2)
                for row, char in enumerate(chunk):
                    draw.text(
                        (start_x - column * step_x, y + row * step_y),
                        char,
                        font=font,
                        fill=style.fill,
                        stroke_width=stroke,
                        stroke_fill=style.stroke_fill,
                    )
        else:
            font, size, lines, line_height = _fit_horizontal(
                font_path, draw, text, width, height, style.font_size
            )
            stroke = round(size * style.stroke_ratio)
            y = top + max(0, (height - len(lines) * line_height) / 2)
            for line in lines:
                line_width = draw.textbbox(
                    (0, 0), line, font=font, stroke_width=stroke
                )[2]
                draw.text(
                    (left + max(0, (width - line_width) / 2), y),
                    line,
                    font=font,
                    fill=style.fill,
                    stroke_width=stroke,
                    stroke_fill=style.stroke_fill,
                )
                y += line_height

    def render_page(
        self,
        source_path: Path,
        page: PageOCR,
        output_path: Path,
        preserve_sfx: bool = True,
    ) -> dict:
        original = np.array(Image.open(source_path).convert("RGB"))
        rgb = original.copy()
        styles: dict[str, TextStyle] = {}
        active = []
        for source_unit in page.units:
            if (
                source_unit.skip
                or not source_unit.zh
                or (preserve_sfx and source_unit.is_sfx)
            ):
                continue
            unit = deepcopy(source_unit)
            source_boxes = unit.erase_boxes or [unit.bbox]
            safe_boxes = prune_detached_orthogonal_outliers(source_boxes)
            if safe_boxes != source_boxes:
                unit.erase_boxes = safe_boxes
                unit.bbox = [
                    min(box[0] for box in safe_boxes),
                    min(box[1] for box in safe_boxes),
                    max(box[2] for box in safe_boxes),
                    max(box[3] for box in safe_boxes),
                ]
                width = unit.bbox[2] - unit.bbox[0]
                height = unit.bbox[3] - unit.bbox[1]
                pad_x = max(10, round(width * 0.04))
                pad_y = max(10, round(height * 0.03))
                unit.crop_bbox = [
                    max(0, unit.bbox[0] - pad_x),
                    max(0, unit.bbox[1] - pad_y),
                    min(page.width, unit.bbox[2] + pad_x),
                    min(page.height, unit.bbox[3] + pad_y),
                ]
            x0, y0, x1, y1 = unit.bbox
            unit.bbox = [
                max(0, min(page.width, x0)),
                max(0, min(page.height, y0)),
                max(0, min(page.width, x1)),
                max(0, min(page.height, y1)),
            ]
            cx0, cy0, cx1, cy1 = unit.crop_bbox or unit.bbox
            unit.crop_bbox = [
                max(0, min(page.width, cx0)),
                max(0, min(page.height, cy0)),
                max(0, min(page.width, cx1)),
                max(0, min(page.height, cy1)),
            ]
            if unit.bbox[2] > unit.bbox[0] and unit.bbox[3] > unit.bbox[1]:
                active.append(unit)
        for unit in active:
            style = self._analyze_style(original, unit)
            styles[unit.id] = style
            profile = getattr(self, "cleanup_profile", "artwork")
            if profile == "quality" and (
                style.outlined or style.background_std >= 10 or style.fill == "white"
            ):
                rgb = self._erase_with_lama(rgb, unit, style)
                continue
            if profile == "quality":
                # Plain black dialogue on a white balloon does not need neural
                # inpainting, but its furigana often sits just outside the main
                # OCR box.  Clean one expanded reviewed crop with the
                # edge-connected component guard so balloon borders survive.
                cx0, cy0, cx1, cy1 = unit.crop_bbox or unit.bbox
                # OCR boxes frequently cover the main kanji column but omit a
                # neighbouring kana column, so the bands reach a little further
                # than the box itself: long along the text, short across it.
                # They are the lettering's own ring -- the mask is clipped to
                # them, so a lock of hair, a hat brim or the balloon's own
                # outline inside the searched rectangle can no longer come back
                # inpainted.  The perpendicular pad is wider than the mask
                # dilation, so no glyph loses its antialiased edge.
                along = max(24, round(style.font_size * 1.2))
                across = max(8, round(style.font_size * 0.35))
                if (cx1 - cx0) >= (cy1 - cy0):
                    pad_x, pad_y = along, across
                else:
                    pad_x, pad_y = across, along
                bands = [
                    [
                        max(0, ex0 - pad_x),
                        max(0, ey0 - pad_y),
                        min(page.width, ex1 + pad_x),
                        min(page.height, ey1 + pad_y),
                    ]
                    for ex0, ey0, ex1, ey1 in (unit.erase_boxes or [unit.bbox])
                ]
                # The searched rectangle stays generous: the paper branch and
                # the glyph polarity are read from it, and a crop holding
                # nothing but lettering makes both statistics meaningless --
                # shrunk to the bands, the area cap also starts dropping whole
                # glyphs, which leaves the source text showing through.
                margin = min(128, max(96, round(style.font_size * 0.8)))
                bounded = [
                    max(0, cx0 - margin),
                    max(0, cy0 - margin),
                    min(page.width, cx1 + margin),
                    min(page.height, cy1 + margin),
                ]
                rgb, _, _ = self._erase(rgb, bounded, focus=unit.bbox, bands=bands)
                continue
            # Grouping several OCR lines is useful for coherent recognition and
            # typesetting, but erasing their union also destroys every drawing
            # between those lines. Only clean the original tight detector boxes.
            erase_boxes = unit.erase_boxes or [unit.bbox]
            for erase_box in erase_boxes:
                ex0, ey0, ex1, ey1 = erase_box
                erase_width, erase_height = ex1 - ex0, ey1 - ey0
                pad_x = min(24, max(8, round(erase_width * 0.06)))
                pad_y = min(24, max(8, round(erase_height * 0.04)))
                bounded = [
                    max(0, min(page.width, ex0 - pad_x)),
                    max(0, min(page.height, ey0 - pad_y)),
                    max(0, min(page.width, ex1 + pad_x)),
                    max(0, min(page.height, ey1 + pad_y)),
                ]
                if bounded[2] > bounded[0] and bounded[3] > bounded[1]:
                    # Only the artwork profile reaches here; quality took one of
                    # the branches above.  The window is barely wider than one
                    # detector line, so the component cap has to be relative to
                    # that window and generous enough to hold a whole letter.
                    rgb, _, _ = self._erase(rgb, bounded, area_ratio=0.12)
        rendered = Image.fromarray(rgb)
        for unit in active:
            # The detector box is the default typesetting region; a box too
            # short for readable CJK grows into its balloon instead of dropping
            # to an unreadable size.  Rendering still happens inside a bounded
            # crop, so antialiasing and stroke pixels cannot spill past it.
            # Sample the balloon colour from the untouched page, not from the
            # already-erased buffer: inpainted pixels no longer describe the
            # original background and would let the growth run past the border.
            box = self._typeset_box(
                original,
                unit,
                styles[unit.id],
                avoid=[
                    other.crop_bbox or other.bbox
                    for other in active
                    if other.id != unit.id
                ],
            )
            # PIL wants integers and this box is geometry, not pixel
            # indices: round once here so the crop, the drawing and the paste
            # all speak the same language whatever _typeset_box returns.
            x0, y0, x1, y1 = (int(round(value)) for value in box)
            region = rendered.crop((x0, y0, x1, y1))
            self._draw(region, unit.zh, [0, 0, x1 - x0, y1 - y0], styles[unit.id])
            rendered.paste(region, (x0, y0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".webp":
            # libwebp's exhaustive method 6 is several times slower on full
            # manga pages without improving decoded pixels.  Method 2 remains
            # lossless and keeps first-run batch rendering practical.
            rendered.save(output_path, "WEBP", lossless=True, quality=100, method=2)
        else:
            rendered.save(output_path, "PNG", compress_level=7)
        return {
            "source": source_path.name,
            "output": output_path.name,
            "width": page.width,
            "height": page.height,
            "translated_units": len(active),
        }

    def render_book(
        self,
        source_dir: Path,
        pages: list[PageOCR],
        output_dir: Path,
        preserve_sfx: bool = True,
        output_format: str = "webp",
    ) -> list[dict]:
        manifest = []
        for page in pages:
            source = source_dir / page.file
            suffix = ".webp" if output_format == "webp" else ".png"
            output = output_dir / f"{source.stem}{suffix}"
            manifest.append(self.render_page(source, page, output, preserve_sfx))
        (output_dir / "translation_manifest.json").write_text(
            json.dumps({"pages": manifest}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest
