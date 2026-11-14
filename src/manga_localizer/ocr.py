from __future__ import annotations

import base64
import difflib
import json
import re
import time
import unicodedata
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_CONTEXT_TOKENS,
    OLLAMA_MAX_OUTPUT_TOKENS,
    AppPaths,
    configure_model_caches,
    ollama_timeout,
)
from .model_manager import ModelDependencyError
from .sfx import lookup_sfx


def manga_force_cpu(device: str, cuda_available: bool) -> bool:
    """Keep Manga OCR on CUDA independently from Paddle's device backend."""
    return device == "cpu" or not cuda_available


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
# The user can declare what the book is written in.  "auto" keeps the detector's
# own Latin/Japanese arbitration, which is right for mixed books; declaring the
# script makes the recognizer choice deterministic and stops MangaOCR from
# inventing kana on an English page or Paddle from dropping kana-only lines on a
# Japanese one.
SOURCE_LANGUAGES = ("auto", "en", "ja")
KATAKANA_RE = re.compile(r"^[\u30a0-\u30ffー・…ッっ♡♥！？!?.．]+$")
EXPLICIT_SKIP_REASONS = frozenset({"duplicate", "noise", "decorative", "preserve"})
URL_RE = re.compile(
    r"https?://\S+|www\.\S+|[A-Za-z0-9][A-Za-z0-9.-]*\.(?:com|net|org|jp|io|app|co|me|tv)/",
    re.IGNORECASE,
)
UNRESOLVED_SKIP_REASON = "unresolved"
DEFAULT_OLLAMA_VISION_MODEL = DEFAULT_OLLAMA_MODEL
# Every vision schema states the same confidence range.  A plain maximum is what
# llama.cpp's schema-to-grammar conversion actually enforces: ``exclusiveMaximum``
# is dropped there, which let a decoder answer 354.5 for a confidence once.  So
# the bound stays inclusive here and the adapters decide separately whether the
# number means anything (see ``scores_are_informative``).
SCORE_SCHEMA = {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
}


def scores_are_informative(scores: list[float]) -> bool:
    """Whether a vision model's confidence numbers separate text from noise.

    llama.cpp's schema-to-grammar decoder answers a flat -0.0127 for every
    region it proposes -- real dialogue and screentone alike, twice in a row --
    so a 0.72 threshold there rejects exactly the omissions the recall pass
    exists to find.  The daemon's 0.71-1.00 spread still carries meaning.  When
    the numbers carry none, the reading itself has to be the filter.
    """
    if not scores:
        return False
    if all(score <= 0.0 for score in scores):
        return False
    saturated = sum(1 for score in scores if score >= 0.99)
    if saturated * 2 > len(scores):
        return False
    return max(scores) - min(scores) >= 0.05


SFX_KINDS = (
    "heartbeat",
    "impact",
    "engine",
    "rumble",
    "movement",
    "friction",
    "liquid",
    "breath",
    "vocalization",
    "ambience",
    "mechanical",
    "other",
    "none",
)


@contextmanager
def _suppress_optional_ccache_warning():
    """Hide Paddle's compile-cache notice without hiding inference warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"No ccache found\..*",
            category=UserWarning,
        )
        yield


def natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def likely_sfx_text(text: str) -> bool:
    """Conservatively recognize short sound effects from OCR text alone.

    MangaOCR occasionally prefixes a repeated katakana sound with one or two
    hallucinated hiragana characters.  Treating those strings as dialogue
    creates conspicuous nonsense translations such as phonetic gibberish.
    Mixed-kana text is accepted only when a katakana bigram actually repeats,
    so ordinary phrases containing a loanword are left as dialogue.
    """
    clean = text.strip()
    if not clean or len(clean) > 14:
        return False
    if KATAKANA_RE.fullmatch(clean):
        return True
    if re.search(r"[\u3400-\u9fff]", clean):
        return False
    if re.fullmatch(r"[A-Za-z]{2,14}(?:\s+[A-Za-z]{2,14}){0,3}[!?！？.…]*", clean):
        # English pages write their effects as words: GRRRRR, poof, AUGH!
        # They also double them ("sniff snif"), so allow a few space-separated
        # words and let the lookup decide whether every word is an effect.
        return lookup_sfx(clean) is not None
    katakana = "".join(re.findall(r"[ァ-ヺー]", clean))
    hiragana_count = len(re.findall(r"[ぁ-ゖ]", clean))
    if len(katakana) < 4 or hiragana_count > 3:
        return False
    return any(
        katakana.count(katakana[index : index + 2]) >= 2
        for index in range(len(katakana) - 1)
    )


def repeated_sfx_core(text: str) -> str:
    """Extract a detector-supported repeated katakana core from mixed OCR noise."""
    runs = re.findall(r"[ァ-ヺーッ]{4,}", text)
    if not runs:
        return ""
    core = max(runs, key=len)
    return (
        core if any(core.count(char) >= 3 for char in set(core) - {"ッ", "ー"}) else ""
    )


def semantic_sfx_classification(text: str, score: float, vision_is_sfx: bool) -> bool:
    """Resolve unsafe short-fragment and VLM role classifications conservatively.

    A vision model can read an isolated brush-stroke fragment as a grammatical
    word and then invite the translator to invent dialogue around it.  Very
    short, low-confidence OCR is therefore preserved as display lettering/noise
    instead of translated.  Conversely, a confident hiragana phrase ending in
    a Japanese comma is ordinary narration/dialogue even when the VLM labels it
    as an effect (for example ``ついに、``).
    """
    clean = re.sub(r"\s+", "", text.strip())
    semantic_chars = re.sub(r"[．。…、，,!?！？:：♡♥〰〜～]", "", clean)
    if likely_sfx_text(clean):
        return True
    hiragana_count = len(re.findall(r"[ぁ-ゖ]", semantic_chars))
    katakana_count = len(re.findall(r"[ァ-ヺ]", semantic_chars))
    # Punctuation is stronger linguistic evidence than an absent detector
    # score. Paddle's detection-only payload commonly reports ``0`` even for
    # perfectly readable text, so short dialogue such as ``いや、`` and
    # ``ん？`` must not be silently preserved as sound effects.
    has_kanji = bool(re.search(r"[\u3400-\u9fff]", semantic_chars))
    has_dialogue_punctuation = bool(re.search(r"[．。…、，,?？]", clean))
    if has_kanji:
        return False
    if hiragana_count and (
        hiragana_count >= 2
        or hiragana_count + katakana_count >= 3
        or has_dialogue_punctuation
    ):
        return False
    if katakana_count and has_dialogue_punctuation:
        return False
    if score < 0.70 and len(semantic_chars) <= 3:
        return True
    return vision_is_sfx


def likely_dialogue_text(text: str) -> bool:
    """Recognize grammatical Japanese that must not become protected artwork.

    This is intentionally narrower than general Japanese detection.  It catches
    short hiragana-bearing phrases such as ``ここは`` while leaving kanji-only
    shirt slogans, logos, and katakana sound effects to the visual role model.
    """
    clean = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", text)
    if likely_sfx_text(text) or not clean:
        return False
    hiragana_count = len(re.findall(r"[ぁ-ゖ]", clean))
    return hiragana_count >= 2 and len(clean) >= 3


def tiny_low_confidence_nontext(page: PageOCR, unit: TextUnit) -> bool:
    """Identify tiny detector fragments that cannot support their OCR string."""
    width = max(1, unit.bbox[2] - unit.bbox[0])
    height = max(1, unit.bbox[3] - unit.bbox[1])
    semantic = re.sub(r"[\s.．。…、，,!?！？:：♡♥〰〜～『』「」]", "", unit.ja)
    if not semantic or unit.score > 0.05:
        return False
    unbalanced_delimiter = unit.ja.count("『") != unit.ja.count("』") or unit.ja.count(
        "「"
    ) != unit.ja.count("」")
    return (
        unbalanced_delimiter
        and height <= max(32, round(page.height * 0.012))
        and width * height / max(1, len(semantic)) < 420
    )


def malformed_tiny_ocr(page: PageOCR, unit: TextUnit) -> bool:
    """Reject implausible repeated-kanji OCR found in tiny artwork patches."""
    width = max(1, unit.bbox[2] - unit.bbox[0])
    height = max(1, unit.bbox[3] - unit.bbox[1])
    compact = re.sub(r"\s+", "", unit.ja)
    return bool(
        unit.score < 0.8
        and width <= page.width * 0.02
        and height <= page.height * 0.04
        and re.match(r"^([\u3400-\u9fff])\1", compact)
    )


PUBLICATION_METADATA_MARKERS = (
    "発行",
    "サークル",
    "漫画",
    "イラスト",
    "ロゴ",
    "デザイン",
    "協力",
    "印刷",
    "著者",
    "作者",
    "Twitter",
)


def page_spanning_display_artwork(page: PageOCR, unit: TextUnit) -> bool:
    """Detect cover artwork that OCR misread as one page-filling caption.

    A cover plate carries stylized display lettering, not a balloon.  Paddle
    groups the display lines into a single detector region spanning a sixth to a
    third of the page, and MangaOCR -- which only knows Japanese -- answers with
    a plausible-looking date for it ("２０１８年０９月１０日" for the three art
    lines of "Display of Passion").  Without this rule the unit looked like an
    ordinary translatable caption: the quality cleaner erased every art line and
    typeset "2018 年 09 月 10 日" across the cover.

    The same misread appears on inner title plates, where the display block is
    smaller (13% of the page instead of 30%) and the hallucinated text is longer,
    so the threshold has to leave room below the cover case.

    One region covering an eighth of the page while every character claims more
    than 0.8% of the page area cannot be a speech bubble or a caption: dialogue
    glyphs sit around 0.02-0.05% of a page, so this demands lettering roughly
    five times larger than a balloon font while also covering a large block.
    Both measures are normalized by page area, so the rule holds at any scan
    resolution.  On the 2248 units currently on disk the two measures separate
    cleanly: the artwork regions score 0.296/0.027 and 0.134/0.009, and the next
    candidate -- long dialogue blocks with small type -- drops to 0.002.

    Merged cover titles are excluded: those are meant to be translated in place,
    and display sound effects are excluded because ``oversized_decorative_sfx``
    owns them.
    """
    if unit.special == "cover_title" or unit.is_sfx:
        return False
    semantic = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", unit.ja)
    if not semantic:
        return False
    page_area = max(1, page.width * page.height)
    box_area = _area(unit.bbox)
    if box_area < page_area * 0.12:
        return False
    return box_area / len(semantic) >= page_area * 0.008


def classify_non_story_text(page: PageOCR) -> PageOCR:
    """Preserve publication metadata and tiny edge folios as non-story text.

    Built-in OCR deliberately groups nearby horizontal lines so ordinary text
    blocks retain one layout region.  On colophons that produces a large
    attribution block, while a printed page number can be misread as one kana
    plus a digit.  Neither is safe to translate or erase: author/contact data
    must stay byte-for-byte visible and folios are decorative navigation.

    The classifier requires both semantic and geometric evidence.  It does
    not silently skip an ordinary dialogue unit merely because one metadata
    word appears in the story.
    """
    page_area = max(1, page.width * page.height)
    for unit in page.units:
        if unit.skip:
            continue
        normalized = unicodedata.normalize("NFKC", unit.ja)
        compact = re.sub(r"[\s.．。…、，,!?！？:：♡♥〰〜～・]", "", normalized)
        width = max(1, unit.bbox[2] - unit.bbox[0])
        height = max(1, unit.bbox[3] - unit.bbox[1])
        near_side = (
            unit.bbox[0] <= page.width * 0.08
            or unit.bbox[2] >= page.width * 0.92
        )
        near_page_end = (
            unit.bbox[1] <= page.height * 0.08
            or unit.bbox[3] >= page.height * 0.90
        )
        tiny_edge_folio = (
            near_side
            and near_page_end
            and width <= page.width * 0.05
            and height <= page.height * 0.05
            and width * height <= page_area * 0.001
            and bool(re.fullmatch(r"[ぁ-ゖァ-ヺー]?\d{1,4}", compact))
        )
        if tiny_edge_folio:
            unit.zh = ""
            unit.skip = True
            unit.skip_reason = "decorative"
            unit.special = "page_folio"
            continue

        if page_spanning_display_artwork(page, unit):
            unit.zh = ""
            unit.skip = True
            unit.skip_reason = "preserve"
            unit.special = "artwork_text"
            continue

        markers = {
            marker for marker in PUBLICATION_METADATA_MARKERS if marker in normalized
        }
        erase_boxes = unit.erase_boxes or [unit.bbox]
        horizontal_rows = sum(
            (box[2] - box[0]) >= max(1, box[3] - box[1]) * 1.8
            for box in erase_boxes
        )
        publication_block = (
            len(markers) >= 2
            and len(erase_boxes) >= 3
            and horizontal_rows >= max(3, round(len(erase_boxes) * 0.7))
            and width >= page.width * 0.20
            and height >= page.height * 0.06
        )
        if publication_block:
            unit.zh = ""
            unit.skip = True
            unit.skip_reason = "preserve"
            unit.special = "publication_metadata"
            continue
        if link_block_text(unit.ja):
            # Credit pages print several links in one detector region.  A
            # translation can only echo or summarise them, both of which fail
            # the quality contract, so the unit used to stay unresolved and
            # failed the whole book.  Keep the printed links instead.
            unit.zh = ""
            unit.skip = True
            unit.skip_reason = "preserve"
            unit.special = "link_block"
    return page


def duplicate_tiny_fragment(page: PageOCR, unit: TextUnit) -> bool:
    """Detect ruby fragments and VLM crop echoes of a stronger detector unit."""
    width = max(1, unit.bbox[2] - unit.bbox[0])
    height = max(1, unit.bbox[3] - unit.bbox[1])
    is_tiny = width <= page.width * 0.04 and height <= page.height * 0.03
    area = width * height

    def semantic(text: str) -> str:
        normalized = text.translate(
            str.maketrans(
                "ぁぃぅぇぉゃゅょっァィゥェォャュョッ",
                "あいうえおやゆよつアイウエオヤユヨツ",
            )
        )
        return re.sub(r"[^ぁ-ゖァ-ヺー\u3400-\u9fff]", "", normalized)

    compact = semantic(unit.ja)
    kanji = "".join(re.findall(r"[\u3400-\u9fff]", compact))
    if len(compact) < 2:
        return False
    for other in page.units:
        if other.id == unit.id:
            continue
        other_area = max(1, other.bbox[2] - other.bbox[0]) * max(
            1, other.bbox[3] - other.bbox[1]
        )
        other_compact = semantic(other.ja)
        repeated = compact == other_compact or (
            len(compact) >= 3 and compact in other_compact
        )
        repeated = repeated or (len(kanji) >= 2 and kanji in other_compact)
        if not repeated:
            continue
        if is_tiny and other_area >= area * 4:
            return True
        if len(compact) >= 5 and other_area >= area * 4:
            return True
        other_width = max(1, other.bbox[2] - other.bbox[0])
        other_height = max(1, other.bbox[3] - other.bbox[1])
        weak_union_box = len(unit.erase_boxes) <= 1 and width >= height * 0.8
        stronger_vertical_boxes = (
            len(other.erase_boxes) >= 2 and other_height >= other_width * 1.4
        )
        if len(compact) >= 5 and weak_union_box and stronger_vertical_boxes:
            return True
        if (
            len(compact) >= 5
            and other_area >= area * 1.6
            and len(other.erase_boxes) > len(unit.erase_boxes)
            and other_height >= other_width * 1.2
        ):
            return True
    return False


def overlapping_sfx_echo(page: PageOCR, unit: TextUnit) -> bool:
    """Reject a short OCR echo inside a larger, visually confirmed SFX group."""
    compact = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", unit.ja)
    if (
        unit.is_sfx
        or unit.special == "semantic_dialogue"
        or not re.fullmatch(r"[ァ-ヺーッ]{2,5}", compact)
    ):
        return False
    x0, y0, x1, y1 = unit.bbox
    area = max(1, (x1 - x0) * (y1 - y0))
    for other in page.units:
        confirmed_non_dialogue = (
            other.special in {"decorative_sfx", "artwork_text"}
            or other.special.startswith("sfx:")
        )
        if other.id == unit.id or not confirmed_non_dialogue:
            continue
        ox0, oy0, ox1, oy1 = other.bbox
        intersection = max(0, min(x1, ox1) - max(x0, ox0)) * max(
            0, min(y1, oy1) - max(y0, oy0)
        )
        other_area = max(1, (ox1 - ox0) * (oy1 - oy0))
        if intersection / area >= 0.25 and other_area >= area * 3:
            return True
    return False


# A detector box that neither Paddle pass could read still reaches MangaOCR,
# which never reports "no text": pointed at artwork it invents kana that pass
# JP_RE, so a ribbon swirl became "２０１８年０" and a cutie mark became "その".
# Real lettering, including stylised display type, survives a change of crop;
# an invention does not.  Measured on 16 text-less candidates from two books:
# the three real ones (a cover title, its repeat, a wooden sign) agreed at
# 0.75-1.00, while all thirteen inventions agreed at 0.00-0.40, so 0.5 splits
# them with room to spare.
RESCUE_READING_AGREEMENT = 0.5


def rescue_reading_is_stable(readings: list[str]) -> bool:
    """Whether MangaOCR read the same text from several crops of one region."""
    normalized = [
        "".join(character for character in value if character.isalnum())
        for value in readings
    ]
    if not all(normalized):
        return False
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            agreement = difflib.SequenceMatcher(None, left, right).ratio()
            if agreement < RESCUE_READING_AGREEMENT:
                return False
    return True


def prefer_semantic_ocr(current: str, candidate: str, score: float) -> bool:
    """Accept contextual OCR corrections without replacing complete text by a prefix."""
    if score < 0.72 or not JP_RE.search(candidate):
        return False

    def normalize(value: str) -> str:
        return re.sub(r"[\s．。…、，,!?！？:：]", "", value)

    current_clean = normalize(current)
    candidate_clean = normalize(candidate)
    if not current_clean:
        return True
    # Vision models sometimes stop after the first clause while assigning high
    # confidence. If the proposed text is merely a short prefix/subsequence of
    # a longer detector crop, MangaOCR contains strictly more source evidence.
    if len(candidate_clean) < len(current_clean) * 0.72:
        return False
    agreement = difflib.SequenceMatcher(None, current_clean, candidate_clean).ratio()
    contained = current_clean in candidate_clean or candidate_clean in current_clean
    return contained or agreement >= 0.5


def semantic_text_agreement(current: str, candidate: str) -> bool:
    """Compare OCR strings without requiring Japanese script.

    Scene text and clothing logos can be Latin, so the Japanese-only semantic
    correction gate is not suitable when confirming an ``artwork`` role.
    """

    def normalize(value: str) -> str:
        return re.sub(r"\W+", "", value).casefold()

    current_clean = normalize(current)
    candidate_clean = normalize(candidate)
    if not current_clean or not candidate_clean:
        return False
    return (
        current_clean == candidate_clean
        or difflib.SequenceMatcher(None, current_clean, candidate_clean).ratio() >= 0.8
    )


def _rereads_known_text(
    text: str, known_text: set[str], known_bigrams: set[str]
) -> bool:
    """Detect a garbled re-read of a line the exact detector already owns.

    The recall pass re-reads whole pages, so it returns lines MangaOCR already
    recognised with one or two characters mangled ("何か理由が…ちゃんとなんて
    話を" for "…ちゃんと話を", "平気よの" for "…平気よ").  Neither containment
    nor a strict ratio catches every case, so a claim is treated as a re-read
    when most of its bigrams appear in the page's known text.  Two-character
    effects stay exempt: ``がー`` shares nothing, and a real omission must keep
    failing the book.  Dropping a re-read costs one untranslated line at worst;
    keeping one fails the entire book.
    """
    if len(text) < 3:
        return False
    bigrams = {text[index : index + 2] for index in range(len(text) - 1)}
    if not bigrams:
        return False
    coverage = len(bigrams & known_bigrams) / len(bigrams)
    if coverage >= 0.6:
        return True
    return any(
        difflib.SequenceMatcher(None, text, known).ratio() >= 0.5
        for known in known_text
    )


def merge_semantic_missing(*batches: list[dict]) -> list[dict]:
    """Union repeated visual omissions without merging two equal nearby sounds."""
    merged: list[dict] = []
    for item in (entry for batch in batches for entry in batch):
        try:
            box = [int(value) for value in item["bbox"]]
        except (KeyError, TypeError, ValueError):
            box = []
        text = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", str(item.get("text", "")))
        duplicate = False
        for known in merged:
            known_text = re.sub(
                r"[\s．。…、，,!?！？:：♡♥〰〜～]",
                "",
                str(known.get("text", "")),
            )
            try:
                known_box = [int(value) for value in known["bbox"]]
            except (KeyError, TypeError, ValueError):
                known_box = []
            if text != known_text or len(box) != 4 or len(known_box) != 4:
                continue
            overlap = _axis_overlap(box[0], box[2], known_box[0], known_box[2]) * (
                _axis_overlap(box[1], box[3], known_box[1], known_box[3])
            )
            if overlap >= min(_area(box), _area(known_box)) * 0.35:
                duplicate = True
                if float(item.get("score", 0.0)) > float(known.get("score", 0.0)):
                    known.update(item)
                break
        if not duplicate:
            merged.append(dict(item))
    return merged


def link_block_text(text: str) -> bool:
    """A block of URLs is not prose and must not be translated.

    Credit pages put several links in one detector region.  The model can only
    echo them or summarise them, and both are rejected, so the unit stays
    unresolved and fails the whole book."""
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    links = URL_RE.findall(text)
    if len(links) >= 3:
        return True
    return bool(links) and sum(len(item) for item in links) >= len(compact) * 0.4


def oversized_text_region(page: PageOCR, unit: TextUnit) -> bool:
    """Identify detector regions too broad to be safe destructive text masks."""
    semantic = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", unit.ja)
    return bool(
        unit.special != "cover_title"
        and not unit.is_sfx
        and semantic
        and _area(unit.bbox) >= page.width * page.height * 0.06
        and _area(unit.bbox) / len(semantic) > 40_000
    )


def oversized_decorative_sfx(page: PageOCR, unit: TextUnit) -> bool:
    """Preserve page-spanning display SFX instead of destructively replacing it."""
    semantic = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", unit.ja)
    return bool(
        unit.is_sfx
        and semantic
        and _area(unit.bbox) >= page.width * page.height * 0.06
        and _area(unit.bbox) / len(semantic) > 40_000
    )


def _adjacent_column_boxes(
    gray: np.ndarray,
    box: list[int],
    column_width: int,
    known_boxes: list[list[int]],
) -> list[list[int]]:
    """Vertical ink columns next to ``box`` that no recognised box covers.

    Paddle scores the thin outer column of a multi-column Japanese bubble below
    its detection floor once the page is downscaled, so a three-column bubble
    can arrive with only its inner columns boxed and the outer column left
    untranslated.  Lowering that floor globally fixes the bubble but drops real
    text on other pages, so the missing columns are located from this page's own
    pixels: a single contiguous glyph-width run of ink beside the group, which
    no detector or recognizer box claims at all.  Only geometry is returned;
    ``analyze`` still has to read Japanese from it.
    """
    height, width = gray.shape
    x0, y0, x1, y1 = box
    if y1 - y0 < max(8, column_width) * 1.5:
        return []
    top = max(0, y0 - max(10, round((y1 - y0) * 0.08)))
    bottom = min(height, y1 + max(10, round((y1 - y0) * 0.08)))
    columns: list[list[int]] = []
    edge = x0
    for _ in range(2):
        near, far = sorted((edge, edge - max(8, column_width)))
        if near >= width or far <= 0:
            break
        strip = gray[top:bottom, max(0, near) : min(width, far)]
        if strip.size == 0:
            break
        probe = [max(0, near), top, min(width, far), bottom]
        if any(
            _axis_overlap(probe[0], probe[2], other[0], other[2])
            * _axis_overlap(probe[1], probe[3], other[1], other[3])
            >= _area(probe) * 0.25
            for other in known_boxes
        ):
            break
        active = (strip < 110).mean(axis=0) >= 0.02
        runs: list[tuple[int, int]] = []
        start = None
        for index, flagged in enumerate(active):
            if flagged and start is None:
                start = index
            elif not flagged and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, len(active)))
        runs = [run for run in runs if run[1] - run[0] >= 6]
        # A single contiguous glyph-width run is a text column; a gutter, a
        # panel border or scattered artwork is not, so scanning stops there.
        if len(runs) != 1:
            break
        left = max(0, near) + runs[0][0]
        right = max(0, near) + runs[0][1]
        columns.append([left, top, right, bottom])
        edge = left
    return columns


def _best_recognized_text(group: list[dict]) -> tuple[str, float]:
    """Reassemble one region group's recognized text in reading order.

    A bubble is detected as several lines and each line as several boxes.  Only
    the longest single box used to survive, which truncated every bubble to one
    fragment ("I SHALL GIVE DUE TO THE" for a two-line speech balloon).  Segments
    are joined top-to-bottom, left-to-right, and Latin words keep a space between
    segments because Japanese-style separation would merge "HAVE TO" and "WEAR".
    """
    pieces = [
        (item["box"], str(item.get("text", "")).strip(), float(item.get("score", 0.0)))
        for item in group
        if str(item.get("text", "")).strip()
    ]
    if not pieces:
        return "", 0.0
    pieces.sort(key=lambda piece: (piece[0][1], piece[0][0]))
    texts = [text for _box, text, _score in pieces]
    latin = sum(len(re.findall(r"[A-Za-z\u00c0-\u024f]", text)) for text in texts)
    japanese = sum(len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text)) for text in texts)
    joiner = " " if latin > japanese else ""
    return joiner.join(texts), max(score for _box, _text, score in pieces)


def is_latin_page(regions: list[dict]) -> bool:
    """A page whose recognized text is overwhelmingly Latin is an English book."""
    latin = sum(
        len(re.findall(r"[A-Za-z\u00c0-\u024f]", str(item.get("text", ""))))
        for item in regions
    )
    script = sum(
        len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", str(item.get("text", ""))))
        for item in regions
    )
    return latin > 0 and latin > script * 2


def language_label(source_language: str) -> str:
    """Human name of a declared source language, for prompts and logs."""
    return {"en": "English", "ja": "Japanese"}.get(source_language, "auto")


def language_label_zh(source_language: str) -> str:
    """The same label in the language the Chinese semantic prompts are written in."""
    return {"en": "英语", "ja": "日语"}.get(source_language, "日语")


def page_reads_as_latin(regions: list[dict], source_language: str = "auto") -> bool:
    """Whether this page's regions should be read as Latin rather than Japanese.

    An explicit source language wins over the detector's own guess: on a declared
    English book a page of shouted Cyrillic-looking effects or a half-empty page
    no longer has to score as "mostly Latin" before Paddle's text is trusted, and
    on a declared Japanese book a page that happens to contain a long Latin sign
    no longer throws the whole page onto the Latin path.
    """
    if source_language == "en":
        return True
    if source_language == "ja":
        return False
    return is_latin_page(regions)


def is_source_text(text: str, source_language: str = "auto") -> bool:
    """Whether a reading is written in the declared source language.

    ``auto`` keeps the historical behaviour, where every accepted reading had to
    contain Japanese.  A declared English book instead requires Latin letters --
    the vision model sometimes answers a Japanese caption in English, and that
    had to be dropped rather than treated as the original.
    """
    if source_language == "en":
        return bool(re.search(r"[A-Za-z\u00c0-\u024f]", text))
    return bool(JP_RE.search(text))


def _is_latin_region(text: str, score: float) -> bool:
    """Decide whether a region holds Latin-script dialogue rather than Japanese.

    Paddle's PP-OCRv5 recognizer covers Latin scripts well, while MangaOCR only
    knows Japanese and hallucinates kana for English bubbles.  A region therefore
    keeps Paddle's own text when it has a confident score and its letters are
    overwhelmingly Latin.  Japanese still routes to MangaOCR, which wins on kana.
    """
    clean = re.sub(r"\s+", "", text)
    if len(clean) < 3 or score < 0.75:
        return False
    latin = len(re.findall(r"[A-Za-z\u00c0-\u024f]", clean))
    kana = len(re.findall(r"[\u3040-\u30ff]", clean))
    letters = latin + kana + len(re.findall(r"[\u3400-\u9fff]", clean))
    return letters > 0 and kana == 0 and latin / letters >= 0.6


def _box_iou(first: list[float], second: list[float]) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes."""
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _payload_rows(payload: dict, key: str) -> list:
    """Rows of a payload field as a list, without testing an array's truth."""
    values = payload.get(key)
    if values is None:
        return []
    return [value for value in values]


def better_recognition(primary: dict, inverted: dict) -> dict:
    """Keep the more confident reading of every recognised line.

    ``inverted`` comes from the same page turned into an inverted greyscale, so
    its line boxes sit on top of the primary ones.  Only the strings that read
    better replace the originals; lines only the second pass found are appended.
    Nothing is deleted, which keeps the detector the one source of truth for
    what gets erased.
    """
    boxes = [
        [float(value) for value in row]
        for row in _payload_rows(primary, "rec_boxes")
    ]
    texts = [str(text) for text in _payload_rows(primary, "rec_texts")]
    scores = [float(score) for score in _payload_rows(primary, "rec_scores")]
    texts += [""] * (len(boxes) - len(texts))
    scores += [0.0] * (len(boxes) - len(scores))
    other_boxes = [
        [float(value) for value in row]
        for row in _payload_rows(inverted, "rec_boxes")
    ]
    other_texts = [str(text) for text in _payload_rows(inverted, "rec_texts")]
    other_scores = [float(score) for score in _payload_rows(inverted, "rec_scores")]
    other_texts += [""] * (len(other_boxes) - len(other_texts))
    other_scores += [0.0] * (len(other_boxes) - len(other_scores))
    others = list(zip(other_boxes, other_texts, other_scores))
    for index, box in enumerate(boxes):
        best_score, best_at = scores[index], -1
        for other_index, (candidate, _, score) in enumerate(others):
            if _box_iou(box, candidate) < 0.5:
                continue
            if score > best_score:
                best_score, best_at = score, other_index
        if best_at >= 0:
            boxes[index], texts[index], scores[index] = (
                others[best_at][0],
                others[best_at][1],
                best_score,
            )
    for candidate, other_text, other_score in others:
        if any(_box_iou(candidate, box) >= 0.5 for box in boxes):
            continue
        boxes.append(candidate)
        texts.append(other_text)
        scores.append(other_score)
    merged = dict(primary)
    merged["rec_boxes"] = boxes
    merged["rec_texts"] = texts
    merged["rec_scores"] = scores
    return merged


def merge_region_candidates(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Union detector candidates without letting recognition delete text boxes.

    The detection-only payload has no text, while the recognition payload has
    both.  A duplicate used to be dropped outright, which threw the recognized
    text away and left MangaOCR to guess at every bubble.  Duplicates now donate
    their text and score to the surviving detector box instead.
    """
    merged = list(primary)
    for candidate in secondary:
        candidate_box = candidate["box"]
        overlapping = [
            item
            for item in merged
            if _axis_overlap(
                candidate_box[0], candidate_box[2], item["box"][0], item["box"][2]
            )
            * _axis_overlap(
                candidate_box[1], candidate_box[3], item["box"][1], item["box"][3]
            )
            >= min(_area(candidate_box), _area(item["box"])) * 0.65
        ]
        if not overlapping:
            merged.append(dict(candidate))
            continue
        for item in overlapping:
            text = str(candidate.get("text", "")).strip()
            if text and text not in str(item.get("text", "")):
                item["text"] = f"{item.get('text', '')} {text}".strip()
            item["score"] = max(
                float(item.get("score", 0.0)), float(candidate.get("score", 0.0))
            )
    return merged


def deduplicate_nested_region_groups(
    groups: list[list[dict]],
) -> list[tuple[list[dict], list[int]]]:
    """Keep the broadest recovered group when another proposal nests inside it."""
    accepted: list[tuple[list[dict], list[int]]] = []
    for group in sorted(groups, key=lambda item: _area(_union_box(item)), reverse=True):
        group_box = _union_box(group)
        nested = any(
            _axis_overlap(group_box[0], group_box[2], box[0], box[2])
            * _axis_overlap(group_box[1], group_box[3], box[1], box[3])
            >= min(_area(group_box), _area(box)) * 0.65
            for _accepted, box in accepted
        )
        if not nested:
            accepted.append((group, group_box))
    return accepted


def _effect_text_repeats(child: str, parent: str) -> bool:
    """Is the smaller effect just a re-read of the larger one?

    Geometry alone is not evidence of duplication.  A reaction mark such as
    ``びっ`` sits inside the same patch of screentone as a neighbouring
    ``バアン`` while being a different sound, and dropping it loses a
    translation the reader expects.  Only text that the larger unit already
    carries may be discarded.
    """
    child_clean = re.sub(r"[\W_]+", "", child)
    parent_clean = re.sub(r"[\W_]+", "", parent)
    if not child_clean or not parent_clean:
        return False
    if child_clean in parent_clean:
        return True
    return difflib.SequenceMatcher(None, child_clean, parent_clean).ratio() >= 0.8


def prune_nested_duplicate_units(page: PageOCR) -> PageOCR:
    """Drop smaller effect units recovered inside an existing broad effect."""
    kept: list[TextUnit] = []
    for unit in sorted(page.units, key=lambda item: _area(item.bbox), reverse=True):
        area = _area(unit.bbox)
        nested_effect = any(
            unit.is_sfx
            and parent.is_sfx
            and _area(parent.bbox) >= area * 1.5
            and _axis_overlap(
                unit.bbox[0], unit.bbox[2], parent.bbox[0], parent.bbox[2]
            )
            * _axis_overlap(unit.bbox[1], unit.bbox[3], parent.bbox[1], parent.bbox[3])
            >= area * 0.85
            and _effect_text_repeats(unit.ja, parent.ja)
            for parent in kept
        )
        if not nested_effect:
            kept.append(unit)
    page.units = sorted(kept, key=lambda unit: (unit.bbox[1] // 180, -unit.bbox[0]))
    for index, unit in enumerate(page.units, start=1):
        unit.id = f"p{page.page:03d}u{index:02d}"
    return page


def ink_tile_hints(
    page: PageOCR,
    gray: np.ndarray,
    *,
    columns: int = 3,
    rows: int = 4,
) -> list[dict]:
    """Deterministic recall pass over page areas the detector never covered.

    A vision model that proposes nothing leaves ``recover_missing`` with
    nothing to verify, so small sound effects stay lost however good the
    recovery is.  llama.cpp does exactly that on a dense vertical page: two
    runs in a row answered with the same two invented boxes, scores included.
    Scanning fixed ink-bearing tiles does not guess -- every tile still has to
    survive Paddle's detector and an independent MangaOCR read before it can
    become a unit -- and it does not depend on a confidence number the decoder
    never actually varied.
    """
    height, width = gray.shape[:2]
    if width <= 0 or height <= 0 or not page.units:
        return []
    known = [
        erase_box for unit in page.units for erase_box in (unit.erase_boxes or [unit.bbox])
    ]
    tile_w, tile_h = width / columns, height / rows
    hints: list[dict] = []
    for row in range(rows):
        for column in range(columns):
            box = [
                round(column * tile_w),
                round(row * tile_h),
                round((column + 1) * tile_w),
                round((row + 1) * tile_h),
            ]
            region = gray[box[1] : box[3], box[0] : box[2]]
            if region.size == 0 or float((region < 140).mean()) < 0.02:
                continue
            covered = max(
                (
                    _axis_overlap(box[0], box[2], other[0], other[2])
                    * _axis_overlap(box[1], box[3], other[1], other[3])
                    / max(1, _area(box))
                    for other in known
                ),
                default=0.0,
            )
            if covered >= 0.55:
                continue
            hints.append(
                {
                    "text": "",
                    "bbox": [
                        round(box[0] * 1000 / width),
                        round(box[1] * 1000 / height),
                        round(box[2] * 1000 / width),
                        round(box[3] * 1000 / height),
                    ],
                    "score": 0.8,
                    "is_sfx": True,
                }
            )
    return hints


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Source folder does not exist: {folder}")
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=natural_key,
    )


@dataclass
class TextUnit:
    id: str
    bbox: list[int]
    crop_bbox: list[int]
    ja: str
    score: float
    is_sfx: bool = False
    zh: str = ""
    skip: bool = False
    # A skipped unit is only complete when a reviewer records an explicit,
    # machine-checkable reason. Legacy ``skip: true`` values are imported as
    # ``unresolved`` so a missed translation can never silently pass QA.
    skip_reason: str = ""
    # Keep the detector's tight line boxes. ``bbox`` is the union used for
    # translation layout, while these boxes constrain destructive cleanup to
    # the pixels that actually contained source text.
    erase_boxes: list[list[int]] = field(default_factory=list)
    # Local-only audit trail for failed/repair translation attempts. Persisted
    # in draft transcripts so a quality-gate failure remains diagnosable.
    translation_attempts: list[str] = field(default_factory=list)
    # Optional semantic layout hint recorded by a reviewed transcript.  It is
    # never required for ordinary OCR output, but lets cover titles and other
    # display lettering keep their intended composition without page-specific
    # hard-coding in the renderer.
    special: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageOCR:
    page: int
    file: str
    width: int
    height: int
    units: list[TextUnit]
    # Full-page vision may see text outside detector-owned geometry. These
    # findings are audit-only: without exact detector boxes they must never be
    # erased or rendered automatically.
    semantic_missing: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["units"] = [unit.to_dict() for unit in self.units]
        return payload


def _area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _axis_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _connects(a: dict, b: dict) -> bool:
    abox, bbox = a["box"], b["box"]
    aw, ah = abox[2] - abox[0], abox[3] - abox[1]
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x_gap = _axis_gap(abox[0], abox[2], bbox[0], bbox[2])
    y_gap = _axis_gap(abox[1], abox[3], bbox[1], bbox[3])
    x_overlap = _axis_overlap(abox[0], abox[2], bbox[0], bbox[2])
    y_overlap = _axis_overlap(abox[1], abox[3], bbox[1], bbox[3])
    a_vertical, b_vertical = ah >= aw * 1.5, bh >= bw * 1.5
    a_horizontal, b_horizontal = aw >= ah * 1.5, bw >= bh * 1.5
    # A large horizontal garment/logo region can sit a few pixels from a
    # vertical speech balloon.  Proximity alone must not merge orthogonal text
    # systems into one destructive unit. Ruby remains compatible because it is
    # normally narrow/neutral rather than a large strongly horizontal block.
    if (a_vertical and b_horizontal) or (a_horizontal and b_vertical):
        return False
    # Gaps are scaled to the text itself.  A fixed 58/72 px budget merges
    # separate bubbles on dense English pages, which pulls several lines of
    # dialogue (and their erase boxes) into one unit and scrambles the reading
    # order.  Line spacing inside one bubble stays well under one line height.
    if ah >= aw * 1.25 and bh >= bw * 1.25:
        slack = max(10, round(min(aw, bw) * 0.9))
        return x_gap <= slack and y_overlap / max(1, min(ah, bh)) >= 0.22
    if aw >= ah * 1.25 and bw >= bh * 1.25:
        slack = max(10, round(min(ah, bh) * 0.9))
        return y_gap <= slack and x_overlap / max(1, min(aw, bw)) >= 0.35
    return x_gap <= 36 and y_gap <= 36


def prune_detached_orthogonal_outliers(boxes: list[list[int]]) -> list[list[int]]:
    """Drop a large detached scene-text box attached to a tighter text group.

    This is a preservation-only guard for legacy/reviewed transcripts as well
    as detector output.  It never invents geometry: at least two smaller,
    parallel boxes must agree, while the candidate is both orthogonal, more
    than four times their median area, and non-overlapping.
    """
    if len(boxes) < 3:
        return [list(box) for box in boxes]

    def orientation(box: list[int]) -> str:
        width = max(1, box[2] - box[0])
        height = max(1, box[3] - box[1])
        if width >= height * 1.5:
            return "horizontal"
        if height >= width * 1.5:
            return "vertical"
        return "neutral"

    areas = sorted(_area(box) for box in boxes)
    median_area = areas[len(areas) // 2]
    kept: list[list[int]] = []
    for candidate in boxes:
        candidate_orientation = orientation(candidate)
        peers = [
            other
            for other in boxes
            if other is not candidate
            and orientation(other) not in {"neutral", candidate_orientation}
            and _area(other) * 3 <= _area(candidate)
        ]
        detached = all(
            _axis_overlap(candidate[0], candidate[2], peer[0], peer[2])
            * _axis_overlap(candidate[1], candidate[3], peer[1], peer[3])
            == 0
            for peer in peers
        )
        is_outlier = (
            candidate_orientation != "neutral"
            and len(peers) >= 2
            and _area(candidate) > median_area * 4
            and detached
        )
        if not is_outlier:
            kept.append(list(candidate))
    return kept or [list(box) for box in boxes]


def _groups(regions: list[dict]) -> list[list[dict]]:
    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if _connects(regions[i], regions[j]):
                union(i, j)
    buckets: dict[int, list[dict]] = {}
    for index, region in enumerate(regions):
        buckets.setdefault(find(index), []).append(region)
    return list(buckets.values())


def _union_box(items: list[dict]) -> list[int]:
    return [
        min(item["box"][0] for item in items),
        min(item["box"][1] for item in items),
        max(item["box"][2] for item in items),
        max(item["box"][3] for item in items),
    ]


def _cover_title_units(
    units: list[TextUnit], page_width: int, page_height: int
) -> list[TextUnit]:
    """Return a conservative full-page display-title group, if one exists."""
    if not 2 <= len(units) <= 6:
        return []
    page_area = max(1, page_width * page_height)
    box = [
        min(unit.bbox[0] for unit in units),
        min(unit.bbox[1] for unit in units),
        max(unit.bbox[2] for unit in units),
        max(unit.bbox[3] for unit in units),
    ]
    width, height = box[2] - box[0], box[3] - box[1]
    total_area = sum(_area(unit.bbox) for unit in units)
    centers = [(unit.bbox[0] + unit.bbox[2]) / 2 for unit in units]
    if not (
        height >= page_height * 0.55
        and page_width * 0.18 <= width <= page_width * 0.58
        and width * height >= page_area * 0.15
        and total_area >= page_area * 0.05
        and min(centers) >= page_width * 0.24
        and max(centers) <= page_width * 0.76
        and sum(len(unit.ja) for unit in units) >= 6
    ):
        return []
    return units


def _light_on_dark_regions(image: Image.Image) -> list[dict]:
    """Find title-like light lettering that general-purpose OCR often misses.

    Paddle's detector is strongest on dark text over light balloons.  Manga
    title cards and captions frequently reverse that polarity, so this adds a
    conservative, model-independent candidate pass.  It only proposes
    horizontal or vertical runs made from several bright components surrounded
    by a mostly dark local background; MangaOCR still has to recognize Japanese
    before a candidate becomes a TextUnit.
    """
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    page_area = height * width
    bright = (gray >= 200).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    accepted = np.zeros_like(bright)
    min_component_area = max(80, round(page_area * 0.00001))
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if area < min_component_area or box_width < 7 or box_height < 7:
            continue
        if box_width > width * 0.35 or box_height > height * 0.35:
            continue
        pad = max(12, round(max(box_width, box_height) * 0.18))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + box_width + pad), min(height, y + box_height + pad)
        local = gray[y0:y1, x0:x1]
        if local.size == 0 or float((local < 105).mean()) < 0.45:
            continue
        accepted[labels == label] = 1

    if not accepted.any():
        return []
    kernels = (
        np.ones(
            (max(5, round(height * 0.006)), max(15, round(width * 0.055))), np.uint8
        ),
        np.ones(
            (max(15, round(height * 0.035)), max(5, round(width * 0.008))), np.uint8
        ),
    )
    candidates: list[dict] = []
    for kernel in kernels:
        joined = cv2.morphologyEx(accepted, cv2.MORPH_CLOSE, kernel)
        joined_count, joined_labels, joined_stats, _ = cv2.connectedComponentsWithStats(
            joined, 8
        )
        for label in range(1, joined_count):
            x, y, box_width, box_height, _ = (
                int(value) for value in joined_stats[label]
            )
            box_area = box_width * box_height
            if box_area < page_area * 0.0015 or box_area > page_area * 0.25:
                continue
            aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
            if aspect < 1.8:
                continue
            member_count = len(np.unique(labels[joined_labels == label])) - 1
            if member_count < 3:
                continue
            local = gray[y : y + box_height, x : x + box_width]
            if float((local < 105).mean()) < 0.4:
                continue
            box = [x, y, x + box_width, y + box_height]
            if any(
                _axis_overlap(box[0], box[2], item["box"][0], item["box"][2])
                * _axis_overlap(box[1], box[3], item["box"][1], item["box"][3])
                >= min(_area(box), _area(item["box"])) * 0.7
                for item in candidates
            ):
                continue
            candidates.append({"box": box, "text": "", "score": 0.5, "reverse": True})
    return candidates


def _outlined_light_text_regions_near(
    image: Image.Image,
    coarse: list[int],
    score: float,
    minimum_components: int = 2,
) -> list[dict]:
    """Recover source-derived boxes for white-filled, dark-outlined display text.

    Manga effects commonly use white glyph interiors with a black outline over
    mid-tone artwork.  Paddle can miss the whole word, while full-page vision
    supplies only an intentionally unsafe coarse hint.  This fallback searches
    around that hint for bright connected glyph interiors, verifies that each
    has both a dark outline and tonal artwork nearby, and returns padded boxes
    derived from the source pixels.  It never promotes the VLM box itself to an
    erase region; MangaOCR still has to recognize Japanese from the resulting
    group before ``recover_missing`` accepts it.
    """
    gray = np.asarray(image.convert("L"))
    page_height, page_width = gray.shape
    width = max(1, coarse[2] - coarse[0])
    height = max(1, coarse[3] - coarse[1])
    search = [
        max(0, coarse[0] - round(width * 1.5)),
        max(0, coarse[1] - round(height * 1.5)),
        min(page_width, coarse[2] + round(width * 1.5)),
        min(page_height, coarse[3] + round(height * 1.5)),
    ]
    local = gray[search[1] : search[3], search[0] : search[2]]
    if local.size == 0:
        return []
    bright = (local >= 220).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    center_x = (coarse[0] + coarse[2]) / 2
    center_y = (coarse[1] + coarse[3]) / 2
    minimum_area = max(80, round(width * height * 0.008))
    regions: list[dict] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if (
            area < minimum_area
            or box_width < 8
            or box_height < 8
            or area / max(1, box_width * box_height) < 0.15
        ):
            continue
        absolute_x = search[0] + x
        absolute_y = search[1] + y
        component_x = absolute_x + box_width / 2
        component_y = absolute_y + box_height / 2
        if (
            abs(component_x - center_x) / width > 2.2
            or abs(component_y - center_y) / height > 2.2
        ):
            continue
        ring = max(6, round(max(box_width, box_height) * 0.16))
        ring_box = [
            max(0, absolute_x - ring),
            max(0, absolute_y - ring),
            min(page_width, absolute_x + box_width + ring),
            min(page_height, absolute_y + box_height + ring),
        ]
        neighborhood = gray[ring_box[1] : ring_box[3], ring_box[0] : ring_box[2]]
        dark_ratio = float((neighborhood < 70).mean())
        midtone_ratio = float(((neighborhood >= 70) & (neighborhood < 205)).mean())
        if dark_ratio < 0.05 or midtone_ratio < 0.12:
            continue
        outline_pad = max(4, round(max(box_width, box_height) * 0.1))
        regions.append(
            {
                "box": [
                    max(0, absolute_x - outline_pad),
                    max(0, absolute_y - outline_pad),
                    min(page_width, absolute_x + box_width + outline_pad),
                    min(page_height, absolute_y + box_height + outline_pad),
                ],
                "text": "",
                "score": round(score, 4),
                "outlined_fallback": True,
            }
        )
    # A single highlight is too easy to confuse with artwork glare. Display
    # lettering produces multiple aligned interiors even when one Japanese
    # character contains several disconnected white pieces.
    if len(regions) < minimum_components:
        return []
    return regions


#: Recognition confidence a box needs before its string is taken as text on a
#: page that is read as Japanese.
EFFECT_SCORE_FLOOR = 0.72
#: The same floor on a page whose text is Latin.  Paddle's recognizer is a
#: Chinese/English one, so its Latin strings stay trustworthy well below the
#: Japanese floor: an audit of six English pages found six boxes in
#: [0.58, 0.72) that clear the small-box floor and every one of them was real
#: text ("'SUP", "puMP", "S-SLOW", "$I'm--!", "MY DICKYOU CUmE.", ".HEH"),
#: while everything lower was a one- or two-glyph mis-reading the 650 px^2 floor
#: drops anyway.  The "'SUP" bubble (34x19 px, score 0.67) was read correctly
#: and thrown away by the Japanese floor, and no later stage can recover a box
#: the detector was never allowed to keep.
LATIN_EFFECT_SCORE_FLOOR = 0.58
#: Recognition confidence below which a non-empty string is treated as noise.
#: White lettering with a dark outline comes back as garbage in this range
#: ("ASAAACAEAE" at 0.14, "Sn aJOONONHM" at 0.43) while the same lines read at
#: 0.94-0.99 from an inverted copy of the page, so a page carrying any such line
#: is read a second time.
NOISY_READ_SCORE = 0.6

# A flat 650 px^2 floor dropped the small print of a low-resolution scan and
# kept it on a large one.  The floor is a share of the page instead, so it
# describes how much of the sheet a box covers rather than how many pixels.
#
# It is clamped to the constant it replaces at the top: text height grows with
# the scan while area grows with its square, so a strict ratio would ask a
# 1600x1132 page for 1177 px^2 and start dropping boxes the old 650 kept.  This
# change is allowed to bring lines back, never to take one away.
# Measured on the sample: page 47 (800x566, 96 lines of 7-11 px type) keeps
# 40 -> 59 lines, and two 800x1132 pages keep everything they kept before.
MIN_REGION_AREA = 650
MIN_REGION_AREA_RATIO = 0.00065


def region_area_floor(width: int, height: int) -> int:
    """Smallest recognised box worth keeping on a page of this size."""
    scaled = round(width * height * MIN_REGION_AREA_RATIO)
    return max(MIN_REGION_AREA // 2, min(MIN_REGION_AREA, scaled))


class PaddleMangaOCR:
    """Paddle text localization followed by manga-specific Japanese recognition.

    ``quality`` uses Paddle's recognition boxes as a second signal.  MangaOCR
    remains the Japanese source of truth, but the recognizer filters detector
    noise and produces line boxes that group substantially better on manga.
    ``fast`` keeps the detection-only path for previews and low-end machines.
    """

    def __init__(
        self,
        paths: AppPaths,
        device: str = "auto",
        profile: str = "quality",
        source_language: str = "auto",
    ):
        configure_model_caches(paths)
        try:
            from manga_ocr import MangaOcr

            # Paddle imports its C++ extension helper even for ordinary inference.
            # ccache is only useful when compiling custom extensions, so do not
            # present its absence as a broken runtime component to desktop users.
            with _suppress_optional_ccache_warning():
                from paddleocr import PaddleOCR, TextDetection
        except ImportError as exc:
            raise ModelDependencyError(
                "OCR dependencies are missing. Run scripts/bootstrap with an ML profile."
            ) from exc

        resolved_device = self._resolve_device(device)
        if profile not in {"quality", "fast"}:
            raise ValueError(f"Unknown OCR profile: {profile}")
        if source_language not in SOURCE_LANGUAGES:
            raise ValueError(f"Unknown source language: {source_language}")
        self.profile = profile
        self.source_language = source_language
        with _suppress_optional_ccache_warning():
            # Recognition-independent detection is the source of truth.  A
            # Chinese recognizer can score valid Japanese too low, so its
            # richer output is additive and can never remove detector boxes.
            self.detector = TextDetection(
                model_name="PP-OCRv5_mobile_det",
                device=resolved_device,
                enable_mkldnn=False,
            )
            self.recognition_detector = None
            if profile == "quality":
                self.recognition_detector = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_server_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    device=resolved_device,
                    text_recognition_batch_size=32,
                    enable_mkldnn=False,
                )
        try:
            import torch

            torch_cuda = torch.cuda.is_available()
        except (ImportError, RuntimeError):
            torch_cuda = False
        self.reader = MangaOcr(force_cpu=manga_force_cpu(device, torch_cuda))

    @staticmethod
    def _paddle_gpu_available() -> bool:
        try:
            import paddle

            return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except Exception:
            return False

    @staticmethod
    def _resolve_device(device: str) -> str:
        # The settings panel offers "GPU 0" on every machine, but Paddle on
        # Windows is CPU- or CUDA-only.  Given "gpu:0" without CUDA it only
        # warns "device not available, switching to CPU" and then dies inside
        # oneDNN's PIR conversion (ConvertPirAttribute2RuntimeAttribute).  A
        # silent CPU fallback is what the user asked for by picking "auto".
        if device != "auto":
            if device.startswith("gpu") or device == "cuda":
                return device if PaddleMangaOCR._paddle_gpu_available() else "cpu"
            return device
        return "gpu:0" if PaddleMangaOCR._paddle_gpu_available() else "cpu"

    @staticmethod
    def _payload(result) -> dict:
        raw = result.json if hasattr(result, "json") else result
        if callable(raw):
            raw = raw()
        return raw.get("res", raw)

    def _read_inverted(self, image: Image.Image) -> dict:
        """Recognise the page again from an inverted greyscale copy.

        Paddle's recogniser is trained on dark glyphs, so white lettering with a
        dark outline on a dark panel came back as noise while the inverted copy
        read the same five lines at 0.94-0.99.  The floating point result is
        inlined here to keep the second pass off the page when it is not needed.
        """
        grey = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        inverted = cv2.cvtColor(255 - grey, cv2.COLOR_GRAY2RGB)
        results = list(self.recognition_detector.predict(inverted))
        return self._payload(results[0]) if results else {}

    def _independent_reading(self, crop: Image.Image) -> str:
        """Read one recovered crop without the vision model that proposed it.

        Recovery only trusts a proposal once something other than the proposer
        can read text inside the geometry the detector just produced.  That
        used to be MangaOCR's kana, which is the one question a declared English
        book cannot answer -- so an English page could never be recovered at
        all, and handwriting Paddle half-remembered stayed English.  Paddle's
        own recogniser answers it in Latin script instead.
        """
        if self._declared_language() == "en":
            if self.recognition_detector is None:
                return ""
            results = list(self.recognition_detector.predict(np.asarray(crop)))
            payload = self._payload(results[0]) if results else {}
            return " ".join(
                text
                for text in (
                    str(value).strip() for value in _payload_rows(payload, "rec_texts")
                )
                if text
            )
        return self.reader(crop).strip()

    @staticmethod
    def _regions(
        payload: dict,
        effect_score: float = EFFECT_SCORE_FLOOR,
        area_floor: int = MIN_REGION_AREA,
    ) -> list[dict]:
        # A recognition payload carries dt_polys *and* rec_texts.  Preferring the
        # detector polys dropped every recognized string, so each region reached
        # MangaOCR as an empty box and the Japanese-only reader hallucinated the
        # text.  Recognized text always wins when it is available.
        # ``rec_boxes`` can arrive as a numpy array, whose truth value is
        # ambiguous; test its length instead of the object itself.
        if len(payload.get("rec_boxes") or []):
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            boxes = payload.get("rec_boxes", [])
            selected = []
            for index, raw_box in enumerate(boxes):
                box = [int(value) for value in raw_box]
                text = str(texts[index]).strip() if index < len(texts) else ""
                score = float(scores[index]) if index < len(scores) else 0.0
                # Small print inside a balloon ("um,", "TH-") covers only ~400
                # px^2, so a flat 650 px^2 floor dropped the balloon's first and
                # last lines: the middle was translated and those two stayed
                # painted in English around the Chinese text.  A confidently
                # recognized string is real text at any size.
                if _area(box) < area_floor and not (score >= 0.8 and len(text) >= 2):
                    continue
                # One glyph spanning a whole drawing ("X" across 160x137 px of
                # artwork) is a mis-recognition, not a caption: real text runs
                # 50-200 px^2 per character.  Strip the text so the box is
                # treated as a text-less detector region instead of a bubble.
                compact = re.sub(r"\s+", "", text)
                if compact and _area(box) / len(compact) > 2_000:
                    text = ""
                has_japanese = bool(JP_RE.search(text))
                kana_count = len(re.findall(r"[\u3040-\u30ff]", text))
                large_detector_box = _area(box) >= 5_000
                # A short English line ("TODAY?", "UMM...") covers barely 1,100
                # px^2, so the old 1,200 px^2 floor dropped it: the balloon kept
                # its first and last lines, the group only erased the middle
                # ones, and the leftovers showed around the Chinese text.
                # ``effect_score`` is the caller's floor: a page read as
                # Latin passes the lower one, because there the
                # recognized string is the evidence and the confidence
                # tuned on Japanese pages is not.
                confident_effect = score >= effect_score and bool(text)
                if not (has_japanese or large_detector_box or confident_effect):
                    continue
                if (
                    has_japanese
                    and score < 0.58
                    and not (score >= 0.42 and kana_count >= 2)
                    and not large_detector_box
                ):
                    continue
                selected.append({"box": box, "text": text, "score": score})
            return selected
        if "dt_polys" in payload:
            selected = []
            scores = payload.get("dt_scores", [])
            for index, polygon in enumerate(payload.get("dt_polys", [])):
                xs = [int(point[0]) for point in polygon]
                ys = [int(point[1]) for point in polygon]
                box = [min(xs), min(ys), max(xs), max(ys)]
                if _area(box) < area_floor:
                    continue
                score = float(scores[index]) if index < len(scores) else 0.0
                selected.append(
                    {
                        "box": box,
                        "text": "",
                        "score": score,
                    }
                )
            return selected
        return []

    @staticmethod
    def is_cover_title_candidate(page: PageOCR) -> bool:
        return bool(_cover_title_units(page.units, page.width, page.height))

    def refine_cover_title(self, image_path: Path, page: PageOCR) -> PageOCR:
        if self._declared_language() == "en":
            # The merge exists to give MangaOCR one crop of a stylised Japanese
            # title.  On a declared English book MangaOCR would invent kana for
            # it, so the per-box Paddle readings must be left as they are.
            return page
        candidates = _cover_title_units(page.units, page.width, page.height)
        if not candidates:
            return page
        image = Image.open(image_path).convert("RGB")
        box = [
            min(unit.bbox[0] for unit in candidates),
            min(unit.bbox[1] for unit in candidates),
            max(unit.bbox[2] for unit in candidates),
            max(unit.bbox[3] for unit in candidates),
        ]
        pad_x = max(16, round((box[2] - box[0]) * 0.04))
        pad_y = max(16, round((box[3] - box[1]) * 0.02))
        crop_box = [
            max(0, box[0] - pad_x),
            max(0, box[1] - pad_y),
            min(page.width, box[2] + pad_x),
            min(page.height, box[3] + pad_y),
        ]
        refined = self.reader(image.crop(tuple(crop_box))).strip()
        if not JP_RE.search(refined):
            return page
        erase_boxes = [
            erase_box
            for unit in candidates
            for erase_box in (unit.erase_boxes or [unit.bbox])
        ]
        merged = TextUnit(
            id=candidates[0].id,
            bbox=box,
            crop_bbox=crop_box,
            ja=refined,
            score=max(unit.score for unit in candidates),
            is_sfx=False,
            erase_boxes=erase_boxes,
            special="cover_title",
        )
        return PageOCR(page.page, page.file, page.width, page.height, [merged])

    def _adjacent_column_reading(
        self,
        image: Image.Image,
        box: list[int],
        group: list[dict],
        known_boxes: list[list[int]],
        reading: str,
    ) -> tuple[str, list[list[int]]]:
        """Re-read a vertical bubble together with the outer columns it lost.

        The group's own boxes give the column width, so the strip beside them is
        about one column wide.  The widened crop is read once rather than column
        by column: reading the columns separately re-reads the characters the
        group already covered ("私に" twice in "わざわざ私に" + "私に化けてくる").
        The new reading is accepted only when it keeps the original one as a
        prefix and says more, so a worse re-read leaves the unit untouched.
        """
        widths = sorted(item["box"][2] - item["box"][0] for item in group)
        if not widths:
            return "", []
        column_width = widths[len(widths) // 2]
        gray = np.asarray(image.convert("L"))
        columns = _adjacent_column_boxes(gray, box, column_width, known_boxes)
        if not columns:
            return "", []
        widened = [
            min([box[0]] + [column[0] for column in columns]),
            min([box[1]] + [column[1] for column in columns]),
            max([box[2]] + [column[2] for column in columns]),
            max([box[3]] + [column[3] for column in columns]),
        ]
        pad_x = max(6, round((widened[2] - widened[0]) * 0.03))
        pad_y = max(6, round((widened[3] - widened[1]) * 0.03))
        crop = image.crop(
            (
                max(0, widened[0] - pad_x),
                max(0, widened[1] - pad_y),
                min(image.width, widened[2] + pad_x),
                min(image.height, widened[3] + pad_y),
            )
        )
        widened_text = self.reader(crop).strip()
        if not JP_RE.search(widened_text):
            return "", []
        original = re.sub(r"\s+", "", reading)
        candidate = re.sub(r"\s+", "", widened_text)
        if not candidate.startswith(original[: max(1, len(original) // 2)]):
            return "", []
        if len(candidate) <= len(original):
            return "", []
        if not self._rescue_reading_is_stable(image, widened, False, widened_text):
            return "", []
        return widened_text, columns

    def _rescue_reading_is_stable(
        self, image: Image.Image, box: list[int], reverse: bool, reading: str
    ) -> bool:
        """Whether a text-less box still reads the same way from shifted crops.

        Called only on the rescue path, where neither Paddle pass produced a
        string and MangaOCR alone proposed ``reading``.  Real lettering -- even
        display type the recognizers cannot parse -- keeps yielding the same
        characters when the crop grows, shrinks and zooms; an invention does not.
        """
        pad_x = max(10, round((box[2] - box[0]) * 0.04))
        pad_y = max(10, round((box[3] - box[1]) * 0.03))
        readings = [reading]
        for local_pad_x, local_pad_y, scale in (
            (max(2, pad_x // 3), max(2, pad_y // 3), 1.0),
            (round(pad_x * 2.5), round(pad_y * 2.5), 1.0),
            (pad_x, pad_y, 1.5),
        ):
            variant_box = (
                max(0, box[0] - local_pad_x),
                max(0, box[1] - local_pad_y),
                min(image.width, box[2] + local_pad_x),
                min(image.height, box[3] + local_pad_y),
            )
            variant = image.crop(variant_box)
            if reverse:
                variant = ImageOps.invert(variant)
            if scale != 1.0:
                variant = variant.resize(
                    (round(variant.width * scale), round(variant.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            readings.append(self.reader(variant).strip())
        return rescue_reading_is_stable(readings)

    def _declared_language(self) -> str:
        """The user-declared input language, defaulting to auto-detection.

        Instances rebuilt through ``__new__`` in tests predate the setting, so the
        attribute is looked up defensively instead of read directly.
        """
        return getattr(self, "source_language", "auto")

    def _reads_latin(self, text: str, score: float) -> bool:
        """Whether a recognized string should be taken as the Latin original.

        With the input language declared the answer is fixed: on an English book
        Paddle's Latin reading is always preferred, and on a Japanese book a
        stray Latin sign must never pull a kana-only line onto the Latin path.
        """
        declared = self._declared_language()
        if declared == "en":
            return bool(str(text).strip())
        if declared == "ja":
            return False
        return _is_latin_region(text, score)

    def analyze(self, image_path: Path, page_number: int) -> PageOCR:
        image = Image.open(image_path).convert("RGB")
        floor = region_area_floor(image.width, image.height)
        results = list(self.detector.predict(str(image_path)))
        payload = self._payload(results[0]) if results else {}
        recognition_payload: dict = {}
        regions = self._regions(payload, area_floor=floor)
        if self.recognition_detector is not None:
            recognition_results = list(
                self.recognition_detector.predict(str(image_path))
            )
            recognition_payload = (
                self._payload(recognition_results[0]) if recognition_results else {}
            )
            if any(
                float(score) < NOISY_READ_SCORE and len(str(text).strip()) >= 3
                for text, score in zip(
                    recognition_payload.get("rec_texts") or [],
                    recognition_payload.get("rec_scores") or [],
                )
            ):
                # A page carrying noise is read again from an inverted copy.
                # Pages whose lines all scored well never pay for that pass.
                recognition_payload = better_recognition(
                    recognition_payload, self._read_inverted(image)
                )
            regions = merge_region_candidates(
                regions, self._regions(recognition_payload, area_floor=floor)
            )
        for candidate in _light_on_dark_regions(image):
            candidate_box = candidate["box"]
            overlaps_detector = any(
                _axis_overlap(
                    candidate_box[0], candidate_box[2], item["box"][0], item["box"][2]
                )
                * _axis_overlap(
                    candidate_box[1], candidate_box[3], item["box"][1], item["box"][3]
                )
                >= min(_area(candidate_box), _area(item["box"])) * 0.65
                for item in regions
            )
            if not overlaps_detector:
                regions.append(candidate)
        # On an English page the detector's text-less boxes are artwork (a mane, a
        # garment, a panel border).  They not only inflated the unit to the
        # drawing's size -- which made the typesetter pick a huge font and the
        # cleaner erase the art -- they also bridged two separate balloons into
        # one unit, so a caption and a speech bubble were translated as a single
        # line.  Paddle's recognizer reads Latin text reliably, so a Latin page
        # keeps only the boxes that actually carry a string.
        if page_reads_as_latin(regions, self._declared_language()):
            # A Latin page is read by Paddle's own recognizer, so a low
            # score is no longer evidence of artwork: merge the boxes
            # below the Japanese floor back in before the text-less ones
            # are dropped.
            regions = merge_region_candidates(
                regions,
                self._regions(
                    recognition_payload,
                    LATIN_EFFECT_SCORE_FLOOR,
                    area_floor=floor,
                ),
            )
            regions = [item for item in regions if str(item.get("text", "")).strip()]
        groups = _groups(regions)
        groups.sort(
            key=lambda group: (_union_box(group)[1] // 180, -_union_box(group)[0])
        )
        units: list[TextUnit] = []
        for index, group in enumerate(groups, start=1):
            # MangaOCR only knows Japanese.  Given an English bubble it does not
            # fail: it invents plausible kana ("いや、そういうことですが、") that
            # pass JP_RE and silently replace the real line, so every English
            # page came out as fluent nonsense.  Paddle's recognizer reads Latin
            # text reliably, so a confidently Latin-only region keeps Paddle's
            # own text; Japanese regions keep MangaOCR, which is far better at
            # kana than the Chinese/English recognizer.
            paddle_text, paddle_score = _best_recognized_text(group)
            if self._reads_latin(paddle_text, paddle_score):
                # On an English page the detector's text-less boxes are often
                # artwork (a mane, a garment) that merely sat next to a caption.
                # Keeping them inflated the unit to the artwork's size, so the
                # typesetter chose a huge font and the translation spilled out
                # of the balloon, and the quality cleanup erased the drawing.
                text_boxes = [
                    item["box"] for item in group if str(item.get("text", "")).strip()
                ]
                group = [
                    item
                    for item in group
                    if str(item.get("text", "")).strip()
                    or any(
                        _axis_overlap(item["box"][0], item["box"][2], tbox[0], tbox[2])
                        * _axis_overlap(item["box"][1], item["box"][3], tbox[1], tbox[3])
                        >= min(_area(item["box"]), _area(tbox)) * 0.65
                        for tbox in text_boxes
                    )
                ]
            box = _union_box(group)
            reverse = any(item.get("reverse", False) for item in group)
            pad_x = max(10, round((box[2] - box[0]) * 0.04))
            pad_y = max(10, round((box[3] - box[1]) * 0.03))
            crop_box = [
                max(0, box[0] - pad_x),
                max(0, box[1] - pad_y),
                min(image.width, box[2] + pad_x),
                min(image.height, box[3] + pad_y),
            ]
            crop = image.crop(tuple(crop_box))
            if reverse:
                crop = ImageOps.invert(crop)
            # A detector fragment that Paddle could not read at all is noise
            # (paper grain, a hair strand, a panel border).  MangaOCR still
            # invents kana for it, and those kana pass JP_RE, so a 52x16 speck
            # became a translated line.  Real text, including small SFX, always
            # carries a recognized string from one of the two passes.
            if not paddle_text and _area(box) < 4_000:
                continue
            if self._declared_language() == "en":
                # A declared English book never needs the Japanese recognizer:
                # given Latin bubbles it invents fluent kana, and given artwork it
                # invents dates.  Paddle's own reading is the original text, so a
                # box it could not read is a box with no text at all.
                if not paddle_text.strip():
                    continue
                refined = paddle_text
            else:
                manga_text = self.reader(crop).strip()
                if self._reads_latin(paddle_text, paddle_score):
                    refined = paddle_text
                elif JP_RE.search(manga_text):
                    # Large display lettering can defeat both Paddle passes, so the
                    # area gate above deliberately lets MangaOCR rescue a text-less
                    # box -- but MangaOCR never says "nothing", and its inventions
                    # also pass JP_RE.  Ask for the same reading from shifted crops.
                    if not paddle_text and not self._rescue_reading_is_stable(
                        image, box, reverse, manga_text
                    ):
                        continue
                    refined = manga_text
                elif self._reads_latin(manga_text, paddle_score):
                    refined = manga_text
                else:
                    continue
            score = max((item["score"] for item in group), default=0.0)
            clean = refined.strip()
            # A vertical Japanese bubble is one column per detector box.  Paddle
            # scores its thin outer column below the detection floor once the
            # page is downscaled, so that column stayed in Japanese beside the
            # translated text.  The group is re-read with that column folded in;
            # the new reading is only taken when it extends the old one.
            erase_boxes = [item["box"] for item in group]
            if (
                JP_RE.search(clean)
                and not self._reads_latin(paddle_text, paddle_score)
                and not reverse
            ):
                widened_text, extra_boxes = self._adjacent_column_reading(
                    image, box, group, [item["box"] for item in regions], clean
                )
                if widened_text:
                    clean = widened_text
                    erase_boxes = [item["box"] for item in group] + list(extra_boxes)
                    box = [
                        min(item[0] for item in erase_boxes),
                        min(item[1] for item in erase_boxes),
                        max(item[2] for item in erase_boxes),
                        max(item[3] for item in erase_boxes),
                    ]
                    crop_box = [
                        max(0, box[0] - pad_x),
                        max(0, box[1] - pad_y),
                        min(image.width, box[2] + pad_x),
                        min(image.height, box[3] + pad_y),
                    ]
            units.append(
                TextUnit(
                    id=f"p{page_number:03d}u{index:02d}",
                    bbox=box,
                    crop_bbox=crop_box,
                    ja=clean,
                    score=round(score, 4),
                    is_sfx=likely_sfx_text(clean),
                    erase_boxes=erase_boxes,
                )
            )
        page = PageOCR(page_number, image_path.name, image.width, image.height, units)
        return self.refine_cover_title(image_path, page)

    def recover_missing(
        self, image_path: Path, page: PageOCR, missing: list[dict]
    ) -> PageOCR:
        """Recover VLM-reported omissions through enlarged detector-owned crops.

        Full-page vision is useful for recall but its coordinates are too coarse
        for destructive editing.  Each proposed region is therefore enlarged,
        run through the exact Paddle detector again, and accepted only when an
        independent read of that geometry finds text in the declared source
        language.  A declared English book used to be refused outright, because
        the confirming reader was MangaOCR and it can only answer in kana; the
        Paddle recogniser that already reads the rest of the page confirms those
        crops instead, so handwriting Paddle only half catches still gets a
        second chance without the vision model ever owning a box.
        """
        if not missing:
            return page
        image = Image.open(image_path).convert("RGB")
        recovered_regions: list[dict] = []
        existing_boxes = [unit.bbox for unit in page.units] + [
            erase_box for unit in page.units for erase_box in unit.erase_boxes
        ]
        # Longer readings subsume fragmentary reports from repeated VLM passes
        # (for example ``ボォ`` before ``ボ``).
        ordered_missing = sorted(
            missing[:40],
            key=lambda item: len(
                re.sub(
                    r"[\s．。…、，,!?！？:：♡♥〰〜～]",
                    "",
                    str(item.get("text", "")),
                )
            ),
            reverse=True,
        )
        for item in ordered_missing:
            try:
                normalized = [int(value) for value in item["bbox"]]
            except (KeyError, TypeError, ValueError):
                continue
            if len(normalized) != 4:
                continue
            coarse = [
                round(normalized[0] * image.width / 1000),
                round(normalized[1] * image.height / 1000),
                round(normalized[2] * image.width / 1000),
                round(normalized[3] * image.height / 1000),
            ]
            if coarse[2] <= coarse[0] or coarse[3] <= coarse[1]:
                continue
            pad_x = max(12, round((coarse[2] - coarse[0]) * 0.12))
            pad_y = max(12, round((coarse[3] - coarse[1]) * 0.12))
            crop_box = [
                max(0, coarse[0] - pad_x),
                max(0, coarse[1] - pad_y),
                min(image.width, coarse[2] + pad_x),
                min(image.height, coarse[3] + pad_y),
            ]
            crop = image.crop(tuple(crop_box))
            scale = max(2.0, min(4.0, 1200 / max(crop.width, crop.height)))
            enlarged = crop.resize(
                (round(crop.width * scale), round(crop.height * scale)),
                Image.Resampling.LANCZOS,
            )
            outlined_regions = (
                _outlined_light_text_regions_near(
                    image,
                    coarse,
                    float(item.get("score", 0.0)),
                    minimum_components=1,
                )
                if bool(item.get("is_sfx", False))
                else []
            )
            semantic_text = str(item.get("text", "")).strip()

            # A full-page VLM can read an outlined sound more accurately than
            # MangaOCR while still proposing only coarse geometry.  When that
            # proposal overlaps detector-owned erase components of an existing
            # SFX unit, enrich the reading but never enlarge the erase mask from
            # the VLM box.  This also repairs a prior artwork-role hallucination.
            if bool(item.get("is_sfx", False)) and semantic_sfx_classification(
                semantic_text, float(item.get("score", 0.0)), True
            ):
                coarse_area = max(1, _area(coarse))
                exact_sfx_matches: list[tuple[float, TextUnit]] = []
                for unit in page.units:
                    if not (unit.is_sfx or likely_sfx_text(unit.ja)):
                        continue
                    overlap_area = sum(
                        _axis_overlap(coarse[0], coarse[2], box[0], box[2])
                        * _axis_overlap(coarse[1], coarse[3], box[1], box[3])
                        for box in (unit.erase_boxes or [unit.bbox])
                    )
                    coverage = min(coarse_area, overlap_area) / coarse_area
                    group_overlap = (
                        _axis_overlap(coarse[0], coarse[2], unit.bbox[0], unit.bbox[2])
                        * _axis_overlap(
                            coarse[1], coarse[3], unit.bbox[1], unit.bbox[3]
                        )
                        / coarse_area
                    )
                    short_group_member = bool(
                        len(re.sub(r"\W+", "", semantic_text)) <= 3
                        and len(unit.erase_boxes) >= 2
                        and group_overlap >= 0.8
                    )
                    if coverage >= 0.25 or short_group_member:
                        exact_sfx_matches.append((max(coverage, group_overlap), unit))
                if exact_sfx_matches:
                    repaired = max(exact_sfx_matches, key=lambda pair: pair[0])[1]
                    current_clean = re.sub(
                        r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", repaired.ja
                    )
                    candidate_clean = re.sub(
                        r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", semantic_text
                    )
                    if candidate_clean and candidate_clean not in current_clean:
                        repaired.ja = f"{repaired.ja}\n{semantic_text}".strip()
                    repaired.score = max(
                        repaired.score, round(float(item.get("score", 0.0)), 4)
                    )
                    repaired.is_sfx = True
                    repaired.skip = False
                    repaired.skip_reason = ""
                    if repaired.special in {"artwork_text", "ocr_noise"}:
                        repaired.special = ""
                    continue
            matching_units = [
                unit
                for unit in page.units
                if sum(
                    1
                    for region in outlined_regions
                    if _axis_overlap(
                        region["box"][0],
                        region["box"][2],
                        unit.bbox[0],
                        unit.bbox[2],
                    )
                    * _axis_overlap(
                        region["box"][1],
                        region["box"][3],
                        unit.bbox[1],
                        unit.bbox[3],
                    )
                    >= _area(region["box"]) * 0.35
                )
                >= (1 if unit.score <= 0.1 else 2)
            ]
            if (
                JP_RE.search(semantic_text)
                and matching_units
                and any(unit.is_sfx or unit.score <= 0.6 for unit in matching_units)
            ):
                repaired = min(
                    matching_units,
                    key=lambda unit: abs(
                        (unit.bbox[0] + unit.bbox[2]) / 2 - (coarse[0] + coarse[2]) / 2
                    )
                    + abs(
                        (unit.bbox[1] + unit.bbox[3]) / 2 - (coarse[1] + coarse[3]) / 2
                    ),
                )
                repaired.ja = semantic_text
                repaired.score = max(
                    repaired.score, round(float(item.get("score", 0.0)), 4)
                )
                repaired.is_sfx = True
                repaired.skip = False
                repaired.skip_reason = ""
                relevant_outlined = [
                    region["box"]
                    for region in outlined_regions
                    if _axis_overlap(
                        region["box"][0],
                        region["box"][2],
                        repaired.bbox[0],
                        repaired.bbox[2],
                    )
                    * _axis_overlap(
                        region["box"][1],
                        region["box"][3],
                        repaired.bbox[1],
                        repaired.bbox[3],
                    )
                    >= min(_area(region["box"]), _area(repaired.bbox)) * 0.2
                    and _area(region["box"]) <= _area(repaired.bbox) * 2.0
                    and region["box"][2] - region["box"][0]
                    <= (repaired.bbox[2] - repaired.bbox[0]) * 2.0
                    and region["box"][3] - region["box"][1]
                    <= (repaired.bbox[3] - repaired.bbox[1]) * 2.0
                ]
                repaired.erase_boxes = [
                    *repaired.erase_boxes,
                    *[
                        box
                        for box in relevant_outlined
                        if box not in repaired.erase_boxes
                    ],
                ]
                repaired.bbox = [
                    min(box[0] for box in repaired.erase_boxes),
                    min(box[1] for box in repaired.erase_boxes),
                    max(box[2] for box in repaired.erase_boxes),
                    max(box[3] for box in repaired.erase_boxes),
                ]
                repair_pad = max(
                    10,
                    round(
                        max(
                            repaired.bbox[2] - repaired.bbox[0],
                            repaired.bbox[3] - repaired.bbox[1],
                        )
                        * 0.06
                    ),
                )
                repaired.crop_bbox = [
                    max(0, repaired.bbox[0] - repair_pad),
                    max(0, repaired.bbox[1] - repair_pad),
                    min(image.width, repaired.bbox[2] + repair_pad),
                    min(image.height, repaired.bbox[3] + repair_pad),
                ]
                if repaired.special in {"artwork_text", "ocr_noise"}:
                    repaired.special = ""
                continue
            item_regions: list[dict] = []
            results = list(self.detector.predict(np.asarray(enlarged)))
            payload = self._payload(results[0]) if results else {}
            for region in self._regions(payload):
                local = region["box"]
                mapped = [
                    crop_box[0] + round(local[0] / scale),
                    crop_box[1] + round(local[1] / scale),
                    crop_box[0] + round(local[2] / scale),
                    crop_box[1] + round(local[3] / scale),
                ]
                if _area(mapped) < 180:
                    continue
                covered = any(
                    _axis_overlap(mapped[0], mapped[2], box[0], box[2])
                    * _axis_overlap(mapped[1], mapped[3], box[1], box[3])
                    >= min(_area(mapped), _area(box)) * 0.65
                    for box in existing_boxes
                )
                if covered:
                    continue
                item_regions.append(
                    {
                        "box": mapped,
                        "text": "",
                        "score": region["score"],
                        "semantic_text": semantic_text,
                        "semantic_is_sfx": bool(item.get("is_sfx", False)),
                    }
                )
            if not item_regions and bool(item.get("is_sfx", False)):
                item_regions = outlined_regions
                for region in item_regions:
                    region["semantic_text"] = str(item.get("text", "")).strip()
                    region["semantic_is_sfx"] = True
                item_regions = [
                    region
                    for region in item_regions
                    if not any(
                        _axis_overlap(
                            region["box"][0], region["box"][2], box[0], box[2]
                        )
                        * _axis_overlap(
                            region["box"][1], region["box"][3], box[1], box[3]
                        )
                        >= min(_area(region["box"]), _area(box)) * 0.65
                        for box in existing_boxes
                    )
                ]
            recovered_regions.extend(item_regions)

        recovered_regions = merge_region_candidates([], recovered_regions)
        accepted_groups = deduplicate_nested_region_groups(_groups(recovered_regions))
        for group, box in accepted_groups:
            pad_x = max(10, round((box[2] - box[0]) * 0.06))
            pad_y = max(10, round((box[3] - box[1]) * 0.05))
            crop_box = [
                max(0, box[0] - pad_x),
                max(0, box[1] - pad_y),
                min(image.width, box[2] + pad_x),
                min(image.height, box[3] + pad_y),
            ]
            refined = self._independent_reading(image.crop(tuple(crop_box)))
            if not is_source_text(refined, self._declared_language()):
                continue
            semantic_candidates = {
                str(item.get("semantic_text", "")).strip()
                for item in group
                if is_source_text(
                    str(item.get("semantic_text", "")), self._declared_language()
                )
            }
            expected_sfx = any(
                bool(item.get("semantic_is_sfx", False)) for item in group
            )
            refined_is_sfx = semantic_sfx_classification(refined, 0.8, expected_sfx)
            if expected_sfx and not refined_is_sfx:
                continue
            # For source-derived outlined-glyph geometry, MangaOCR is the
            # independent confirmation that the pixels are Japanese text; the
            # full-page visual read can then supply the exact SFX spelling.
            # This avoids keeping a MangaOCR near-miss such as レキット when
            # the visual audit consistently saw ムチッ, without ever trusting
            # the VLM's coarse coordinates for destructive editing.
            if len(semantic_candidates) == 1 and expected_sfx and refined_is_sfx:
                refined = semantic_candidates.pop()
            page.units.append(
                TextUnit(
                    id="",
                    bbox=box,
                    crop_bbox=crop_box,
                    ja=refined,
                    score=round(max(item["score"] for item in group), 4),
                    is_sfx=likely_sfx_text(refined),
                    erase_boxes=[item["box"] for item in group],
                )
            )
        page.units.sort(key=lambda unit: (unit.bbox[1] // 180, -unit.bbox[0]))
        for index, unit in enumerate(page.units, start=1):
            unit.id = f"p{page.page:03d}u{index:02d}"
        return page


class OllamaVisionOCR:
    """Optional local vision-OCR adapter using Ollama's native chat endpoint.

    The specialized Paddle/Manga OCR path remains the accuracy-first default.
    This adapter exists so users with an Ollama vision model can use the same
    local service boundary for OCR and translation without any cloud API.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 300,
        source_language: str = "auto",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout
        if source_language not in SOURCE_LANGUAGES:
            raise ValueError(f"Unknown source language: {source_language}")
        self.source_language = source_language
        if not self.model:
            raise ValueError("An Ollama vision model is required for Ollama OCR")

    def _image_bytes(self, image_path: Path) -> bytes:
        """The page as the vision service wants it.

        Kept as its own method because the transport decides the format: Ollama
        accepts the WebP files this app stores, llama.cpp answers 400 for them,
        so the llama.cpp subclass converts instead of re-encoding for both.
        """
        return image_path.read_bytes()

    def _chat(self, images: list[bytes], prompt: str, schema: dict) -> dict:
        payload_base = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        base64.b64encode(content).decode("ascii")
                        for content in images
                    ],
                }
            ],
        }
        retry_delays = (0.5, 2.0)
        # The first attempt stays on the historical budget so ordinary pages are
        # not slowed down; a truncated reply escalates towards the shared output
        # ceiling instead of failing after three tries.  A dense text page can
        # legitimately need more than 16K of JSON.
        prediction_budgets = (
            8192,
            16_384,
            32_768,
            OLLAMA_MAX_OUTPUT_TOKENS,
        )
        for attempt in range(len(prediction_budgets)):
            options = {
                "temperature": 0,
                "num_ctx": OLLAMA_CONTEXT_TOKENS,
                "num_predict": prediction_budgets[attempt],
            }
            if attempt:
                # Dense pages can make the model degenerate into emitting the
                # same glyph with ever-growing coordinates ("オ" at y=3173,
                # y=7413, y=15493…) until the budget runs out, so a bigger
                # budget alone never recovers.  Penalising repeats on the
                # retry breaks the loop; the first attempt stays untouched so
                # ordinary pages keep the upstream sampling.
                options["repeat_penalty"] = 1.3
                options["repeat_last_n"] = 256
            body = json.dumps(
                {
                    **payload_base,
                    # OCR pages need bounded generation, not the model's full
                    # 262K KV cache. Escalating the output budget only after a
                    # truncated response keeps ordinary pages responsive while
                    # allowing a transiently verbose structured generation to
                    # recover without accepting partial JSON.
                    "options": options,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            request = Request(
                f"{self.base_url}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                # The budget escalates to the shared 64K ceiling, so the wait has
                # to escalate with it: a fixed 300 s would abort exactly the
                # dense page whose reply needed the larger budget.
                attempt_timeout = max(
                    self.timeout, ollama_timeout(prediction_budgets[attempt])
                )
                with urlopen(request, timeout=attempt_timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("done_reason") == "length":
                    # Truncation is answered by the next (larger) budget, not by
                    # the transport-retry count, so walk the whole plan.
                    if attempt < len(prediction_budgets) - 1:
                        time.sleep(retry_delays[min(attempt, len(retry_delays) - 1)])
                        continue
                    raise RuntimeError(
                        "Ollama OCR output was truncated before valid JSON completed "
                        f"after budgets {', '.join(map(str, prediction_budgets))}"
                    )
                break
            except HTTPError as exc:
                detail = exc.read(2000).decode("utf-8", errors="replace").strip()
                if exc.code >= 500 and attempt < len(retry_delays):
                    time.sleep(retry_delays[attempt])
                    continue
                message = f"HTTP {exc.code}: {detail or exc.reason}"
                raise RuntimeError(f"Ollama OCR request failed: {message}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < len(retry_delays):
                    time.sleep(retry_delays[attempt])
                    continue
                raise RuntimeError(f"Ollama OCR request failed: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"Ollama OCR request failed: {exc}") from exc
        content = str(payload.get("message", {}).get("content", ""))
        return self._json_payload(content)

    @staticmethod
    def _json_payload(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Ollama OCR did not return JSON")
            return json.loads(content[start : end + 1])

    def analyze(self, image_path: Path, page_number: int) -> PageOCR:
        with Image.open(image_path) as image:
            width, height = image.size
        prompt = (
            f"Detect every {language_label(self.source_language) if self.source_language != 'auto' else 'Japanese'} "
            "text region in this manga page, including dialogue, "
            "narration and sound effects. Furigana annotating kanji is reading evidence, "
            "not a second copy of the text: return the main kanji expression once. "
            # Not 0.0: llama.cpp compiles the schema below into a decoding grammar
            # and copies a boundary value out of the example verbatim, so every box
            # came back at 0.0 and the score gates downstream silently closed
            # (``prefer_semantic_ocr`` needs 0.72, a hybrid page then loses whole
            # speech bubbles).  The daemon ignores the example, so this keeps both
            # transports comparable instead of flattering one of them.
            "Return JSON only as "
            '{"regions":[{"bbox":[x0,y0,x1,y1],"text":"...",'
            '"score":0.9,"is_sfx":false}]}. Coordinates must be integer source-image '
            f"pixels within width={width}, height={height}. Do not translate or omit text."
        )
        schema = {
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "text": {"type": "string"},
                            "score": SCORE_SCHEMA,
                            "is_sfx": {"type": "boolean"},
                        },
                        "required": ["bbox", "text", "score", "is_sfx"],
                    },
                }
            },
            "required": ["regions"],
        }
        regions = self._chat([self._image_bytes(image_path)], prompt, schema).get(
            "regions", []
        )
        units: list[TextUnit] = []
        for region in regions[:300]:
            try:
                raw = [float(value) for value in region["bbox"]]
                if len(raw) != 4:
                    continue
                if all(0 <= value <= 1 for value in raw):
                    raw = [
                        raw[0] * width,
                        raw[1] * height,
                        raw[2] * width,
                        raw[3] * height,
                    ]
                box = [
                    max(0, min(width, round(raw[0]))),
                    max(0, min(height, round(raw[1]))),
                    max(0, min(width, round(raw[2]))),
                    max(0, min(height, round(raw[3]))),
                ]
                text = str(region.get("text", "")).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if (
                box[2] <= box[0]
                or box[3] <= box[1]
                or not is_source_text(text, self.source_language)
            ):
                continue
            pad_x = max(8, round((box[2] - box[0]) * 0.04))
            pad_y = max(8, round((box[3] - box[1]) * 0.03))
            units.append(
                TextUnit(
                    id="",
                    bbox=box,
                    crop_bbox=[
                        max(0, box[0] - pad_x),
                        max(0, box[1] - pad_y),
                        min(width, box[2] + pad_x),
                        min(height, box[3] + pad_y),
                    ],
                    ja=text,
                    score=float(region.get("score", 0.75)),
                    is_sfx=bool(region.get("is_sfx", False))
                    or (bool(KATAKANA_RE.fullmatch(text)) and len(text) <= 14),
                    erase_boxes=[box],
                )
            )
        units.sort(key=lambda unit: (unit.bbox[1] // 180, -unit.bbox[0]))
        for index, unit in enumerate(units, start=1):
            unit.id = f"p{page_number:03d}u{index:02d}"
        return PageOCR(page_number, image_path.name, width, height, units)

    @staticmethod
    def _annotated_image(image: Image.Image, page: PageOCR) -> bytes:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        line_width = max(4, round(min(image.size) * 0.003))
        label_height = max(28, round(min(image.size) * 0.024))
        for index, unit in enumerate(page.units, start=1):
            x0, y0, x1, y1 = unit.bbox
            draw.rectangle((x0, y0, x1, y1), outline="#ff2d2d", width=line_width)
            label = str(index)
            label_box = draw.textbbox((0, 0), label)
            label_width = max(label_height, label_box[2] - label_box[0] + 12)
            label_y = max(0, y0 - label_height)
            draw.rectangle(
                (x0, label_y, min(image.width, x0 + label_width), y0),
                fill="#ff2d2d",
            )
            draw.text((x0 + 6, label_y + 3), label, fill="white")
        output = BytesIO()
        annotated.save(output, "JPEG", quality=92, subsampling=0)
        return output.getvalue()

    def refine(self, image_path: Path, page: PageOCR) -> tuple[PageOCR, list[dict]]:
        """Correct detector-bound OCR using full-page vision without changing geometry.

        The original page and a numbered overlay are sent together. The VLM may
        correct text and classify SFX, but exact bounding/erase boxes remain the
        detector's exclusive responsibility. Unboxed findings are returned for
        audit and can never enter destructive rendering through this method.
        """
        if not page.units:
            return page, []
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        ids = [unit.id for unit in page.units]
        candidates = "\n".join(
            f"{index}. {unit.id}: {unit.ja}"
            for index, unit in enumerate(page.units, start=1)
        )
        prompt = f"""你会收到同一张{language_label_zh(self.source_language)}漫画的两幅图：第一幅是无标记原图，第二幅有红色编号框。
逐框结合整页上下文校正 OCR。红框和数字是程序标记，不属于漫画文字。
必须为每个 id 返回一项；不要翻译。text 只写框内{language_label_zh(self.source_language)}，保留标点和语气符号。
汉字旁或上方的振假名只用于确认读音，不得把“振假名＋汉字正文”重复写入 text；保留汉字正文一次。只有不属于汉字注音的独立假名才抄写。
role 必须区分 dialogue（对白）、narration（旁白）、sfx（拟声/效果字）、artwork（衣服印花、商标、招牌或画面道具文字）和 noise（误检）。
衣服、包装和背景物件上的装饰字属于 artwork，不能当对白；一个框同时含对白和 artwork 时只抄对白并选 dialogue。
仅 role=sfx 时填写最接近的 effect_kind，否则填 none。
如果页面上还有未被任何红框覆盖的可见{language_label_zh(self.source_language)}，写入 missing，bbox 使用 0 到 1000 的归一化整数；没有则返回空数组。

当前机器 OCR 候选：
{candidates}"""
        item_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "enum": ids},
                "text": {"type": "string"},
                "score": SCORE_SCHEMA,
                "role": {
                    "type": "string",
                    "enum": ["dialogue", "narration", "sfx", "artwork", "noise"],
                },
                "effect_kind": {"type": "string", "enum": list(SFX_KINDS)},
            },
            "required": ["id", "text", "score", "role", "effect_kind"],
        }
        missing_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "score": SCORE_SCHEMA,
                "is_sfx": {"type": "boolean"},
            },
            "required": ["text", "bbox", "score", "is_sfx"],
        }
        schema = {
            "type": "object",
            "properties": {
                "regions": {"type": "array", "items": item_schema},
                "missing": {"type": "array", "items": missing_schema},
            },
            "required": ["regions", "missing"],
        }
        payload = self._chat(
            [self._image_bytes(image_path), self._annotated_image(image, page)],
            prompt,
            schema,
        )
        by_id = {
            str(item.get("id")): item
            for item in payload.get("regions", [])
            if isinstance(item, dict) and item.get("id") in ids
        }
        for unit in page.units:
            item = by_id.get(unit.id)
            vision_is_sfx = unit.is_sfx
            if item:
                candidate = str(item.get("text", "")).strip()
                score = float(item.get("score", 0.0))
                role = str(item.get("role", "")).lower()
                if not role and "is_sfx" in item:
                    role = "sfx" if bool(item.get("is_sfx")) else "dialogue"
                # The exact detector has already established that this sparse,
                # page-spanning group is the cover/title display text.  A VLM
                # often calls stylized titles "artwork" when viewed in isolation;
                # that semantic label must not turn translatable title text into
                # a protected logo or sound effect.
                if unit.special == "cover_title":
                    role = "narration"
                agrees = prefer_semantic_ocr(unit.ja, candidate, score)
                role_agrees = agrees or bool(
                    role == "artwork" and semantic_text_agreement(unit.ja, candidate)
                )
                source_looks_like_sfx = unit.is_sfx or likely_sfx_text(unit.ja)
                source_looks_like_dialogue = likely_dialogue_text(unit.ja)
                if (
                    unit.special != "cover_title"
                    and role in {"dialogue", "narration"}
                    and (agrees or semantic_text_agreement(unit.ja, candidate))
                ):
                    # Read the page, not the crop.  ``agrees`` demands Japanese,
                    # so a Latin aside never satisfies it here; the text match is
                    # what confirms the page-level pass actually found words and
                    # not texture.  Either way the page-level dialogue verdict is
                    # recorded so the tight-crop audit cannot demote the unit to
                    # ``artwork_text`` later -- that demotion is how a two-word
                    # Spanish aside ("SUPONGO...") stayed untranslated while the
                    # caption beside it was translated off the same page.
                    unit.special = "page_dialogue"
                if (
                    role in {"artwork", "noise"}
                    and score >= 0.92
                    and role_agrees
                    and not source_looks_like_sfx
                    and not source_looks_like_dialogue
                ):
                    unit.zh = ""
                    unit.skip = True
                    unit.skip_reason = "preserve" if role == "artwork" else "noise"
                    unit.special = "artwork_text" if role == "artwork" else "ocr_noise"
                    continue
                if not agrees:
                    if role in {"sfx", "artwork", "noise"} and score >= 0.85:
                        # Preserve the detector text and ask the tight crop to
                        # adjudicate the role.  Previously this disagreement
                        # was silently ignored and obvious effects could enter
                        # the dialogue translator as invented speech.
                        unit.special = "semantic_role_conflict"
                    continue
                unit.ja = candidate
                unit.score = max(unit.score, round(score, 4))
                vision_is_sfx = role == "sfx"
                effect_kind = str(item.get("effect_kind", "none")).lower()
                if vision_is_sfx and effect_kind in SFX_KINDS and effect_kind != "none":
                    unit.special = f"sfx:{effect_kind}"
            unit.is_sfx = semantic_sfx_classification(
                unit.ja,
                unit.score,
                vision_is_sfx,
            )
        candidates = [
            item
            for item in payload.get("missing", [])
            if isinstance(item, dict)
            and is_source_text(str(item.get("text", "")), self.source_language)
        ]
        # Same reason as ``find_missing``: an inarticulate decoder must not be
        # able to veto the recall pass with a number it never varied.
        if scores_are_informative(
            [float(item.get("score", 0.0)) for item in candidates]
        ):
            missing = [
                item for item in candidates if float(item.get("score", 0.0)) >= 0.75
            ]
        else:
            missing = candidates
        return page, missing

    def find_missing(self, image_path: Path, page: PageOCR) -> list[dict]:
        """Run a recall-only visual pass, isolated from per-id OCR correction."""
        if not page.units:
            return []
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        missing_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "score": SCORE_SCHEMA,
                "is_sfx": {"type": "boolean"},
            },
            "required": ["text", "bbox", "score", "is_sfx"],
        }
        schema = {
            "type": "object",
            "properties": {
                "missing": {"type": "array", "items": missing_schema},
            },
            "required": ["missing"],
        }
        prompt = f"""第一张是{language_label_zh(self.source_language)}漫画原图，第二张有红色编号框。此任务只做漏字召回，不校正红框 id。
逐处检查原图里所有没有被红框完整覆盖、或只被红框覆盖一部分的可见文字，特别注意白色填充黑色描边、斜排、竖排、拟声词和标点效果字。
missing.text 必须逐字抄写完整{language_label_zh(self.source_language)}，不翻译；bbox 使用 0 到 1000 的整页归一化整数并包住完整文字。已经被红框完整覆盖的文字不要重复。没有遗漏才返回空数组。"""
        payload = self._chat(
            [self._image_bytes(image_path), self._annotated_image(image, page)],
            prompt,
            schema,
        )
        candidates = [
            item
            for item in payload.get("missing", [])
            if isinstance(item, dict)
            and is_source_text(str(item.get("text", "")), self.source_language)
        ]
        # The confidence gate only means something when the decoder's numbers
        # actually vary (see ``scores_are_informative``).  llama.cpp answered a
        # flat -0.0127 for both of its proposals here, so keeping the gate would
        # discard the whole recall pass on that transport.
        if scores_are_informative([float(item.get("score", 0.0)) for item in candidates]):
            return [item for item in candidates if float(item.get("score", 0.0)) >= 0.72]
        return candidates

    @staticmethod
    def _local_crop_is_suspicious(unit: TextUnit) -> bool:
        """Flag OCR text whose length is implausible for its exact geometry."""
        if unit.special == "semantic_role_conflict":
            return True
        text = re.sub(r"\s+", "", unit.ja)
        if not text:
            return True
        area_per_character = _area(unit.bbox) / max(1, len(text))
        return (
            unit.score <= 0.05
            or area_per_character < 350
            or area_per_character > 20_000
        )

    @staticmethod
    def _tight_crop_bytes(image: Image.Image, unit: TextUnit) -> bytes:
        x0, y0, x1, y1 = unit.bbox
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        pad_x = max(12, round(width * 0.08))
        pad_y = max(12, round(height * 0.08))
        crop = image.crop(
            (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(image.width, x1 + pad_x),
                min(image.height, y1 + pad_y),
            )
        )
        scale = min(4.0, max(1.0, 800 / max(crop.width, crop.height)))
        if scale > 1:
            crop = crop.resize(
                (round(crop.width * scale), round(crop.height * scale)),
                Image.Resampling.LANCZOS,
            )
        output = BytesIO()
        crop.save(output, "PNG")
        return output.getvalue()

    @staticmethod
    def _normalize_local_crop_text(value: str) -> str:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        kept = []
        for index, line in enumerate(lines):
            if (
                len(line) <= 2
                and index + 1 < len(lines)
                and lines[index + 1].startswith(line)
            ):
                continue
            kept.append(line)
        return "\n".join(kept)

    def refine_local_crops(
        self, image_path: Path, page: PageOCR, batch_size: int = 1
    ) -> PageOCR:
        """Audit OCR text and semantic role from tight, ordered source crops.

        Full-page VLM passes are useful for story context, but small numbered
        regions can be assigned to the wrong id.  This second pass removes that
        ambiguity: each image contains exactly one detector-owned region.  The
        specialized OCR remains authoritative unless geometry is suspicious or
        the crop auditor strongly disagrees about dialogue versus sound effect.
        """
        if not page.units:
            return page
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        # Normalize legacy/reviewed detector groups before producing audit
        # crops. Otherwise one broad horizontal shirt/logo box can dominate a
        # vertical dialogue crop and make the VLM protect the entire unit.
        for unit in page.units:
            if not unit.erase_boxes:
                continue
            cleaned = prune_detached_orthogonal_outliers(unit.erase_boxes)
            if cleaned == unit.erase_boxes:
                continue
            unit.erase_boxes = cleaned
            unit.bbox = [
                min(box[0] for box in cleaned),
                min(box[1] for box in cleaned),
                max(box[2] for box in cleaned),
                max(box[3] for box in cleaned),
            ]
            pad = max(
                10,
                round(
                    max(
                        unit.bbox[2] - unit.bbox[0],
                        unit.bbox[3] - unit.bbox[1],
                    )
                    * 0.04
                ),
            )
            unit.crop_bbox = [
                max(0, unit.bbox[0] - pad),
                max(0, unit.bbox[1] - pad),
                min(page.width, unit.bbox[2] + pad),
                min(page.height, unit.bbox[3] + pad),
            ]
        audit_units = [
            unit
            for unit in page.units
            if (
                unit.skip
                and (
                    unit.skip_reason != "preserve"
                    or not JP_RE.search(unit.ja)
                    or unit.score < 0.65
                    or self._local_crop_is_suspicious(unit)
                )
            )
            or not JP_RE.search(unit.ja)
            or (
                not unit.is_sfx
                and (unit.score < 0.65 or self._local_crop_is_suspicious(unit))
            )
            or (unit.is_sfx and unit.score < 0.65 and not likely_sfx_text(unit.ja))
        ]
        for start in range(0, len(audit_units), max(1, batch_size)):
            units = audit_units[start : start + max(1, batch_size)]
            ids = [unit.id for unit in units]
            item_schema = {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "enum": ids},
                    "text": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": [
                            "dialogue",
                            "narration",
                            "sfx",
                            "artwork",
                            "noise",
                            "nontext",
                        ],
                    },
                    "effect_kind": {"type": "string", "enum": list(SFX_KINDS)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["id", "text", "role", "effect_kind", "confidence"],
            }
            schema = {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": item_schema},
                },
                "required": ["items"],
            }
            order = "、".join(ids)
            prompt = f"""你会依次收到 {len(units)} 张漫画文字框的紧密局部放大图，对应 id 顺序为：{order}。
每张图只抄写局部内真正可见的日文，禁止根据故事或相邻图片补全。
汉字旁或上方的振假名只用于确认读音，不得与汉字正文重复抄写；保留汉字正文一次。
role 选择 dialogue（对白）、narration（旁白）、sfx（拟声/效果字）、artwork（衣服印花、商标、招牌或道具文字）或 noise（误检）。看不清就选 noise。
仅 role=sfx 时根据画面填写 effect_kind，否则填写 none。衣服或背景物件上的文字不能当对白。
必须为每个 id 返回一项。"""
            payload = self._chat(
                [self._tight_crop_bytes(image, unit) for unit in units],
                prompt,
                schema,
            )
            by_id = {
                str(item.get("id")): item
                for item in payload.get("items", [])
                if isinstance(item, dict) and item.get("id") in ids
            }
            for unit in units:
                item = by_id.get(unit.id)
                if not item:
                    continue
                role = str(item.get("role", "")).lower()
                confidence = float(item.get("confidence", 0.0))
                candidate = self._normalize_local_crop_text(str(item.get("text", "")))
                suspicious = self._local_crop_is_suspicious(unit)
                cover_title = unit.special == "cover_title"
                if cover_title:
                    role = "narration"
                if role in {"artwork", "noise", "nontext"}:
                    agrees = prefer_semantic_ocr(unit.ja, candidate, confidence)
                    latin_logo = bool(
                        role == "artwork"
                        and suspicious
                        and confidence >= 0.95
                        and re.fullmatch(
                            r"[A-Za-z0-9][A-Za-z0-9 ._&'/-]{3,}", candidate
                        )
                    )
                    artwork_agrees = (
                        agrees
                        or bool(
                            role == "artwork"
                            and semantic_text_agreement(unit.ja, candidate)
                        )
                        or latin_logo
                    )
                    source_looks_like_sfx = unit.is_sfx or bool(
                        JP_RE.search(unit.ja) and likely_sfx_text(unit.ja)
                    )
                    source_looks_like_dialogue = likely_dialogue_text(unit.ja)
                    if source_looks_like_sfx and unit.special in {
                        "artwork_text",
                        "ocr_noise",
                    }:
                        unit.skip = False
                        unit.skip_reason = ""
                        unit.special = ""
                    # ``page_dialogue`` means the full-page pass read words here
                    # and found them readable; stylised lettering seen alone in a
                    # tight crop is not evidence strong enough to overrule that.
                    if (
                        confidence >= 0.9
                        and artwork_agrees
                        and not source_looks_like_sfx
                        and (not source_looks_like_dialogue or latin_logo)
                        and unit.special != "page_dialogue"
                    ):
                        unit.zh = ""
                        unit.skip = True
                        unit.skip_reason = "preserve" if role == "artwork" else "noise"
                        unit.special = (
                            "artwork_text" if role == "artwork" else "ocr_noise"
                        )
                    elif (
                        not candidate
                        and confidence >= 0.8
                        and unit.score <= 0.55
                        and not source_looks_like_sfx
                    ):
                        unit.zh = ""
                        unit.skip = True
                        unit.skip_reason = "noise"
                        unit.special = "ocr_noise"
                    continue
                if role == "sfx":
                    threshold = 0.68 if suspicious or unit.is_sfx else 0.84
                    if confidence >= threshold:
                        original_ja = unit.ja
                        agrees = prefer_semantic_ocr(unit.ja, candidate, confidence)
                        core = repeated_sfx_core(unit.ja)
                        candidate_fit = _area(unit.bbox) / max(
                            1,
                            len(
                                re.sub(
                                    r"[\s．。…、，,!?！？:：♡♥〰〜～]",
                                    "",
                                    candidate,
                                )
                            ),
                        )
                        candidate_is_sfx = semantic_sfx_classification(
                            candidate, confidence, True
                        )
                        if (
                            confidence >= 0.9
                            and not candidate_is_sfx
                            and JP_RE.search(candidate)
                            and candidate_fit >= 250
                        ):
                            unit.ja = candidate
                            unit.score = max(unit.score, round(confidence, 4))
                            unit.is_sfx = False
                            unit.special = "semantic_dialogue"
                            if unit.skip_reason in {"duplicate", "noise", "preserve"}:
                                unit.skip = False
                                unit.skip_reason = ""
                            continue
                        if core:
                            unit.ja = core
                        elif (
                            confidence >= max(0.82, threshold)
                            and JP_RE.search(candidate)
                            and candidate_fit >= 250
                            and (not unit.is_sfx or agrees)
                            and (
                                unit.is_sfx
                                or likely_sfx_text(candidate)
                                or bool(re.fullmatch(r"[ァ-ヺーッ♡♥♪]+", candidate))
                            )
                        ):
                            unit.ja = candidate
                            unit.score = max(unit.score, round(confidence, 4))
                        if unit.ja != original_ja and unit.skip_reason in {
                            "duplicate",
                            "noise",
                            "preserve",
                        }:
                            unit.skip = False
                            unit.skip_reason = ""
                            if unit.special in {
                                "ocr_duplicate",
                                "ocr_noise",
                                "artwork_text",
                            }:
                                unit.special = ""
                        unit.is_sfx = True
                        effect_kind = str(item.get("effect_kind", "none")).lower()
                        if effect_kind in SFX_KINDS and effect_kind != "none":
                            unit.special = f"sfx:{effect_kind}"
                    continue
                if role not in {"dialogue", "narration"} or confidence < 0.82:
                    continue
                if not JP_RE.search(candidate):
                    continue
                if unit.is_sfx:
                    semantic = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", candidate)
                    has_sentence_evidence = (
                        len(semantic) >= 4
                        or bool(re.search(r"[．。…、，,?？]", candidate))
                        or (
                            bool(re.search(r"[\u3400-\u9fff]", semantic))
                            and bool(re.search(r"[ぁ-ゖ]", semantic))
                        )
                    )
                    if not has_sentence_evidence:
                        continue
                # Tight-crop text may replace the detector recognizer only when
                # the old geometry/text contract is already suspect, or when it
                # recovers a region previously classified as an effect.
                candidate_fit = _area(unit.bbox) / max(
                    1,
                    len(re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", candidate)),
                )
                current_length = len(
                    re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", unit.ja)
                )
                candidate_length = len(
                    re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", candidate)
                )
                if (
                    confidence >= 0.9
                    and candidate_fit >= 250
                    and candidate_length >= current_length * 0.72
                ):
                    unit.ja = candidate
                    unit.score = max(unit.score, round(confidence, 4))
                    unit.is_sfx = False
                if confidence >= 0.88:
                    if unit.special in {"artwork_text", "ocr_noise"}:
                        unit.skip = False
                        unit.skip_reason = ""
                    if not cover_title:
                        unit.special = "semantic_dialogue"
        for unit in page.units:
            if unit.special == "page_dialogue":
                # Served its purpose in the crop audit above; the stored page
                # keeps the same special values it always had.  Cleared before
                # the branches below so an early ``continue`` cannot leak it.
                unit.special = ""
            if unit.special == "cover_title":
                unit.is_sfx = False
                unit.skip = False
                unit.skip_reason = ""
                continue
            unit.is_sfx = semantic_sfx_classification(unit.ja, unit.score, unit.is_sfx)
            if not unit.is_sfx and unit.special.startswith("sfx:"):
                unit.special = ""
            if oversized_text_region(page, unit):
                unit.zh = ""
                unit.skip = True
                unit.skip_reason = "preserve"
                unit.special = "artwork_text"
                continue
            if oversized_decorative_sfx(page, unit):
                unit.zh = ""
                unit.skip = True
                unit.skip_reason = "decorative"
                unit.special = "decorative_sfx"
                continue
            sfx_echo = overlapping_sfx_echo(page, unit)
            if unit.special != "semantic_dialogue" and (
                tiny_low_confidence_nontext(page, unit)
                or malformed_tiny_ocr(page, unit)
                or duplicate_tiny_fragment(page, unit)
                or sfx_echo
            ):
                duplicate = duplicate_tiny_fragment(page, unit) or sfx_echo
                unit.zh = ""
                unit.skip = True
                unit.skip_reason = "duplicate" if duplicate else "noise"
                unit.special = "ocr_duplicate" if duplicate else "ocr_noise"
        return page

    @staticmethod
    def filter_covered_missing(page: PageOCR, missing: list[dict]) -> list[dict]:
        """Discard second-pass VLM omissions already disproved by exact OCR."""

        def normalized_text(value: str) -> str:
            return re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", value)

        known_text = {
            normalized_text(unit.ja)
            for unit in page.units
            if len(normalized_text(unit.ja)) >= 2
        }
        known_bigrams: set[str] = set()
        for value in known_text:
            known_bigrams.update(
                value[index : index + 2] for index in range(len(value) - 1)
            )
        known_boxes = [
            erase_box
            for unit in page.units
            for erase_box in (unit.erase_boxes or [unit.bbox])
        ]
        unresolved = []
        for item in missing:
            text = normalized_text(str(item.get("text", "")))
            if text and any(
                text == known or text in known or known in text for known in known_text
            ):
                continue
            # The recall pass re-reads lines the detector already owns and
            # garbles a character or two ("何か理由が…ちゃんとなんて話を" for
            # "…ちゃんと話を").  Exact containment then misses and a fully
            # covered line aborted the book, so compare loosely as well.
            if text and _rereads_known_text(text, known_text, known_bigrams):
                continue
            try:
                raw = [int(value) for value in item["bbox"]]
            except (KeyError, TypeError, ValueError):
                unresolved.append(item)
                continue
            if len(raw) != 4:
                unresolved.append(item)
                continue
            box = [
                round(raw[0] * page.width / 1000),
                round(raw[1] * page.height / 1000),
                round(raw[2] * page.width / 1000),
                round(raw[3] * page.height / 1000),
            ]
            if bool(item.get("is_sfx", False)) and any(
                oversized_decorative_sfx(page, unit)
                and _axis_overlap(box[0], box[2], unit.bbox[0], unit.bbox[2])
                * _axis_overlap(box[1], box[3], unit.bbox[1], unit.bbox[3])
                >= _area(box) * 0.8
                for unit in page.units
            ):
                continue
            covered = any(
                _axis_overlap(box[0], box[2], known[0], known[2])
                * _axis_overlap(box[1], box[3], known[1], known[3])
                >= min(_area(box), _area(known)) * 0.5
                for known in known_boxes
            )
            if not covered:
                unresolved.append(item)
        return unresolved


class LlamaVisionOCR(OllamaVisionOCR):
    """The same vision OCR, answered by a ``llama-server`` we started ourselves.

    Only the transport changes, so this subclasses the Ollama adapter and keeps
    every prompt, schema, clamping rule and role filter exactly as it was -- that
    is what makes an A/B comparison meaningful.  Two things do differ:

    * The request is OpenAI-shaped (``/v1/chat/completions`` with image parts and
      ``response_format.json_schema``), which lives in ``llama_client``.
    * llama.cpp's image loader rejects WebP, so pages are converted to PNG first;
      every page this app stores is WebP, so skipping that step fails on page 1.

    ``think`` is accepted and ignored: the server is started with ``-rea off``,
    which is the server-side twin of Ollama's ``think: false``.

    The default wait is longer than the daemon adapter's 300 s because the same
    refine call on the densest page finished inside 300 s there and ran past it
    here, and the transport is the only thing that changed; the client scales the
    wait further with each escalated budget.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 600,
        source_language: str = "auto",
        context_tokens: int = OLLAMA_CONTEXT_TOKENS,
        max_output_tokens: int = OLLAMA_MAX_OUTPUT_TOKENS,
    ):
        super().__init__(
            base_url, model, timeout=timeout, source_language=source_language
        )
        # Imported here: ``llama_client`` imports ``translator``, which imports
        # this module, so a module-level import would close the circle.
        from .llama_client import LlamaChatClient

        self.client = LlamaChatClient(
            base_url,
            self.model,
            context_tokens=context_tokens,
            max_output_tokens=max_output_tokens,
        )

    def _image_bytes(self, image_path: Path) -> bytes:
        from .llama_client import png_bytes

        return png_bytes(image_path)

    def _chat(self, images: list[bytes], prompt: str, schema: dict) -> dict:
        return self.client.chat_vision(images, prompt, schema, timeout=self.timeout)


class HybridMangaOCR:
    """Quality OCR: Paddle/MangaOCR geometry plus full-page vision semantics."""

    def __init__(
        self,
        paths: AppPaths,
        base_url: str,
        model: str = DEFAULT_OLLAMA_VISION_MODEL,
        device: str = "auto",
        profile: str = "quality",
        source_language: str = "auto",
        semantics: "OllamaVisionOCR | None" = None,
    ):
        self.geometry = PaddleMangaOCR(paths, device, profile, source_language)
        # Injected so the exact same geometry/semantics split can run against
        # either vision transport; the default keeps the Ollama path untouched.
        if semantics is not None:
            self.semantics = semantics
        else:
            self.semantics = OllamaVisionOCR(
                base_url, model, source_language=source_language
            )
        self.profile = profile
        self.source_language = source_language
        self.last_missing: list[dict] = []

    @staticmethod
    def _coarse_page_as_missing(page: PageOCR) -> list[dict]:
        return [
            {
                "text": unit.ja,
                "bbox": [
                    round(unit.bbox[0] * 1000 / page.width),
                    round(unit.bbox[1] * 1000 / page.height),
                    round(unit.bbox[2] * 1000 / page.width),
                    round(unit.bbox[3] * 1000 / page.height),
                ],
                "score": unit.score,
                "is_sfx": unit.is_sfx,
            }
            for unit in page.units
        ]

    def analyze(self, image_path: Path, page_number: int) -> PageOCR:
        page = self.geometry.analyze(image_path, page_number)
        if page.units:
            page, first_missing = self.semantics.refine(image_path, page)
            second_missing: list[dict] = []
            if getattr(self, "profile", "quality") == "quality":
                if hasattr(self.semantics, "find_missing"):
                    second_missing = self.semantics.find_missing(image_path, page)
                else:
                    page, second_missing = self.semantics.refine(image_path, page)
            self.last_missing = merge_semantic_missing(first_missing, second_missing)
        else:
            # An empty exact-detector result is not evidence that a page has no
            # text. Use full-page vision only to propose recovery crops; raw VLM
            # coordinates still never become erase geometry.
            coarse = self.semantics.analyze(image_path, page_number)
            self.last_missing = self._coarse_page_as_missing(coarse)
        pending_missing = self.semantics.filter_covered_missing(
            page, list(self.last_missing)
        )
        if page.units and getattr(self, "profile", "quality") == "quality":
            # Proposals and confidence are not the same thing.  llama.cpp
            # proposed two regions on a dense page whose three real sound
            # effects it never mentioned, and a decoder that proposes nothing
            # leaves recovery with nothing to verify.  Scan the uncovered ink
            # tiles as well: recovery still demands detector geometry plus an
            # independent MangaOCR read, so a tile is a proposal, never a unit.
            pending_missing = pending_missing + ink_tile_hints(
                page, np.asarray(Image.open(image_path).convert("L"))
            )
        if pending_missing:
            # Remove exact textual/geometry coverage before invoking expensive
            # regional recovery. Full-page VLMs often repeat one long vertical
            # line as several shorter "missing" fragments.
            page = self.geometry.recover_missing(image_path, page, pending_missing)
            if page.units:
                # Recovery already requires detector-owned geometry plus an
                # independent MangaOCR read. Repeating the full-page VLM here
                # added minutes per page and could reintroduce stale omissions.
                self.last_missing = self.semantics.filter_covered_missing(
                    page, pending_missing
                )
            # Whatever recovery could not confirm is not actionable: the claim
            # has no detector-owned geometry, so nothing can be erased and the
            # translator has nowhere to put the line.  A full-page VLM produces
            # such claims constantly — screentone hatching read as "ただだ",
            # panel borders as "効果か:", a known line re-read with one kana
            # changed — and each one used to abort the whole book.  Mark them
            # instead of dropping them: the page keeps the record (it lands in
            # the transcript and the work directory) and the run continues.
            for item in self.last_missing:
                item["unconfirmed"] = True
        page = prune_nested_duplicate_units(page)
        if getattr(self, "profile", "quality") == "quality" and hasattr(
            self.semantics, "refine_local_crops"
        ):
            page = self.semantics.refine_local_crops(image_path, page)
            self.last_missing = self.semantics.filter_covered_missing(
                page, self.last_missing
            )
        page.semantic_missing = list(self.last_missing)
        return page

    def refine_cover_title(self, image_path: Path, page: PageOCR) -> PageOCR:
        page = self.geometry.refine_cover_title(image_path, page)
        page, self.last_missing = self.semantics.refine(image_path, page)
        for unit in page.units:
            if unit.special == "page_dialogue":
                # No crop audit follows this path, so the marker has nothing to
                # guard here; dropping it keeps it out of the stored transcript.
                unit.special = ""
        page.semantic_missing = list(self.last_missing)
        return page
