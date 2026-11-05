from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SfxMeaning:
    kind: str
    chinese: str


_PUNCTUATION_RE = re.compile(r"[\s．。…、，,!?！？:：♡♥〰〜～・]+")


def _compact(source: str) -> str:
    return _PUNCTUATION_RE.sub("", source).replace("ッ", "ッ")


def _emphasis(source: str) -> str:
    if re.search(r"[!！]|ッ\s*$", source.strip()):
        return "！"
    if re.search(r"[…．。]{2,}", source):
        return "……"
    return ""


# English sound words that a model reads as a word and transliterates into
# nonsense Chinese (Grrrrr -> 格勒勒).  Keep the list to onomatopoeia that has a
# settled Chinese convention; anything else still goes to the model.
_LATIN_SFX = {
    "grr": SfxMeaning("vocalization", "咕噜噜"),
    "grrr": SfxMeaning("vocalization", "咕噜噜"),
    "argh": SfxMeaning("vocalization", "呃啊！"),
    "augh": SfxMeaning("vocalization", "啊啊！"),
    "agh": SfxMeaning("vocalization", "啊！"),
    "ugh": SfxMeaning("vocalization", "呃……"),
    "hmph": SfxMeaning("vocalization", "哼！"),
    "gasp": SfxMeaning("vocalization", "呼——"),
    "sigh": SfxMeaning("vocalization", "唉……"),
    "poof": SfxMeaning("impact", "噗"),
    "bam": SfxMeaning("impact", "砰！"),
    "bang": SfxMeaning("impact", "砰！"),
    "thud": SfxMeaning("impact", "咚"),
    "crash": SfxMeaning("impact", "哗啦！"),
    "slam": SfxMeaning("impact", "砰！"),
    "clang": SfxMeaning("impact", "铛！"),
    "knock": SfxMeaning("impact", "咚咚"),
    "creak": SfxMeaning("mechanical", "嘎吱"),
    "rustle": SfxMeaning("movement", "窸窣"),
    "splash": SfxMeaning("wet", "哗啦"),
    "gulp": SfxMeaning("swallow", "咕咚"),
    "munch": SfxMeaning("wet", "咔嚓咔嚓"),
    "sniff": SfxMeaning("vocalization", "吸溜"),
    "wheeze": SfxMeaning("vocalization", "呼哧呼哧"),
    "zzz": SfxMeaning("vocalization", "呼……呼……"),
}


def _latin_sfx(compact: str) -> SfxMeaning | None:
    """Match a repeated Latin sound word such as GRRRRR or Poof!."""
    word = compact.lower()
    if not re.fullmatch(r"[a-z]{2,14}", word):
        return None
    while len(word) >= 2:
        if word in _LATIN_SFX:
            return _LATIN_SFX[word]
        if len(set(word[-2:])) != 1:
            return None
        word = word[:-1]
    return None


def _effect_word(word: str) -> SfxMeaning | None:
    """Match one Latin effect, tolerating an OCR-clipped spelling ("nif")."""
    meaning = _latin_sfx(word)
    if meaning is not None:
        return meaning
    for known, candidate in _LATIN_SFX.items():
        if len(word) >= 3 and word in known and len(known) - len(word) <= 2:
            return candidate
    return None


def _latin_sfx_phrase(source: str) -> SfxMeaning | None:
    """English pages often double an effect: "sniff snif", "poof poof"."""
    words = re.findall(r"[A-Za-z]+", source)
    if not 2 <= len(words) <= 4:
        return None
    meanings = [_effect_word(word.lower()) for word in words]
    if any(meaning is None for meaning in meanings):
        return None
    first = meanings[0]
    if all(meaning.chinese == first.chinese for meaning in meanings):
        return SfxMeaning(first.kind, first.chinese * len(meanings))
    return first


def lookup_sfx(source: str) -> SfxMeaning | None:
    """Return a conservative, language-level manga SFX interpretation.

    These rules cover productive Japanese sound families instead of book- or
    character-specific strings.  They provide a semantic prior for the local
    model and a deterministic answer only where the Chinese convention is
    unambiguous enough to beat unconstrained phonetic generation.
    """

    compact = _compact(source)
    if not compact:
        return None

    latin = _latin_sfx(compact)
    if latin is not None:
        return latin
    phrase = _latin_sfx_phrase(source)
    if phrase is not None:
        return phrase

    heartbeat_source = compact.rstrip("ッ")
    heartbeat_hits = re.findall(r"(?:ドキン?|ドクン?|バクン?)", heartbeat_source)
    if heartbeat_hits and "".join(heartbeat_hits) == heartbeat_source:
        repeated = "".join("怦咚" for _ in range(min(4, len(heartbeat_hits))))
        return SfxMeaning("heartbeat", repeated + _emphasis(source))
    if compact.count("バ") >= 2 and "バクン" in compact:
        repeated = "".join("怦咚" for _ in range(min(4, compact.count("バ"))))
        return SfxMeaning("heartbeat", repeated + _emphasis(source))

    if re.fullmatch(r"[ドトッ]{4,}", compact):
        return SfxMeaning("rumble", "轰隆隆" + _emphasis(source))
    if re.fullmatch(r"ゴ{3,}", compact):
        return SfxMeaning("rumble", "隆隆隆" + _emphasis(source))
    if re.fullmatch(r"(?:ブロ+ン?|ブル+ン?)+", compact):
        return SfxMeaning("engine", "轰隆" + _emphasis(source))

    if re.fullmatch(r"キャッ+", compact):
        return SfxMeaning("vocalization", "呀！")
    if re.fullmatch(r"ギャッ+", compact):
        return SfxMeaning("vocalization", "哇啊！")
    if re.fullmatch(r"ヒッ+", compact):
        return SfxMeaning("vocalization", "咿！")

    exact = {
        "ゴッ": SfxMeaning("impact", "咚" + _emphasis(source)),
        "ドン": SfxMeaning("impact", "咚" + _emphasis(source)),
        "ドーン": SfxMeaning("impact", "轰" + _emphasis(source)),
        "バン": SfxMeaning("impact", "砰" + _emphasis(source)),
        "バタン": SfxMeaning("impact", "砰" + _emphasis(source)),
        "ガタン": SfxMeaning("impact", "哐当" + _emphasis(source)),
        "ガン": SfxMeaning("impact", "铛" + _emphasis(source)),
        "カチャ": SfxMeaning("mechanical", "咔哒" + _emphasis(source)),
        "カチ": SfxMeaning("mechanical", "咔" + _emphasis(source)),
        "カチッ": SfxMeaning("mechanical", "咔哒！"),
        "パチ": SfxMeaning("snap", "啪" + _emphasis(source)),
        "パン": SfxMeaning("snap", "啪" + _emphasis(source)),
        "ズル": SfxMeaning("movement", "滋溜" + _emphasis(source)),
        "ズズ": SfxMeaning("movement", "簌簌" + _emphasis(source)),
        "ゴクン": SfxMeaning("swallow", "咕咚" + _emphasis(source)),
        "ゴクリ": SfxMeaning("swallow", "咕咚" + _emphasis(source)),
        "ペロ": SfxMeaning("lick", "舔" + _emphasis(source)),
        "クチュ": SfxMeaning("wet", "咕啾" + _emphasis(source)),
        "グチュ": SfxMeaning("wet", "咕啾" + _emphasis(source)),
    }
    if compact in exact:
        return exact[compact]

    if re.fullmatch(r"(?:パン){2,}", compact):
        return SfxMeaning("repeated_impact", "啪啪" + _emphasis(source))
    if re.fullmatch(r"(?:カチ){2,}", compact):
        return SfxMeaning("mechanical", "咔哒咔哒" + _emphasis(source))
    return None
