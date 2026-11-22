#!/usr/bin/env python
"""Pixel probe: print what a coordinate looks like before and after render.

Written after a colour regression that no test caught, and after a human
misread a screenshot twice in the same session.  The rule this exists to
enforce: never argue about colour from a downscaled picture.  Ask the pixels.

    python scripts/pixel_probe.py <source-or-book> --point 720 190
    python scripts/pixel_probe.py <source> --box 717 168 787 207
    python scripts/pixel_probe.py <source> --unit p015u01 --book <rendered book>

With a rendered book, the same coordinate or the same unit is measured in both
images and the two readings are printed side by side, so "the colour is wrong"
becomes a number instead of an impression.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _hue_name(colour) -> str:
    spread = int(max(colour) - min(colour))
    if spread < 40:
        return "neutral"
    r, g, b = (float(v) / 255.0 for v in colour)
    hue = colorsys.rgb_to_hls(r, g, b)[0] * 360.0
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 260:
        return "blue"
    if hue < 320:
        return "purple"
    return "magenta"


def report(label: str, arr: np.ndarray, box) -> None:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = arr[y0:y1, x0:x1]
    if not crop.size:
        print(f"  {label:9} empty region {box}")
        return
    flat = crop.reshape(-1, 3).astype(int)
    level = flat @ np.array([0.299, 0.587, 0.114])
    dark = flat[level <= np.percentile(level, 10)]
    mean = dark.mean(axis=0)
    colour = tuple(int(v) for v in mean)
    spread = int(colour[0:3].__len__() and max(colour) - min(colour))
    print(
        f"  {label:9} box={x0},{y0},{x1},{y1} size={x1 - x0}x{y1 - y0} "
        f"paper={int(np.median(level)):3d} "
        f"glyph={colour} spread={spread:3d} -> {_hue_name(colour)}"
    )


def find_page(folder: Path, stem: str) -> Path | None:
    for suffix in (".webp", ".png", ".jpg", ".jpeg"):
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def units_of(book: Path, stem: str) -> dict:
    cache = book / ".manga-localizer-work" / f"{stem}.ocr.json"
    if not cache.is_file():
        return {}
    payload = json.loads(cache.read_text(encoding="utf-8"))
    return {row["id"]: [int(round(v)) for v in row["bbox"]] for row in payload["units"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="one page image, or a book folder")
    parser.add_argument("--rendered", type=Path, help="the rendered book folder")
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"),
                        help="sample a small window centred here")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--unit", help="unit id, resolved through the OCR cache")
    parser.add_argument("--page", help="page stem for --unit, e.g. 00000015")
    args = parser.parse_args()

    if args.source.is_dir():
        stem = args.page or "00000001"
        page = find_page(args.source, stem)
    else:
        stem = args.source.stem
        page = args.source
    if page is None:
        print(f"no page {stem} in {args.source}", file=sys.stderr)
        return 2

    would_be = args.rendered or (args.source if args.source.is_dir() else None)
    source_arr = np.asarray(Image.open(page).convert("RGB"))

    box = None
    if args.unit:
        # The OCR cache lives beside the *rendered* book, not the source art, so
        # try every folder the caller gave us before giving up.
        roots = []
        if args.rendered is not None:
            roots.append(args.rendered)
        roots.append(args.source if args.source.is_dir() else args.source.parent)
        for root in roots:
            box = units_of(root, stem).get(args.unit)
            if box is not None:
                break
        if box is None:
            tried = ", ".join(str(root) for root in roots)
            print(f"unit {args.unit} not found near: {tried}", file=sys.stderr)
            return 2
    elif args.box:
        box = args.box
    elif args.point:
        x, y = args.point
        box = [x - 3, y - 3, x + 4, y + 4]
    else:
        parser.error("one of --point, --box or --unit is required")

    print(f"page {page.name}  ({source_arr.shape[1]}x{source_arr.shape[0]})")
    report("source", source_arr, box)

    if would_be is not None:
        rendered_page = find_page(would_be, stem)
        if rendered_page is None:
            print(f"  (no rendered {stem} in {would_be})")
        else:
            report("rendered", np.asarray(Image.open(rendered_page).convert("RGB")), box)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
