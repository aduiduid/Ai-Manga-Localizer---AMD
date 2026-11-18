"""Read a source page even when the file on disk is not quite whole.

A ripped WebP can end in the middle of its pixel payload: the RIFF header and
the ``VP8 `` chunk still announce their full lengths, but the bytes stop early.
Pillow's ``open()`` happily parses the header -- it reports the right size --
and only raises when ``load()`` asks libwebp to build a decoder, which fails
because the chunk it was promised is not there:

    OSError: could not create decoder object

That is a fatal error in the middle of OCR for one single page of a 255 page
book.  The payload is usually intact up to the cut, so zero-filling the file to
the length it declares makes both Pillow and OpenCV decode a complete picture.
This module tries that repair first, then a straight OpenCV decode, and only
then gives up with a message naming the file.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# A truncated page is normally a few hundred kilobytes.  Anything claiming more
# than this is a corrupt length field, not a page, and padding it would allocate
# gigabytes for nothing.
MAX_REPAIR_BYTES = 512 * 1024 * 1024


class UnreadableImage(OSError):
    """A source page that no decoder in the process could open."""


def _riff_end(data: bytes) -> int:
    """Byte length the RIFF header and its chunks claim."""
    if len(data) < 12 or data[:4] != b"RIFF":
        return 0
    total = int.from_bytes(data[4:8], "little") + 8
    offset = 12
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        end = offset + 8 + size + (size & 1)
        if end <= offset:
            break
        total = max(total, end)
        offset = end
    return total


def _padded(data: bytes) -> bytes | None:
    """``data`` grown with zeros to the length it declares, when it is short."""
    end = _riff_end(data)
    if end <= len(data) or end > MAX_REPAIR_BYTES:
        return None
    return data + b"\x00" * (end - len(data))


def _first_decode(path: Path):
    image = Image.open(path)
    image.load()
    return image


def _recover(path: Path, raw: bytes):
    """Second and third attempts: pad the truncated file, then raw OpenCV."""
    padded = _padded(raw)
    if padded is not None:
        try:
            image = Image.open(BytesIO(padded))
            image.load()
            return image
        except OSError:
            pass
        decoded = cv2.imdecode(np.frombuffer(padded, np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if decoded is not None:
        return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    return None


def open_image(path: Path) -> Image.Image:
    """Open and fully decode ``path``, repairing a truncated file if needed."""
    path = Path(path)
    try:
        return _first_decode(path)
    except OSError as error:
        failure = error
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise UnreadableImage(
            f"无法读取图片：{path.name}（{error}）。请确认该文件仍然存在后重跑。"
        ) from error
    recovered = _recover(path, raw)
    if recovered is not None:
        return recovered
    raise UnreadableImage(
        f"无法读取图片：{path.name}。该文件已损坏（{failure}），"
        "补零修复与备用解码都没能读出画面。请重新获取这一页的源文件后重跑；"
        "其余页不受影响。"
    ) from failure


def load_rgb(path: Path) -> Image.Image:
    """Decode ``path`` as RGB."""
    return open_image(path).convert("RGB")


def load_gray(path: Path) -> Image.Image:
    """Decode ``path`` as an 8-bit greyscale image."""
    return open_image(path).convert("L")


__all__ = ["MAX_REPAIR_BYTES", "UnreadableImage", "load_gray", "load_rgb", "open_image"]
