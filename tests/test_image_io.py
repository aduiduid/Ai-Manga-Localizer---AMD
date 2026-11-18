"""Reading pages that arrive half-written: a truncated WebP still has to decode."""

import io
import struct

import numpy as np
import pytest
from PIL import Image

from manga_localizer.image_io import UnreadableImage, load_rgb, open_image


def _webp_bytes(size=(480, 360), lossless=False) -> bytes:
    """A page-sized WebP.

    A tiny 120x90 file survives the cut inside a single block and Pillow reads
    it anyway, so it proves nothing about the repair; a real page is what the
    broken file in the report looked like.
    """
    noise = np.random.default_rng(7).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(noise)
    buffer = io.BytesIO()
    if lossless:
        image.save(buffer, "WEBP", lossless=True)
    else:
        image.save(buffer, "WEBP", quality=92)
    return buffer.getvalue()


def _truncate(data: bytes, keep: float) -> bytes:
    return data[: int(len(data) * keep)]


@pytest.mark.parametrize("lossless", [False, True])
def test_a_truncated_webp_is_padded_back_to_its_declared_length(tmp_path, lossless):
    """Half a page still beats a failed 255-page book.

    The reported page declared a 77 KB ``VP8 `` chunk in a 36 KB file; libwebp
    refuses to build a decoder until the chunk length is satisfied, and the run
    died on page 34 of 255.  Padding the tail with zeros decodes the picture.
    """
    full = _webp_bytes(lossless=lossless)
    broken = tmp_path / "00000034.webp"
    broken.write_bytes(_truncate(full, 0.7))
    image = load_rgb(broken)
    assert image.mode == "RGB"
    assert image.size == (480, 360)
    # A zero-filled tail still decodes the picture, not a grey placeholder.
    assert float(np.asarray(image).std()) > 20


def test_a_broken_page_names_itself_in_chinese(tmp_path):
    """The message has to say which page died, or 515 pages are unreadable."""
    broken = tmp_path / "00000099.webp"
    broken.write_bytes(b"RIFF" + struct.pack("<I", 64) + b"WEBPVP8 " + bytes(12))
    with pytest.raises(UnreadableImage) as caught:
        load_rgb(broken)
    message = str(caught.value)
    assert "00000099.webp" in message
    assert "无法读取图片" in message


def test_a_missing_page_is_reported_rather_than_raised_raw(tmp_path):
    with pytest.raises(UnreadableImage) as caught:
        open_image(tmp_path / "gone.webp")
    assert "gone.webp" in str(caught.value)
