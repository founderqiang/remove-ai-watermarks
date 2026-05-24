"""Tests for open invisible-watermark (imwatermark) detection.

Each known scheme is round-tripped: embed its exact upstream pattern with the
encoder, then assert the detector names it. Skipped entirely if the optional
``invisible-watermark`` package is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from remove_ai_watermarks.invisible_watermark import (
    _BITS_48,
    _SD1_STRING,
    _bits_match,
    _bytes_match_frac,
    detect_invisible_watermark,
    is_available,
)

pytestmark = pytest.mark.skipif(not is_available(), reason="invisible-watermark not installed")


def _base_image() -> np.ndarray:
    # imwatermark needs enough DWT coefficients; use a 512x512 textured image.
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)


def _write_bits_watermark(tmp_path: Path, message: int) -> Path:
    from imwatermark import WatermarkEncoder

    bits = [int(b) for b in format(message, "048b")]
    enc = WatermarkEncoder()
    enc.set_watermark("bits", bits)
    wm = enc.encode(_base_image(), "dwtDct")
    path = tmp_path / "wm.png"
    cv2.imwrite(str(path), wm)
    return path


class TestHelpers:
    def test_bits_match_exact(self):
        assert _bits_match(0b1010, 0b1010, width=4) == 4

    def test_bits_match_one_off(self):
        assert _bits_match(0b1010, 0b1011, width=4) == 3

    def test_bytes_match_identical(self):
        assert _bytes_match_frac(_SD1_STRING, _SD1_STRING) == 1.0

    def test_bytes_match_length_mismatch_is_zero(self):
        assert _bytes_match_frac(b"abc", b"abcd") == 0.0


class TestDetect:
    def test_detects_sdxl(self, tmp_path: Path):
        path = _write_bits_watermark(tmp_path, _BITS_48["Stable Diffusion XL"])
        assert detect_invisible_watermark(path) == "Stable Diffusion XL"

    def test_detects_flux(self, tmp_path: Path):
        path = _write_bits_watermark(tmp_path, _BITS_48["FLUX.2 (Black Forest Labs)"])
        assert detect_invisible_watermark(path) == "FLUX.2 (Black Forest Labs)"

    def test_detects_sd1_string(self, tmp_path: Path):
        from imwatermark import WatermarkEncoder

        enc = WatermarkEncoder()
        enc.set_watermark("bytes", _SD1_STRING)
        wm = enc.encode(_base_image(), "dwtDct")
        path = tmp_path / "sd1.png"
        cv2.imwrite(str(path), wm)
        assert detect_invisible_watermark(path) == "Stable Diffusion 1.x / 2.x"

    def test_clean_image_is_none(self, tmp_path: Path):
        path = tmp_path / "clean.png"
        cv2.imwrite(str(path), _base_image())
        assert detect_invisible_watermark(path) is None

    def test_unreadable_file_is_none(self, tmp_path: Path):
        path = tmp_path / "not_image.png"
        path.write_bytes(b"not a png")
        assert detect_invisible_watermark(path) is None
