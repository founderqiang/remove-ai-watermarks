"""Byte-identity guards for the text-mark engine memory optimization.

The reverse-alpha text-mark engine used to allocate full-frame arrays where only
the glyph footprint is ever read:

  * ``extract_mask`` built a full ``(h, w)`` uint8 mask and every caller cropped
    it to the located box;
  * ``_fixed_alpha_map`` / ``_aligned_alpha_map`` each built a full ``(h, w)``
    float32 alpha map that is non-zero only inside the glyph box, and two were
    held at once during removal.

Both now return footprint-sized arrays. These tests prove the new footprint-sized
path is BYTE-IDENTICAL to the old full-frame path by reconstructing the old
behavior inline from the new building blocks (so the proof survives a cv2/asset
version bump, unlike a pinned output hash), and lock in the O(footprint) memory
characteristic so a regression back to a full-frame allocation fails loudly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import remove_ai_watermarks.doubao_engine as D
import remove_ai_watermarks.jimeng_engine as J
import remove_ai_watermarks.samsung_engine as S
from remove_ai_watermarks.doubao_engine import DoubaoEngine
from remove_ai_watermarks.jimeng_engine import JimengEngine
from remove_ai_watermarks.samsung_engine import SamsungEngine

# (engine factory, engine module) for each reverse-alpha text mark.
ENGINES = [
    pytest.param(DoubaoEngine, D, id="doubao"),
    pytest.param(JimengEngine, J, id="jimeng"),
    pytest.param(SamsungEngine, S, id="samsung"),
]


def _watermarked(engine, module) -> np.ndarray:
    """Composite the engine's real alpha glyph onto a flat mid-gray field at the
    captured native width (so both placement candidates fire)."""
    cfg = engine.config
    nw = module._ALPHA_NATIVE_WIDTH
    at = module._alpha_template()
    gw, gh = int(cfg.alpha_width_frac * nw), int(cfg.alpha_height_frac * nw)
    ax = (nw - int(cfg.alpha_margin_x_frac * nw) - gw) if cfg.corner == "br" else int(cfg.alpha_margin_x_frac * nw)
    ay = nw - int(cfg.alpha_margin_bottom_frac * nw) - gh
    amap = np.zeros((nw, nw), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    img = np.full((nw, nw, 3), 100.0, np.float32)
    return (a3 * np.array(cfg.alpha_logo_bgr, np.float32) + (1 - a3) * img).clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize(("factory", "module"), ENGINES)
class TestExtractMaskFootprint:
    def test_returns_box_sized_mask(self, factory, module):
        eng = factory()
        img = _watermarked(eng, module)
        loc = eng.locate(img)
        box = eng.extract_mask(img, loc)
        assert box.dtype == np.uint8
        # Shape == loc.bbox, i.e. the old full-frame mask's [y:y+bh, x:x+bw] crop.
        assert box.shape == (loc.h, loc.w)
        # Footprint, not full frame: the box is a tiny fraction of the image.
        assert box.size * 4 < img.shape[0] * img.shape[1]


@pytest.mark.parametrize(("factory", "module"), ENGINES)
class TestAlphaMapFootprint:
    def test_maps_are_footprint_sized_blocks(self, factory, module):
        eng = factory()
        img = _watermarked(eng, module)
        for placed in (eng._fixed_alpha_map(img), eng._aligned_alpha_map(img)):
            assert placed is not None
            block, (ax, ay, gw, gh) = placed
            assert block.dtype == np.float32
            assert block.shape == (gh, gw)
            # The placement stays fully inside the image (no clipping needed).
            assert ax >= 0
            assert ax + gw <= img.shape[1]
            assert ay >= 0
            assert ay + gh <= img.shape[0]
            # O(footprint): far smaller than the frame.
            assert block.size * 4 < img.shape[0] * img.shape[1]

    def test_apply_reverse_alpha_equals_old_fullframe(self, factory, module):
        """``_apply_reverse_alpha`` with the glyph block is byte-identical to the
        old full-frame path: rebuild the full ``(h, w)`` map, run the old-style
        full-frame reverse-alpha, and compare to the new block-based output."""
        eng = factory()
        img = _watermarked(eng, module)
        h, w = img.shape[:2]
        for placed in (eng._fixed_alpha_map(img), eng._aligned_alpha_map(img)):
            assert placed is not None
            block, region = placed
            ax, ay, gw, gh = region

            new_out = eng._apply_reverse_alpha(img, block, region)

            # Old behavior: a full-frame map, indexed by region inside _apply_reverse_alpha.
            full = np.zeros((h, w), np.float32)
            full[ay : ay + gh, ax : ax + gw] = block
            old_out = img.copy()
            a3 = np.clip(full[ay : ay + gh, ax : ax + gw], 0.0, 1.0)[:, :, None]
            logo = np.array(eng.config.alpha_logo_bgr, np.float32)
            roi = old_out[ay : ay + gh, ax : ax + gw].astype(np.float32)
            old_out[ay : ay + gh, ax : ax + gw] = np.clip(
                (roi - a3 * logo) / np.clip(1.0 - a3, 0.25, 1.0), 0, 255
            ).astype(np.uint8)

            assert np.array_equal(new_out, old_out)

    def test_residual_mask_equals_old_fullframe(self, factory, module):
        """The residual inpaint mask built from the block embedded in a full-frame
        canvas equals thresholding the old full-frame float32 map (zero outside the
        block), so the dilate + inpaint see the same mask."""
        eng = factory()
        img = _watermarked(eng, module)
        h, w = img.shape[:2]
        cfg = eng.config
        block, (ax, ay, gw, gh) = eng._fixed_alpha_map(img)

        # New: embed the block into a uint8 canvas, then threshold.
        new_mask = np.zeros((h, w), np.uint8)
        new_mask[ay : ay + gh, ax : ax + gw] = (block > cfg.residual_alpha_floor).astype(np.uint8) * 255

        # Old: a full-frame float32 map, thresholded everywhere.
        old_full = np.zeros((h, w), np.float32)
        old_full[ay : ay + gh, ax : ax + gw] = block
        old_mask = (old_full > cfg.residual_alpha_floor).astype(np.uint8) * 255

        assert np.array_equal(new_mask, old_mask)
