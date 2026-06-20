"""Reverse-alpha over-subtraction guard for the visible text-mark engines.

Ported from the Gemini sparkle fix (commit 41f6797) to Doubao/Jimeng/Samsung
(retained-corpus mining 2026-06-20, roadmap P0#8): on a dark or mid-tone
background the captured alpha can over-estimate THIS image's mark opacity, and
reverse-alpha leaves a darker-than-background glyph ghost (a "dark pit") instead
of recovering the true pixels. The guard predicts the reverse-alpha output per
pixel and, when the glyph body lands far below the local ring, reconstructs the
footprint from the original surroundings instead of shipping the pit.

These assert visual residual (pixel levels vs the local background), not just a
detector re-fire -- a dark pit can clear the NCC detector while still looking wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from remove_ai_watermarks import image_io
from remove_ai_watermarks._text_mark_engine import _OVERSUB_DARK_MARGIN
from remove_ai_watermarks.doubao_engine import DoubaoEngine
from remove_ai_watermarks.jimeng_engine import JimengEngine
from remove_ai_watermarks.samsung_engine import SamsungEngine

_ENGINES = [DoubaoEngine, JimengEngine, SamsungEngine]


def _compose(engine, bg: float, opacity_gain: float, w: int = 1024, h: int = 1024):
    """Composite the engine's captured mark onto a flat ``bg`` at ``opacity_gain``.

    ``opacity_gain < 1`` makes the mark FAINTER than the capture, so reverse-alpha
    at the full captured alpha over-subtracts into a dark pit -- the case the guard
    must catch. Returns ``(watermarked_uint8, alpha_block, region)`` where the block
    and region are exactly what the engine's reverse-alpha receives.
    """
    img = np.full((h, w, 3), float(bg), np.float32)
    block, (ax, ay, gw, gh) = engine._fixed_alpha_map(img)
    a = np.clip(block * opacity_gain, 0.0, 0.99)[:, :, None]
    logo = np.array(engine.config.alpha_logo_bgr, np.float32)
    img[ay : ay + gh, ax : ax + gw] = img[ay : ay + gh, ax : ax + gw] * (1 - a) + logo * a
    return np.clip(img, 0, 255).astype(np.uint8), block, (ax, ay, gw, gh)


def _body_vs_ring(out, region, block) -> tuple[float, float]:
    """Median luma of the glyph body vs the local background ring in ``out``."""
    ax, ay, gw, gh = region
    g = out.astype(np.float32).mean(axis=2)
    body = block >= 0.15
    pad = max(4, int(gh * 0.6))
    ry1, ry2 = max(0, ay - pad), min(g.shape[0], ay + gh + pad)
    rx1, rx2 = max(0, ax - pad), min(g.shape[1], ax + gw + pad)
    ring = g[ry1:ry2, rx1:rx2]
    fy1, fy2, fx1, fx2 = ay - ry1, ay - ry1 + gh, ax - rx1, ax - rx1 + gw
    ring_mask = np.ones(ring.shape, dtype=bool)
    ring_mask[fy1:fy2, fx1:fx2] = False
    core = float(np.median(g[ay : ay + gh, ax : ax + gw][body]))
    return core, float(np.median(ring[ring_mask]))


@pytest.mark.parametrize("Engine", _ENGINES, ids=lambda e: e.__name__)
class TestOversubtractionGuard:
    @pytest.mark.parametrize(("bg", "gain"), [(120, 0.45), (150, 0.4), (90, 0.5)])
    def test_guard_trips_on_faint_mark(self, Engine, bg, gain):
        eng = Engine()
        wm, block, region = _compose(eng, bg, gain)
        assert eng._reverse_alpha_oversubtracts(image_io.to_bgr(wm), block, region)

    @pytest.mark.parametrize("bg", [255, 200, 128, 60])
    def test_guard_skips_clean_full_strength_mark(self, Engine, bg):
        # A cleanly captured (gain 1.0) mark predicts back to the background, so the
        # guard must NOT trip -- no regression of the common clean-removal path.
        eng = Engine()
        wm, block, region = _compose(eng, bg, 1.0)
        assert not eng._reverse_alpha_oversubtracts(image_io.to_bgr(wm), block, region)

    @pytest.mark.parametrize(("bg", "gain"), [(120, 0.45), (150, 0.4)])
    def test_faint_removal_leaves_no_dark_pit(self, Engine, bg, gain):
        # End-to-end acceptance (roadmap P0#8): after removal the glyph footprint is
        # not a region more than _OVERSUB_DARK_MARGIN below the local background.
        eng = Engine()
        wm, block, region = _compose(eng, bg, gain)
        out = eng.remove_watermark_reverse_alpha(wm)
        core, ring_bg = _body_vs_ring(out, region, block)
        assert core >= ring_bg - _OVERSUB_DARK_MARGIN, f"dark pit: body {core:.0f} vs ring {ring_bg:.0f}"

    def test_clean_mark_removal_unchanged_by_guard(self, Engine, monkeypatch):
        # On a clean mark the guard must be a no-op: forcing it off yields the same
        # output (the guard only ever diverts the over-subtraction case).
        eng = Engine()
        wm, _block, _region = _compose(eng, 200, 1.0)
        guarded = eng.remove_watermark_reverse_alpha(wm)
        monkeypatch.setattr(type(eng), "_reverse_alpha_oversubtracts", lambda self, *a, **k: False)
        unguarded = eng.remove_watermark_reverse_alpha(wm)
        assert np.array_equal(guarded, unguarded)


@pytest.mark.parametrize("Engine", _ENGINES, ids=lambda e: e.__name__)
def test_guard_recovers_pit_on_textured_background(Engine):
    """The guard's footprint inpaint reconstructs from the ORIGINAL surroundings,
    so a faint mark over-subtracted on a textured background recovers to roughly the
    local content level rather than a glyph-shaped dark ghost."""
    eng = Engine()
    w = h = 1024
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 120 + 35 * np.sin(xx / 80.0) + 25 * np.cos(yy / 60.0)
    bg_img = np.clip(np.stack([base, base * 0.95, base * 1.05], axis=-1), 0, 255)
    block, (ax, ay, gw, gh) = eng._fixed_alpha_map(bg_img)
    a = np.clip(block * 0.45, 0.0, 0.99)[:, :, None]
    logo = np.array(eng.config.alpha_logo_bgr, np.float32)
    bg_img[ay : ay + gh, ax : ax + gw] = bg_img[ay : ay + gh, ax : ax + gw] * (1 - a) + logo * a
    wm = np.clip(bg_img, 0, 255).astype(np.uint8)
    out = eng.remove_watermark_reverse_alpha(wm).astype(np.float32)
    # Compare the recovered glyph body to the clean texture under the mark.
    clean = np.clip(np.stack([base, base * 0.95, base * 1.05], axis=-1), 0, 255)
    body = block >= 0.15
    region_out = out[ay : ay + gh, ax : ax + gw].mean(axis=2)
    region_clean = clean[ay : ay + gh, ax : ax + gw].mean(axis=2)
    err = float(np.abs(region_out[body] - region_clean[body]).mean())
    assert err < 25.0, f"glyph body not recovered (mean abs err {err:.1f})"
