"""A mark the `tophat` front-end DETECTS must also be MASKABLE.

Corpus-found 2026-07-20: Doubao's detection moved to the continuous `tophat` front-end
(which does not binarize, and that is where its recall 89% -> 92% came from), but the
removal mask still came from the BINARIZED glyph blob. A mark faint enough to be found
only by the continuous response therefore produced an empty binary blob, `localize`
returned mask=None, and `remove()` was a silent no-op: `identify` reported
`visible_doubao` while `visible` said "no visible mark" on the same file. Measured on the
full corpus parity sweep: 57 of 60 sampled still-detected Doubao marks were untouched
no-ops, ~8% of all Doubao detections.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.doubao_engine import DoubaoEngine


def _faint_mark_image(w: int = 900, h: int = 1200, alpha: float = 0.06) -> np.ndarray:
    """A mid-gray frame carrying the REAL Doubao glyph shape at very low opacity.

    The shape has to be genuine or the NCC detector will not fire and the test would be
    exercising nothing; the low ``alpha`` is what keeps the binarizing path from finding
    a blob. Composited with the same forward model the marks use:
    ``stamped = (1-a)*bg + a*white``.
    """
    from remove_ai_watermarks._text_mark_engine import load_alpha_template

    tmpl = load_alpha_template("doubao_alpha.png")
    if tmpl is None:
        pytest.skip("doubao alpha asset unavailable")
    img = np.full((h, w, 3), 120, np.uint8)
    eng = DoubaoEngine()
    loc = eng.locate(img)
    base = eng.scale_base(img)
    gw = max(eng.config.min_gw, int(eng.config.alpha_width_frac * base))
    gh = max(4, int(eng.config.alpha_height_frac * base))
    a = cv2.resize(tmpl, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) * alpha
    x = loc.x + (loc.w - gw) // 2
    y = loc.y + (loc.h - gh) // 2
    roi = img[y : y + gh, x : x + gw].astype(np.float32)
    a3 = a[..., None]
    img[y : y + gh, x : x + gw] = np.clip(roi * (1 - a3) + 255.0 * a3, 0, 255).astype(np.uint8)
    return img


class TestFaintMarkIsMaskable:
    def test_binary_glyph_blob_is_empty_on_a_faint_mark(self):
        """The premise: this is the input class the binarizing path cannot segment."""
        eng = DoubaoEngine()
        img = _faint_mark_image()
        loc = eng.locate(img)
        glyph = eng.extract_mask(img, loc)
        assert int((glyph > 0).sum()) < eng._MIN_GLYPH_PIXELS

    def test_footprint_mask_is_not_empty_when_the_continuous_response_has_signal(self):
        """The fix: a faint mark must still yield a removal mask, without --no-detect.

        Without it `localize` returns None and removal silently does nothing while
        `identify` keeps reporting the mark.
        """
        eng = DoubaoEngine()
        img = _faint_mark_image()
        mask = eng.footprint_mask(img, force=False)
        assert mask is not None, "a detectable faint mark produced no removal mask"
        assert int((mask > 0).sum()) > 0

    def test_a_clean_frame_still_produces_no_mask(self):
        """The guard: the fallback must not turn every flat corner into a fill."""
        eng = DoubaoEngine()
        clean = cv2.GaussianBlur(np.full((1200, 900, 3), 120, np.uint8), (5, 5), 0)
        assert eng.footprint_mask(clean, force=False) is None

    @pytest.mark.parametrize("alpha", [0.5, 0.9])
    def test_a_bold_mark_is_unaffected(self, alpha: float):
        """A mark the binary path already segments must keep its tight glyph box."""
        eng = DoubaoEngine()
        img = _faint_mark_image(alpha=alpha)
        mask = eng.footprint_mask(img, force=False)
        assert mask is not None
        assert int((mask > 0).sum()) > 0
