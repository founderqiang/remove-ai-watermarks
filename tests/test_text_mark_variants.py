"""Synthetic regressions for independently gated text-mark layouts."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks.doubao_engine import DoubaoEngine, _OutlinedDoubaoEngine
from remove_ai_watermarks.microsoft_engine import MicrosoftEngine
from remove_ai_watermarks.samsung_engine import SamsungEngine
from remove_ai_watermarks.yuanbao_engine import YuanbaoEngine


def _stamp_variant(engine, variant_index: int, *, width: int = 1280, height: int = 960):
    variant = engine._variants[variant_index]
    config = variant.config
    background = 90 if config.asset_name != "microsoft_text_alpha.png" else 220
    foreground = 235 if background < 128 else 35
    clean = np.full((height, width, 3), background, np.uint8)
    alpha = _text_mark_engine.load_alpha_template(config.asset_name)
    assert alpha is not None
    base = variant.scale_base(clean)
    glyph_width = int(config.alpha_width_frac * base)
    glyph_height = int(config.alpha_height_frac * base)
    alpha = cv2.resize(alpha, (glyph_width, glyph_height), interpolation=cv2.INTER_LINEAR)
    loc = variant.locate(clean)
    x = loc.x if config.corner in {"bl", "tl"} else loc.x + loc.w - glyph_width
    y = loc.y if config.corner in {"tl", "tr"} else loc.y + loc.h - glyph_height
    if config.asset_name == "samsung_ko_alpha.png":
        x += round(1.3 * glyph_height)
    marked = clean.copy()
    if config.asset_name == "microsoft_text_alpha.png":
        pad_x, pad_y = int(width * 0.012), int(width * 0.014)
        cv2.rectangle(
            marked,
            (x - pad_x, y - pad_y),
            (x + glyph_width + pad_x, y + glyph_height + pad_y),
            (242, 238, 232),
            -1,
        )
    region = marked[y : y + glyph_height, x : x + glyph_width].astype(np.float32)
    opacity = alpha[:, :, None]
    marked[y : y + glyph_height, x : x + glyph_width] = np.clip(
        region * (1.0 - opacity) + foreground * opacity, 0, 255
    ).astype(np.uint8)
    return clean, marked, (x, y, glyph_width, glyph_height)


@pytest.mark.parametrize(
    ("engine", "asset"),
    [
        (DoubaoEngine(), "doubao_outline_alpha.png"),
        (YuanbaoEngine(), "yuanbao_compact_alpha.png"),
        (SamsungEngine(), "samsung_ko_alpha.png"),
        (MicrosoftEngine(), "microsoft_text_alpha.png"),
    ],
)
def test_variant_detection_retains_winning_asset(engine, asset: str) -> None:
    _clean, marked, _box = _stamp_variant(engine, 0)
    detection = engine.detect(marked)
    assert detection.detected
    assert detection.template_asset == asset


def test_microsoft_text_variant_masks_the_enclosing_pill() -> None:
    clean, marked, _box = _stamp_variant(MicrosoftEngine(), 0)
    detection = MicrosoftEngine().detect(marked)
    mask = MicrosoftEngine().footprint_mask(marked, detection=detection)
    assert mask is not None
    changed = np.any(clean != marked, axis=2)
    assert np.count_nonzero(mask[changed]) / np.count_nonzero(changed) >= 0.98


def test_samsung_korean_variant_masks_the_leading_sparkle_area() -> None:
    _clean, marked, (x, y, _width, height) = _stamp_variant(SamsungEngine(), 0)
    detection = SamsungEngine().detect(marked)
    mask = SamsungEngine().footprint_mask(marked, detection=detection)
    assert mask is not None
    sparkle_x = x - height // 2
    sparkle_y = y + height // 2
    assert mask[sparkle_y, sparkle_x] == 255


def test_low_confidence_cjk_layouts_require_matching_provenance() -> None:
    doubao = DoubaoEngine()._variants[0].config
    yuanbao_contrast, yuanbao_gray = (variant.config for variant in YuanbaoEngine()._variants)
    assert (doubao.detect_ncc_threshold, doubao.provenance_ncc_factor) == (0.50, 0.25)
    assert (yuanbao_contrast.detect_ncc_threshold, yuanbao_contrast.provenance_ncc_factor) == (0.50, 0.62)
    assert (yuanbao_gray.detect_ncc_threshold, yuanbao_gray.provenance_ncc_factor) == (0.50, 0.50)


def test_doubao_provenance_does_not_confirm_a_colored_template_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean, marked, _box = _stamp_variant(DoubaoEngine(), 0)
    gray = cv2.cvtColor(marked, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blend = np.clip((gray - 90) / 145, 0, 1)[:, :, None]
    background = np.array([30, 60, 120], np.float32)
    foreground = np.array([60, 180, 250], np.float32)
    colored_shape = (background * (1 - blend) + foreground * blend).astype(np.uint8)

    assert not DoubaoEngine().detect(colored_shape, provenance=True).detected
    monkeypatch.setattr(_OutlinedDoubaoEngine, "_outline_pixels_ok", lambda _cls, _patch: True)
    assert DoubaoEngine().detect(colored_shape, provenance=True).detected


def test_workbuddy_provenance_relaxes_threshold_for_tencent_tc260_code() -> None:
    from remove_ai_watermarks._internal.tc260_producers import producer_codes_for_mark
    from remove_ai_watermarks.workbuddy_engine import WorkBuddyEngine

    wb = WorkBuddyEngine().config
    assert producer_codes_for_mark("workbuddy") == ("911101085636549482",)
    assert (wb.detect_ncc_threshold, wb.provenance_ncc_factor) == (0.60, 0.90)
