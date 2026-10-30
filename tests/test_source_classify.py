"""Tests for the abstaining 10-provider source-pipeline classifier (source_classify)."""

from __future__ import annotations

import json
import re
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, PngImagePlugin

import remove_ai_watermarks as raiw
from remove_ai_watermarks import source_classify
from remove_ai_watermarks.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def _write_synthetic_v3_npz(
    path: Path,
    *,
    schema: str = source_classify.MODEL_SCHEMA,
    target_class: str = "openai",
    target_global_bias: float = 4.0,
    target_sub_bias: float = -3.0,
    target_sub_alpha: float = 0.85,
    router_threshold: float = 1.0,
    stage2_win_idx: int = 1,
) -> Path:
    """Create a minimal valid pickle-free multi-provider-source-v3 .npz fixture."""
    d = source_classify.FEATURE_DIM
    payload: dict[str, Any] = {
        "schema": np.asarray(schema),
        "feature_dim": np.asarray(d, dtype=np.int64),
        "provider_classes": np.asarray(source_classify.ALL_PROVIDER_CLASSES, dtype="<U32"),
        "damping_gamma": np.asarray(0.60, dtype=np.float64),
        "mean": np.zeros(d, dtype=np.float64),
        "scale": np.ones(d, dtype=np.float64),
        "router_thresholds": np.full(len(source_classify.ALL_PROVIDER_CLASSES), router_threshold, dtype=np.float64),
    }

    def add_mlp(prefix: str, bias_val: float) -> None:
        payload[f"{prefix}_w0"] = np.zeros((4, d), dtype=np.float32)
        payload[f"{prefix}_b0"] = np.zeros(4, dtype=np.float32)
        payload[f"{prefix}_w1"] = np.zeros((2, 4), dtype=np.float32)
        payload[f"{prefix}_b1"] = np.zeros(2, dtype=np.float32)
        payload[f"{prefix}_w2"] = np.zeros((1, 2), dtype=np.float32)
        payload[f"{prefix}_b2"] = np.asarray([bias_val], dtype=np.float32)

    def add_unit(prefix: str, bias_val: float, *, is_sub: bool, alpha: float = 0.0) -> None:
        payload[f"{prefix}_blend_weights"] = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        payload[f"{prefix}_base_offset"] = np.asarray(0.0, dtype=np.float64)
        if is_sub:
            payload[f"{prefix}_pool_alpha"] = np.asarray(alpha, dtype=np.float64)
        payload[f"{prefix}_soup_count"] = np.asarray(1, dtype=np.int16)
        add_mlp(f"{prefix}_s0", bias_val)
        add_mlp(f"{prefix}_et_mlp", 0.0)
        payload[f"{prefix}_hgb_baseline"] = np.asarray(0.0, dtype=np.float64)
        payload[f"{prefix}_hgb_max_depth"] = np.asarray(1, dtype=np.int16)
        payload[f"{prefix}_hgb_feat_idx"] = np.zeros((1, 1), dtype=np.int16)
        payload[f"{prefix}_hgb_thresholds"] = np.asarray([[np.inf]], dtype=np.float64)
        payload[f"{prefix}_hgb_left"] = np.zeros((1, 1), dtype=np.uint8)
        payload[f"{prefix}_hgb_right"] = np.zeros((1, 1), dtype=np.uint8)
        payload[f"{prefix}_hgb_values"] = np.zeros((1, 1), dtype=np.float64)

    desc_classes: list[str] = []
    desc_names: list[str] = []
    for cls_name in source_classify.ALL_PROVIDER_CLASSES:
        is_target = cls_name == target_class
        g_bias = target_global_bias if is_target else -3.0
        add_unit(f"s1_{cls_name}_global", g_bias, is_sub=False)
        payload[f"s1_{cls_name}_pool_recenter"] = np.asarray(0.0, dtype=np.float64)
        desc_classes.append(cls_name)
        desc_names.append("__global__")

        sub_name = f"{cls_name}:v_sub_a"
        payload[f"s1_{cls_name}_sub_names"] = np.asarray([sub_name], dtype="<U48")
        s_bias = target_sub_bias if is_target else -3.0
        s_alpha = target_sub_alpha if is_target else 0.0
        add_unit(f"s1_{cls_name}_sub0", s_bias, is_sub=True, alpha=s_alpha)
        desc_classes.append(cls_name)
        desc_names.append(sub_name)

    payload["unit_descriptor_classes"] = np.asarray(desc_classes, dtype="<U32")
    payload["unit_descriptor_names"] = np.asarray(desc_names, dtype="<U48")

    payload["stage2_version_classes"] = np.asarray([target_class], dtype="<U32")
    subs = [f"{target_class}:v1", f"{target_class}:v2"]
    payload[f"s2_{target_class}_subfamilies"] = np.asarray(subs, dtype="<U48")
    base_s2 = np.asarray([-1.0, 1.0] if stage2_win_idx == 1 else [1.0, -1.0], dtype=np.float64)
    payload[f"s2_{target_class}_hgb_baseline"] = base_s2
    payload[f"s2_{target_class}_hgb_max_depth"] = np.asarray(1, dtype=np.int16)
    payload[f"s2_{target_class}_hgb_feat_idx"] = np.zeros((1, 2, 1), dtype=np.int16)
    payload[f"s2_{target_class}_hgb_thresholds"] = np.full((1, 2, 1), np.inf, dtype=np.float64)
    payload[f"s2_{target_class}_hgb_left"] = np.zeros((1, 2, 1), dtype=np.uint8)
    payload[f"s2_{target_class}_hgb_right"] = np.zeros((1, 2, 1), dtype=np.uint8)
    payload[f"s2_{target_class}_hgb_values"] = np.zeros((1, 2, 1), dtype=np.float64)

    np.savez(path, **payload)
    return path


def test_weights_revision_and_sha256_are_pinned_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", source_classify.WEIGHTS_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", source_classify.MODEL_SHA256)
    assert raiw.classify_source is source_classify.classify_source
    assert raiw.SourceClassification is source_classify.SourceClassification


def test_log_spectrum_discards_translation_phase() -> None:
    from remove_ai_watermarks._internal.source_spectral import phase_free_log_spectrum

    rng = np.random.default_rng(17)
    channel = rng.normal(size=(32, 32))

    original = phase_free_log_spectrum(channel)
    translated = phase_free_log_spectrum(np.roll(channel, shift=(5, -7), axis=(0, 1)))

    assert np.allclose(original, translated, rtol=1e-12, atol=1e-12)


def test_spectral_feature_is_finite_and_has_the_frozen_width() -> None:
    from remove_ai_watermarks._internal.source_spectral import feature_from_image, make_geometry

    rng = np.random.default_rng(20260916)
    pixels = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    feature = feature_from_image(Image.fromarray(pixels), make_geometry())

    assert feature.shape == (768,)
    assert np.all(np.isfinite(feature))


def test_augment_892d_to_956d_produces_finite_956d_for_1d_and_2d() -> None:
    rng = np.random.default_rng(42)
    vec_892 = rng.normal(size=892).astype(np.float64)
    out_1d = source_classify.augment_892d_to_956d(vec_892)
    assert out_1d.shape == (956,)
    assert np.all(np.isfinite(out_1d))

    mat_892 = np.stack([vec_892, vec_892 * 0.5], axis=0)
    out_2d = source_classify.augment_892d_to_956d(mat_892)
    assert out_2d.shape == (2, 956)
    np.testing.assert_allclose(out_2d[0], out_1d)

    with pytest.raises(ValueError, match="Expected 892-d"):
        source_classify.augment_892d_to_956d(np.zeros(100, dtype=np.float64))


def test_small_image_returns_feature_unavailable_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_load(_weights_path: Path | None = None) -> Any:
        raise AssertionError("weights must not load when image is smaller than 256x256")

    monkeypatch.setattr(source_classify, "_load_model", _unexpected_load)
    tiny = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    result = source_classify.classify_source(tiny)
    assert result.label == "unknown"
    assert result.reason == "feature_unavailable"
    assert result.version_subfamily is None
    assert result.scores == {}
    assert result.watermark_truth == "unknown"


def test_positive_part_subunit_pooling_and_threshold_mutation(tmp_path: Path) -> None:
    x_956 = np.zeros(source_classify.FEATURE_DIM, dtype=np.float64)

    # Case 1: global_bias = -1.5, sub_bias = -2.0 (negative sub-unit score must NOT clamp z to 0.0!)
    w_neg = _write_synthetic_v3_npz(
        tmp_path / "neg.npz",
        target_class="openai",
        target_global_bias=-1.5,
        target_sub_bias=-2.0,
        target_sub_alpha=1.0,
        router_threshold=0.5,
    )
    res_neg = source_classify.classify_source_features(x_956, weights_path=w_neg)
    assert res_neg.label == "unknown"
    assert res_neg.reason == "abstained"
    assert res_neg.pooled_scores["openai"] == pytest.approx(-1.5)

    # Case 2: global_bias = -1.5, sub_bias = +3.0, alpha = 0.8 -> pooled = +2.4 >= threshold 1.0
    w_pos = _write_synthetic_v3_npz(
        tmp_path / "pos.npz",
        target_class="openai",
        target_global_bias=-1.5,
        target_sub_bias=3.0,
        target_sub_alpha=0.8,
        router_threshold=1.0,
        stage2_win_idx=1,
    )
    res_pos = source_classify.classify_source_features(x_956, weights_path=w_pos)
    assert res_pos.label == "openai"
    assert res_pos.reason == "classified"
    assert res_pos.pooled_scores["openai"] == pytest.approx(2.4)
    assert res_pos.version_subfamily == "openai:v2"
    assert res_pos.fine_subfamily == "openai:v_sub_a"
    assert res_pos.watermark_truth == "unknown"

    # Case 3: raising router_threshold above 2.4 forces abstention
    w_hi = _write_synthetic_v3_npz(
        tmp_path / "hi.npz",
        target_class="openai",
        target_global_bias=-1.5,
        target_sub_bias=3.0,
        target_sub_alpha=0.8,
        router_threshold=3.0,
    )
    res_hi = source_classify.classify_source_features(x_956, weights_path=w_hi)
    assert res_hi.label == "unknown"
    assert res_hi.reason == "abstained"

    # Confidence must increase monotonically with AI evidence across abstained and classified regimes:
    # res_neg (excess = -2.0) < res_hi (excess = -0.6) < 0.5 <= res_pos (excess = +1.4)
    assert 0.0 < res_neg.confidence < res_hi.confidence < 0.5 <= res_pos.confidence < 1.0


def test_canonical_provider_routing_and_version_head(tmp_path: Path) -> None:
    x_956 = np.zeros(source_classify.FEATURE_DIM, dtype=np.float64)
    w_sd = _write_synthetic_v3_npz(
        tmp_path / "sd.npz",
        target_class="stable_diffusion",
        target_global_bias=3.5,
        router_threshold=1.0,
        stage2_win_idx=0,
    )
    res = source_classify.classify_source_features(x_956, weights_path=w_sd)
    assert res.label == "stable_diffusion"
    assert res.version_subfamily == "stable_diffusion:v1"
    assert set(res.pooled_scores) == set(source_classify.ALL_PROVIDER_CLASSES)


def test_stage15_hinge_meta_router_disambiguation(tmp_path: Path) -> None:
    x_956 = np.zeros(source_classify.FEATURE_DIM, dtype=np.float64)
    w_base = _write_synthetic_v3_npz(
        tmp_path / "base_meta.npz",
        target_class="openai",
        target_global_bias=2.0,
        target_sub_bias=2.5,
        target_sub_alpha=0.0,
        router_threshold=1.0,
    )
    # Without router_meta_w, openai wins with pooled_z = 2.0
    res_base = source_classify.classify_source_features(x_956, weights_path=w_base)
    assert res_base.label == "openai"

    # Add router_meta_tau and router_meta_w where openai:v_sub_a hinge (index 1, value 2.5 - 0.5 = 2.0)
    # suppresses openai (-1.0 * 2.0 = -2.0) and boosts firefly (+2.5 * 2.0 = +5.0 -> +2.0).
    with np.load(w_base, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    n_units = len(payload["unit_descriptor_classes"])
    n_cls = len(source_classify.ALL_PROVIDER_CLASSES)
    tau = np.zeros(n_units, dtype=np.float64)
    tau[1] = 0.5  # openai:v_sub_a has raw score 2.5 -> h_u[1] = 2.0
    w_meta = np.zeros((n_units + n_cls, n_cls), dtype=np.float64)
    openai_idx = source_classify.ALL_PROVIDER_CLASSES.index("openai")
    firefly_idx = source_classify.ALL_PROVIDER_CLASSES.index("firefly")
    w_meta[1, openai_idx] = -1.0  # delta = -2.0 -> pooled_z["openai"] = 0.0
    w_meta[1, firefly_idx] = 2.5  # delta = +5.0 -> pooled_z["firefly"] = -3.0 + 5.0 = +2.0
    payload["router_meta_tau"] = tau
    payload["router_meta_w"] = w_meta
    w_routed = tmp_path / "routed_meta.npz"
    np.savez(w_routed, **payload)

    res_routed = source_classify.classify_source_features(x_956, weights_path=w_routed)
    assert res_routed.label == "firefly"
    assert res_routed.pooled_scores["openai"] == pytest.approx(0.0)
    assert res_routed.pooled_scores["firefly"] == pytest.approx(2.0)


def test_metadata_does_not_change_pixel_decision(tmp_path: Path) -> None:
    weights = _write_synthetic_v3_npz(
        tmp_path / "meta.npz",
        target_class="openai",
        target_global_bias=3.2,
        router_threshold=1.0,
    )
    rng = np.random.default_rng(11)
    pixels = Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8), mode="RGB")
    clean = tmp_path / "clean.png"
    tagged = tmp_path / "tagged.png"
    pixels.save(clean)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "arbitrary metadata")
    pixels.save(tagged, pnginfo=metadata)

    first = source_classify.classify_source(clean, weights_path=weights)
    second = source_classify.classify_source(tagged, weights_path=weights)

    assert first.label == second.label == "openai"
    assert first.scores == second.scores


def test_load_model_rejects_bad_schema_and_hash_mismatch(tmp_path: Path) -> None:
    bad = _write_synthetic_v3_npz(tmp_path / "bad_schema.npz", schema="wrong-schema")
    with pytest.raises(ValueError, match="Unsupported source-classifier model schema"):
        source_classify._load_model(bad)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        source_classify._verify_sha256(bad, "0" * 64)


def test_cli_classify_source_json_and_text_and_missing_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img_path = tmp_path / "sample.png"
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(img_path)

    weights = _write_synthetic_v3_npz(
        tmp_path / "cli_v3.npz",
        target_class="google",
        target_global_bias=3.2,
        router_threshold=1.0,
        stage2_win_idx=1,
    )
    monkeypatch.setenv(source_classify.WEIGHTS_ENV, str(weights))

    runner = CliRunner()
    res_json = runner.invoke(main, ["classify-source", str(img_path), "--json"])
    assert res_json.exit_code == 0, res_json.output
    payload = json.loads(res_json.output)
    assert payload["label"] == "google"
    assert payload["version_subfamily"] == "google:v2"
    assert payload["watermark_truth"] == "unknown"

    res_text = runner.invoke(main, ["classify-source", str(img_path)])
    assert res_text.exit_code == 0, res_text.output
    assert "Label: google" in res_text.output
    assert "not SynthID detection" in res_text.output

    monkeypatch.setattr(source_classify, "is_available", lambda: False)
    res_missing = runner.invoke(main, ["classify-source", str(img_path)])
    assert res_missing.exit_code == 1
    assert "source-classify" in res_missing.output


def test_identify_does_not_import_source_classify(tmp_path: Path) -> None:
    from remove_ai_watermarks.identify import identify

    img_path = tmp_path / "plain.png"
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8), mode="RGB").save(img_path)
    sys.modules.pop("remove_ai_watermarks.source_classify", None)
    identify(img_path, check_visible=False, check_invisible=False)
    assert "remove_ai_watermarks.source_classify" not in sys.modules
