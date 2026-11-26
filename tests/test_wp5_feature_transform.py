"""WP5 tests: log1p/clip scaler, empty-feature policy, per-feature eval write.

Functions
---------
test_log1p_transform_inverse_roundtrip_on_observed
    log1p features round-trip through transform → inverse.
test_clip_nonnegative_on_restore
    Negative restored values clip to 0 when enabled.
test_legacy_scaler_npz_loads_without_transform_keys
    NPZ with only mean/std/eps loads as no-transform.
test_empty_feature_raises_when_fallback_disabled
    Zero train observations raise when fallback is false.
test_empty_feature_falls_back_when_legacy_enabled
    Zero observations keep identity mean 0 / std 1 when fallback is true.
test_per_feature_mae_cell_weighted_equals_pooled
    Cell-count-weighted per-feature MAE equals pooled masked MAE.
test_benchmark_writes_per_feature_csv
    Artifact paths expose ``per_feature.csv`` and a writer creates it.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from src.data.normalize import FeatureScaler, FeatureScalerOptions
from src.data.schema import (
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
)
from src.eval.m2_val_impute import OriginalScaleRestorer
from src.eval.metrics import (
    EvalTensors,
    empirical_crps_per_feature,
    masked_mae,
    masked_mae_per_feature,
    masked_rmse_per_feature,
)
from src.eval.m3_panels import M3ArtifactPaths
from src.eval.m3_table import MetricRecord, ProtocolKey, RunMeta, ScoreBlock, SeedPair

_SEQ = 4
_N_WINDOWS = 2
_ROUNDTRIP_ATOL = 1e-4
_STATS_ATOL = 1e-5
_FEATURE_NAMES = ("linear_f", "skew_f")
_LOG1P_FEATURES = ("skew_f",)
_LINEAR_IDX = 0
_SKEW_IDX = 1
_BATCH = 1
_K = 2
_FEAT = 2


def _build_scaler(**kwargs) -> FeatureScaler:
    """Construct a ``FeatureScaler`` via ``FeatureScalerOptions``.

    Parameters
    ----------
    **kwargs
        Fields forwarded to ``FeatureScalerOptions``.

    Returns
    -------
    FeatureScaler
        Constructed scaler.

    Raises
    ------
    pytest.fail.Exception
        When the WP5 options API is missing.
    """
    try:
        return FeatureScaler(options=FeatureScalerOptions(**kwargs))
    except TypeError as exc:
        pytest.fail(f"WP5 FeatureScalerOptions API missing: {exc}")


def test_log1p_transform_inverse_roundtrip_on_observed() -> None:
    """Observed cells round-trip with per-feature log1p before z-score."""
    values = np.asarray(
        [
            [[1.0, 2.0], [3.0, 8.0], [5.0, 15.0], [0.0, 0.0]],
            [[2.0, 4.0], [4.0, 10.0], [6.0, 20.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    mask = np.ones_like(values, dtype=np.float32)
    mask[:, -1, :] = MASK_MISSING
    values = np.where(mask == MASK_OBSERVED, values, MISSING_VALUE_SENTINEL)

    scaler = _build_scaler(
        log1p_features=_LOG1P_FEATURES,
        feature_names=_FEATURE_NAMES,
        clip_nonnegative=False,
        allow_empty_feature_fallback=True,
    )
    scaler.fit(values, mask)
    normalized = scaler.transform(values, mask)
    restored = scaler.inverse_transform(normalized, mask)

    observed = mask == MASK_OBSERVED
    assert bool(
        jnp.allclose(
            restored[observed],
            jnp.asarray(values)[observed],
            atol=_ROUNDTRIP_ATOL,
        )
    )
    missing = mask == MASK_MISSING
    assert bool(jnp.all(normalized[missing] == MISSING_VALUE_SENTINEL))
    assert bool(jnp.all(restored[missing] == MISSING_VALUE_SENTINEL))


def test_clip_nonnegative_on_restore() -> None:
    """Negative restored values are clipped to 0 when enabled."""
    values = np.asarray(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=np.float32,
    )
    mask = np.ones_like(values, dtype=np.float32)
    scaler = _build_scaler(
        log1p_features=(),
        feature_names=_FEATURE_NAMES,
        clip_nonnegative=True,
        allow_empty_feature_fallback=True,
    )
    scaler.fit(values, mask)
    # Large negative z → negative original units before clip.
    neg_norm = jnp.asarray(
        [[[-50.0, -50.0], [-50.0, -50.0]]],
        dtype=jnp.float32,
    )
    restored = scaler.inverse_transform(neg_norm, mask)
    assert bool(jnp.all(restored >= 0.0))
    assert bool(jnp.allclose(restored, 0.0, atol=_STATS_ATOL))


def test_legacy_scaler_npz_loads_without_transform_keys(
        tmp_path: Path,
) -> None:
    """NPZ with only mean/std/eps loads as no transform / no clip."""
    mean = np.asarray([1.0, 2.0], dtype=np.float32)
    std = np.asarray([0.5, 1.5], dtype=np.float32)
    path = tmp_path / "feature_scaler.npz"
    np.savez_compressed(path, mean=mean, std=std, eps=np.float32(1e-6))

    loaded = FeatureScaler.load(path)
    assert bool(jnp.allclose(loaded.mean, mean))
    assert bool(jnp.allclose(loaded.std, std))
    assert tuple(loaded.log1p_features) == ()
    assert not bool(np.any(np.asarray(loaded.log1p_mask)))
    assert loaded.clip_nonnegative is False


def test_empty_feature_raises_when_fallback_disabled() -> None:
    """Zero observed cells in the fit slice raise when fallback is false."""
    values = np.ones((_N_WINDOWS, _SEQ, 2), dtype=np.float32)
    mask = np.ones_like(values, dtype=np.float32)
    mask[:, :, _SKEW_IDX] = MASK_MISSING
    values[:, :, _SKEW_IDX] = MISSING_VALUE_SENTINEL

    scaler = _build_scaler(
        allow_empty_feature_fallback=False,
        feature_names=_FEATURE_NAMES,
        fit_slice_label="train split",
    )
    with pytest.raises(ValueError, match="train split|skew_f"):
        scaler.fit(values, mask)


def test_empty_feature_falls_back_when_legacy_enabled() -> None:
    """Legacy fallback keeps identity stats on an empty feature."""
    values = np.ones((_N_WINDOWS, _SEQ, 2), dtype=np.float32)
    mask = np.ones_like(values, dtype=np.float32)
    mask[:, :, _SKEW_IDX] = MASK_MISSING
    values[:, :, _SKEW_IDX] = MISSING_VALUE_SENTINEL

    scaler = _build_scaler(
        allow_empty_feature_fallback=True,
        feature_names=_FEATURE_NAMES,
        fit_slice_label="train split",
    )
    scaler.fit(values, mask)
    assert float(scaler.mean[_SKEW_IDX]) == pytest.approx(0.0, abs=_STATS_ATOL)
    assert float(scaler.std[_SKEW_IDX]) == pytest.approx(1.0, abs=_STATS_ATOL)
    assert float(scaler.mean[_LINEAR_IDX]) == pytest.approx(
        1.0, abs=_STATS_ATOL
    )


def _asymmetric_target_tensors() -> EvalTensors:
    """Tensors with different per-feature target errors."""
    truth = jnp.asarray(
        [[[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]]],
        dtype=jnp.float32,
    )
    mean = jnp.asarray(
        [[[0.0, 11.0], [1.0, 22.0], [2.0, 33.0]]],
        dtype=jnp.float32,
    )
    mask = jnp.asarray(
        [
            [
                [MASK_OBSERVED, MASK_MISSING],
                [MASK_MISSING, MASK_MISSING],
                [MASK_OBSERVED, MASK_OBSERVED],
            ]
        ],
        dtype=jnp.float32,
    )
    samples = jnp.broadcast_to(
        mean[:, None, :, :],
        (_BATCH, _K, truth.shape[1], _FEAT),
    )
    return EvalTensors(
        truth=truth,
        mean=mean,
        samples=samples,
        mask=mask,
    )


def test_per_feature_mae_cell_weighted_equals_pooled() -> None:
    """Cell-count-weighted per-feature MAE equals pooled masked MAE."""
    tensors = _asymmetric_target_tensors()
    per_feature = masked_mae_per_feature(tensors)
    weights = tensors.missing_weights()
    counts = jnp.sum(weights, axis=(0, 1))
    pooled_from_parts = jnp.sum(per_feature * counts) / jnp.sum(counts)
    assert float(pooled_from_parts) == pytest.approx(
        float(masked_mae(tensors)),
        abs=_STATS_ATOL,
    )


def test_per_feature_rmse_and_crps_differ_across_features() -> None:
    """Per-feature RMSE/CRPS are not a pooled value copied onto each row."""
    tensors = _asymmetric_target_tensors()
    per_rmse = masked_rmse_per_feature(tensors)
    per_crps = empirical_crps_per_feature(tensors)
    assert float(per_rmse[0]) != pytest.approx(float(per_rmse[1]), abs=1e-6)
    assert float(per_crps[0]) != pytest.approx(float(per_crps[1]), abs=1e-6)


def test_log1p_mean_after_restore_not_expm1_of_norm_mean() -> None:
    """Original-unit mean is mean(expm1(samples)), not expm1(mean)."""
    values = np.asarray([[[1.0, 2.0], [3.0, 8.0]]], dtype=np.float32)
    mask = np.ones_like(values, dtype=np.float32)
    scaler = _build_scaler(
        log1p_features=_LOG1P_FEATURES,
        feature_names=_FEATURE_NAMES,
    ).fit(values, mask)
    samples_n = jnp.asarray(
        [[[[0.0, -1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, -1.0]]]],
        dtype=jnp.float32,
    )
    restorer = OriginalScaleRestorer(scaler)
    mean_from_samples, samples = restorer.restore_mean_from_samples(
        samples_n, mask
    )
    wrong = restorer.restore_windows(
        jnp.mean(samples_n, axis=1),
        mask,
    )
    skew_gap = float(
        jnp.max(jnp.abs(mean_from_samples[..., _SKEW_IDX] - wrong[..., _SKEW_IDX]))
    )
    assert skew_gap > 1e-3
    assert bool(
        jnp.allclose(
            mean_from_samples,
            jnp.mean(samples, axis=1),
            atol=_STATS_ATOL,
        )
    )


def test_original_unit_means_expm1_log1p_features() -> None:
    """Baseline means use ``expm1`` on log1p features."""
    values = np.asarray([[[1.0, np.e - 1.0]]], dtype=np.float32)
    mask = np.ones_like(values, dtype=np.float32)
    scaler = _build_scaler(
        log1p_features=_LOG1P_FEATURES,
        feature_names=_FEATURE_NAMES,
    ).fit(values, mask)
    means = scaler.original_unit_means()
    assert float(means[_SKEW_IDX]) == pytest.approx(np.e - 1.0, abs=1e-4)
    assert float(means[_LINEAR_IDX]) == pytest.approx(1.0, abs=1e-4)


def test_feature_names_roundtrip_in_npz(tmp_path: Path) -> None:
    """Saved scaler NPZ restores real feature names."""
    values = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
    mask = np.ones_like(values, dtype=np.float32)
    scaler = _build_scaler(
        log1p_features=_LOG1P_FEATURES,
        feature_names=_FEATURE_NAMES,
    ).fit(values, mask)
    path = tmp_path / "feature_scaler.npz"
    scaler.save(path)
    loaded = FeatureScaler.load(path)
    assert loaded.feature_names == _FEATURE_NAMES
    assert loaded.log1p_features == _LOG1P_FEATURES


def test_metrics_by_rate_includes_norm_columns() -> None:
    """Long-form rows expose ``*_norm`` columns next to original metrics."""
    record = MetricRecord(
        method="timefill",
        protocol=ProtocolKey(name="mcar", mask_rate=0.1, block_len=None),
        scores=ScoreBlock(
            mae=1.0,
            rmse=2.0,
            crps=0.5,
            coverage_50=0.5,
            coverage_90=0.9,
        ),
        meta=RunMeta(
            seeds=SeedPair(mask=0, sample=1),
            split="test",
            k_samples=8,
            point_metrics="ensemble",
        ),
        n_target_cells=4,
        scores_norm=ScoreBlock(
            mae=0.1,
            rmse=0.2,
            crps=0.05,
            coverage_50=0.5,
            coverage_90=0.9,
        ),
    )
    payload = record.as_dict()
    assert payload["mae"] == 1.0
    assert payload["mae_norm"] == 0.1
    assert payload["rmse_norm"] == 0.2


def test_per_feature_rows_uses_names_and_differing_metrics() -> None:
    """``_per_feature_rows`` keeps real names and per-feature RMSE/CRPS."""
    from src.eval.m3_benchmark import M3BenchmarkRunner, MethodScoreInput
    from src.eval.m3_data import DataBundle, SplitWindowBundle

    tensors = _asymmetric_target_tensors()
    windows = SplitWindowBundle(
        values_n=jnp.zeros_like(tensors.truth),
        m_native=jnp.ones_like(tensors.truth),
        split_name="test",
    )
    scaler = _build_scaler(feature_names=_FEATURE_NAMES).fit(
        np.asarray([[[1.0, 2.0]]], dtype=np.float32),
        np.ones((1, 1, 2), dtype=np.float32),
    )
    runner = object.__new__(M3BenchmarkRunner)
    runner.data = DataBundle(windows=windows, scaler=scaler)
    runner.grid = type(
        "Grid",
        (),
        {
            "methods": ("timefill",),
            "point_estimate": "mean",
            "seeds": SeedPair(mask=0, sample=1),
        },
    )()
    pack = MethodScoreInput(
        truth=tensors.truth,
        mean=tensors.mean,
        samples=tensors.samples,
        metric_mask=tensors.mask,
    )
    rows = M3BenchmarkRunner._per_feature_rows(
        runner,
        ProtocolKey(name="mcar", mask_rate=0.3, block_len=None),
        {"timefill": pack},
    )
    assert [row["feature"] for row in rows] == list(_FEATURE_NAMES)
    assert rows[0]["rmse"] != pytest.approx(rows[1]["rmse"], abs=1e-6)
    assert rows[0]["crps"] != pytest.approx(rows[1]["crps"], abs=1e-6)


def test_data_bundle_falls_back_to_config_feature_names() -> None:
    """Legacy scaler without names uses bench-config feature names."""
    from src.eval.m3_data import DataBundle, SplitWindowBundle

    values_n = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    masks = jnp.ones_like(values_n)
    windows = SplitWindowBundle(
        values_n=values_n,
        m_native=masks,
        split_name="test",
    )
    scaler = FeatureScaler()
    scaler.mean = jnp.zeros((2,), dtype=jnp.float32)
    scaler.std = jnp.ones((2,), dtype=jnp.float32)
    scaler.log1p_mask = jnp.zeros((2,), dtype=bool)
    data = DataBundle(
        windows=windows,
        scaler=scaler,
        config_feature_names=_FEATURE_NAMES,
    )
    assert data.feature_names() == _FEATURE_NAMES


def test_benchmark_writes_per_feature_csv(tmp_path: Path) -> None:
    """``M3ArtifactPaths`` exposes per-feature CSV and a writer creates it."""
    from src.eval import m3_table as table_mod

    paths = M3ArtifactPaths(tmp_path)
    if not hasattr(paths, "per_feature_csv"):
        pytest.fail("WP5 M3ArtifactPaths.per_feature_csv is missing")
    out = paths.per_feature_csv()
    assert out.name == "per_feature.csv"

    write_fn = getattr(table_mod, "write_per_feature_csv", None)
    if write_fn is None:
        pytest.fail("WP5 write_per_feature_csv is missing")

    rows = [
        {
            "method": "timefill",
            "protocol": "mcar",
            "feature": "skew_f",
            "mae": 1.0,
            "rmse": 2.0,
            "crps": 0.5,
            "coverage_50": 0.5,
            "coverage_90": 0.9,
            "n_target_cells": 3,
        }
    ]
    written = write_fn(out, rows)
    path = Path(written) if written is not None else out
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "feature" in text
    assert "skew_f" in text
