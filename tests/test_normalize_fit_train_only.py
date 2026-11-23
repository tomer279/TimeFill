"""WP2/WP5 tests: train-only FeatureScaler fit, A8 leakage guards, smoke.

Functions
---------
test_scaler_fit_ignores_perturbed_held_out
    Train-slice stats unchanged after val/test are scaled by 1000.
test_missing_cells_with_nan_and_huge_do_not_pollute_stats
    Huge / NaN under ``m==0`` (no sentinel overwrite) match a fit that
    drops those cells.
test_inverse_transform_roundtrip_on_observed
    Observed cells round-trip through transform → inverse.
test_scaler_save_load_checkpoint_convention
    Scaler NPZ lives under checkpoints/ with the locked filename.
test_real_batch_normalize_score_net_smoke
    Train/val real batches normalize then score-net forward stays finite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.data.normalize import (
    FeatureScaler,
    default_scaler_filename,
    fit_scaler_on_train_split,
    resolve_scaler_path,
)
from src.data.real import (
    BeijingCsvLoader,
    BeijingLoadSpec,
    EntityAwareSplitter,
    RealDataPaths,
    RealWindowBuilder,
    SplitFractions,
    WindowSpec,
    default_beijing_features,
)
from src.data.schema import (
    FEATURE_AXIS,
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
)
from src.models.score_net import ConditionalScoreNet, ScoreNetConfig
from src.train.artifacts import resolve_artifact_paths

_SEQ_LEN = 8
_N_HOURS = 24
_SEED = 0
_FEATURES = default_beijing_features()
_N_FEATURES = len(_FEATURES)
_GAP_FEATURE_IDX = 0
_ROUNDTRIP_ATOL = 1e-4
_STATS_ATOL = 1e-5
_DIFFUSION_TIME = 0.5
_BATCH_TAKE = 2
_N_TRAIN = 4
_N_HELD_OUT = 4
_TRAIN_VALUE = 2.0
_HELD_OUT_VALUE = 5.0
_PERTURB_SCALE = 1000.0
_HUGE_MISSING = 1.0e6


def _write_site_csv(
        path: Path,
        station: str,
        n_hours: int,
        gap_hour: int,
) -> None:
    """Write a tiny PRSA-style CSV for hermetic tests.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    station : str
        Site / entity id.
    n_hours : int
        Number of hourly rows starting at 2013-03-01 00:00.
    gap_hour : int
        Hour index at which feature 0 is written as ``NA``.
    """
    fieldnames = [
        "No",
        "year",
        "month",
        "day",
        "hour",
        *list(_FEATURES),
        "station",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for hour_idx in range(n_hours):
            row = {
                "No": str(hour_idx + 1),
                "year": "2013",
                "month": "3",
                "day": "1",
                "hour": str(hour_idx),
                "station": station,
            }
            for feat_idx, name in enumerate(_FEATURES):
                if hour_idx == gap_hour and feat_idx == _GAP_FEATURE_IDX:
                    row[name] = "NA"
                else:
                    row[name] = str(float(10 + feat_idx + 0.1 * hour_idx))
            writer.writerow(row)


@pytest.fixture
def window_dataset(tmp_path: Path):
    """Hermetic Beijing windows from tiny site CSVs."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_site_csv(raw_dir / "site_a.csv", "SiteA", _N_HOURS, gap_hour=2)
    _write_site_csv(raw_dir / "site_b.csv", "SiteB", _N_HOURS, gap_hour=5)
    _write_site_csv(raw_dir / "site_c.csv", "SiteC", _N_HOURS, gap_hour=1)
    paths = RealDataPaths(
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
    )
    sites = BeijingCsvLoader(BeijingLoadSpec()).load_directory(paths)
    builder = RealWindowBuilder(
        WindowSpec(seq_len=_SEQ_LEN, stride=_SEQ_LEN)
    )
    return builder.build(sites)


def test_scaler_fit_ignores_perturbed_held_out() -> None:
    """``fit_scaler_on_train_split`` ignores ×1000 held-out windows (A8).

    Notes
    -----
    Mutation check (walkthrough): temporarily change
    ``fit_scaler_on_train_split`` to fit all windows and expect failure.
    """
    from src.data.real.windows import RealWindowDataset

    rng = np.random.default_rng(_SEED)
    train_x = rng.normal(
        loc=_TRAIN_VALUE,
        scale=1.0,
        size=(_N_TRAIN, _SEQ_LEN, _N_FEATURES),
    ).astype(np.float32)
    held_x = rng.normal(
        loc=_HELD_OUT_VALUE,
        scale=1.0,
        size=(_N_HELD_OUT, _SEQ_LEN, _N_FEATURES),
    ).astype(np.float32)
    values = np.concatenate([train_x, held_x], axis=0)
    masks = np.ones_like(values, dtype=np.float32)
    entity_ids = np.arange(values.shape[0], dtype=np.int64)
    dataset = RealWindowDataset(
        values=values,
        native_masks=masks,
        entity_ids=entity_ids,
    )
    train_idx = np.arange(_N_TRAIN, dtype=np.int64)
    baseline = fit_scaler_on_train_split(dataset, train_idx)

    perturbed = np.array(values, copy=True)
    perturbed[_N_TRAIN:] *= _PERTURB_SCALE
    perturbed_dataset = RealWindowDataset(
        values=perturbed,
        native_masks=masks,
        entity_ids=entity_ids,
    )
    refit = fit_scaler_on_train_split(perturbed_dataset, train_idx)
    assert bool(jnp.allclose(refit.mean, baseline.mean, atol=_STATS_ATOL))
    assert bool(jnp.allclose(refit.std, baseline.std, atol=_STATS_ATOL))
    leaked = FeatureScaler().fit(
        perturbed_dataset.values,
        perturbed_dataset.native_masks,
    )
    assert float(jnp.max(jnp.abs(leaked.mean - baseline.mean))) > 1.0


def test_missing_cells_with_nan_and_huge_do_not_pollute_stats() -> None:
    """Huge values and NaN under ``m==0`` must not enter moments (A8)."""
    values = np.ones((2, 4, _N_FEATURES), dtype=np.float32)
    mask = np.ones_like(values, dtype=np.float32)
    values[:, 0, :] = _HUGE_MISSING
    values[:, 1, :] = np.nan
    mask[:, 0, :] = MASK_MISSING
    mask[:, 1, :] = MASK_MISSING
    # Intentionally leave NaN / huge under missing (no sentinel overwrite).

    # Same moments as fitting only the observed time steps (all ones).
    clean_values = np.ones((2, 2, _N_FEATURES), dtype=np.float32)
    clean_mask = np.ones_like(clean_values, dtype=np.float32)
    expected = FeatureScaler().fit(clean_values, clean_mask)

    scaler = FeatureScaler().fit(values, mask)
    assert bool(jnp.allclose(scaler.mean, expected.mean, atol=_STATS_ATOL))
    assert bool(jnp.allclose(scaler.std, expected.std, atol=_STATS_ATOL))
    assert bool(jnp.all(jnp.isfinite(scaler.mean)))
    assert bool(jnp.all(jnp.isfinite(scaler.std)))


def test_inverse_transform_roundtrip_on_observed() -> None:
    """Observed cells round-trip through transform → inverse_transform."""
    values = np.linspace(
        0.0,
        1.0,
        num=2 * 4 * _N_FEATURES,
        dtype=np.float32,
    ).reshape(2, 4, _N_FEATURES)
    mask = np.ones_like(values, dtype=np.float32)
    mask[:, -1, :] = MASK_MISSING
    values = np.where(mask == MASK_OBSERVED, values, MISSING_VALUE_SENTINEL)

    scaler = FeatureScaler().fit(values, mask)
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


def test_scaler_save_load_checkpoint_convention(tmp_path: Path) -> None:
    """Scaler NPZ lives under checkpoints/ with the locked filename."""
    paths = resolve_artifact_paths(str(tmp_path / "artifacts" / "m2"))
    paths.ensure()
    scaler_path = resolve_scaler_path(paths.checkpoint_dir())
    assert scaler_path.name == default_scaler_filename()

    values = np.arange(2 * 4 * _N_FEATURES, dtype=np.float32).reshape(
        2, 4, _N_FEATURES
    )
    mask = np.ones_like(values, dtype=np.float32)
    original = FeatureScaler().fit(values, mask)
    original.save(scaler_path)
    loaded = FeatureScaler.load(scaler_path)
    assert bool(jnp.allclose(loaded.mean, original.mean))
    assert bool(jnp.allclose(loaded.std, original.std))


def test_real_batch_normalize_score_net_smoke(window_dataset) -> None:
    """Train/val real batches normalize then score-net forward stays finite."""
    splitter = EntityAwareSplitter(
        SplitFractions(0.70, 0.15, 0.15, seed=_SEED)
    )
    split = splitter.split_by_index(window_dataset.shape()[0])
    scaler = fit_scaler_on_train_split(window_dataset, split.train_idx)

    n_train = min(_BATCH_TAKE, int(split.train_idx.size))
    train_batch = window_dataset.as_batch(split.train_idx[:n_train])
    if split.val_idx.size > 0:
        n_val = min(_BATCH_TAKE, int(split.val_idx.size))
        val_batch = window_dataset.as_batch(split.val_idx[:n_val])
    else:
        val_batch = train_batch

    x_train = scaler.transform(train_batch.x, train_batch.m)
    x_val = scaler.transform(val_batch.x, val_batch.m)
    assert bool(jnp.all(jnp.isfinite(x_train)))
    assert bool(jnp.all(jnp.isfinite(x_val)))
    assert int(x_train.shape[FEATURE_AXIS]) == _N_FEATURES

    net = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        ScoreNetConfig(n_features=_N_FEATURES),
    )
    score = net(x_val, _DIFFUSION_TIME, val_batch.m, x_val)
    assert score.shape == x_val.shape
    assert bool(jnp.all(jnp.isfinite(score)))
