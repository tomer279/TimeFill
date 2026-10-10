"""WP2 tests: train-only FeatureScaler fit, round-trip, score-net smoke."""

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


def test_scaler_ignores_val_when_fitting() -> None:
    """Fit stats match train observed cells and ignore a shifted val set."""
    n_train, seq_len = 4, 8
    train_x = np.full(
        (n_train, seq_len, _N_FEATURES),
        2.0,
        dtype=np.float32,
    )
    train_m = np.ones_like(train_x, dtype=np.float32)
    # Val tensors intentionally unused: fit must not see them.
    _val_x = np.full((4, seq_len, _N_FEATURES), 1000.0, dtype=np.float32)
    _val_m = np.ones_like(_val_x, dtype=np.float32)
    del _val_x, _val_m

    scaler = FeatureScaler().fit(train_x, train_m)
    assert bool(jnp.allclose(scaler.mean, 2.0, atol=_STATS_ATOL))
    assert bool(jnp.allclose(scaler.std, 1.0, atol=_STATS_ATOL))


def test_missing_cells_do_not_pollute_stats() -> None:
    """Huge values under m==0 must not affect mean/std."""
    values = np.ones((2, 4, _N_FEATURES), dtype=np.float32)
    mask = np.ones_like(values, dtype=np.float32)
    values[:, 0, :] = 1.0e6
    mask[:, 0, :] = MASK_MISSING
    values = np.where(mask == MASK_OBSERVED, values, MISSING_VALUE_SENTINEL)

    scaler = FeatureScaler().fit(values, mask)
    assert bool(jnp.allclose(scaler.mean, 1.0, atol=_STATS_ATOL))
    assert bool(jnp.all(scaler.std >= 1.0 - _STATS_ATOL))


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
