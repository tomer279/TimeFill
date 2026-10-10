"""WP1 tests: Beijing real loader, native masks, composition, splits."""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.data.real import (
    BeijingCsvLoader,
    BeijingLoadSpec,
    EntityAwareSplitter,
    MaskComposer,
    RealDataPaths,
    RealWindowBuilder,
    SplitFractions,
    WindowSpec,
    artificial_target_mask,
    compose_train_mask,
    default_beijing_features,
    load_processed_dataset,
    save_processed_dataset,
)
from src.data.schema import (
    FEATURE_AXIS,
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    TIME_AXIS,
)

_SEQ_LEN = 8
_N_HOURS = 24
_SEED = 0
_MASK_RATE = 0.5
_BLOCK_LEN = 3
_FEATURES = default_beijing_features()
_MASK_COMPARE_TOL = 1e-6
_GAP_FEATURE_IDX = 0


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
def tiny_raw_dir(tmp_path: Path) -> Path:
    """Create three tiny site CSVs under a temporary raw directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_site_csv(raw_dir / "site_a.csv", "SiteA", _N_HOURS, gap_hour=2)
    _write_site_csv(raw_dir / "site_b.csv", "SiteB", _N_HOURS, gap_hour=5)
    _write_site_csv(raw_dir / "site_c.csv", "SiteC", _N_HOURS, gap_hour=1)
    return raw_dir


@pytest.fixture
def window_dataset(tiny_raw_dir: Path):
    """Load fixture CSVs and build non-overlapping windows."""
    paths = RealDataPaths(
        raw_dir=tiny_raw_dir,
        processed_dir=tiny_raw_dir.parent / "processed",
    )
    sites = BeijingCsvLoader(BeijingLoadSpec()).load_directory(paths)
    builder = RealWindowBuilder(
        WindowSpec(seq_len=_SEQ_LEN, stride=_SEQ_LEN)
    )
    return builder.build(sites)


def test_window_shapes_b_t_f(window_dataset) -> None:
    """Stacked windows and batches use `(N/B, T, F)` with F=6."""
    n_windows, seq_len, n_features = window_dataset.shape()
    assert seq_len == _SEQ_LEN
    assert n_features == len(_FEATURES)
    assert n_windows > 0
    assert window_dataset.values.shape == (n_windows, seq_len, n_features)
    assert window_dataset.native_masks.shape == window_dataset.values.shape

    batch = window_dataset.as_batch(np.arange(min(2, n_windows)))
    assert batch.x.ndim == 3
    assert batch.x.shape[TIME_AXIS] == _SEQ_LEN
    assert batch.x.shape[FEATURE_AXIS] == len(_FEATURES)
    assert batch.m.shape == batch.x.shape


def test_native_missing_sentinel_and_mask(tiny_raw_dir: Path) -> None:
    """Native NA cells become sentinel 0.0 with m_native == 0."""
    path = tiny_raw_dir / "site_a.csv"
    series = BeijingCsvLoader(BeijingLoadSpec()).load_csv(path)
    gap_hour = 2
    assert series.native_mask[gap_hour, _GAP_FEATURE_IDX] == MASK_MISSING
    assert series.values[gap_hour, _GAP_FEATURE_IDX] == MISSING_VALUE_SENTINEL
    assert series.native_mask[gap_hour + 1, _GAP_FEATURE_IDX] == MASK_OBSERVED


def test_artificial_never_observes_native_missing(window_dataset) -> None:
    """Restricted artificial masks stay 0 wherever native is missing."""
    n_take = min(4, window_dataset.shape()[0])
    values = jnp.asarray(window_dataset.values[:n_take])
    m_native = jnp.asarray(window_dataset.native_masks[:n_take])
    composer = MaskComposer(rate=_MASK_RATE, block_len=_BLOCK_LEN)
    key = jax.random.PRNGKey(_SEED)
    _x_obs, m_train, m_art = composer.compose_random(key, values, m_native)

    native_missing = jnp.abs(m_native - MASK_MISSING) < _MASK_COMPARE_TOL
    assert bool(jnp.all(m_art[native_missing] == MASK_MISSING))
    assert bool(jnp.all(m_train[native_missing] == MASK_MISSING))

    expected = compose_train_mask(m_native, m_art)
    assert bool(jnp.allclose(m_train, expected))

    targets = artificial_target_mask(m_native, m_art)
    target_cells = targets == MASK_OBSERVED
    assert bool(jnp.all(m_native[target_cells] == MASK_OBSERVED))


def test_train_val_indices_disjoint(window_dataset) -> None:
    """Index and entity splits expose disjoint train/val/test indices."""
    fractions = SplitFractions(
        train=0.70,
        val=0.15,
        test=0.15,
        seed=_SEED,
    )
    splitter = EntityAwareSplitter(fractions)
    n_windows = window_dataset.shape()[0]
    index_split = splitter.split_by_index(n_windows)
    index_split.assert_disjoint()
    assert index_split.n_total() == n_windows

    entity_split = splitter.split_by_entity(window_dataset.entity_ids)
    entity_split.assert_disjoint()
    assert entity_split.n_total() == n_windows

    train_ents = set(
        window_dataset.entity_ids[entity_split.train_idx].tolist()
    )
    val_ents = set(window_dataset.entity_ids[entity_split.val_idx].tolist())
    test_ents = set(
        window_dataset.entity_ids[entity_split.test_idx].tolist()
    )
    assert not (train_ents & val_ents)
    assert not (train_ents & test_ents)
    assert not (val_ents & test_ents)


def test_save_load_processed_roundtrip(
        window_dataset,
        tmp_path: Path,
) -> None:
    """Processed NPZ round-trips values, masks, and split indices."""
    paths = RealDataPaths(
        raw_dir=tmp_path / "raw_unused",
        processed_dir=tmp_path / "processed",
    )
    splitter = EntityAwareSplitter(
        SplitFractions(0.70, 0.15, 0.15, seed=_SEED)
    )
    split = splitter.split_by_index(window_dataset.shape()[0])
    out_dir = save_processed_dataset(window_dataset, split, paths)
    loaded, loaded_split = load_processed_dataset(out_dir)
    assert loaded.shape() == window_dataset.shape()
    assert np.allclose(loaded.values, window_dataset.values)
    assert np.allclose(loaded.native_masks, window_dataset.native_masks)
    assert np.array_equal(loaded_split.train_idx, split.train_idx)
