"""WP0 tests: benchmark split pin — refuse on mismatch, warn when unpinned."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from src.data.normalize import FeatureScaler
from src.data.real import (
    RealDataPaths,
    RealWindowDataset,
    SplitIndices,
    load_processed_dataset,
    save_processed_dataset,
)
from src.eval.m3_data import SplitLoadRequest, load_split_window_bundle

_SEED = 0
_N_WINDOWS = 6
_SEQ = 4
_FEAT = 2


def _split_pin_api():
    """Import WP0 split-pin helpers (missing until implementation lands)."""
    from src.eval.split_pin import (
        build_split_manifest,
        write_split_manifest,
    )

    return build_split_manifest, write_split_manifest


def _write_processed(tmp_path: Path, seed: int) -> tuple[Path, SplitIndices]:
    """Write a tiny processed NPZ for pin integration tests."""
    values = (
        np.arange(_N_WINDOWS * _SEQ * _FEAT, dtype=np.float32).reshape(
            _N_WINDOWS, _SEQ, _FEAT
        )
        + float(seed)
    )
    masks = np.ones_like(values, dtype=np.float32)
    entity_ids = np.asarray(
        [f"e{i % 3}" for i in range(_N_WINDOWS)],
    )
    dataset = RealWindowDataset(
        values=values,
        native_masks=masks,
        entity_ids=entity_ids,
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(_N_WINDOWS)
    split = SplitIndices(
        train_idx=np.asarray(order[:3], dtype=np.int64),
        val_idx=np.asarray(order[3:4], dtype=np.int64),
        test_idx=np.asarray(order[4:], dtype=np.int64),
    )
    paths = RealDataPaths(
        raw_dir=tmp_path / "raw_unused",
        processed_dir=tmp_path / "processed",
    )
    return save_processed_dataset(dataset, split, paths), split


def _fit_scaler(processed_dir: Path) -> FeatureScaler:
    """Fit a scaler on all windows (hermetic; pin tests ignore leakage)."""
    dataset, _split = load_processed_dataset(processed_dir)
    return FeatureScaler().fit(dataset.values, dataset.native_masks)


def test_benchmark_load_raises_when_reprepared_with_different_seed(
        tmp_path: Path,
) -> None:
    """Pinned checkpoint refuses a windows.npz rebuilt with another split seed."""
    build_split_manifest, write_split_manifest = _split_pin_api()
    processed_a, split_a = _write_processed(tmp_path / "a", seed=_SEED)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    manifest = build_split_manifest(
        processed_dir=processed_a,
        split_mode="index",
        split_seed=_SEED,
        fractions={"train": 0.5, "val": 0.1667, "test": 0.3333},
        split=split_a,
    )
    write_split_manifest(ckpt_dir, manifest)

    processed_b, _split_b = _write_processed(tmp_path / "b", seed=_SEED + 1)
    scaler = _fit_scaler(processed_b)
    request = SplitLoadRequest(
        processed_dir=processed_b,
        scaler=scaler,
        split_name="test",
        max_windows=None,
        checkpoint_dir=ckpt_dir,
    )
    with pytest.raises(ValueError, match="index_sha256|windows_sha256|split"):
        load_split_window_bundle(request)


def test_legacy_checkpoint_without_manifest_warns_but_loads(
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
) -> None:
    """No split_manifest.json → warn, ``split_pinned=False``, load still works."""
    processed_dir, _split = _write_processed(tmp_path, seed=_SEED)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    scaler = _fit_scaler(processed_dir)
    request = SplitLoadRequest(
        processed_dir=processed_dir,
        scaler=scaler,
        split_name="test",
        max_windows=None,
        checkpoint_dir=ckpt_dir,
    )
    with caplog.at_level(logging.WARNING):
        bundle = load_split_window_bundle(request)
    assert bundle.split_name == "test"
    assert bundle.values_n.shape[0] >= 1
    assert any(
        "split_pinned" in rec.message or "split manifest" in rec.message.lower()
        for rec in caplog.records
    )
    assert getattr(bundle, "split_pinned", None) is False


def test_matching_manifest_sets_split_pinned_true(
        tmp_path: Path,
) -> None:
    """Matching manifest → ``split_pinned=True``; truncate keeps the flag."""
    build_split_manifest, write_split_manifest = _split_pin_api()
    processed_dir, split = _write_processed(tmp_path, seed=_SEED)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    write_split_manifest(
        ckpt_dir,
        build_split_manifest(
            processed_dir=processed_dir,
            split_mode="index",
            split_seed=_SEED,
            fractions={"train": 0.5, "val": 0.1667, "test": 0.3333},
            split=split,
        ),
    )
    scaler = _fit_scaler(processed_dir)
    request = SplitLoadRequest(
        processed_dir=processed_dir,
        scaler=scaler,
        split_name="test",
        max_windows=None,
        checkpoint_dir=ckpt_dir,
    )
    bundle = load_split_window_bundle(request)
    assert bundle.split_pinned is True
    truncated = bundle.truncate(1)
    assert truncated.split_pinned is True
    assert truncated.n_windows() == 1
