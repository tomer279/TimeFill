"""WP0 tests: split_manifest.json round-trip and pin verification (A5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.real import (
    RealDataPaths,
    RealWindowDataset,
    SplitIndices,
    save_processed_dataset,
)

_SEED = 0
_N_WINDOWS = 6
_SEQ = 4
_FEAT = 2


def _split_pin_api():
    """Import WP0 split-pin helpers (missing until implementation lands)."""
    from src.eval.split_pin import (
        build_split_manifest,
        verify_split_manifest,
        write_split_manifest,
    )

    return build_split_manifest, write_split_manifest, verify_split_manifest


def _write_tiny_processed(tmp_path: Path, seed: int) -> tuple[Path, SplitIndices]:
    """Write a tiny ``windows.npz`` with a deterministic index split.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temp root.
    seed : int
        Split seed used for the shuffled index cut.

    Returns
    -------
    processed_dir : pathlib.Path
        Directory containing ``windows.npz``.
    split : SplitIndices
        Train / val / test indices written into the NPZ.
    """
    values = np.arange(
        _N_WINDOWS * _SEQ * _FEAT, dtype=np.float32
    ).reshape(_N_WINDOWS, _SEQ, _FEAT)
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
    out_dir = save_processed_dataset(dataset, split, paths)
    return out_dir, split


def test_split_manifest_round_trip(tmp_path: Path) -> None:
    """Build → write → verify succeeds against the matching processed dir."""
    build_split_manifest, write_split_manifest, verify_split_manifest = (
        _split_pin_api()
    )
    processed_dir, split = _write_tiny_processed(tmp_path, seed=_SEED)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    manifest = build_split_manifest(
        processed_dir=processed_dir,
        split_mode="index",
        split_seed=_SEED,
        fractions={"train": 0.5, "val": 0.1667, "test": 0.3333},
        split=split,
    )
    path = write_split_manifest(ckpt_dir, manifest)
    assert path.name == "split_manifest.json"
    verify_split_manifest(manifest, processed_dir)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["split_mode"] == "index"
    assert on_disk["split_seed"] == _SEED
    assert "index_sha256" in on_disk
    assert "windows_sha256" in on_disk


def test_verify_raises_on_index_hash_mismatch(tmp_path: Path) -> None:
    """Pin check names the index hash field when window indices change."""
    build_split_manifest, _, verify_split_manifest = _split_pin_api()
    processed_dir, split = _write_tiny_processed(tmp_path, seed=_SEED)
    manifest = build_split_manifest(
        processed_dir=processed_dir,
        split_mode="index",
        split_seed=_SEED,
        fractions={"train": 0.5, "val": 0.1667, "test": 0.3333},
        split=split,
    )
    other_dir, _ = _write_tiny_processed(tmp_path / "other", seed=_SEED + 1)
    with pytest.raises(ValueError, match="index_sha256"):
        verify_split_manifest(manifest, other_dir)


def test_verify_raises_on_windows_content_hash_mismatch(
        tmp_path: Path,
) -> None:
    """Pin check names ``windows_sha256`` when values/masks change."""
    build_split_manifest, _, verify_split_manifest = _split_pin_api()
    processed_dir, split = _write_tiny_processed(tmp_path, seed=_SEED)
    manifest = build_split_manifest(
        processed_dir=processed_dir,
        split_mode="index",
        split_seed=_SEED,
        fractions={"train": 0.5, "val": 0.1667, "test": 0.3333},
        split=split,
    )
    npz_path = processed_dir / "windows.npz"
    with np.load(npz_path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["values"] = payload["values"] * 2.0
    np.savez_compressed(npz_path, **payload)
    with pytest.raises(ValueError, match="windows_sha256"):
        verify_split_manifest(manifest, processed_dir)
