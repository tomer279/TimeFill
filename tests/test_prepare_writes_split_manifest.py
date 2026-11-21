"""WP0 tests: prepare path writes split_manifest and clears stale best."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.real import (
    RealDataPaths,
    RealWindowDataset,
    SplitIndices,
    save_processed_dataset,
)
from src.data.real_config import load_real_m2_config
from src.eval.split_pin import read_split_manifest, verify_split_manifest

_REPO = Path(__file__).resolve().parents[1]
_REAL_CFG = _REPO / "configs" / "real_m2.yaml"
_SEED = 0
_N_WINDOWS = 6
_SEQ = 8
_FEAT = 6
_BLOCK_LEN = 4


def _write_processed(tmp_path: Path) -> Path:
    """Write a tiny F=6 processed NPZ for prepare-path tests."""
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
    rng = np.random.default_rng(_SEED)
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
    return save_processed_dataset(dataset, split, paths)


def test_prepare_artifacts_writes_split_manifest(tmp_path: Path) -> None:
    """``_prepare_artifacts_and_pool`` writes a verifiable split pin."""
    from src.train.real_loop import _prepare_artifacts_and_pool

    processed = _write_processed(tmp_path)
    artifact_root = tmp_path / "artifacts"
    ckpt_dir = artifact_root / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    stale = ckpt_dir / "best_score_net.eqx"
    stale.write_bytes(b"stale")
    (ckpt_dir / "model_config.json").write_text("{}", encoding="utf-8")

    config = load_real_m2_config(_REAL_CFG)
    config.data.paths.processed_dir = str(processed)
    config.data.window.seq_len = _SEQ
    config.data.window.stride = _SEQ
    config.data.batch.block_len = _BLOCK_LEN
    config.train.protocol.artifact_dir = str(artifact_root)
    config.validate()

    paths, _pool = _prepare_artifacts_and_pool(config)
    manifest_path = paths.checkpoint_dir() / "split_manifest.json"
    assert manifest_path.is_file()
    assert not paths.best_checkpoint().is_file()
    assert not (paths.checkpoint_dir() / "model_config.json").is_file()

    manifest = read_split_manifest(paths.checkpoint_dir())
    assert manifest is not None
    verify_split_manifest(manifest, processed)


def test_real_m7_5_train_guard_raises() -> None:
    """``real_m7_5.yaml`` is parse-only until WP1+ consumers land."""
    from src.train.real_loop import _assert_wp0_consumable

    config = load_real_m2_config(_REPO / "configs" / "real_m7_5.yaml")
    with pytest.raises(NotImplementedError, match="M7.5 flags"):
        _assert_wp0_consumable(config)


def test_prepare_refuses_frozen_artifact_dir() -> None:
    """Frozen claim roots and their subdirs are refused before clear/write."""
    from src.train.real_loop import _assert_not_frozen_artifact_dir
    from src.train.artifacts import resolve_artifact_paths

    for artifact_dir in ("artifacts/m2", "artifacts/m7/retrain"):
        paths = resolve_artifact_paths(artifact_dir)
        with pytest.raises(RuntimeError, match="frozen claim dir"):
            _assert_not_frozen_artifact_dir(paths)


def test_model_guard_rejects_non_default_n_layers() -> None:
    """Partial ``model:`` overrides must not slip past the unused-flag guard."""
    from src.models.score_net import ScoreNetConfig
    from src.train.real_loop import _assert_wp0_consumable

    config = load_real_m2_config(_REAL_CFG)
    config.model = ScoreNetConfig(
        n_features=config.data.n_features,
        n_layers=6,
    )
    with pytest.raises(NotImplementedError, match="model"):
        _assert_wp0_consumable(config)
