"""Split-manifest pin helpers for Milestone 7.5 (A5).

Functions
---------
build_split_manifest
    Build a pin dict from processed windows + split metadata.
write_split_manifest
    Write ``split_manifest.json`` under a checkpoint directory.
read_split_manifest
    Load a manifest dict or return ``None`` if missing.
verify_split_manifest
    Recompute hashes from ``windows.npz`` and raise on mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.real.io import load_processed_dataset
from src.data.real.specs import (
    PROCESSED_ARRAYS_NAME,
    PROCESSED_ENTITY_NAME,
    PROCESSED_MASKS_NAME,
    PROCESSED_VALUES_NAME,
)
from src.data.real.splits import SplitIndices

_MANIFEST_NAME = "split_manifest.json"


def build_split_manifest(
        processed_dir: Path,
        split_mode: str,
        split_seed: int,
        fractions: Mapping[str, float],
        split: SplitIndices,
) -> dict[str, Any]:
    """Build a split pin dict for a checkpoint directory.

    Parameters
    ----------
    processed_dir : pathlib.Path
        Directory containing ``windows.npz``.
    split_mode : str
        ``entity`` or ``index`` (metadata only; not re-verified).
    split_seed : int
        Split RNG seed (metadata only).
    fractions : Mapping of str to float
        ``train`` / ``val`` / ``test`` fractions.
    split : SplitIndices
        Disjoint index arrays written into the NPZ.

    Returns
    -------
    dict of str to Any
        Manifest payload including ``index_sha256`` and ``windows_sha256``.

    Notes
    -----
    shortcut: mode/seed/fractions are not verified against NPZ; upgrade when
    prepare writes them into the bundle.
    """
    dataset, _loaded = load_processed_dataset(processed_dir)
    return {
        "split_mode": str(split_mode),
        "split_seed": int(split_seed),
        "fractions": {
            "train": float(fractions["train"]),
            "val": float(fractions["val"]),
            "test": float(fractions["test"]),
        },
        "entities": {
            "train": _entity_list(dataset.entity_ids, split.train_idx),
            "val": _entity_list(dataset.entity_ids, split.val_idx),
            "test": _entity_list(dataset.entity_ids, split.test_idx),
        },
        "index_sha256": _index_sha256(split),
        "windows_sha256": _windows_sha256(processed_dir),
    }


def write_split_manifest(
        checkpoint_dir: Path,
        manifest: Mapping[str, Any],
) -> Path:
    """Write ``split_manifest.json`` under ``checkpoint_dir``.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Typically ``.../checkpoints``.
    manifest : Mapping of str to Any
        Payload from ``build_split_manifest``.

    Returns
    -------
    pathlib.Path
        Path to the written JSON file.
    """
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _MANIFEST_NAME
    path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_split_manifest(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Load a split manifest or return ``None`` when absent.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory that may contain ``split_manifest.json``.

    Returns
    -------
    dict of str to Any or None
        Parsed manifest, or ``None`` if the file is missing.
    """
    path = Path(checkpoint_dir) / _MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_split_manifest(
        manifest: Mapping[str, Any],
        processed_dir: Path,
) -> None:
    """Recompute pin hashes and raise if they disagree with ``manifest``.

    Parameters
    ----------
    manifest : Mapping of str to Any
        Previously written pin payload.
    processed_dir : pathlib.Path
        Current processed windows directory.

    Raises
    ------
    ValueError
        When ``index_sha256`` or ``windows_sha256`` differs; message names
        the field.
    """
    _dataset, split = load_processed_dataset(processed_dir)
    expected_index = _index_sha256(split)
    if manifest.get("index_sha256") != expected_index:
        raise ValueError(
            "split pin mismatch on field index_sha256: "
            f"manifest={manifest.get('index_sha256')!r} "
            f"current={expected_index!r}"
        )
    expected_windows = _windows_sha256(processed_dir)
    if manifest.get("windows_sha256") != expected_windows:
        raise ValueError(
            "split pin mismatch on field windows_sha256: "
            f"manifest={manifest.get('windows_sha256')!r} "
            f"current={expected_windows!r}"
        )


def _index_sha256(split: SplitIndices) -> str:
    """Hash sorted train/val/test index arrays."""
    digest = hashlib.sha256()
    for arr in (
        np.asarray(split.train_idx, dtype=np.int64),
        np.asarray(split.val_idx, dtype=np.int64),
        np.asarray(split.test_idx, dtype=np.int64),
    ):
        digest.update(np.sort(arr).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def _windows_sha256(processed_dir: Path) -> str:
    """Hash values/masks/entities only (not split index arrays)."""
    path = Path(processed_dir) / PROCESSED_ARRAYS_NAME
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as handle:
        for key in (
            PROCESSED_VALUES_NAME,
            PROCESSED_MASKS_NAME,
            PROCESSED_ENTITY_NAME,
        ):
            digest.update(np.ascontiguousarray(handle[key]).tobytes())
            digest.update(b"|")
    return digest.hexdigest()


def _entity_list(entity_ids: np.ndarray, indices: np.ndarray) -> list[str]:
    """Return sorted unique entity ids for the given window indices."""
    if indices.size == 0:
        return []
    selected = entity_ids[np.asarray(indices, dtype=np.int64)]
    return sorted({str(item) for item in selected.tolist()})
