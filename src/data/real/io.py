"""Serialize / load processed M2 window NPZ bundles.

Functions
---------
save_processed_dataset
    Write windows + split indices under `data/processed/m2/`.
load_processed_dataset
    Load a previously prepared NPZ bundle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.real.specs import (
    PROCESSED_ARRAYS_NAME,
    PROCESSED_ENTITY_NAME,
    PROCESSED_MASKS_NAME,
    PROCESSED_TEST_NAME,
    PROCESSED_TRAIN_NAME,
    PROCESSED_VAL_NAME,
    PROCESSED_VALUES_NAME,
    RealDataPaths,
)
from src.data.real.splits import SplitIndices
from src.data.real.windows import RealWindowDataset


def save_processed_dataset(
        dataset: RealWindowDataset,
        split: SplitIndices,
        paths: RealDataPaths,
) -> Path:
    """Write windows and split indices under ``paths.processed_dir``.

    Parameters
    ----------
    dataset : RealWindowDataset
        Window stack to serialize.
    split : SplitIndices
        Disjoint split indices.
    paths : RealDataPaths
        Provides ``processed_dir``.

    Returns
    -------
    pathlib.Path
        Directory containing ``windows.npz``.
    """
    split.assert_disjoint()
    out_dir = paths.ensure_processed_dir()
    np.savez_compressed(
        out_dir / PROCESSED_ARRAYS_NAME,
        **{
            PROCESSED_VALUES_NAME: dataset.values,
            PROCESSED_MASKS_NAME: dataset.native_masks,
            PROCESSED_ENTITY_NAME: dataset.entity_ids,
            PROCESSED_TRAIN_NAME: split.train_idx,
            PROCESSED_VAL_NAME: split.val_idx,
            PROCESSED_TEST_NAME: split.test_idx,
        },
    )
    return out_dir


def load_processed_dataset(
        processed_dir: Path,
) -> tuple[RealWindowDataset, SplitIndices]:
    """Load windows and split indices written by ``save_processed_dataset``.

    Parameters
    ----------
    processed_dir : pathlib.Path
        Directory containing ``windows.npz``.

    Returns
    -------
    dataset : RealWindowDataset
        Loaded window stack.
    split : SplitIndices
        Loaded disjoint indices.
    """
    arrays_path = Path(processed_dir) / PROCESSED_ARRAYS_NAME
    with np.load(arrays_path, allow_pickle=False) as handle:
        dataset = RealWindowDataset(
            values=handle[PROCESSED_VALUES_NAME],
            native_masks=handle[PROCESSED_MASKS_NAME],
            entity_ids=handle[PROCESSED_ENTITY_NAME],
        )
        split = SplitIndices(
            train_idx=handle[PROCESSED_TRAIN_NAME],
            val_idx=handle[PROCESSED_VAL_NAME],
            test_idx=handle[PROCESSED_TEST_NAME],
        )
    split.assert_disjoint()
    return dataset, split
