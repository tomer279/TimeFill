"""Normalized train/val window pools for M2 real-data training.

Classes
-------
RealBatchPool
    Normalized train/val window pools with native masks.

Functions
---------
load_normalized_pool
    Load processed windows, fit scaler, return normalized pools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.data.normalize import FeatureScaler, fit_scaler_on_train_split
from src.data.real import load_processed_dataset
from src.data.real_config import RealM2Config
from src.data.schema import MASK_OBSERVED
from src.train.real_loss import RealDSMBatch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MASK_TOL = 1e-6


@dataclass(frozen=True)
class RealBatchPool:
    """Normalized real windows split into train / val pools.

    Public methods
    --------------
    sample_train
        Draw a random train ``RealDSMBatch``.
    fixed_val_batch
        Deterministic leading val batch (padded by repeat if needed).

    Instance variables
    ------------------
    train_values : jax.Array
        Train windows ``(N_train, T, F)``.
    train_masks : jax.Array
        Train native masks.
    val_values : jax.Array
        Val windows ``(N_val, T, F)``.
    val_masks : jax.Array
        Val native masks.
    """

    train_values: jnp.ndarray
    train_masks: jnp.ndarray
    val_values: jnp.ndarray
    val_masks: jnp.ndarray

    def sample_train(self, key: jax.Array, batch_size: int) -> RealDSMBatch:
        """Sample a train mini-batch with replacement if needed.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        batch_size : int
            Requested ``B``.

        Returns
        -------
        RealDSMBatch
            Sampled values and native masks.
        """
        return _take_batch(
            key,
            self.train_values,
            self.train_masks,
            batch_size,
        )

    def fixed_val_batch(self, batch_size: int) -> RealDSMBatch:
        """Return a fixed val batch from the leading val windows.

        Parameters
        ----------
        batch_size : int
            Requested ``B``.

        Returns
        -------
        RealDSMBatch
            Deterministic val batch.
        """
        n_val = int(self.val_values.shape[0])
        if n_val <= 0:
            raise ValueError("val pool is empty")
        idx = np.arange(batch_size, dtype=np.int64) % n_val
        return RealDSMBatch(
            x0=self.val_values[idx],
            m_native=self.val_masks[idx],
        )


def load_normalized_pool(
        config: RealM2Config,
) -> tuple[RealBatchPool, FeatureScaler]:
    """Load processed NPZ, fit scaler on train, return normalized pools.

    Parameters
    ----------
    config : RealM2Config
        Validated real-m2 config.

    Returns
    -------
    pool : RealBatchPool
        Normalized train/val pools.
    scaler : FeatureScaler
        Train-fitted scaler (also written by ``run_m2_training``).
    """
    processed = config.data.resolve_processed_dir(_REPO_ROOT)
    dataset, split = load_processed_dataset(processed)
    scaler = fit_scaler_on_train_split(dataset, split.train_idx)
    values_n = np.asarray(
        scaler.transform(dataset.values, dataset.native_masks),
        dtype=np.float32,
    )
    masks = np.asarray(dataset.native_masks, dtype=np.float32)
    _assert_observed_finite(values_n, masks)
    train_idx = np.asarray(split.train_idx, dtype=np.int64)
    val_idx = np.asarray(split.val_idx, dtype=np.int64)
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("train and val splits must be non-empty")
    pool = RealBatchPool(
        train_values=jnp.asarray(values_n[train_idx], dtype=jnp.float32),
        train_masks=jnp.asarray(masks[train_idx], dtype=jnp.float32),
        val_values=jnp.asarray(values_n[val_idx], dtype=jnp.float32),
        val_masks=jnp.asarray(masks[val_idx], dtype=jnp.float32),
    )
    return pool, scaler


def _assert_observed_finite(
        values: np.ndarray,
        masks: np.ndarray,
) -> None:
    """Raise if any natively observed cell is non-finite.

    Parameters
    ----------
    values : numpy.ndarray
        Window values.
    masks : numpy.ndarray
        Native masks.
    """
    observed = np.abs(masks - MASK_OBSERVED) < _MASK_TOL
    bad = observed & ~np.isfinite(values)
    if np.any(bad):
        raise ValueError(
            "non-finite values found on natively observed cells; "
            "check preparation / scaling"
        )


def _take_batch(
        key: jax.Array,
        values: jnp.ndarray,
        masks: jnp.ndarray,
        batch_size: int,
) -> RealDSMBatch:
    """Index a random mini-batch from a window pool.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    values : jax.Array
        Pool values ``(N, T, F)``.
    masks : jax.Array
        Pool native masks.
    batch_size : int
        Requested ``B``.

    Returns
    -------
    RealDSMBatch
        Indexed batch.
    """
    n_windows = int(values.shape[0])
    replace = batch_size > n_windows
    idx = jax.random.choice(
        key,
        n_windows,
        shape=(batch_size,),
        replace=replace,
    )
    return RealDSMBatch(x0=values[idx], m_native=masks[idx])
