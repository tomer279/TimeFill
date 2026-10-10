"""Fixed-length window stacking for M2 real hourly series.

Classes
-------
RealWindowDataset
    Stacked windows `(N, T, F)` plus entity ids and native masks.
RealWindowBuilder
    Cut non-overlapping windows from hourly site series.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from src.data.real.beijing import HourlySiteSeries
from src.data.real.specs import WindowSpec
from src.data.schema import FEATURE_AXIS, TIME_AXIS, TimeSeriesBatch


@dataclass(frozen=True)
class RealWindowDataset:
    """Stacked fixed-length windows with native masks and entity ids.

    Public methods
    --------------
    shape
        Return ``(N, T, F)``.
    subset
        Return a dataset restricted to the given window indices.
    as_batch
        Pack selected windows into a ``TimeSeriesBatch`` (native mask).

    Instance variables
    ------------------
    values : numpy.ndarray
        Ground-truth windows ``(N, T, F)`` (sentinel on native missing).
    native_masks : numpy.ndarray
        Native masks ``(N, T, F)``.
    entity_ids : numpy.ndarray
        Entity id per window, shape ``(N,)``.
    """

    values: np.ndarray
    native_masks: np.ndarray
    entity_ids: np.ndarray

    def shape(self) -> tuple[int, int, int]:
        """Return the stacked window shape ``(N, T, F)``.

        Returns
        -------
        tuple of int
            Number of windows, ``T``, and ``F``.
        """
        n_windows = int(self.values.shape[0])
        seq_len = int(self.values.shape[TIME_AXIS])
        n_features = int(self.values.shape[FEATURE_AXIS])
        return n_windows, seq_len, n_features

    def subset(self, indices: np.ndarray) -> RealWindowDataset:
        """Return windows at ``indices``.

        Parameters
        ----------
        indices : numpy.ndarray
            1-D integer index array.

        Returns
        -------
        RealWindowDataset
            Subset with the same ``T`` and ``F``.
        """
        idx = np.asarray(indices, dtype=np.int64)
        return RealWindowDataset(
            values=self.values[idx],
            native_masks=self.native_masks[idx],
            entity_ids=self.entity_ids[idx],
        )

    def as_batch(self, indices: np.ndarray) -> TimeSeriesBatch:
        """Build a JAX ``TimeSeriesBatch`` from selected windows.

        Parameters
        ----------
        indices : numpy.ndarray
            1-D integer index array (batch axis).

        Returns
        -------
        TimeSeriesBatch
            ``x`` / ``m`` from ground truth and **native** masks.
        """
        subset = self.subset(indices)
        return TimeSeriesBatch(
            x=jnp.asarray(subset.values, dtype=jnp.float32),
            m=jnp.asarray(subset.native_masks, dtype=jnp.float32),
        )


class RealWindowBuilder:
    """Cut fixed-length windows from hourly site series.

    Default ``stride == seq_len`` yields non-overlapping windows so a
    later index / entity split cannot leak shared timesteps.

    Public methods
    --------------
    build
        Stack windows from many sites into a ``RealWindowDataset``.
    count_windows
        Number of windows available for a series of length ``L``.

    Instance variables
    ------------------
    window_spec : WindowSpec
        ``seq_len`` / ``stride`` geometry.
    """

    def __init__(self, window_spec: WindowSpec) -> None:
        """Store and validate window geometry.

        Parameters
        ----------
        window_spec : WindowSpec
            Positive ``seq_len`` and ``stride``.
        """
        window_spec.validate()
        self.window_spec = window_spec

    def count_windows(self, n_steps: int) -> int:
        """Return how many windows fit in ``n_steps`` hours.

        Parameters
        ----------
        n_steps : int
            Hourly series length ``L``.

        Returns
        -------
        int
            Non-negative window count.
        """
        seq_len = self.window_spec.seq_len
        stride = self.window_spec.stride
        if n_steps < seq_len:
            return 0
        return 1 + (n_steps - seq_len) // stride

    def build(self, sites: list[HourlySiteSeries]) -> RealWindowDataset:
        """Stack windows from all sites.

        Parameters
        ----------
        sites : list of HourlySiteSeries
            Hourly grids (possibly multiple entities).

        Returns
        -------
        RealWindowDataset
            Arrays shaped ``(N, T, F)``.

        Raises
        ------
        ValueError
            If no windows can be formed.
        """
        value_blocks: list[np.ndarray] = []
        mask_blocks: list[np.ndarray] = []
        entity_blocks: list[str] = []
        seq_len = self.window_spec.seq_len
        stride = self.window_spec.stride
        for site in sites:
            n_win = self.count_windows(site.n_steps())
            for win_idx in range(n_win):
                start = win_idx * stride
                stop = start + seq_len
                value_blocks.append(site.values[start:stop])
                mask_blocks.append(site.native_mask[start:stop])
                entity_blocks.append(site.entity_id)
        if not value_blocks:
            raise ValueError("no windows produced; check seq_len vs series")
        return RealWindowDataset(
            values=np.stack(value_blocks, axis=0).astype(np.float32),
            native_masks=np.stack(mask_blocks, axis=0).astype(np.float32),
            entity_ids=np.asarray(entity_blocks),
        )
