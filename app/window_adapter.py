"""Adapt validated M4 CSV frames into ``(x, m)`` windows.

Native NaNs / empty cells are already sentinel ``0.0`` with ``m == 0`` in
``ValidatedFrame``. This module only applies the M4 pad / multi-window
policy from ``ASSUMPTIONS.md`` (Milestone 4): non-overlapping windows of
length ``T``, right-pad short series and trailing remainders with
sentinel ``0.0`` and ``m = 0``, then optionally cap with ``max_windows``.

Padded cells are never ground-truth imputation targets.

Classes
-------
WindowAdaptRequest
    Frozen window length and optional ``max_windows`` cap.
AdaptedWindows
    Stacked ``(B, T, F)`` values / masks ready for ``FeatureScaler.transform``.

Functions
---------
adapt_validated_frame
    Cut / pad non-overlapping windows from a ``ValidatedFrame``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from app.schema_validate import ValidatedFrame
from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_MISSING,
    MISSING_VALUE_SENTINEL,
    TIME_AXIS,
    TimeSeriesBatch,
)

_ROW_AXIS = 0
_FEATURE_AXIS_LF = 1
_DEFAULT_SEQ_LEN = 64


@dataclass(frozen=True)
class WindowAdaptRequest:
    """Knobs for CSV → window adaptation (M4 pad / cap policy).

    Public methods
    --------------
    validate
        Check ``seq_len`` and ``max_windows``.
    effective_max_windows
        Return the cap, or ``None`` when uncapped.

    Instance variables
    ------------------
    seq_len : int
        Window length ``T`` (default 64).
    max_windows : int or None
        Optional UI latency cap (``smoke.max_windows``); ``None`` = no cap.
    """

    seq_len: int = _DEFAULT_SEQ_LEN
    max_windows: int | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` if request knobs are invalid."""
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError(
                f"max_windows must be positive when set, got {self.max_windows}"
            )

    def effective_max_windows(self) -> int | None:
        """Return the window cap used by the adapter.

        Returns
        -------
        int or None
            Positive cap, or ``None`` if uncapped.
        """
        return self.max_windows


@dataclass(frozen=True)
class AdaptedWindows:
    """Stacked fixed-length windows under the M1/M2 tensor contract.

    Public methods
    --------------
    n_windows
        Batch size ``B``.
    as_batch
        Pack into a JAX ``TimeSeriesBatch``.

    Instance variables
    ------------------
    values : numpy.ndarray
        Float32 ``(B, T, F)``; missing / pad cells use sentinel ``0.0``.
    native_mask : numpy.ndarray
        Float32 ``(B, T, F)``; ``1`` observed, ``0`` missing or padded.
    feature_names : tuple of str
        Feature axis labels (frozen order).
    entity_ids : numpy.ndarray
        Object / unicode entity id per window, shape ``(B,)``.
    """

    values: np.ndarray
    native_mask: np.ndarray
    feature_names: tuple[str, ...]
    entity_ids: np.ndarray

    def n_windows(self) -> int:
        """Return the number of windows ``B``.

        Returns
        -------
        int
            ``values.shape[0]``.
        """
        return int(self.values.shape[BATCH_AXIS])

    def as_batch(self) -> TimeSeriesBatch:
        """Return a JAX ``TimeSeriesBatch`` for downstream impute.

        Returns
        -------
        TimeSeriesBatch
            ``x`` / ``m`` shaped ``(B, T, F)``.
        """
        return TimeSeriesBatch(
            x=jnp.asarray(self.values, dtype=jnp.float32),
            m=jnp.asarray(self.native_mask, dtype=jnp.float32),
        )


def adapt_validated_frame(
        frame: ValidatedFrame,
        request: WindowAdaptRequest,
) -> AdaptedWindows:
    """Build non-overlapping ``(B, T, F)`` windows with M4 pad policy.

    Shorter-than-``T`` series and trailing remainders are **right-padded**
    with sentinel ``0.0`` and ``m = 0``. Output is ready for
    ``FeatureScaler.transform`` (impute wiring is WP2+).

    Parameters
    ----------
    frame : ValidatedFrame
        Schema-validated table in original units.
    request : WindowAdaptRequest
        ``seq_len`` and optional ``max_windows`` cap.

    Returns
    -------
    AdaptedWindows
        Stack ready for ``FeatureScaler.transform``.

    Raises
    ------
    ValueError
        If the frame has no rows, feature count mismatches, or knobs are bad.
    """
    request.validate()
    if frame.n_rows() <= 0:
        raise ValueError("cannot adapt an empty ValidatedFrame")
    n_features = frame.n_features()
    if n_features != len(frame.feature_names):
        raise ValueError(
            "feature_count mismatch: values have "
            f"{n_features} features but feature_names has "
            f"{len(frame.feature_names)}"
        )
    value_blocks, mask_blocks = _cut_padded_windows(
        frame.values,
        frame.native_mask,
        request.seq_len,
    )
    max_windows = request.effective_max_windows()
    if max_windows is not None:
        value_blocks = value_blocks[:max_windows]
        mask_blocks = mask_blocks[:max_windows]
    entity_label = frame.entity_id if frame.entity_id is not None else ""
    entity_ids = np.asarray(
        [entity_label] * len(value_blocks),
        dtype=object,
    )
    stacked_values = np.stack(value_blocks, axis=BATCH_AXIS).astype(np.float32)
    stacked_masks = np.stack(mask_blocks, axis=BATCH_AXIS).astype(np.float32)
    _assert_window_rank3(stacked_values, request.seq_len, n_features)
    return AdaptedWindows(
        values=stacked_values,
        native_mask=stacked_masks,
        feature_names=frame.feature_names,
        entity_ids=entity_ids,
    )


def _assert_window_rank3(
        values: np.ndarray,
        seq_len: int,
        n_features: int,
) -> None:
    """Check stacked windows have shape ``(B, T, F)``.

    Parameters
    ----------
    values : numpy.ndarray
        Stacked window values.
    seq_len : int
        Expected ``T``.
    n_features : int
        Expected ``F``.
    """
    if values.ndim != 3:
        raise ValueError(f"expected rank-3 windows, got shape {values.shape}")
    if int(values.shape[TIME_AXIS]) != seq_len:
        raise ValueError(
            f"expected seq_len={seq_len}, got T={values.shape[TIME_AXIS]}"
        )
    if int(values.shape[FEATURE_AXIS]) != n_features:
        raise ValueError(
            f"expected F={n_features}, got F={values.shape[FEATURE_AXIS]}"
        )


def _cut_padded_windows(
        values: np.ndarray,
        native_mask: np.ndarray,
        seq_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Cut non-overlapping windows; right-pad short / remainder chunks.

    Parameters
    ----------
    values : numpy.ndarray
        Float32 ``(L, F)`` values.
    native_mask : numpy.ndarray
        Float32 ``(L, F)`` native masks.
    seq_len : int
        Window length ``T``.

    Returns
    -------
    value_blocks : list of numpy.ndarray
        Each ``(T, F)``.
    mask_blocks : list of numpy.ndarray
        Each ``(T, F)``.
    """
    n_rows = int(values.shape[_ROW_AXIS])
    n_features = int(values.shape[_FEATURE_AXIS_LF])
    value_blocks: list[np.ndarray] = []
    mask_blocks: list[np.ndarray] = []
    start = 0
    while start < n_rows:
        stop = min(start + seq_len, n_rows)
        chunk_values = values[start:stop]
        chunk_mask = native_mask[start:stop]
        pad = seq_len - int(chunk_values.shape[_ROW_AXIS])
        if pad > 0:
            chunk_values = _right_pad_values(chunk_values, pad, n_features)
            chunk_mask = _right_pad_mask(chunk_mask, pad, n_features)
        value_blocks.append(chunk_values)
        mask_blocks.append(chunk_mask)
        start += seq_len
    return value_blocks, mask_blocks


def _right_pad_values(
        chunk: np.ndarray,
        pad: int,
        n_features: int,
) -> np.ndarray:
    """Right-pad a ``(L, F)`` value chunk with the missing sentinel.

    Parameters
    ----------
    chunk : numpy.ndarray
        Leading real rows.
    pad : int
        Number of pad rows.
    n_features : int
        Feature count ``F``.

    Returns
    -------
    numpy.ndarray
        Length-``T`` value window.
    """
    pad_block = np.full(
        (pad, n_features),
        float(MISSING_VALUE_SENTINEL),
        dtype=np.float32,
    )
    return np.concatenate([chunk.astype(np.float32), pad_block], axis=_ROW_AXIS)


def _right_pad_mask(
        chunk: np.ndarray,
        pad: int,
        n_features: int,
) -> np.ndarray:
    """Right-pad a ``(L, F)`` mask chunk with ``MASK_MISSING``.

    Parameters
    ----------
    chunk : numpy.ndarray
        Leading real rows.
    pad : int
        Number of pad rows.
    n_features : int
        Feature count ``F``.

    Returns
    -------
    numpy.ndarray
        Length-``T`` mask window.
    """
    pad_block = np.full(
        (pad, n_features),
        float(MASK_MISSING),
        dtype=np.float32,
    )
    return np.concatenate([chunk.astype(np.float32), pad_block], axis=_ROW_AXIS)
