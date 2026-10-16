"""Adapt validated CSV frames into ``(x, m)`` windows.

Native NaNs / empty cells are already sentinel ``0.0`` with ``m == 0`` in
``ValidatedFrame``. Shared Exact / Flexible pad policy (ASSUMPTIONS M4 / M6):
non-overlapping windows of length ``T``, right-pad short series and trailing
remainders with sentinel ``0.0`` and ``m = 0``, then optionally cap with
``max_windows``.

**Shared (Exact Beijing + Flexible):** ``adapt_validated_frame`` /
``WindowAdaptRequest`` pad / multi-window / cap. Exact M4 callers keep using
this entry only (no min-length / sparsity reject; short series still pad).

**Flexible-only (M6):** ``adapt_flexible_frame`` optionally bins wall-clock
timestamps to an hourly last-value-in-bin grid, then enforces ASSUMPTIONS M6
reject thresholds (too short / too sparse / schema-invalid) with user-facing
``ValueError`` messages before the shared pad path. Padded cells are never
ground-truth imputation targets.

Classes
-------
WindowAdaptRequest
    Frozen window length and optional ``max_windows`` cap.
AdaptedWindows
    Stacked ``(B, T, F)`` values / masks ready for ``FeatureScaler.transform``.
IngestRejectPolicy
    M6 flexible reject thresholds (min rows / observed rates).
FlexibleAdaptRequest
    Flexible-only knobs: window request, reject policy, optional hourly bin.

Functions
---------
adapt_validated_frame
    Cut / pad non-overlapping windows (Exact + shared pad core).
default_ingest_reject_policy
    Build ASSUMPTIONS defaults from ``seq_len``.
bin_validated_frame_hourly
    Hourly last-value-in-bin grid when timestamps are present.
adapt_flexible_frame
    Optional bin → reject → shared ``adapt_validated_frame``.
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
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    TIME_AXIS,
    TimeSeriesBatch,
)

_ROW_AXIS = 0
_FEATURE_AXIS_LF = 1
_DEFAULT_SEQ_LEN = 64
_MIN_ROWS_DIVISOR = 4
_DEFAULT_MIN_OBSERVED_GLOBAL = 0.10
_DEFAULT_MIN_OBSERVED_PER_FEATURE = 0.05
_MASK_COMPARE_TOL = 1e-6
_HOUR_DELTA = np.timedelta64(1, "h")
_HOUR_UNIT = "h"


@dataclass(frozen=True)
class WindowAdaptRequest:
    """Knobs for CSV → window adaptation (M4 / shared pad / cap policy).

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


@dataclass(frozen=True)
class IngestRejectPolicy:
    """ASSUMPTIONS M6 reject thresholds for flexible ingest (native frame).

    Applied **after** optional hourly binning and **before** pad / window.
    Exact Beijing ``adapt_validated_frame`` does **not** use this policy.

    Public methods
    --------------
    validate
        Check threshold ranges.
    assert_acceptable
        Raise user-facing ``ValueError`` when the frame fails reject rules.

    Instance variables
    ------------------
    min_rows : int
        Reject if usable grid length ``L < min_rows``.
    min_observed_rate_global : float
        Reject if ``mean(m_native) <`` this rate.
    min_observed_rate_per_feature : float
        Reject if any feature observed rate is below this value.
    """

    min_rows: int
    min_observed_rate_global: float = _DEFAULT_MIN_OBSERVED_GLOBAL
    min_observed_rate_per_feature: float = (
        _DEFAULT_MIN_OBSERVED_PER_FEATURE
    )

    def validate(self) -> None:
        """Raise ``ValueError`` if reject knobs are inconsistent."""
        if self.min_rows < 1:
            raise ValueError(
                f"min_rows must be >= 1, got {self.min_rows}"
            )
        if not 0.0 <= self.min_observed_rate_global <= 1.0:
            raise ValueError(
                "min_observed_rate_global must be in [0, 1], got "
                f"{self.min_observed_rate_global}"
            )
        if not 0.0 <= self.min_observed_rate_per_feature <= 1.0:
            raise ValueError(
                "min_observed_rate_per_feature must be in [0, 1], got "
                f"{self.min_observed_rate_per_feature}"
            )

    def assert_acceptable(self, frame: ValidatedFrame) -> None:
        """Reject too-short / too-sparse / schema-invalid native frames.

        Parameters
        ----------
        frame : ValidatedFrame
            Regular-grid frame (post optional binning).

        Raises
        ------
        ValueError
            User-facing reject; never invents Beijing columns.
        """
        self.validate()
        n_rows = frame.n_rows()
        n_features = frame.n_features()
        if n_rows <= 0:
            raise ValueError(
                "schema-invalid: cannot adapt an empty series "
                "(no usable grid rows after optional binning)"
            )
        if n_features <= 0:
            raise ValueError(
                "schema-invalid: zero feature columns "
                "(columns are not remapped to Beijing names)"
            )
        if n_features != len(frame.feature_names):
            raise ValueError(
                "schema-invalid: feature_count mismatch: values have "
                f"{n_features} features but feature_names has "
                f"{len(frame.feature_names)}"
            )
        if n_rows < self.min_rows:
            raise ValueError(
                f"series too short: usable length L={n_rows} is below "
                f"min_rows={self.min_rows} (after optional binning). "
                "Pad-only tiny series are rejected."
            )
        global_rate = float(np.mean(frame.native_mask))
        if global_rate < self.min_observed_rate_global:
            raise ValueError(
                f"series too sparse: global observed rate "
                f"{global_rate:.4f} is below "
                f"min_observed_rate_global="
                f"{self.min_observed_rate_global}"
            )
        per_feature = np.mean(
            frame.native_mask,
            axis=_ROW_AXIS,
        ).astype(np.float64)
        for feat_idx, name in enumerate(frame.feature_names):
            rate = float(per_feature[feat_idx])
            if rate < self.min_observed_rate_per_feature:
                raise ValueError(
                    f"series too sparse: feature {name!r} observed rate "
                    f"{rate:.4f} is below "
                    f"min_observed_rate_per_feature="
                    f"{self.min_observed_rate_per_feature}"
                )


@dataclass(frozen=True)
class FlexibleAdaptRequest:
    """Flexible-ingest adapt knobs (M6 WP2).

    Shared pad / multi-window rules live in ``window``. Reject thresholds and
    optional hourly binning are flexible-only and do not change Exact M4
    ``adapt_validated_frame`` behavior.

    Public methods
    --------------
    validate
        Validate nested window and reject knobs.
    as_window_request
        Return the shared ``WindowAdaptRequest``.

    Instance variables
    ------------------
    window : WindowAdaptRequest
        ``seq_len`` and optional ``max_windows``.
    reject : IngestRejectPolicy
        M6 too-short / too-sparse thresholds.
    apply_hourly_bin : bool
        If ``True`` and timestamps exist, bin to hourly last-value-in-bin;
        if timestamps are absent, row order is the grid.
    """

    window: WindowAdaptRequest
    reject: IngestRejectPolicy
    apply_hourly_bin: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` if nested knobs are invalid."""
        self.window.validate()
        self.reject.validate()

    def as_window_request(self) -> WindowAdaptRequest:
        """Return the shared window request used for pad / cut.

        Returns
        -------
        WindowAdaptRequest
            Nested window knobs.
        """
        return self.window


@dataclass(frozen=True)
class _HourlyGridBuffers:
    """Mutable hourly bin destination (internal).

    Public methods
    --------------
    n_steps
        Grid length.
    n_features
        Feature count.

    Instance variables
    ------------------
    grid : numpy.ndarray
        Hour axis.
    values : numpy.ndarray
        Destination values ``(L, F)``.
    masks : numpy.ndarray
        Destination masks ``(L, F)``.
    """

    grid: np.ndarray
    values: np.ndarray
    masks: np.ndarray

    def n_steps(self) -> int:
        """Return the hourly grid length.

        Returns
        -------
        int
            ``grid.shape[0]``.
        """
        return int(self.grid.shape[_ROW_AXIS])

    def n_features(self) -> int:
        """Return the feature count ``F``.

        Returns
        -------
        int
            ``values.shape[1]``.
        """
        return int(self.values.shape[_FEATURE_AXIS_LF])


def default_ingest_reject_policy(seq_len: int) -> IngestRejectPolicy:
    """Return ASSUMPTIONS M6 reject defaults for a window length.

    ``min_rows = max(1, seq_len // 4)`` (16 when ``seq_len = 64``).
    Observed-rate floors match ``configs/ingest_m6.yaml``.

    Parameters
    ----------
    seq_len : int
        Window length ``T`` (must be positive).

    Returns
    -------
    IngestRejectPolicy
        Locked flexible reject thresholds.

    Raises
    ------
    ValueError
        If ``seq_len`` is not positive.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    min_rows = max(1, seq_len // _MIN_ROWS_DIVISOR)
    return IngestRejectPolicy(
        min_rows=min_rows,
        min_observed_rate_global=_DEFAULT_MIN_OBSERVED_GLOBAL,
        min_observed_rate_per_feature=_DEFAULT_MIN_OBSERVED_PER_FEATURE,
    )


def bin_validated_frame_hourly(frame: ValidatedFrame) -> ValidatedFrame:
    """Resample a frame onto a contiguous hourly last-value-in-bin grid.

    Empty hours become native missing (``m = 0``, sentinel ``0.0``). Duplicate
    hours keep the last **observed** value per feature. Timestamps are for
    binning only; irregular wall-clock SDE dynamics remain deferred.

    Parameters
    ----------
    frame : ValidatedFrame
        Frame with parseable wall-clock ``timestamps``.

    Returns
    -------
    ValidatedFrame
        Dense hourly grid with the same feature names / entity.

    Raises
    ------
    ValueError
        If timestamps are missing, length-mismatched, or not datetime-like.
    """
    if frame.timestamps is None:
        raise ValueError(
            "schema-invalid: hourly binning requires timestamps; "
            "got None (use row order instead)"
        )
    n_rows = frame.n_rows()
    if int(frame.timestamps.shape[0]) != n_rows:
        raise ValueError(
            "schema-invalid: timestamps length "
            f"{frame.timestamps.shape[0]} != n_rows={n_rows}"
        )
    if n_rows <= 0:
        raise ValueError(
            "schema-invalid: cannot bin an empty ValidatedFrame"
        )
    stamps = _as_hourly_timestamps(frame.timestamps)
    order = np.argsort(stamps, kind="mergesort")
    stamps_sorted = stamps[order]
    values_sorted = frame.values[order]
    masks_sorted = frame.native_mask[order]
    buffers = _allocate_hourly_grid(
        stamps_sorted[0],
        stamps_sorted[-1],
        frame.n_features(),
    )
    _scatter_last_observed(
        stamps_sorted,
        values_sorted,
        masks_sorted,
        buffers,
    )
    return ValidatedFrame(
        values=buffers.values,
        native_mask=buffers.masks,
        feature_names=frame.feature_names,
        timestamps=buffers.grid,
        entity_id=frame.entity_id,
    )


def adapt_flexible_frame(
        frame: ValidatedFrame,
        request: FlexibleAdaptRequest,
) -> AdaptedWindows:
    """Flexible path: optional hourly bin, reject policy, then shared pad.

    Does **not** rename features to Beijing names. Callers that need Exact
    Beijing pretrained impute should keep using ``adapt_validated_frame``.

    Parameters
    ----------
    frame : ValidatedFrame
        Flexible-schema (or any) validated table in original units.
    request : FlexibleAdaptRequest
        Window, reject, and bin knobs.

    Returns
    -------
    AdaptedWindows
        ``(B, T, F)`` windows; pad cells have ``m = 0``.

    Raises
    ------
    ValueError
        User-facing too-short / too-sparse / schema-invalid rejects, or
        invalid knobs.
    """
    request.validate()
    grid_frame = frame
    if request.apply_hourly_bin and frame.timestamps is not None:
        grid_frame = bin_validated_frame_hourly(frame)
    request.reject.assert_acceptable(grid_frame)
    return adapt_validated_frame(
        grid_frame,
        request.as_window_request(),
    )


def adapt_validated_frame(
        frame: ValidatedFrame,
        request: WindowAdaptRequest,
) -> AdaptedWindows:
    """Build non-overlapping ``(B, T, F)`` windows with shared pad policy.

    Shorter-than-``T`` series and trailing remainders are **right-padded**
    with sentinel ``0.0`` and ``m = 0``. Exact Beijing / M4 does **not**
    apply M6 min-length or sparsity rejects here (use
    ``adapt_flexible_frame`` for those).

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


def _as_hourly_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Cast frame timestamps to ``datetime64[h]``.

    Parameters
    ----------
    timestamps : numpy.ndarray
        Length-``L`` wall-clock labels.

    Returns
    -------
    numpy.ndarray
        Hour-resolution timestamps.

    Raises
    ------
    ValueError
        If values are not datetime-like.
    """
    try:
        stamps = np.asarray(timestamps, dtype=f"datetime64[{_HOUR_UNIT}]")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "schema-invalid: timestamps are present but not parseable "
            "as datetimes for hourly binning"
        ) from exc
    if np.any(np.isnat(stamps)):
        raise ValueError(
            "schema-invalid: timestamps contain NaT after hourly cast"
        )
    return stamps


def _allocate_hourly_grid(
        start: np.datetime64,
        stop: np.datetime64,
        n_features: int,
) -> _HourlyGridBuffers:
    """Allocate a dense hourly grid filled with native missing.

    Parameters
    ----------
    start : numpy.datetime64
        Inclusive start hour.
    stop : numpy.datetime64
        Inclusive stop hour.
    n_features : int
        Feature count ``F``.

    Returns
    -------
    _HourlyGridBuffers
        Grid axis plus sentinel / missing buffers.
    """
    grid = np.arange(start, stop + _HOUR_DELTA, _HOUR_DELTA)
    values = np.full(
        (grid.size, n_features),
        float(MISSING_VALUE_SENTINEL),
        dtype=np.float32,
    )
    masks = np.full(
        (grid.size, n_features),
        float(MASK_MISSING),
        dtype=np.float32,
    )
    return _HourlyGridBuffers(grid=grid, values=values, masks=masks)


def _scatter_last_observed(
        stamps: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        buffers: _HourlyGridBuffers,
) -> None:
    """Scatter sorted rows onto ``buffers``; later observed values overwrite.

    Parameters
    ----------
    stamps : numpy.ndarray
        Sorted source timestamps.
    values : numpy.ndarray
        Sorted source values.
    masks : numpy.ndarray
        Sorted source native masks.
    buffers : _HourlyGridBuffers
        Destination grid mutated in place.
    """
    start = buffers.grid[0]
    for row_idx, ts in enumerate(stamps):
        grid_idx = int((ts - start) / _HOUR_DELTA)
        observed = masks[row_idx] >= (MASK_OBSERVED - _MASK_COMPARE_TOL)
        buffers.values[grid_idx, observed] = values[row_idx, observed]
        buffers.masks[grid_idx, observed] = MASK_OBSERVED
