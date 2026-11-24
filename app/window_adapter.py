"""Adapt validated CSV frames into ``(x, m)`` windows.

Native NaNs / empty cells are already sentinel ``0.0`` with ``m == 0`` in
``ValidatedFrame``. Shared Exact / Flexible pad policy (ASSUMPTIONS M4 / M6):
non-overlapping windows of length ``T``, right-pad short series and trailing
remainders with sentinel ``0.0`` and ``m = 0``, then optionally cap with
``max_windows``.

**Shared (Exact Beijing + Flexible):** ``adapt_validated_frame`` /
``WindowAdaptRequest`` pad / multi-window / cap (no min-length / sparsity
reject; short series still pad).

**Exact with timestamps (M7.5):** ``adapt_exact_frame`` hourly-bins (same
last-value path + duplicate-hour reject) then shared pad. Without timestamps,
row order only.

**Flexible-only (M6):** ``adapt_flexible_frame`` optionally bins, then
enforces ASSUMPTIONS M6 reject thresholds before the shared pad path.
Padded cells are never ground-truth imputation targets.

Classes
-------
WindowAdaptRequest
    Window length and optional ``max_windows``.
WindowPadInfo
    Pad/cap provenance for honest export.
AdaptedWindows
    Stacked ``(B, T, F)`` values/masks.
IngestRejectPolicy
    M6 flexible reject thresholds.
FlexibleAdaptRequest
    Flexible window/reject/bin knobs.

Functions
---------
adapt_validated_frame
    Cut / pad non-overlapping windows (Exact + shared pad core).
adapt_exact_frame
    Exact path: optional hourly bin, then ``adapt_validated_frame``.
default_ingest_reject_policy
    ASSUMPTIONS defaults from ``seq_len``.
bin_validated_frame_hourly
    Hourly last-value-in-bin grid when timestamps are present.
multi_row_hour_warnings
    Preview warnings when hours contain multiple raw rows.
adapt_flexible_frame
    Bin → reject → shared pad.
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
_PAD_ROW_INDEX = -1
# shortcut: fixed 0.01 threshold, promote to ingest_m6.yaml if product needs a dial.
_MAX_DUPLICATE_HOUR_SHARE = 0.01
_MIN_ROWS_PER_DUP_HOUR = 2


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
class WindowPadInfo:
    """Pad / cap provenance for honest download export.

    Public methods
    --------------
    is_partial
        True when fewer windows were kept than ``uncapped_n_windows``.
    take
        Slice provenance along the batch axis (e.g. smoke indices).

    Instance variables
    ------------------
    valid_len : numpy.ndarray
        ``(B,)`` real timesteps per window (excludes right-pad).
    row_index : numpy.ndarray
        ``(B, T)`` source/grid row; ``-1`` on pad cells.
    timestamps : numpy.ndarray or None
        ``(B, T)`` wall-clock labels; ``NaT`` / ``None`` on pad when present.
    source_n_rows : int or None
        Grid length before window/cap.
    uncapped_n_windows : int or None
        Window count before ``max_windows`` slice.
    """

    valid_len: np.ndarray
    row_index: np.ndarray
    timestamps: np.ndarray | None = None
    source_n_rows: int | None = None
    uncapped_n_windows: int | None = None

    def is_partial(self, n_kept: int) -> bool:
        """Return whether the window cap dropped trailing windows.

        Parameters
        ----------
        n_kept : int
            Number of windows retained after the cap.

        Returns
        -------
        bool
            True when ``n_kept`` is below ``uncapped_n_windows``.
        """
        if self.uncapped_n_windows is None:
            return False
        return int(n_kept) < int(self.uncapped_n_windows)

    def take(self, indices: np.ndarray) -> "WindowPadInfo":
        """Return provenance for a batch-axis subset.

        Parameters
        ----------
        indices : numpy.ndarray
            Window indices to keep.

        Returns
        -------
        WindowPadInfo
            Sliced ``valid_len`` / ``row_index`` / timestamps.
        """
        stamps = None
        if self.timestamps is not None:
            stamps = np.asarray(self.timestamps[indices])
        return WindowPadInfo(
            valid_len=np.asarray(self.valid_len[indices]),
            row_index=np.asarray(self.row_index[indices]),
            timestamps=stamps,
            source_n_rows=self.source_n_rows,
            uncapped_n_windows=self.uncapped_n_windows,
        )


@dataclass(frozen=True)
class AdaptedWindows:
    """Stacked fixed-length windows under the M1/M2 tensor contract.

    Public methods
    --------------
    n_windows
        Batch size ``B``.
    as_batch
        Pack into a JAX ``TimeSeriesBatch``.
    is_partial
        True when ``max_windows`` dropped trailing windows.

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
    pad : WindowPadInfo or None
        Pad / cap provenance; ``None`` for legacy hand-built batches.
    """

    values: np.ndarray
    native_mask: np.ndarray
    feature_names: tuple[str, ...]
    entity_ids: np.ndarray
    pad: WindowPadInfo | None = None

    @property
    def valid_len(self) -> np.ndarray | None:
        """``(B,)`` real lengths, or ``None`` when provenance is absent."""
        return None if self.pad is None else self.pad.valid_len

    @property
    def row_index(self) -> np.ndarray | None:
        """``(B, T)`` source rows, or ``None`` when provenance is absent."""
        return None if self.pad is None else self.pad.row_index

    @property
    def timestamps(self) -> np.ndarray | None:
        """``(B, T)`` timestamps, or ``None`` when absent."""
        return None if self.pad is None else self.pad.timestamps

    @property
    def source_n_rows(self) -> int | None:
        """Source/grid length before window/cap, if known."""
        return None if self.pad is None else self.pad.source_n_rows

    @property
    def uncapped_n_windows(self) -> int | None:
        """Window count before ``max_windows``, if known."""
        return None if self.pad is None else self.pad.uncapped_n_windows

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

    def is_partial(self) -> bool:
        """Return whether the window cap dropped trailing windows.

        Returns
        -------
        bool
            True when fewer windows were kept than the uncapped cut.
        """
        if self.pad is None:
            return False
        return self.pad.is_partial(self.n_windows())


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
    hours keep the last **observed** value per feature. Exact same-second
    repeats contribute to a duplicate-hour share; above
    ``_MAX_DUPLICATE_HOUR_SHARE`` the frame is rejected. Timestamps are for
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
        If timestamps are missing, length-mismatched, not datetime-like, or
        the duplicate-hour share is too high.
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
    stamps_sec = np.asarray(frame.timestamps, dtype="datetime64[s]")
    stamps = _as_hourly_timestamps(frame.timestamps)
    order = np.argsort(stamps, kind="mergesort")
    stamps_sorted = stamps[order]
    values_sorted = frame.values[order]
    masks_sorted = frame.native_mask[order]
    _assert_duplicate_hour_share(stamps_sorted, stamps_sec[order])
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


def multi_row_hour_warnings(frame: ValidatedFrame) -> tuple[str, ...]:
    """Return preview warnings when hourly bins collapse multiple raw rows.

    Counts hours (after ``datetime64[h]`` cast) with ≥2 raw rows. Exact
    same-second rejects still happen inside ``bin_validated_frame_hourly``;
    this only warns about silent last-value collapses (including sub-hourly).

    Parameters
    ----------
    frame : ValidatedFrame
        Pre-bin parse result (warnings are empty when timestamps are absent).

    Returns
    -------
    tuple of str
        Zero or one user-facing warning string.
    """
    if frame.timestamps is None:
        return ()
    try:
        stamps = _as_hourly_timestamps(frame.timestamps)
    except ValueError:
        return ()
    _uniq, counts = np.unique(stamps, return_counts=True)
    n_multi = int(np.sum(counts >= _MIN_ROWS_PER_DUP_HOUR))
    if n_multi <= 0:
        return ()
    return (
        f"{n_multi} hours had multiple rows; kept last value",
    )


def adapt_exact_frame(
        frame: ValidatedFrame,
        request: WindowAdaptRequest,
) -> AdaptedWindows:
    """Exact Beijing path: hourly bin when timestamps exist, then pad/cut.

    Does **not** apply Flexible ``IngestRejectPolicy`` floors. Multi-entity and
    non-finite rejects happen earlier in schema validation.

    Parameters
    ----------
    frame : ValidatedFrame
        Exact-schema table in original units.
    request : WindowAdaptRequest
        ``seq_len`` / optional ``max_windows``.

    Returns
    -------
    AdaptedWindows
        ``(B, T, F)`` windows; pad cells have ``m = 0``.

    Raises
    ------
    ValueError
        From hourly binning (duplicate hours, bad timestamps) or window knobs.
    """
    grid_frame = frame
    if frame.timestamps is not None:
        grid_frame = bin_validated_frame_hourly(frame)
    return adapt_validated_frame(grid_frame, request)


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
    cut = _cut_padded_windows(
        frame.values,
        frame.native_mask,
        request.seq_len,
        frame.timestamps,
    )
    uncapped_n_windows = len(cut[0])
    max_windows = request.effective_max_windows()
    if max_windows is not None:
        cut = _truncate_cut(cut, max_windows)
    return _adapted_from_cut(
        cut,
        frame,
        request.seq_len,
        uncapped_n_windows,
    )


def _adapted_from_cut(
        cut: _CutBlocks,
        frame: ValidatedFrame,
        seq_len: int,
        uncapped_n_windows: int,
) -> AdaptedWindows:
    """Stack cut blocks into ``AdaptedWindows`` with pad provenance."""
    value_blocks, mask_blocks, valid_lens, row_blocks, stamp_blocks = cut
    entity_label = frame.entity_id if frame.entity_id is not None else ""
    entity_ids = np.asarray(
        [entity_label] * len(value_blocks),
        dtype=object,
    )
    stacked_values = np.stack(value_blocks, axis=BATCH_AXIS).astype(np.float32)
    stacked_masks = np.stack(mask_blocks, axis=BATCH_AXIS).astype(np.float32)
    _assert_window_rank3(
        stacked_values,
        seq_len,
        frame.n_features(),
    )
    stacked_stamps = None
    if stamp_blocks is not None:
        stacked_stamps = np.stack(stamp_blocks, axis=BATCH_AXIS)
    return AdaptedWindows(
        values=stacked_values,
        native_mask=stacked_masks,
        feature_names=frame.feature_names,
        entity_ids=entity_ids,
        pad=WindowPadInfo(
            valid_len=np.asarray(valid_lens, dtype=np.int64),
            row_index=np.stack(row_blocks, axis=BATCH_AXIS),
            timestamps=stacked_stamps,
            source_n_rows=int(frame.n_rows()),
            uncapped_n_windows=int(uncapped_n_windows),
        ),
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


_CutBlocks = tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    list[np.ndarray],
    list[np.ndarray] | None,
]


def _truncate_cut(cut: _CutBlocks, max_windows: int) -> _CutBlocks:
    """Keep the leading ``max_windows`` blocks from a pad cut."""
    values, masks, lengths, rows, stamps = cut
    stamps_out = None if stamps is None else stamps[:max_windows]
    return (
        values[:max_windows],
        masks[:max_windows],
        lengths[:max_windows],
        rows[:max_windows],
        stamps_out,
    )


def _cut_padded_windows(
        values: np.ndarray,
        native_mask: np.ndarray,
        seq_len: int,
        timestamps: np.ndarray | None = None,
) -> _CutBlocks:
    """Cut non-overlapping windows; right-pad short / remainder chunks.

    Parameters
    ----------
    values : numpy.ndarray
        Float32 ``(L, F)`` values.
    native_mask : numpy.ndarray
        Float32 ``(L, F)`` native masks.
    seq_len : int
        Window length ``T``.
    timestamps : numpy.ndarray or None
        Optional length-``L`` timestamps to window with the same pad policy.

    Returns
    -------
    _CutBlocks
        Value / mask / valid_len / row_index / optional timestamp blocks.
    """
    n_rows = int(values.shape[_ROW_AXIS])
    value_blocks: list[np.ndarray] = []
    mask_blocks: list[np.ndarray] = []
    valid_lens: list[int] = []
    row_blocks: list[np.ndarray] = []
    stamp_blocks: list[np.ndarray] | None = (
        [] if timestamps is not None else None
    )
    start = 0
    while start < n_rows:
        chunk = _pad_one_window(
            values,
            native_mask,
            timestamps,
            start,
            seq_len,
        )
        value_blocks.append(chunk[0])
        mask_blocks.append(chunk[1])
        valid_lens.append(chunk[2])
        row_blocks.append(chunk[3])
        if stamp_blocks is not None and chunk[4] is not None:
            stamp_blocks.append(chunk[4])
        start += seq_len
    return (
        value_blocks,
        mask_blocks,
        valid_lens,
        row_blocks,
        stamp_blocks,
    )


def _pad_one_window(
        values: np.ndarray,
        native_mask: np.ndarray,
        timestamps: np.ndarray | None,
        start: int,
        seq_len: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray | None]:
    """Build one right-padded window and its provenance rows/stamps."""
    stop = min(start + seq_len, int(values.shape[_ROW_AXIS]))
    real_len = stop - start
    n_features = int(values.shape[_FEATURE_AXIS_LF])
    chunk_values = values[start:stop]
    chunk_mask = native_mask[start:stop]
    row_idx = np.full(seq_len, _PAD_ROW_INDEX, dtype=np.int64)
    row_idx[:real_len] = np.arange(start, stop, dtype=np.int64)
    pad = seq_len - real_len
    if pad > 0:
        chunk_values = _right_pad_values(chunk_values, pad, n_features)
        chunk_mask = _right_pad_mask(chunk_mask, pad, n_features)
    stamps = None
    if timestamps is not None:
        stamps = _right_pad_timestamps(timestamps[start:stop], pad)
    return chunk_values, chunk_mask, real_len, row_idx, stamps


def _right_pad_timestamps(chunk: np.ndarray, pad: int) -> np.ndarray:
    """Right-pad timestamps with ``NaT`` / ``None``."""
    stamps = np.asarray(chunk)
    if pad <= 0:
        return stamps
    if np.issubdtype(stamps.dtype, np.datetime64):
        pad_block = np.full(pad, np.datetime64("NaT"), dtype=stamps.dtype)
    else:
        pad_block = np.full(pad, None, dtype=object)
        stamps = stamps.astype(object)
    return np.concatenate([stamps, pad_block], axis=0)


def _right_pad_values(
        chunk: np.ndarray,
        pad: int,
        n_features: int,
) -> np.ndarray:
    """Right-pad values with the missing sentinel."""
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
    """Right-pad masks with ``MASK_MISSING``."""
    pad_block = np.full(
        (pad, n_features),
        float(MASK_MISSING),
        dtype=np.float32,
    )
    return np.concatenate([chunk.astype(np.float32), pad_block], axis=_ROW_AXIS)


def _assert_duplicate_hour_share(
        stamps_hourly: np.ndarray,
        stamps_sec: np.ndarray,
) -> None:
    """Reject when too many grid hours have exact duplicate timestamps.

    Sub-hourly cadence (e.g. 00:00 and 00:30) is allowed and bins to last
    observed. Exact same-second repeats count: share = (hours containing an
    exact duplicate stamp) / (dense grid length). Soft below-threshold
    duplicates still keep last-observed.

    Parameters
    ----------
    stamps_hourly : numpy.ndarray
        Sorted ``datetime64[h]`` source stamps.
    stamps_sec : numpy.ndarray
        Sorted ``datetime64[s]`` stamps (same row order).

    Raises
    ------
    ValueError
        If the duplicate-hour share exceeds ``_MAX_DUPLICATE_HOUR_SHARE``.
    """
    uniq_sec, counts = np.unique(stamps_sec, return_counts=True)
    dup_sec = uniq_sec[counts >= _MIN_ROWS_PER_DUP_HOUR]
    if dup_sec.size == 0:
        return
    dup_hours = np.unique(dup_sec.astype(f"datetime64[{_HOUR_UNIT}]"))
    n_dup_hours = int(dup_hours.size)
    n_grid = int((stamps_hourly[-1] - stamps_hourly[0]) / _HOUR_DELTA) + 1
    share = (
        float(n_dup_hours) / float(n_grid) if n_grid > 0 else 0.0
    )
    if share > _MAX_DUPLICATE_HOUR_SHARE:
        raise ValueError(
            "duplicate-hour share "
            f"{share:.4f} exceeds {_MAX_DUPLICATE_HOUR_SHARE} "
            f"({n_dup_hours} of {n_grid} grid hours have duplicate timestamps)"
        )


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
