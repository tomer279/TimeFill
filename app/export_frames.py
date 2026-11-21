"""Build downloadable frames from M4 impute results (no Streamlit).

Classes
-------
MeanFillExportRequest
    Mean-fill tensor, feature names, optional pad provenance.
SampleExportRequest
    Posterior samples, feature names, optional pad provenance.

Functions
---------
mean_fill_dataframe
    Wide best-estimate table; drops pads when provenance is set.
samples_long_dataframe
    Long multi-sample table; drops pads when provenance is set.
mean_fill_request_from_windows
    Build an honest ``MeanFillExportRequest`` from ``AdaptedWindows``.
sample_request_from_windows
    Build an honest ``SampleExportRequest`` (native cells frozen).
covered_row_range
    Inclusive min/max non-pad ``row_index`` for partial filenames.
mean_fill_download_filename
    Mean-fill CSV name (optional stem; partial ``rows_a-b``).
samples_long_download_filename
    Long-samples CSV name (same partial pattern).
truncation_warning_message
    User-facing window-cap truncation warning.
dataframe_to_csv_bytes
    UTF-8 CSV bytes for Streamlit ``download_button``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore  # pylint: disable=import-error

from app.window_adapter import AdaptedWindows
from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_OBSERVED,
    TIME_AXIS,
)

_WINDOW_COL = "window_id"
_T_COL = "t_index"
_ROW_COL = "row_index"
_TIMESTAMP_COL = "timestamp"
_ENTITY_COL = "entity"
_SAMPLE_COL = "sample_id"
_FEATURE_COL = "feature"
_VALUE_COL = "value"
_IMPUTED_COL = "is_imputed"
_IMPUTED_SUFFIX = "_imputed"
_SAMPLE_AXIS = 1
_RANK4_TIME_AXIS = 2
_RANK4_FEATURE_AXIS = 3
_EXPECTED_MEAN_RANK = 3
_EXPECTED_SAMPLE_RANK = 4
_MASK_TOL = 1e-6
_DEFAULT_MEAN_FILL_STEM = "timefill_mean_fill"
_DEFAULT_SAMPLES_LONG_STEM = "timefill_samples_long"


@dataclass(frozen=True)
class MeanFillExportRequest:
    """Inputs for a mean-fill CSV table.

    Public methods
    --------------
    validate
        Check ranks / feature-name length / provenance consistency.
    n_windows
        Batch size ``B``.
    has_provenance
        True when pad provenance is present for honest export.

    Instance variables
    ------------------
    mean_fill : numpy.ndarray
        ``(B, T, F)`` original-unit mean fill.
    feature_names : tuple of str
        Feature axis labels.
    m_native : numpy.ndarray or None
        Native / pad mask; required with ``row_index``.
    valid_len : numpy.ndarray or None
        Optional ``(B,)``; checked against ``row_index`` when set.
    row_index : numpy.ndarray or None
        ``(B, T)`` source rows; ``-1`` on pads unlocks honest export.
    timestamps : numpy.ndarray or None
        Optional ``(B, T)`` timestamps.
    entity : str or None
        Optional entity label broadcast to every row.
    """

    mean_fill: np.ndarray
    feature_names: tuple[str, ...]
    m_native: np.ndarray | None = None
    valid_len: np.ndarray | None = None
    row_index: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    entity: str | None = None

    def has_provenance(self) -> bool:
        """Return whether honest pad-aware export is requested.

        Returns
        -------
        bool
            True when ``row_index`` is set.
        """
        return self.row_index is not None

    def validate(self) -> None:
        """Raise ``ValueError`` if the export request is inconsistent."""
        if self.mean_fill.ndim != _EXPECTED_MEAN_RANK:
            raise ValueError(
                "mean_fill must be rank-3 (B, T, F), "
                f"got {self.mean_fill.shape}"
            )
        n_features = int(self.mean_fill.shape[FEATURE_AXIS])
        if n_features != len(self.feature_names):
            raise ValueError(
                "feature_names length must match F; "
                f"got {len(self.feature_names)} vs {n_features}"
            )
        if not self.has_provenance():
            return
        if self.m_native is None:
            raise ValueError(
                "honest export requires m_native with row_index"
            )
        if self.m_native.shape != self.mean_fill.shape:
            raise ValueError("m_native shape must match mean_fill")
        row_index = np.asarray(self.row_index)
        if row_index.shape[:2] != self.mean_fill.shape[:2]:
            raise ValueError("row_index batch/time must match mean_fill")
        if self.valid_len is not None:
            _assert_valid_len_matches_rows(
                np.asarray(self.valid_len),
                row_index,
            )

    def n_windows(self) -> int:
        """Return ``B``.

        Returns
        -------
        int
            Number of windows.
        """
        return int(self.mean_fill.shape[BATCH_AXIS])


@dataclass(frozen=True)
class SampleExportRequest:
    """Inputs for a multi-sample long-format CSV.

    Public methods
    --------------
    validate
        Check ranks / feature-name length / provenance consistency.
    n_samples
        Posterior sample count ``K``.
    has_provenance
        True when pad provenance is present for honest export.

    Instance variables
    ------------------
    samples : numpy.ndarray
        ``(B, K, T, F)`` original-unit samples.
    feature_names : tuple of str
        Feature axis labels.
    m_native : numpy.ndarray or None
        Native / pad mask; required with ``row_index``.
    valid_len : numpy.ndarray or None
        Optional ``(B,)``; checked against ``row_index`` when set.
    row_index : numpy.ndarray or None
        ``(B, T)`` source rows; ``-1`` on pads unlocks honest export.
    timestamps : numpy.ndarray or None
        Optional ``(B, T)`` timestamps carried into the long CSV.
    """

    samples: np.ndarray
    feature_names: tuple[str, ...]
    m_native: np.ndarray | None = None
    valid_len: np.ndarray | None = None
    row_index: np.ndarray | None = None
    timestamps: np.ndarray | None = None

    def has_provenance(self) -> bool:
        """Return whether honest pad-aware export is requested.

        Returns
        -------
        bool
            True when ``row_index`` is set.
        """
        return self.row_index is not None

    def validate(self) -> None:
        """Raise ``ValueError`` if the export request is inconsistent."""
        if self.samples.ndim != _EXPECTED_SAMPLE_RANK:
            raise ValueError(
                "samples must be rank-4 (B, K, T, F), "
                f"got {self.samples.shape}"
            )
        n_features = int(self.samples.shape[_RANK4_FEATURE_AXIS])
        if n_features != len(self.feature_names):
            raise ValueError(
                "feature_names length must match F; "
                f"got {len(self.feature_names)} vs {n_features}"
            )
        if not self.has_provenance():
            return
        if self.m_native is None:
            raise ValueError(
                "honest export requires m_native with row_index"
            )
        batch = int(self.samples.shape[BATCH_AXIS])
        seq_len = int(self.samples.shape[_RANK4_TIME_AXIS])
        n_features = int(self.samples.shape[_RANK4_FEATURE_AXIS])
        if self.m_native.shape != (batch, seq_len, n_features):
            raise ValueError("m_native shape must match (B, T, F)")
        row_index = np.asarray(self.row_index)
        if row_index.shape[:2] != (batch, seq_len):
            raise ValueError("row_index batch/time must match samples")
        if self.valid_len is not None:
            _assert_valid_len_matches_rows(
                np.asarray(self.valid_len),
                row_index,
            )

    def n_samples(self) -> int:
        """Return ``K``.

        Returns
        -------
        int
            Number of posterior samples.
        """
        return int(self.samples.shape[_SAMPLE_AXIS])


def mean_fill_request_from_windows(
        mean_fill: np.ndarray,
        windows: AdaptedWindows,
) -> MeanFillExportRequest:
    """Build an honest mean-fill export request from adapted windows.

    Parameters
    ----------
    mean_fill : numpy.ndarray
        ``(B, T, F)`` original-unit mean fill.
    windows : AdaptedWindows
        Windows that produced the impute (must carry provenance).

    Returns
    -------
    MeanFillExportRequest
        Request with ``m_native`` / ``row_index`` / optional stamps.

    Raises
    ------
    ValueError
        If ``windows`` lack pad provenance.
    """
    if windows.row_index is None or windows.valid_len is None:
        raise ValueError(
            "AdaptedWindows must carry row_index and valid_len for export"
        )
    entity = _entity_label(windows)
    return MeanFillExportRequest(
        mean_fill=mean_fill,
        feature_names=tuple(windows.feature_names),
        m_native=windows.native_mask,
        valid_len=windows.valid_len,
        row_index=windows.row_index,
        timestamps=windows.timestamps,
        entity=entity,
    )


def sample_request_from_windows(
        samples: np.ndarray,
        windows: AdaptedWindows,
) -> SampleExportRequest:
    """Build an honest long-sample export request from adapted windows.

    Native-observed cells are overwritten with ``windows.values`` so demo
    masks cannot mislabel model draws as observed.

    Parameters
    ----------
    samples : numpy.ndarray
        ``(B, K, T, F)`` original-unit samples.
    windows : AdaptedWindows
        Windows that produced the impute (must carry provenance).

    Returns
    -------
    SampleExportRequest
        Request with frozen native cells and ``row_index``.

    Raises
    ------
    ValueError
        If ``windows`` lack pad provenance.
    """
    if windows.row_index is None or windows.valid_len is None:
        raise ValueError(
            "AdaptedWindows must carry row_index and valid_len for export"
        )
    return SampleExportRequest(
        samples=_freeze_native_in_samples(
            samples,
            windows.values,
            windows.native_mask,
        ),
        feature_names=tuple(windows.feature_names),
        m_native=windows.native_mask,
        valid_len=windows.valid_len,
        row_index=windows.row_index,
        timestamps=windows.timestamps,
    )


def _freeze_native_in_samples(
        samples: np.ndarray,
        observations: np.ndarray,
        m_native: np.ndarray,
) -> np.ndarray:
    """Overwrite ``m_native == 1`` sample cells with native observations."""
    observed = np.abs(m_native - MASK_OBSERVED) < _MASK_TOL
    return np.where(
        observed[:, None, :, :],
        observations[:, None, :, :].astype(np.float32),
        samples.astype(np.float32),
    ).astype(np.float32)


def covered_row_range(windows: AdaptedWindows) -> tuple[int, int]:
    """Return inclusive ``(row_start, row_end)`` over non-pad ``row_index``.

    Parameters
    ----------
    windows : AdaptedWindows
        Windows with pad provenance.

    Returns
    -------
    tuple of int
        Min and max source/grid row among real cells.

    Raises
    ------
    ValueError
        If provenance is missing or every cell is pad.
    """
    if windows.row_index is None:
        raise ValueError("row_index required for covered_row_range")
    real = windows.row_index[windows.row_index >= 0]
    if real.size == 0:
        raise ValueError("no real rows in row_index")
    return int(real.min()), int(real.max())


def mean_fill_download_filename(
        is_partial: bool,
        row_start: int = 0,
        row_end: int = 0,
        stem: str = _DEFAULT_MEAN_FILL_STEM,
) -> str:
    """Return a mean-fill CSV name (full or partial ``rows_a-b``).

    Parameters
    ----------
    is_partial : bool
        Whether ``max_windows`` truncated the series.
    row_start : int
        First covered source/grid row (inclusive).
    row_end : int
        Last covered source/grid row (inclusive).
    stem : str
        Filename stem without ``.csv`` (UI default ``timefill_mean_fill``;
        CLI uses ``mean_fill``).

    Returns
    -------
    str
        Download file name.
    """
    if not is_partial:
        return f"{stem}.csv"
    return f"{stem}_partial_rows_{row_start}-{row_end}.csv"


def samples_long_download_filename(
        is_partial: bool,
        row_start: int = 0,
        row_end: int = 0,
        stem: str = _DEFAULT_SAMPLES_LONG_STEM,
) -> str:
    """Return a long-samples CSV name (full or partial ``rows_a-b``)."""
    return mean_fill_download_filename(
        is_partial,
        row_start,
        row_end,
        stem,
    )


def truncation_warning_message(
        filled_rows: int,
        total_rows: int,
        cap_knob: str = "max_windows",
) -> str:
    """Return the window-cap truncation warning text.

    Parameters
    ----------
    filled_rows : int
        Source/grid rows covered by the kept windows.
    total_rows : int
        Source/grid rows before the cap (Exact: input rows; Flexible: post-bin).
    cap_knob : str
        Config/CLI knob that applied the cap (e.g. ``smoke.max_windows``).

    Returns
    -------
    str
        User-facing warning.
    """
    return (
        f"Filled the first {filled_rows} of {total_rows} rows; "
        f"raise {cap_knob} for more of the series."
    )


def mean_fill_dataframe(request: MeanFillExportRequest) -> pd.DataFrame:
    """Build a wide mean-fill table in original units.

    With provenance: one row per real source/grid index; pads dropped;
    ``*_imputed`` flags native gaps. Without provenance: legacy ``B*T``
    ``window_id`` / ``t_index`` table.

    Parameters
    ----------
    request : MeanFillExportRequest
        Mean-fill tensor and optional provenance.

    Returns
    -------
    pandas.DataFrame
        Download table.
    """
    request.validate()
    if not request.has_provenance():
        return _legacy_mean_fill_dataframe(request)
    return _honest_mean_fill_dataframe(request)


def samples_long_dataframe(request: SampleExportRequest) -> pd.DataFrame:
    """Build a long multi-sample table in original units.

    With provenance: pad cells omitted; ``is_imputed`` from ``m_native``.
    Without provenance: legacy ``B*K*T*F`` rows.

    Parameters
    ----------
    request : SampleExportRequest
        Posterior samples and optional provenance.

    Returns
    -------
    pandas.DataFrame
        Long download table.
    """
    request.validate()
    if not request.has_provenance():
        return _legacy_samples_long_dataframe(request)
    return _honest_samples_long_dataframe(request)


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a dataframe to UTF-8 CSV bytes (no index).

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to download.

    Returns
    -------
    bytes
        CSV payload for ``st.download_button``.
    """
    return frame.to_csv(index=False).encode("utf-8")


def _entity_label(windows: AdaptedWindows) -> str | None:
    """Return a non-empty entity label from the first window, if any."""
    if windows.entity_ids.size <= 0:
        return None
    label = str(windows.entity_ids[0])
    if not label:
        return None
    return label


def _assert_valid_len_matches_rows(
        valid_len: np.ndarray,
        row_index: np.ndarray,
) -> None:
    """Raise when ``valid_len`` disagrees with non-pad ``row_index`` counts."""
    n_windows = int(row_index.shape[BATCH_AXIS])
    if valid_len.shape != (n_windows,):
        raise ValueError(
            f"valid_len must have shape ({n_windows},), got {valid_len.shape}"
        )
    for window_i in range(n_windows):
        n_real = int(np.sum(row_index[window_i] >= 0))
        if n_real != int(valid_len[window_i]):
            raise ValueError(
                "valid_len disagrees with row_index at window "
                f"{window_i}: valid_len={int(valid_len[window_i])} "
                f"vs non-pad={n_real}"
            )


def _legacy_mean_fill_dataframe(
        request: MeanFillExportRequest,
) -> pd.DataFrame:
    """Legacy wide table: one row per ``(window, t)`` including pads."""
    mean_fill = request.mean_fill
    n_windows = int(mean_fill.shape[BATCH_AXIS])
    seq_len = int(mean_fill.shape[TIME_AXIS])
    n_features = int(mean_fill.shape[FEATURE_AXIS])
    flat = mean_fill.reshape(n_windows * seq_len, n_features)
    frame = pd.DataFrame(flat, columns=list(request.feature_names))
    frame.insert(0, _T_COL, np.tile(np.arange(seq_len), n_windows))
    frame.insert(0, _WINDOW_COL, np.repeat(np.arange(n_windows), seq_len))
    return frame


def _honest_mean_fill_dataframe(
        request: MeanFillExportRequest,
) -> pd.DataFrame:
    """Wide table with pads dropped and ``*_imputed`` flags."""
    row_index = np.asarray(request.row_index)
    real = row_index >= 0
    rows = row_index[real]
    values = request.mean_fill[real]
    native = np.asarray(request.m_native)[real]
    order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    values = values[order]
    native = native[order]
    data: dict[str, object] = {_ROW_COL: rows}
    if request.timestamps is not None:
        data[_TIMESTAMP_COL] = np.asarray(request.timestamps)[real][order]
    if request.entity is not None:
        data[_ENTITY_COL] = np.full(len(rows), request.entity, dtype=object)
    for feat_i, name in enumerate(request.feature_names):
        data[name] = values[:, feat_i]
        imputed = np.abs(native[:, feat_i] - MASK_OBSERVED) >= _MASK_TOL
        data[f"{name}{_IMPUTED_SUFFIX}"] = imputed
    return pd.DataFrame(data)


def _legacy_samples_long_dataframe(
        request: SampleExportRequest,
) -> pd.DataFrame:
    """Legacy long table: ``B * K * T * F`` rows including pads."""
    samples = request.samples
    n_windows = int(samples.shape[BATCH_AXIS])
    n_samples = int(samples.shape[_SAMPLE_AXIS])
    seq_len = int(samples.shape[_RANK4_TIME_AXIS])
    n_features = int(samples.shape[_RANK4_FEATURE_AXIS])
    index = pd.MultiIndex.from_product(
        [
            range(n_windows),
            range(n_samples),
            range(seq_len),
        ],
        names=[_WINDOW_COL, _SAMPLE_COL, _T_COL],
    )
    wide = pd.DataFrame(
        samples.reshape(n_windows * n_samples * seq_len, n_features),
        index=index,
        columns=list(request.feature_names),
    )
    wide.columns.name = _FEATURE_COL
    stacked = wide.stack()
    if not isinstance(stacked, pd.Series):
        raise TypeError(
            "expected stack() to return a Series for single-level columns"
        )
    frame = stacked.reset_index(name=_VALUE_COL)
    return frame[
        [_SAMPLE_COL, _WINDOW_COL, _T_COL, _FEATURE_COL, _VALUE_COL]
    ]


def _honest_samples_long_dataframe(
        request: SampleExportRequest,
) -> pd.DataFrame:
    """Long table with pads dropped, ``row_index``, and ``is_imputed``."""
    samples_bt = np.transpose(request.samples, (0, 2, 1, 3))
    row_index = np.asarray(request.row_index)
    real = row_index >= 0
    order = np.argsort(row_index[real], kind="mergesort")
    vals = samples_bt[real][order]
    native = np.asarray(request.m_native)[real][order]
    rows = row_index[real][order]
    stamps = None
    if request.timestamps is not None:
        stamps = np.asarray(request.timestamps)[real][order]
    return _long_frame_from_real_vals(
        vals,
        native,
        request.feature_names,
        rows,
        stamps,
    )


def _long_frame_from_real_vals(
        vals: np.ndarray,
        native: np.ndarray,
        feature_names: tuple[str, ...],
        rows: np.ndarray,
        timestamps: np.ndarray | None = None,
) -> pd.DataFrame:
    """Stack ``(n_real, K, F)`` samples into a long table with time keys."""
    n_real = int(vals.shape[0])
    n_samples = int(vals.shape[1])
    flat = vals.reshape(n_real * n_samples, int(vals.shape[2]))
    wide = pd.DataFrame(flat, columns=list(feature_names))
    wide.insert(0, _SAMPLE_COL, np.tile(np.arange(n_samples), n_real))
    wide.insert(0, "_real", np.repeat(np.arange(n_real), n_samples))
    stacked = wide.set_index([_SAMPLE_COL, "_real"]).stack()
    if not isinstance(stacked, pd.Series):
        raise TypeError(
            "expected stack() to return a Series for single-level columns"
        )
    out = stacked.reset_index(name=_VALUE_COL)
    feature_col = out.columns[2]
    out = out.rename(columns={feature_col: _FEATURE_COL})
    real_ids = out["_real"].to_numpy()
    feat_ids = out[_FEATURE_COL].map(
        {name: feat_i for feat_i, name in enumerate(feature_names)}
    ).to_numpy()
    out[_ROW_COL] = rows[real_ids]
    out[_IMPUTED_COL] = (
        np.abs(native[real_ids, feat_ids] - MASK_OBSERVED) >= _MASK_TOL
    )
    columns = [_SAMPLE_COL, _ROW_COL]
    if timestamps is not None:
        out[_TIMESTAMP_COL] = timestamps[real_ids]
        columns.append(_TIMESTAMP_COL)
    columns.extend([_FEATURE_COL, _VALUE_COL, _IMPUTED_COL])
    return out[columns]
