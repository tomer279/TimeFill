"""Build downloadable frames from M4 impute results (no Streamlit).

Classes
-------
MeanFillExportRequest
    Window tensors plus feature names for the mean-fill table.
SampleExportRequest
    Posterior samples plus feature names for long-format export.

Functions
---------
mean_fill_dataframe
    Wide table: ``window_id``, ``t_index``, feature columns.
samples_long_dataframe
    Long table: ``sample_id``, ``window_id``, ``t_index``, ``feature``, ``value``.
dataframe_to_csv_bytes
    UTF-8 CSV bytes for Streamlit ``download_button``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore  # pylint: disable=import-error

from src.data.schema import BATCH_AXIS, FEATURE_AXIS, TIME_AXIS

_WINDOW_COL = "window_id"
_T_COL = "t_index"
_SAMPLE_COL = "sample_id"
_FEATURE_COL = "feature"
_VALUE_COL = "value"
_SAMPLE_AXIS = 1
_RANK4_TIME_AXIS = 2
_RANK4_FEATURE_AXIS = 3
_EXPECTED_MEAN_RANK = 3
_EXPECTED_SAMPLE_RANK = 4


@dataclass(frozen=True)
class MeanFillExportRequest:
    """Inputs for a mean-fill CSV table.

    Public methods
    --------------
    validate
        Check ranks / feature-name length.
    n_windows
        Batch size ``B``.

    Instance variables
    ------------------
    mean_fill : numpy.ndarray
        ``(B, T, F)`` original-unit mean fill.
    feature_names : tuple of str
        Feature axis labels.
    """

    mean_fill: np.ndarray
    feature_names: tuple[str, ...]

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
        Check ranks / feature-name length.
    n_samples
        Posterior sample count ``K``.

    Instance variables
    ------------------
    samples : numpy.ndarray
        ``(B, K, T, F)`` original-unit samples.
    feature_names : tuple of str
        Feature axis labels.
    """

    samples: np.ndarray
    feature_names: tuple[str, ...]

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

    def n_samples(self) -> int:
        """Return ``K``.

        Returns
        -------
        int
            Number of posterior samples.
        """
        return int(self.samples.shape[_SAMPLE_AXIS])


def mean_fill_dataframe(request: MeanFillExportRequest) -> pd.DataFrame:
    """Build a wide mean-fill table in original units (vectorized).

    Parameters
    ----------
    request : MeanFillExportRequest
        Mean-fill tensor and feature names.

    Returns
    -------
    pandas.DataFrame
        Columns ``window_id``, ``t_index``, then each feature.
    """
    request.validate()
    mean_fill = request.mean_fill
    n_windows = int(mean_fill.shape[BATCH_AXIS])
    seq_len = int(mean_fill.shape[TIME_AXIS])
    n_features = int(mean_fill.shape[FEATURE_AXIS])
    flat = mean_fill.reshape(n_windows * seq_len, n_features)
    frame = pd.DataFrame(flat, columns=list(request.feature_names))
    frame.insert(0, _T_COL, np.tile(np.arange(seq_len), n_windows))
    frame.insert(0, _WINDOW_COL, np.repeat(np.arange(n_windows), seq_len))
    return frame


def samples_long_dataframe(request: SampleExportRequest) -> pd.DataFrame:
    """Build a long multi-sample table in original units (vectorized).

    Parameters
    ----------
    request : SampleExportRequest
        Posterior samples and feature names.

    Returns
    -------
    pandas.DataFrame
        Columns ``sample_id``, ``window_id``, ``t_index``, ``feature``, ``value``.
    """
    request.validate()
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
    frame = stacked.rename(_VALUE_COL).reset_index()
    return frame[
        [_SAMPLE_COL, _WINDOW_COL, _T_COL, _FEATURE_COL, _VALUE_COL]
    ]


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
