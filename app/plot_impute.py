"""Thin Plotly chart builders for M4 impute results (no Streamlit).

Classes
-------
SeriesPlotRequest
    One window / feature slice for gap + mean ± band charts.

Functions
---------
figure_original_with_gaps
    Observed series with missing cells as gaps.
figure_imputed_mean_bands
    Posterior mean with quantile bands; optional sample overlay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go  # type: ignore  # pylint: disable=import-error

from src.data.schema import MASK_OBSERVED

_MASK_COMPARE_TOL = 1e-6
_MAX_OVERLAY_SAMPLES = 5
_TIME_AXIS_1D = 0
_SAMPLE_AXIS = 0


@dataclass(frozen=True)
class SeriesPlotRequest:
    """One ``(T,)`` feature slice for plotting.

    Public methods
    --------------
    validate
        Check 1-D lengths agree.
    time_axis
        ``0 … T-1`` index array.

    Instance variables
    ------------------
    observations : numpy.ndarray
        Original-unit context ``(T,)`` (sentinels on gaps OK).
    mask : numpy.ndarray
        Run mask ``(T,)``; ``1`` observed.
    mean : numpy.ndarray
        Posterior mean ``(T,)``.
    quantile_low : numpy.ndarray
        Lower band ``(T,)``.
    quantile_high : numpy.ndarray
        Upper band ``(T,)``.
    samples : numpy.ndarray or None
        Optional ``(K, T)`` overlay draws.
    feature_name : str
        Series label.
    """

    observations: np.ndarray
    mask: np.ndarray
    mean: np.ndarray
    quantile_low: np.ndarray
    quantile_high: np.ndarray
    samples: np.ndarray | None
    feature_name: str

    def validate(self) -> None:
        """Raise ``ValueError`` if slice shapes are inconsistent."""
        seq_len = int(self.mean.shape[_TIME_AXIS_1D])
        for name, arr in (
            ("observations", self.observations),
            ("mask", self.mask),
            ("quantile_low", self.quantile_low),
            ("quantile_high", self.quantile_high),
        ):
            if arr.ndim != 1 or int(arr.shape[_TIME_AXIS_1D]) != seq_len:
                raise ValueError(
                    f"{name} must be length-{seq_len}, got {arr.shape}"
                )
        if self.samples is not None and self.samples.ndim != 2:
            raise ValueError(
                f"samples must be (K, T) when set, got {self.samples.shape}"
            )

    def time_axis(self) -> np.ndarray:
        """Return integer time indices.

        Returns
        -------
        numpy.ndarray
            ``arange(T)``.
        """
        return np.arange(
            int(self.mean.shape[_TIME_AXIS_1D]),
            dtype=np.int32,
        )


def figure_original_with_gaps(request: SeriesPlotRequest) -> go.Figure:
    """Plot observed values with NaN gaps where ``mask == 0``.

    Parameters
    ----------
    request : SeriesPlotRequest
        Feature slice.

    Returns
    -------
    plotly.graph_objects.Figure
        Line chart with gaps.
    """
    request.validate()
    time_axis = request.time_axis()
    observed = np.asarray(request.observations, dtype=np.float64).copy()
    is_obs = np.abs(request.mask - MASK_OBSERVED) < _MASK_COMPARE_TOL
    observed[~is_obs] = np.nan
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=observed,
            mode="lines+markers",
            name=f"{request.feature_name} (measured)",
            connectgaps=False,
        )
    )
    fig.update_layout(
        title=f"Your data (gaps shown) — {request.feature_name}",
        xaxis_title="Time step",
        yaxis_title="Original units",
    )
    return fig


def figure_imputed_mean_bands(request: SeriesPlotRequest) -> go.Figure:
    """Plot posterior mean ± bands and optional sample overlay.

    Parameters
    ----------
    request : SeriesPlotRequest
        Feature slice including quantiles.

    Returns
    -------
    plotly.graph_objects.Figure
        Mean, band, optional thin sample traces.
    """
    request.validate()
    time_axis = request.time_axis()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=request.quantile_high,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=request.quantile_low,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            name="Uncertainty band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=request.mean,
            mode="lines",
            name="Best estimate",
        )
    )
    if request.samples is not None:
        n_show = min(
            _MAX_OVERLAY_SAMPLES,
            int(request.samples.shape[_SAMPLE_AXIS]),
        )
        for sample_idx in range(n_show):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=request.samples[sample_idx],
                    mode="lines",
                    line={"width": 1},
                    opacity=0.35,
                    name=f"Draw {sample_idx}",
                    showlegend=sample_idx == 0,
                    legendgroup="samples",
                )
            )
    fig.update_layout(
        title=f"Filled series with uncertainty — {request.feature_name}",
        xaxis_title="Time step",
        yaxis_title="Original units",
    )
    return fig
