"""M3 comparison panels, artifact paths, and original-unit baseline packs.

Classes
-------
BaselineFills
    LOCF and linear fills in original units.
OriginalUnitContext
    Truth and observed context in original units.
ScoredContext
    Context plus point-baseline fills.
PanelSpec
    Title and ``(batch, feature)`` indices for a panel.
ComparisonView
    ``ImputationPlotData`` plus baselines for one PNG.
M3ArtifactPaths
    ``artifacts/m3/`` layout (tables + panels).
BenchmarkComparisonPlotter
    Truth vs TimeFill bands vs LOCF / linear on shared gaps.

Functions
---------
point_fill_as_ensemble
    Wrap a point fill as ``(B, 1, T, F)`` for CRPS / coverage.
panel_title
    Build a short comparison-panel title for one protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.data.schema import FEATURE_AXIS
from src.eval.m3_table import ProtocolKey
from src.eval.plots import ImputationPlotData
from src.plot_io import save_figure

# Headless artifact writes (after imports so pylint C0413 stays clean).
plt.switch_backend("Agg")

_METRICS_CSV = "metrics_by_rate.csv"
_METRICS_JSON = "metrics_by_rate.json"
_PANELS_DIR = "panels"
_PANEL_NAME = "comparison_timefill_locf_linear.png"
_SAMPLE_AXIS = 1
_FIRST_BATCH = 0
_FIRST_FEATURE = 0
_FIGSIZE = (9.0, 3.8)
_BAND_ALPHA = 0.25


@dataclass(frozen=True)
class BaselineFills:
    """LOCF and linear point fills in original units.

    Public methods
    --------------
    for_method
        Return the fill for ``locf`` or ``linear``.
    as_ensemble
        Wrap a named fill as ``(B, 1, T, F)``.

    Instance variables
    ------------------
    locf : jax.Array
        LOCF fill ``(B, T, F)``.
    linear : jax.Array
        Linear fill ``(B, T, F)``.
    """

    locf: jnp.ndarray
    linear: jnp.ndarray

    def for_method(self, method: str) -> jnp.ndarray:
        """Return the point fill for a baseline method name.

        Parameters
        ----------
        method : str
            ``locf`` or ``linear``.

        Returns
        -------
        jax.Array
            Point fill ``(B, T, F)``.
        """
        if method == "locf":
            return self.locf
        if method == "linear":
            return self.linear
        raise ValueError(f"unknown baseline method: {method!r}")

    def as_ensemble(self, method: str) -> jnp.ndarray:
        """Wrap a baseline fill as a degenerate ``K=1`` ensemble."""
        return point_fill_as_ensemble(self.for_method(method))


@dataclass(frozen=True)
class OriginalUnitContext:
    """Truth and observed context in original units.

    Public methods
    --------------
    with_baselines
        Pair this context with LOCF / linear fills.
    n_features
        Feature count ``F``.

    Instance variables
    ------------------
    truth : jax.Array
        Ground truth ``(B, T, F)``.
    observations : jax.Array
        Context under ``m_eval`` ``(B, T, F)``.
    """

    truth: jnp.ndarray
    observations: jnp.ndarray

    def with_baselines(self, baselines: BaselineFills) -> "ScoredContext":
        """Attach baseline fills for scoring and plotting.

        Parameters
        ----------
        baselines : BaselineFills
            LOCF / linear fills.

        Returns
        -------
        ScoredContext
            Context plus baselines.
        """
        return ScoredContext(context=self, baselines=baselines)

    def n_features(self) -> int:
        """Return ``F``."""
        return int(self.truth.shape[FEATURE_AXIS])


@dataclass(frozen=True)
class ScoredContext:
    """Original-unit context plus point-baseline fills.

    Public methods
    --------------
    truth
        Ground-truth windows.
    observations
        Observed context windows.

    Instance variables
    ------------------
    context : OriginalUnitContext
        Truth and observations.
    baselines : BaselineFills
        LOCF / linear fills.
    """

    context: OriginalUnitContext
    baselines: BaselineFills

    def truth(self) -> jnp.ndarray:
        """Return ground truth in original units."""
        return self.context.truth

    def observations(self) -> jnp.ndarray:
        """Return observed context in original units."""
        return self.context.observations


@dataclass(frozen=True)
class PanelSpec:
    """Title and slice indices for one comparison panel.

    Public methods
    --------------
    indices
        Return ``(batch_index, feature_index)``.
    with_title
        Return a copy with a new title.

    Instance variables
    ------------------
    title : str
        Axes title.
    batch_index : int
        Window index to plot.
    feature_index : int
        Feature channel to plot.
    """

    title: str
    batch_index: int = _FIRST_BATCH
    feature_index: int = _FIRST_FEATURE

    def indices(self) -> tuple[int, int]:
        """Return ``(batch_index, feature_index)``."""
        return int(self.batch_index), int(self.feature_index)

    def with_title(self, title: str) -> "PanelSpec":
        """Return a copy with an updated title."""
        return PanelSpec(
            title=title,
            batch_index=self.batch_index,
            feature_index=self.feature_index,
        )


@dataclass(frozen=True)
class ComparisonView:
    """Inputs for one multi-method comparison PNG.

    Public methods
    --------------
    series_arrays
        Extract 1-D NumPy series for the configured slice.
    timefill_mean
        Posterior mean from the bands pack.

    Instance variables
    ------------------
    bands : ImputationPlotData
        Truth, observations, TimeFill mean / quantiles, metric mask.
    baselines : BaselineFills
        LOCF and linear fills.
    spec : PanelSpec
        Title and slice indices.
    """

    bands: ImputationPlotData
    baselines: BaselineFills
    spec: PanelSpec

    def timefill_mean(self) -> jnp.ndarray:
        """Return the TimeFill posterior mean tensor."""
        return self.bands.mean

    def series_arrays(self) -> dict[str, np.ndarray]:
        """Slice the configured ``(batch, feature)`` into NumPy series."""
        batch_index, feature_index = self.spec.indices()
        bundle = self.bands.series_bundle(batch_index, feature_index)
        return {
            "time": np.asarray(bundle["time"]),
            "truth": np.asarray(bundle["truth"]),
            "observed": np.asarray(bundle["observed"]),
            "tf_mean": np.asarray(bundle["mean"]),
            "tf_low": np.asarray(bundle["low"]),
            "tf_high": np.asarray(bundle["high"]),
            "locf": np.asarray(
                self.baselines.locf[batch_index, :, feature_index]
            ),
            "linear": np.asarray(
                self.baselines.linear[batch_index, :, feature_index]
            ),
            "is_missing": np.asarray(bundle["is_missing"]),
        }


class M3ArtifactPaths:
    """Resolve M3 table and panel paths under an artifact root.

    Public methods
    --------------
    ensure
        Create root and ``panels/``.
    metrics_csv
        Path for ``metrics_by_rate.csv``.
    metrics_json
        Path for ``metrics_by_rate.json``.
    comparison_panel
        Path for the primary comparison PNG.

    Instance variables
    ------------------
    root : pathlib.Path
        Artifact root (typically ``artifacts/m3``).
    """

    def __init__(self, root: Path | str) -> None:
        """Store the artifact root.

        Parameters
        ----------
        root : path-like
            Directory that will hold M3 outputs.
        """
        self.root = Path(root)

    def ensure(self) -> None:
        """Create artifact root and panels directory."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / _PANELS_DIR).mkdir(parents=True, exist_ok=True)

    def metrics_csv(self) -> Path:
        """Return ``root / metrics_by_rate.csv``."""
        return self.root / _METRICS_CSV

    def metrics_json(self) -> Path:
        """Return ``root / metrics_by_rate.json``."""
        return self.root / _METRICS_JSON

    def comparison_panel(self) -> Path:
        """Return ``root / panels / comparison_....png``."""
        return self.root / _PANELS_DIR / _PANEL_NAME


class BenchmarkComparisonPlotter:
    """Plot truth, TimeFill bands, LOCF, and linear on shared gaps.

    Public methods
    --------------
    save
        Write one comparison PNG from a ``ComparisonView``.
    draw
        Populate a Matplotlib axis from series arrays.
    """

    def save(self, view: ComparisonView, output_path: Path) -> Path:
        """Save a multi-method comparison panel.

        Parameters
        ----------
        view : ComparisonView
            Original-unit series for one protocol.
        output_path : pathlib.Path
            Destination PNG.

        Returns
        -------
        pathlib.Path
            Path written.
        """
        series = view.series_arrays()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=_FIGSIZE)
        self.draw(axis, series, view.spec.title)
        fig.tight_layout()
        save_figure(fig, out)
        plt.close(fig)
        return out

    def draw(
            self,
            axis,
            series: dict[str, np.ndarray],
            title: str,
    ) -> None:
        """Draw comparison curves onto an axis.

        Parameters
        ----------
        axis :
            Matplotlib axes.
        series : dict of str to numpy.ndarray
            Output of ``ComparisonView.series_arrays``.
        title : str
            Axes title.
        """
        time = series["time"]
        axis.plot(time, series["truth"], color="0.65", lw=1.5, label="truth")
        axis.plot(
            time, series["tf_mean"], color="C0", lw=1.4, label="TimeFill mean",
        )
        axis.fill_between(
            time,
            series["tf_low"],
            series["tf_high"],
            color="C0",
            alpha=_BAND_ALPHA,
            label="TimeFill band",
        )
        axis.plot(time, series["locf"], color="C1", lw=1.2, label="LOCF")
        axis.plot(time, series["linear"], color="C3", lw=1.2, label="linear")
        axis.plot(
            time,
            series["observed"],
            color="C2",
            lw=1.0,
            alpha=0.85,
            label="x_obs",
        )
        missing = series["is_missing"]
        if bool(np.any(missing)):
            axis.scatter(
                time[missing],
                series["truth"][missing],
                color="black",
                s=16,
                zorder=3,
                label="targets",
            )
        axis.set_xlabel("t")
        axis.set_ylabel("value (original units)")
        axis.set_title(title)
        axis.legend(loc="best", fontsize=8)


def point_fill_as_ensemble(filled: jnp.ndarray) -> jnp.ndarray:
    """Wrap a point fill as a degenerate ensemble with ``K = 1``.

    Parameters
    ----------
    filled : jax.Array
        Point fill shaped ``(B, T, F)``.

    Returns
    -------
    jax.Array
        Samples shaped ``(B, 1, T, F)``.
    """
    return jnp.expand_dims(filled, axis=_SAMPLE_AXIS)


def panel_title(protocol: ProtocolKey) -> str:
    """Build a short panel title for one protocol.

    Parameters
    ----------
    protocol : ProtocolKey
        MCAR or block protocol.

    Returns
    -------
    str
        Axes title string.
    """
    if protocol.is_block():
        return f"M3 comparison (block_len={protocol.block_len})"
    return f"M3 comparison (MCAR rate={protocol.mask_rate})"


__all__ = [
    "BaselineFills",
    "BenchmarkComparisonPlotter",
    "ComparisonView",
    "M3ArtifactPaths",
    "OriginalUnitContext",
    "PanelSpec",
    "ScoredContext",
    "panel_title",
    "point_fill_as_ensemble",
]
