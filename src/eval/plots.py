"""M1 evaluation figures: forward noise, masks, imputation bands (WP6).

Classes
-------
EvalPlotPaths
    Default PNG paths under an artifact root (typically ``artifacts/m1``).
ImputationPlotData
    Packed tensors for the mean ± quantile-band figure.

Functions
---------
save_forward_noising_plot
    Delegate to ``src.sde.forward_plots.save_forward_noising``.
save_mask_patterns_plot
    Delegate to ``src.data.mask_plots.save_mask_comparison``.
save_imputation_bands
    Plot truth, observations, posterior mean, and predictive bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.data.mask_plots import save_mask_comparison
from src.data.schema import MASK_MISSING, TIME_AXIS
from src.plot_io import save_figure
from src.sde.forward_plots import save_forward_noising
from src.sde.vp_sde import VPSDE

_FORWARD_NAME = "forward_noising.png"
_MASK_NAME = "mask_random_vs_block.png"
_IMPUTE_NAME = "imputation_mean_bands.png"
_FIRST_BATCH = 0
_FIRST_FEATURE = 0
_Q_LOW_INDEX = 0
_Q_HIGH_INDEX = 1
_MASK_COMPARE_TOL = 0.5
_FIGSIZE = (8.0, 3.5)
_BAND_ALPHA = 0.25


class EvalPlotPaths:
    """Resolve WP6 figure paths under an artifact directory.

    Public methods
    --------------
    ensure
        Create the artifact root if needed.
    forward_noising
        Path for the forward-noising PNG.
    mask_patterns
        Path for the random-vs-block mask PNG.
    imputation_bands
        Path for the imputation mean ± bands PNG.

    Instance variables
    ------------------
    root : pathlib.Path
        Artifact root (e.g. ``artifacts/m1``).
    """

    def __init__(self, root: Path | str) -> None:
        """Store the artifact root path.

        Parameters
        ----------
        root : path-like
            Directory that will hold WP6 figures.
        """
        self.root = Path(root)

    def ensure(self) -> None:
        """Create the artifact root directory."""
        self.root.mkdir(parents=True, exist_ok=True)

    def forward_noising(self) -> Path:
        """Return the forward-noising figure path.

        Returns
        -------
        pathlib.Path
            ``root / forward_noising.png``.
        """
        return self.root / _FORWARD_NAME

    def mask_patterns(self) -> Path:
        """Return the mask-comparison figure path.

        Returns
        -------
        pathlib.Path
            ``root / mask_random_vs_block.png``.
        """
        return self.root / _MASK_NAME

    def imputation_bands(self) -> Path:
        """Return the imputation bands figure path.

        Returns
        -------
        pathlib.Path
            ``root / imputation_mean_bands.png``.
        """
        return self.root / _IMPUTE_NAME


@dataclass(frozen=True)
class ImputationPlotData:
    """Inputs for a single-window imputation mean ± band plot.

    Public methods
    --------------
    time_index
        Integer grid ``0 … T-1`` for the plotted window.
    series_bundle
        Extract 1-D series for one ``(batch, feature)`` pair.

    Instance variables
    ------------------
    truth : jax.Array
        Clean ground truth ``(B, T, F)``.
    observations : jax.Array
        Sentinel-filled observations ``(B, T, F)``.
    mean : jax.Array
        Posterior mean ``(B, T, F)``.
    quantiles : jax.Array
        Low/high quantiles ``(B, T, F, 2)``.
    mask : jax.Array
        Observation mask ``(B, T, F)``.
    """

    truth: jnp.ndarray
    observations: jnp.ndarray
    mean: jnp.ndarray
    quantiles: jnp.ndarray
    mask: jnp.ndarray

    def time_index(self) -> jnp.ndarray:
        """Return the regular-grid time index for the window.

        Returns
        -------
        jax.Array
            Float32 vector shaped ``(T,)``.
        """
        seq_len = int(self.truth.shape[TIME_AXIS])
        return jnp.arange(seq_len, dtype=jnp.float32)

    def series_bundle(
            self,
            batch_index: int,
            feature_index: int,
    ) -> dict[str, jnp.ndarray]:
        """Slice one feature series into a plotting dict.

        Parameters
        ----------
        batch_index : int
            Batch item to plot.
        feature_index : int
            Feature channel to plot.

        Returns
        -------
        dict of str to jax.Array
            Keys: ``time``, ``truth``, ``observed``, ``mean``,
            ``low``, ``high``, ``is_missing``.
        """
        return {
            "time": self.time_index(),
            "truth": self.truth[batch_index, :, feature_index],
            "observed": self.observations[batch_index, :, feature_index],
            "mean": self.mean[batch_index, :, feature_index],
            "low": self.quantiles[
                batch_index, :, feature_index, _Q_LOW_INDEX
            ],
            "high": self.quantiles[
                batch_index, :, feature_index, _Q_HIGH_INDEX
            ],
            "is_missing": (
                jnp.abs(
                    self.mask[batch_index, :, feature_index] - MASK_MISSING
                )
                < _MASK_COMPARE_TOL
            ),
        }


def save_forward_noising_plot(
        clean: jnp.ndarray,
        output_path: Path | str,
        sde: VPSDE,
        key: jax.Array,
) -> Path:
    """Save VP-SDE forward-noising panels (WP2 helper wrapper).

    Parameters
    ----------
    clean : jax.Array
        Clean windows ``(B, T, F)`` or ``(T, F)``.
    output_path : path-like
        Destination PNG.
    sde : VPSDE
        Forward process.
    key : jax.Array
        PRNG key for shared ``eps``.

    Returns
    -------
    pathlib.Path
        Path written.
    """
    return save_forward_noising(clean, output_path, sde, key)


def save_mask_patterns_plot(
        clean: jnp.ndarray,
        output_path: Path | str,
        key: jax.Array,
        mask_rate: float,
        block_len: int,
) -> Path:
    """Save random vs block mask comparison (WP1 helper wrapper).

    Parameters
    ----------
    clean : jax.Array
        Clean windows ``(B, T, F)`` or ``(T, F)``.
    output_path : path-like
        Destination PNG.
    key : jax.Array
        PRNG key.
    mask_rate : float
        MCAR rate for the left panel.
    block_len : int
        Block length for the right panel.

    Returns
    -------
    pathlib.Path
        Path written.
    """
    return save_mask_comparison(
        clean,
        output_path,
        key,
        mask_rate,
        block_len,
    )


def save_imputation_bands(
        data: ImputationPlotData,
        output_path: Path | str,
) -> Path:
    """Save truth / observations / posterior mean ± quantile bands.

    Parameters
    ----------
    data : ImputationPlotData
        Batched imputation tensors.
    output_path : path-like
        Destination PNG (parent dirs are created).

    Returns
    -------
    pathlib.Path
        Path written.
    """
    series = data.series_bundle(_FIRST_BATCH, _FIRST_FEATURE)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=_FIGSIZE)
    time = series["time"]
    axis.plot(time, series["truth"], color="0.65", linewidth=1.5, label="truth")
    axis.plot(
        time,
        series["mean"],
        color="C0",
        linewidth=1.4,
        label="posterior mean",
    )
    axis.fill_between(
        time,
        series["low"],
        series["high"],
        color="C0",
        alpha=_BAND_ALPHA,
        label="predictive band",
    )
    axis.plot(
        time,
        series["observed"],
        color="C2",
        linewidth=1.0,
        alpha=0.85,
        label="x_obs",
    )
    is_missing = series["is_missing"]
    if bool(jnp.any(is_missing)):
        axis.scatter(
            time[is_missing],
            series["truth"][is_missing],
            color="C3",
            s=18,
            zorder=3,
            label="missing (truth)",
        )
    axis.set_xlabel("t")
    axis.set_ylabel("value")
    axis.set_title("M1 imputation mean ± bands")
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)
    return out
