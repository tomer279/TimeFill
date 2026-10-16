"""Light Matplotlib helpers for M1 mask visualization.

Functions
---------
save_mask_comparison
    Save a side-by-side random vs block mask figure under `artifacts/m1/`.
default_mask_figure_path
    Return the documented default figure path.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.data.masking import block_mask, random_mask
from src.data.schema import MASK_MISSING, TIME_AXIS
from src.plot_io import save_figure

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIG_PATH = (
    _REPO_ROOT / "artifacts" / "m1" / "mask_random_vs_block.png"
)
_FIRST_BATCH = 0
_FIRST_FEATURE = 0
_N_PANEL_COLS = 2
_N_PANEL_ROWS = 1
_FIGSIZE = (10.0, 3.5)


def default_mask_figure_path() -> Path:
    """Return the documented WP1 mask-comparison figure path.

    Returns
    -------
    pathlib.Path
        ``artifacts/m1/mask_random_vs_block.png`` under the repo root.
    """
    return _DEFAULT_FIG_PATH


def save_mask_comparison(
        clean: jnp.ndarray,
        output_path: Path | str,
        key: jax.Array,
        mask_rate: float,
        block_len: int,
) -> Path:
    """Save side-by-side random vs block masks for one window/feature.

    Parameters
    ----------
    clean : jax.Array
        Clean tensor `(B, T, F)` or a single window `(T, F)`.
    output_path : path-like
        Destination PNG path (parent dirs are created).
    key : jax.Array
        PRNG key (split for the two maskers).
    mask_rate : float
        MCAR missing rate for the left panel.
    block_len : int
        Contiguous block length for the right panel.

    Returns
    -------
    pathlib.Path
        Path written.
    """
    window = _as_batch(clean)
    random_series, block_series = _masked_panel_series(
        window,
        key,
        mask_rate,
        block_len,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        _N_PANEL_ROWS,
        _N_PANEL_COLS,
        figsize=_FIGSIZE,
        sharey=True,
    )
    _draw_panel(
        axes[0],
        random_series,
        f"random MCAR (rate={mask_rate})",
    )
    _draw_panel(
        axes[1],
        block_series,
        f"block (len={block_len})",
    )
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)
    return out


def _masked_panel_series(
        window: jnp.ndarray,
        key: jax.Array,
        mask_rate: float,
        block_len: int,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """Build random- and block-mask series dicts for the first feature.

    Parameters
    ----------
    window : jax.Array
        Rank-3 clean batch ``(B, T, F)``.
    key : jax.Array
        PRNG key (split internally).
    mask_rate : float
        MCAR rate for the random panel.
    block_len : int
        Block length for the block panel.

    Returns
    -------
    random_series : dict of str to jax.Array
        Keys ``time``, ``truth``, ``observed``, ``is_missing``.
    block_series : dict of str to jax.Array
        Same keys for the block masker.
    """
    key_random, key_block = jax.random.split(key)
    x_random, m_random = random_mask(window, mask_rate, key_random)
    x_block, m_block = block_mask(window, block_len, key_block)
    return (
        _feature_panel_series(window, x_random, m_random),
        _feature_panel_series(window, x_block, m_block),
    )


def _feature_panel_series(
        window: jnp.ndarray,
        x_obs: jnp.ndarray,
        mask: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Slice the first batch/feature into a panel series dict.

    Parameters
    ----------
    window : jax.Array
        Clean batch ``(B, T, F)``.
    x_obs : jax.Array
        Masked observations ``(B, T, F)``.
    mask : jax.Array
        Observation mask ``(B, T, F)``.

    Returns
    -------
    dict of str to jax.Array
        Keys ``time``, ``truth``, ``observed``, ``is_missing``.
    """
    return {
        "time": jnp.arange(window.shape[TIME_AXIS]),
        "truth": window[_FIRST_BATCH, :, _FIRST_FEATURE],
        "observed": x_obs[_FIRST_BATCH, :, _FIRST_FEATURE],
        "is_missing": mask[_FIRST_BATCH, :, _FIRST_FEATURE] == MASK_MISSING,
    }


def _as_batch(clean: jnp.ndarray) -> jnp.ndarray:
    """Promote `(T, F)` to `(1, T, F)` when needed.

    Parameters
    ----------
    clean : jax.Array
        Rank-2 window or rank-3 batch.

    Returns
    -------
    jax.Array
        Rank-3 tensor.

    Raises
    ------
    ValueError
        If `clean` is not rank 2 or 3.
    """
    if clean.ndim == 3:
        return clean
    if clean.ndim == 2:
        return clean[None, :, :]
    raise ValueError(
        "clean must have shape (B, T, F) or (T, F); "
        f"got ndim={clean.ndim}"
    )


def _draw_panel(
        axis: Axes,
        series: dict[str, jnp.ndarray],
        title: str,
) -> None:
    """Draw one mask comparison panel from a small series dict.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target axes.
    series : dict of str to jax.Array
        Keys: ``time``, ``truth``, ``observed``, ``is_missing``.
    title : str
        Panel title.
    """
    time = series["time"]
    truth = series["truth"]
    observed = series["observed"]
    is_missing = series["is_missing"]
    axis.plot(time, truth, color="0.7", linewidth=1.5, label="clean")
    axis.plot(time, observed, color="C0", linewidth=1.2, label="x_obs")
    if bool(jnp.any(is_missing)):
        axis.scatter(
            time[is_missing],
            truth[is_missing],
            color="C3",
            s=18,
            zorder=3,
            label="missing (truth)",
        )
    axis.set_title(title)
    axis.set_xlabel("t")
    axis.legend(loc="best", fontsize=8)
