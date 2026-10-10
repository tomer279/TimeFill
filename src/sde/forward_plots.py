"""Matplotlib helpers for M1 VP-SDE forward-noising figures.

Functions
---------
default_forward_figure_path
    Return the documented default figure path under ``artifacts/m1/``.
save_forward_noising
    Plot one clean window at several diffusion times.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.data.schema import TIME_AXIS
from src.plot_io import save_figure
from src.sde.vp_sde import VPSDE

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIG_PATH = _REPO_ROOT / "artifacts" / "m1" / "forward_noising.png"
_FIRST_BATCH = 0
_FIRST_FEATURE = 0
_N_PANEL_ROWS = 1
_FIGSIZE = (12.0, 3.0)
_DEFAULT_TIME_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def default_forward_figure_path() -> Path:
    """Return the documented WP2 forward-noising figure path.

    Returns
    -------
    pathlib.Path
        ``artifacts/m1/forward_noising.png`` under the repo root.
    """
    return _DEFAULT_FIG_PATH


def save_forward_noising(
        clean: jnp.ndarray,
        output_path: Path | str,
        sde: VPSDE,
        key: jax.Array,
) -> Path:
    """Save a clean window noised at several diffusion times.

    Panels use fractions of ``sde.t_diff`` in
    ``{0, 0.25, 0.5, 0.75, 1}``. At ``t ≈ t_diff`` the series should look
    near-Gaussian (VP prior).

    Parameters
    ----------
    clean : jax.Array
        Clean tensor ``(B, T, F)`` or a single window ``(T, F)``.
    output_path : path-like
        Destination PNG path (parent dirs are created).
    sde : VPSDE
        Forward process providing ``forward_sample``.
    key : jax.Array
        PRNG key for a single shared ``eps`` across panels.

    Returns
    -------
    pathlib.Path
        Path written.
    """
    window = _as_batch(clean)
    eps = jax.random.normal(key, shape=window.shape, dtype=jnp.float32)
    times = tuple(
        fraction * sde.t_diff for fraction in _DEFAULT_TIME_FRACTIONS
    )
    time_grid = jnp.arange(window.shape[TIME_AXIS])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        _N_PANEL_ROWS,
        len(times),
        figsize=_FIGSIZE,
        sharey=True,
    )
    for axis, time_value in zip(axes, times):
        noised = sde.forward_sample(window, time_value, eps)
        series = noised[_FIRST_BATCH, :, _FIRST_FEATURE]
        axis.plot(time_grid, series, color="C0", linewidth=1.2)
        axis.set_title(f"t={time_value:.2f}")
        axis.set_xlabel("index")
    axes[0].set_ylabel("value")
    fig.suptitle("VP-SDE forward noising (near-Gaussian at t=t_diff)")
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)
    return out


def _as_batch(clean: jnp.ndarray) -> jnp.ndarray:
    """Promote ``(T, F)`` to ``(1, T, F)`` when needed.

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
        If ``clean`` is not rank 2 or 3.
    """
    if clean.ndim == 3:
        return clean
    if clean.ndim == 2:
        return clean[None, :, :]
    raise ValueError(
        "clean must have shape (B, T, F) or (T, F); "
        f"got ndim={clean.ndim}"
    )
