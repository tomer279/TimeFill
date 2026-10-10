"""Save the WP2 VP-SDE forward-noising figure.

Writes ``artifacts/m1/forward_noising.png`` using `configs/toy_m1.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import jax

from src.data.config import load_config
from src.data.synthetic import generate_sine_windows
from src.sde.forward_plots import (
    default_forward_figure_path,
    save_forward_noising,
)
from src.sde.vp_sde import VPSDE

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "configs" / "toy_m1.yaml"
_PRNG_SEED = 0


def main() -> Path:
    """Generate one sine window, forward-noise it, and save the figure.

    Returns
    -------
    pathlib.Path
        Path to the written PNG.
    """
    config = load_config(_CONFIG_PATH)
    key = jax.random.PRNGKey(_PRNG_SEED)
    key_data, key_eps = jax.random.split(key)
    shape = (
        config.data.batch_size,
        config.data.seq_len,
        config.data.n_features,
    )
    clean = generate_sine_windows(key_data, shape)
    sde = VPSDE(config.sde)
    out = save_forward_noising(
        clean,
        default_forward_figure_path(),
        sde,
        key_eps,
    )
    print(f"Wrote forward-noising figure to {out}")
    return out


if __name__ == "__main__":
    main()
