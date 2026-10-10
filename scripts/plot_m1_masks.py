"""Save the WP1 random-vs-block mask comparison figure.

Writes ``artifacts/m1/mask_random_vs_block.png`` using `configs/toy_m1.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import jax

from src.data.config import load_config
from src.data.mask_plots import default_mask_figure_path, save_mask_comparison
from src.data.synthetic import generate_sine_windows

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "configs" / "toy_m1.yaml"
_PRNG_SEED = 0


def main() -> Path:
    """Generate one sine window, mask it two ways, and save the figure.

    Returns
    -------
    pathlib.Path
        Path to the written PNG.
    """
    config = load_config(_CONFIG_PATH)
    key = jax.random.PRNGKey(_PRNG_SEED)
    key_data, key_plot = jax.random.split(key)
    shape = (
        config.data.batch_size,
        config.data.seq_len,
        config.data.n_features,
    )
    clean = generate_sine_windows(key_data, shape)
    out = save_mask_comparison(
        clean,
        default_mask_figure_path(),
        key_plot,
        config.data.mask_rate,
        config.data.block_len,
    )
    print(f"Wrote mask comparison figure to {out}")
    return out


if __name__ == "__main__":
    main()
