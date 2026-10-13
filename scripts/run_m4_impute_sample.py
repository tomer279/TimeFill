"""WP2: impute the bundled M4 sample from the M2 checkpoint (no Streamlit).

Prerequisites
-------------
- ``artifacts/m2/checkpoints/best_score_net.eqx``
- ``artifacts/m2/checkpoints/feature_scaler.npz``
- Bundled CSV at ``data.sample_csv`` in the app config

Example
-------
python scripts/run_m4_impute_sample.py --config configs/app_m4.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.session_impute import impute_bundled_sample

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "app_m4.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the M4 bundled-sample impute smoke.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed flags.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to configs/app_m4.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run bundled-sample imputation and print a short summary.

    Parameters
    ----------
    argv : list of str or None, optional
        Optional CLI args.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    result = impute_bundled_sample(args.config)
    print(
        f"windows={result.n_windows()} "
        f"{result.meta.describe()} "
        f"observed_cells={result.observed_cell_count()} "
        f"mean_fill_shape={tuple(result.context.mean_fill.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
