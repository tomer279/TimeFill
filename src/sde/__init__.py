"""VP-SDE forward process helpers for M1.

Classes
-------
VPSDE
    Variance-preserving SDE with closed-form Gaussian marginals.

Functions
---------
default_forward_figure_path
    Return the documented WP2 forward-noising figure path.
save_forward_noising
    Save a clean-window → multi-``t`` forward-noising figure.
"""

from src.sde.forward_plots import (
    default_forward_figure_path,
    save_forward_noising,
)
from src.sde.vp_sde import VPSDE

__all__ = [
    "VPSDE",
    "default_forward_figure_path",
    "save_forward_noising",
]
