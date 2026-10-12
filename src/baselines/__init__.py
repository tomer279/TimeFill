"""Point baselines for the M3 benchmark (LOCF and linear interpolation).

Modules
-------
protocol
    Shared ``(x, m)`` preparation and train-feature-mean helpers.
locf
    Last-observation-carried-forward imputer.
linear
    Per-feature linear interpolation on observed time indices.

Classes
-------
LocfImputer
    Carry the latest earlier observation; otherwise the train-feature mean.
LinearImputer
    Interpolate bracketed gaps; otherwise the train-feature mean.

Functions
---------
locf_impute
    Functional LOCF fill for ``(values, mask, feature_means)``.
linear_impute
    Functional linear fill for ``(values, mask, feature_means)``.

Constants
---------
EDGE_POLICY_TRAIN_FEATURE_MEAN
    Locked edge-policy name from ``configs/bench_m3.yaml``.
"""

from src.baselines.linear import LinearImputer, linear_impute
from src.baselines.locf import LocfImputer, locf_impute
from src.baselines.protocol import EDGE_POLICY_TRAIN_FEATURE_MEAN

__all__ = [
    "EDGE_POLICY_TRAIN_FEATURE_MEAN",
    "LinearImputer",
    "LocfImputer",
    "linear_impute",
    "locf_impute",
]
