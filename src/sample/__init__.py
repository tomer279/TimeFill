"""Reverse-time SDE sampling and imputation for M1 (WP5).

Classes
-------
ImputeSolverSettings
    Diffrax step count, ``t_eps``, quantiles, and noise-init strategy.
ImputeContext
    Bundled score model, ``VPSDE``, and solver settings for imputation.
ImputeQuery
    Observations, mask, timestamps, PRNG key, and sample count for one call.
ReverseSDEImputer
    Conditional reverse-SDE sampler returning samples / mean / quantiles.

Functions
---------
context_from_toy_config
    Build an ``ImputeContext`` from a model and ``ToyM1Config``.
context_from_real_config
    Build an ``ImputeContext`` from a model and ``RealM2Config``.
grid_timestamps
    Default regular-grid timestamps ``0 … T-1`` for M1.
"""

from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    context_from_real_config,
    context_from_toy_config,
    grid_timestamps,
)

__all__ = [
    "ImputeContext",
    "ImputeQuery",
    "ImputeSolverSettings",
    "ReverseSDEImputer",
    "context_from_real_config",
    "context_from_toy_config",
    "grid_timestamps",
]
