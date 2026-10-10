"""Reverse-time SDE sampling and imputation for M1 (WP5).

Classes
-------
ImputeSolverSettings
    Diffrax step count, ``t_eps``, quantiles, and noise-init strategy.
ImputeContext
    Bundled score model, ``VPSDE``, and solver settings for imputation.
ReverseSDEImputer
    Conditional reverse-SDE sampler returning samples / mean / quantiles.

Functions
---------
context_from_toy_config
    Build an ``ImputeContext`` from a model and ``ToyM1Config``.
grid_timestamps
    Default regular-grid timestamps ``0 … T-1`` for M1.
"""

from src.sample.reverse_sde import (
    ImputeContext,
    ImputeSolverSettings,
    ReverseSDEImputer,
    context_from_toy_config,
    grid_timestamps,
)

__all__ = [
    "ImputeContext",
    "ImputeSolverSettings",
    "ReverseSDEImputer",
    "context_from_toy_config",
    "grid_timestamps",
]
