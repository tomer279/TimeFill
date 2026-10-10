"""Equinox model components for M1 conditional score matching.

Classes
-------
ScoreNetConfig
    Hyperparameters for the conditional temporal score network.
TimeConditioner
    Sinusoidal diffusion-time embedding projected to TCN width.
ConditionalScoreNet
    Dilated TCN score model conditioned on mask and observations.

Functions
---------
default_score_net_config
    Build a default `ScoreNetConfig` for a given feature count `F`.
"""

from src.models.score_net import (
    ConditionalScoreNet,
    ScoreNetConfig,
    TimeConditioner,
    default_score_net_config,
)

__all__ = [
    "ConditionalScoreNet",
    "ScoreNetConfig",
    "TimeConditioner",
    "default_score_net_config",
]
