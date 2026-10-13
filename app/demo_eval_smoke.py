"""Tiny in-app demo-eval smoke (not the M3 report grid).

Classes
-------
DemoEvalSmokeRequest
    Windows plus smoke ``K`` / steps / MCAR rate.
DemoEvalSmokeResult
    Compact metrics table plus run meta.

Functions
---------
run_demo_eval_smoke
    Apply MCAR on native observations, impute, score artificial targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from app.session_impute import (
    DemoMaskRequest,
    SessionImputeRequest,
    SessionImputer,
)
from app.window_adapter import AdaptedWindows
from src.eval.m2_val_impute import metric_observation_mask
from src.eval.metrics import EvalTensors, ImputationEvaluator, MetricsTable

_MODE_RANDOM = "random"


@dataclass(frozen=True)
class DemoEvalSmokeRequest:
    """Knobs for one UI demo-eval smoke run.

    Public methods
    --------------
    validate
        Check positivity / rate bounds.
    describe
        Short label for the metrics caption.

    Instance variables
    ------------------
    windows : AdaptedWindows
        Original-unit windows (treated as ground truth where native).
    mcar_rate : float
        Artificial MCAR rate.
    k_samples : int
        Tiny ``K`` (``smoke.K_samples``).
    solver_steps : int
        Tiny solver steps.
    mask_seed : int
        Demo corruption seed.
    block_len : int
        Unused for MCAR; kept for ``DemoMaskRequest`` parity.
    """

    windows: AdaptedWindows
    mcar_rate: float
    k_samples: int
    solver_steps: int
    mask_seed: int
    block_len: int

    def validate(self) -> None:
        """Raise ``ValueError`` if smoke knobs are invalid."""
        if self.windows.n_windows() <= 0:
            raise ValueError("windows must be non-empty")
        if not 0.0 < self.mcar_rate <= 1.0:
            raise ValueError(
                f"mcar_rate must be in (0, 1], got {self.mcar_rate}"
            )
        if self.k_samples <= 0 or self.solver_steps <= 0:
            raise ValueError("k_samples and solver_steps must be positive")

    def describe(self) -> str:
        """Return a short smoke-run label.

        Returns
        -------
        str
            Rate / ``K`` / steps summary.
        """
        return (
            f"smoke MCAR={self.mcar_rate:.2f} "
            f"K={self.k_samples} steps={self.solver_steps}"
        )


@dataclass(frozen=True)
class DemoEvalSmokeResult:
    """Compact smoke metrics (label as demo, not M3).

    Public methods
    --------------
    as_rows
        List of ``(metric, value)`` for a Streamlit table.
    caption
        User-facing disclaimer string.

    Instance variables
    ------------------
    metrics : MetricsTable
        MAE / RMSE / CRPS / coverage.
    label : str
        Run description from the request.
    """

    metrics: MetricsTable
    label: str

    def as_rows(self) -> list[tuple[str, float]]:
        """Return ordered metric rows.

        Returns
        -------
        list of tuple
            ``(name, value)`` pairs.
        """
        return self.metrics.rows()

    def caption(self) -> str:
        """Return a disclaimer for the UI.

        Returns
        -------
        str
            Smoke-slice warning.
        """
        return (
            f"Demo check only ({self.label}). "
            "Full benchmark tables: scripts/run_m3_benchmark.py"
        )


def run_demo_eval_smoke(
        imputer: SessionImputer,
        request: DemoEvalSmokeRequest,
) -> DemoEvalSmokeResult:
    """Run a tiny MCAR smoke eval on adapted windows.

    Parameters
    ----------
    imputer : SessionImputer
        Loaded pretrained imputer.
    request : DemoEvalSmokeRequest
        Windows and smoke knobs.

    Returns
    -------
    DemoEvalSmokeResult
        Compact metrics on artificial targets with GT.
    """
    request.validate()
    demo_mask = DemoMaskRequest(
        mode=_MODE_RANDOM,
        rate=request.mcar_rate,
        block_len=request.block_len,
        seed=request.mask_seed,
    )
    values = jnp.asarray(request.windows.values, dtype=jnp.float32)
    m_native = jnp.asarray(request.windows.native_mask, dtype=jnp.float32)
    _x_obs, _m_eval, m_artificial = demo_mask.compose(values, m_native)
    result = imputer.impute(
        SessionImputeRequest(
            windows=request.windows,
            demo_mask=demo_mask,
            k_samples=request.k_samples,
            solver_steps=request.solver_steps,
        )
    )
    score_mask = metric_observation_mask(m_native, m_artificial)
    table = ImputationEvaluator().compute(
        EvalTensors(
            truth=values,
            mean=jnp.asarray(result.tensors.mean, dtype=jnp.float32),
            samples=jnp.asarray(result.tensors.samples, dtype=jnp.float32),
            mask=score_mask,
        )
    )
    return DemoEvalSmokeResult(metrics=table, label=request.describe())
