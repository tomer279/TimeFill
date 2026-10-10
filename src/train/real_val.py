"""Lightweight validation imputation snapshots for M2 training.

Classes
-------
M2ValHookBundle
    Reverse-impute val snapshot writer (mean ± bands PNG).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax

from src.data.schema import TIME_AXIS
from src.eval.plots import ImputationPlotData, save_imputation_bands
from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE
from src.train.artifacts import ValSnapshotPayload


@dataclass(frozen=True)
class M2ValHookBundle:
    """Build lightweight val imputation mean ± band PNGs.

    Public methods
    --------------
    as_hook
        Return a ``ValSnapshotPayload -> Path`` callable.
    write_snapshot
        Run reverse impute for one payload and save a PNG.

    Instance variables
    ------------------
    sde : VPSDE
        VP-SDE matching training.
    t_eps : float
        Terminal diffusion time.
    k_samples : int
        Posterior draws for the snapshot.
    solver_steps : int
        Diffrax steps for the snapshot.
    """

    sde: VPSDE
    t_eps: float
    k_samples: int
    solver_steps: int

    def as_hook(self) -> Callable[[ValSnapshotPayload], Path]:
        """Return a hook closure bound to this bundle.

        Returns
        -------
        callable
            ``ValSnapshotPayload -> Path``.
        """
        return self.write_snapshot

    def write_snapshot(self, payload: ValSnapshotPayload) -> Path:
        """Impute a val batch and write mean ± bands.

        Parameters
        ----------
        payload : ValSnapshotPayload
            Model, tensors, step, and output directory.

        Returns
        -------
        pathlib.Path
            Written PNG path.
        """
        payload.out_dir.mkdir(parents=True, exist_ok=True)
        settings = ImputeSolverSettings(
            solver_steps=self.solver_steps,
            t_eps=self.t_eps,
        )
        imputer = ReverseSDEImputer(
            ImputeContext(
                model=payload.model,
                sde=self.sde,
                settings=settings,
            )
        )
        seq_len = int(payload.x0.shape[TIME_AXIS])
        _samples, mean, quantiles = imputer.impute(
            ImputeQuery(
                observations=payload.x_obs,
                mask=payload.mask,
                timestamps=grid_timestamps(seq_len),
                key=jax.random.PRNGKey(payload.step),
                k_samples=self.k_samples,
            )
        )
        out = payload.out_dir / f"val_impute_step_{payload.step}.png"
        return save_imputation_bands(
            ImputationPlotData(
                truth=payload.x0,
                observations=payload.x_obs,
                mean=mean,
                quantiles=quantiles,
                mask=payload.mask,
            ),
            out,
        )
