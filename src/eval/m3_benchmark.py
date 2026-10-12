"""M3 benchmark runner: protocol × method loop and public re-exports.

Classes
-------
RunGrid
    Protocols, methods, seeds, ``K``, and artifact root.
ImputeContextRequest
    Build a reverse-SDE context with overridden solver steps.
MethodScoreInput
    Truth / mean / samples / mask pack for scoring.
TimeFillResult
    Original-unit TimeFill mean, samples, and quantiles.
M3BenchmarkRunner
    End-to-end protocol × method loop.

Modules re-exported
-------------------
m3_table
    Long-form metric row types and assemblers.
m3_data
    Split loading and artificial-mask composition.
m3_panels
    Comparison panels and artifact paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from src.baselines import LinearImputer, LocfImputer
from src.data.schema import TIME_AXIS
from src.eval.m2_val_impute import OriginalScaleRestorer
from src.eval.m3_data import (
    DataBundle,
    ProtocolComposeRequest,
    ProtocolMaskBundle,
    SplitLoadRequest,
    SplitWindowBundle,
    build_protocol_grid,
    compose_protocol_masks,
    count_target_cells,
    load_split_window_bundle,
)
from src.eval.m3_panels import (
    BaselineFills,
    BenchmarkComparisonPlotter,
    ComparisonView,
    M3ArtifactPaths,
    OriginalUnitContext,
    PanelSpec,
    ScoredContext,
    panel_title,
    point_fill_as_ensemble,
)
from src.eval.m3_table import (
    MetricRecord,
    MetricsTableAssembler,
    ProtocolKey,
    RunMeta,
    ScoreBlock,
    SeedPair,
    point_method_names,
)
from src.eval.metrics import EvalTensors, ImputationEvaluator, MetricsTable
from src.eval.plots import ImputationPlotData
from src.models.score_net import ConditionalScoreNet
from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE

_POINT_METHODS = point_method_names()


@dataclass(frozen=True)
class RunGrid:
    """Protocol × method grid and shared run knobs.

    Public methods
    --------------
    n_protocols
        Number of protocols in the grid.
    includes
        Whether a method name is enabled.

    Instance variables
    ------------------
    protocols : tuple of ProtocolKey
        Frozen protocol list.
    methods : tuple of str
        Enabled method names in report order.
    seeds : SeedPair
        Mask and sampler seeds.
    k_samples : int
        TimeFill ensemble size.
    artifact_dir : pathlib.Path
        ``artifacts/m3`` root.
    """

    protocols: tuple[ProtocolKey, ...]
    methods: tuple[str, ...]
    seeds: SeedPair
    k_samples: int
    artifact_dir: Path

    def n_protocols(self) -> int:
        """Return the number of protocols."""
        return len(self.protocols)

    def includes(self, method: str) -> bool:
        """Return whether ``method`` is in the enabled list."""
        return method in self.methods


@dataclass(frozen=True)
class ImputeContextRequest:
    """Build a reverse-SDE context with an overridden solver step count.

    Public methods
    --------------
    build
        Construct and validate an ``ImputeContext``.
    describe
        Short log string.

    Instance variables
    ------------------
    model : ConditionalScoreNet
        Loaded score network.
    sde_cfg :
        ``SdeConfig`` from ``real_m2.yaml``.
    t_eps : float
        Terminal diffusion time.
    solver_steps : int
        Euler–Maruyama steps (bench or smoke).
    """

    model: ConditionalScoreNet
    sde_cfg: object
    t_eps: float
    solver_steps: int

    def build(self) -> ImputeContext:
        """Return a validated ``ImputeContext``."""
        settings = ImputeSolverSettings(
            solver_steps=int(self.solver_steps),
            t_eps=float(self.t_eps),
        )
        context = ImputeContext(
            model=self.model,
            sde=VPSDE(self.sde_cfg),
            settings=settings,
        )
        context.validate()
        return context

    def describe(self) -> str:
        """Return a short solver summary."""
        return f"solver_steps={self.solver_steps}, t_eps={self.t_eps}"


@dataclass(frozen=True)
class MethodScoreInput:
    """Inputs needed to score one method on a protocol.

    Public methods
    --------------
    as_eval_tensors
        Pack into ``EvalTensors``.
    n_targets
        Target-cell count from the metric mask.

    Instance variables
    ------------------
    truth : jax.Array
        Ground truth ``(B, T, F)``.
    mean : jax.Array
        Point / posterior mean ``(B, T, F)``.
    samples : jax.Array
        Ensemble ``(B, K, T, F)``.
    metric_mask : jax.Array
        Observation-style mask (0 = score).
    """

    truth: jnp.ndarray
    mean: jnp.ndarray
    samples: jnp.ndarray
    metric_mask: jnp.ndarray

    def as_eval_tensors(self) -> EvalTensors:
        """Build ``EvalTensors`` for ``ImputationEvaluator``."""
        return EvalTensors(
            truth=self.truth,
            mean=self.mean,
            samples=self.samples,
            mask=self.metric_mask,
        )

    def n_targets(self) -> int:
        """Count cells where the metric mask is missing (scored)."""
        weights = self.as_eval_tensors().missing_weights()
        return int(jnp.sum(weights))


@dataclass(frozen=True)
class TimeFillResult:
    """Original-unit TimeFill impute outputs.

    Public methods
    --------------
    as_plot_data
        Build ``ImputationPlotData`` with truth / obs / metric mask.
    score_input
        Build ``MethodScoreInput`` for metrics.

    Instance variables
    ------------------
    mean : jax.Array
        Posterior mean ``(B, T, F)``.
    samples : jax.Array
        Posterior samples ``(B, K, T, F)``.
    quantiles : jax.Array
        Low/high bands ``(B, T, F, 2)``.
    """

    mean: jnp.ndarray
    samples: jnp.ndarray
    quantiles: jnp.ndarray

    def as_plot_data(
            self,
            scored: ScoredContext,
            metric_mask: jnp.ndarray,
    ) -> ImputationPlotData:
        """Pack TimeFill bands with truth and observations."""
        return ImputationPlotData(
            truth=scored.truth(),
            observations=scored.observations(),
            mean=self.mean,
            quantiles=self.quantiles,
            mask=metric_mask,
        )

    def score_input(
            self,
            truth: jnp.ndarray,
            metric_mask: jnp.ndarray,
    ) -> MethodScoreInput:
        """Build a score pack for TimeFill."""
        return MethodScoreInput(
            truth=truth,
            mean=self.mean,
            samples=self.samples,
            metric_mask=metric_mask,
        )


class _ScoreAndPlotHelpers:
    """Private evaluator + plotter pair (keeps the runner attribute-light).

    Public methods
    --------------
    score
        Compute a ``MetricsTable``.
    save_panel
        Write a comparison PNG.
    """

    def __init__(self) -> None:
        """Create evaluator and plotter instances."""
        self._evaluator = ImputationEvaluator()
        self._plotter = BenchmarkComparisonPlotter()

    def score(self, score_input: MethodScoreInput) -> MetricsTable:
        """Score one method on artificial targets."""
        return self._evaluator.compute(score_input.as_eval_tensors())

    def save_panel(self, view: ComparisonView, output_path: Path) -> Path:
        """Write one comparison panel PNG."""
        return self._plotter.save(view, output_path)


class M3BenchmarkRunner:
    """Run protocol × method imputation and assemble M3 artifacts.

    Public methods
    --------------
    run
        Execute the full grid and write tables + one panel.
    score_method
        Compute a ``MetricsTable`` for one method pack.

    Instance variables
    ------------------
    data : DataBundle
        Windows and scaler.
    context : ImputeContext
        Reverse-SDE impute context.
    grid : RunGrid
        Protocols, methods, seeds, ``K``, artifact root.
    paths : M3ArtifactPaths
        Output paths under ``artifacts/m3``.
    """

    def __init__(
            self,
            data: DataBundle,
            context: ImputeContext,
            grid: RunGrid,
    ) -> None:
        """Attach data, model context, and run grid.

        Parameters
        ----------
        data : DataBundle
            Report-split windows and scaler.
        context : ImputeContext
            Reverse-SDE imputer wiring.
        grid : RunGrid
            Protocol × method knobs.
        """
        self.data = data
        self.context = context
        self.grid = grid
        self.paths = M3ArtifactPaths(grid.artifact_dir)
        self._helpers = _ScoreAndPlotHelpers()

    def run(self) -> list[MetricRecord]:
        """Run all protocols × methods and write artifacts.

        Returns
        -------
        list of MetricRecord
            Long-form table rows.
        """
        self.paths.ensure()
        assembler = MetricsTableAssembler()
        panel_written = False
        for protocol in self.grid.protocols:
            rows, view = self._run_protocol(protocol)
            for row in rows:
                assembler.append(row)
            if not panel_written and view is not None:
                self._helpers.save_panel(view, self.paths.comparison_panel())
                panel_written = True
        assembler.write_csv(self.paths.metrics_csv())
        assembler.write_json(self.paths.metrics_json())
        return assembler.rows()

    def score_method(self, score_input: MethodScoreInput) -> MetricsTable:
        """Score one method on artificial targets.

        Parameters
        ----------
        score_input : MethodScoreInput
            Truth, mean, samples, metric mask.

        Returns
        -------
        MetricsTable
            MAE / RMSE / CRPS / coverage.
        """
        return self._helpers.score(score_input)

    def _run_protocol(
            self,
            protocol: ProtocolKey,
    ) -> tuple[list[MetricRecord], ComparisonView | None]:
        """Impute and score all methods for one protocol."""
        masks = compose_protocol_masks(
            ProtocolComposeRequest(
                protocol=protocol,
                windows=self.data.windows,
                key=jax.random.PRNGKey(self.grid.seeds.mask),
            )
        )
        scored = self._original_unit_context(masks)
        timefill = self._timefill(masks)
        metric_mask = masks.metric_mask()
        n_targets = masks.n_target_cells()
        packs = self._method_packs(scored, timefill, metric_mask)
        rows = [
            self._row_from_score(name, protocol, packs[name], n_targets)
            for name in self.grid.methods
        ]
        view = ComparisonView(
            bands=timefill.as_plot_data(scored, metric_mask),
            baselines=scored.baselines,
            spec=PanelSpec(title=panel_title(protocol)),
        )
        return rows, view

    def _original_unit_context(
            self,
            masks: ProtocolMaskBundle,
    ) -> ScoredContext:
        """Inverse-transform context and run LOCF / linear in original units."""
        restorer = OriginalScaleRestorer(self.data.scaler)
        truth = restorer.restore_windows(
            self.data.windows.values_n,
            masks.m_native,
        )
        observations = restorer.restore_windows(masks.x_obs_n, masks.m_eval)
        means = self.data.feature_means()
        baselines = BaselineFills(
            locf=LocfImputer(means).impute(observations, masks.m_eval),
            linear=LinearImputer(means).impute(observations, masks.m_eval),
        )
        return OriginalUnitContext(truth, observations).with_baselines(
            baselines,
        )

    def _timefill(self, masks: ProtocolMaskBundle) -> TimeFillResult:
        """Impute in normalized space and restore to original units."""
        seq_len = int(masks.x_obs_n.shape[TIME_AXIS])
        imputer = ReverseSDEImputer(self.context)
        samples_n, mean_n, quantiles_n = imputer.impute(
            ImputeQuery(
                observations=masks.x_obs_n,
                mask=masks.m_eval,
                timestamps=grid_timestamps(seq_len),
                key=jax.random.PRNGKey(self.grid.seeds.sample),
                k_samples=self.grid.k_samples,
            )
        )
        restorer = OriginalScaleRestorer(self.data.scaler)
        m_native = masks.m_native
        return TimeFillResult(
            mean=restorer.restore_windows(mean_n, m_native),
            samples=restorer.restore_samples(samples_n, m_native),
            quantiles=restorer.restore_quantiles(quantiles_n, m_native),
        )

    def _method_packs(
            self,
            scored: ScoredContext,
            timefill: TimeFillResult,
            metric_mask: jnp.ndarray,
    ) -> dict[str, MethodScoreInput]:
        """Build score inputs for TimeFill and point baselines."""
        truth = scored.truth()
        return {
            "timefill": timefill.score_input(truth, metric_mask),
            "locf": MethodScoreInput(
                truth=truth,
                mean=scored.baselines.locf,
                samples=scored.baselines.as_ensemble("locf"),
                metric_mask=metric_mask,
            ),
            "linear": MethodScoreInput(
                truth=truth,
                mean=scored.baselines.linear,
                samples=scored.baselines.as_ensemble("linear"),
                metric_mask=metric_mask,
            ),
        }

    def _row_from_score(
            self,
            method: str,
            protocol: ProtocolKey,
            pack: MethodScoreInput,
            n_targets: int,
    ) -> MetricRecord:
        """Build a composed row from scored tensors."""
        table = self.score_method(pack)
        point_policy = (
            "degenerate_k1" if method in _POINT_METHODS else "ensemble"
        )
        return MetricRecord(
            method=method,
            protocol=protocol,
            scores=ScoreBlock.from_table(table),
            meta=RunMeta(
                seeds=self.grid.seeds,
                split=self.data.windows.split_name,
                k_samples=self.grid.k_samples,
                point_metrics=point_policy,
            ),
            n_target_cells=n_targets,
        )


__all__ = [
    "BaselineFills",
    "BenchmarkComparisonPlotter",
    "ComparisonView",
    "DataBundle",
    "ImputeContextRequest",
    "M3ArtifactPaths",
    "M3BenchmarkRunner",
    "MetricRecord",
    "MetricsTableAssembler",
    "MethodScoreInput",
    "OriginalUnitContext",
    "PanelSpec",
    "ProtocolComposeRequest",
    "ProtocolKey",
    "ProtocolMaskBundle",
    "RunGrid",
    "RunMeta",
    "ScoreBlock",
    "ScoredContext",
    "SeedPair",
    "SplitLoadRequest",
    "SplitWindowBundle",
    "TimeFillResult",
    "build_protocol_grid",
    "compose_protocol_masks",
    "count_target_cells",
    "load_split_window_bundle",
    "point_fill_as_ensemble",
]
