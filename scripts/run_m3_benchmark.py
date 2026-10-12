"""M3 WP2: protocol grid × TimeFill / LOCF / linear → artifacts/m3/.

Prerequisites
-------------
- Processed windows from ``scripts/prepare_m2_data.py``.
- M2 checkpoint + scaler at paths in ``configs/bench_m3.yaml``.
- SDE / score-net knobs from ``configs/real_m2.yaml`` (not retrained here).

Example
-------
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml --smoke
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import yaml

# Script entrypoints need the repo root on ``sys.path`` before ``src`` imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pylint: disable=wrong-import-position
from src.data.normalize import FeatureScaler
from src.data.real_config import load_real_m2_config
from src.eval.m3_benchmark import (
    DataBundle,
    ImputeContextRequest,
    M3BenchmarkRunner,
    RunGrid,
    SeedPair,
    SplitLoadRequest,
    build_protocol_grid,
    load_split_window_bundle,
)
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.train.artifacts import load_checkpoint
# pylint: enable=wrong-import-position

_DEFAULT_BENCH = _REPO_ROOT / "configs" / "bench_m3.yaml"
_DEFAULT_REAL = _REPO_ROOT / "configs" / "real_m2.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the M3 benchmark runner.

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
        default=_DEFAULT_BENCH,
        help="Path to bench_m3 YAML",
    )
    parser.add_argument(
        "--real-config",
        type=Path,
        default=_DEFAULT_REAL,
        help="Path to real_m2 YAML (SDE / t_eps / architecture)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke.max_windows / K / solver_steps (and smoke.split)",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class BenchPaths:
    """Filesystem inputs and artifact root for the M3 benchmark.

    Public methods
    --------------
    require_exist
        Fail fast if processed data, checkpoint, or scaler is missing.
    describe
        Short path summary for logs.

    Instance variables
    ------------------
    processed_dir : pathlib.Path
        M2 processed windows directory.
    checkpoint : pathlib.Path
        TimeFill score-net checkpoint.
    scaler_path : pathlib.Path
        ``feature_scaler.npz``.
    artifact_dir : pathlib.Path
        ``artifacts/m3`` root.
    """

    processed_dir: Path
    checkpoint: Path
    scaler_path: Path
    artifact_dir: Path

    def require_exist(self) -> None:
        """Raise ``FileNotFoundError`` when required inputs are absent."""
        if not self.processed_dir.is_dir():
            raise FileNotFoundError(
                "processed windows dir not found: "
                f"{self.processed_dir}; run scripts/prepare_m2_data.py first"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                "TimeFill checkpoint not found: "
                f"{self.checkpoint}; run scripts/train_m2_real.py first"
            )
        if not self.scaler_path.is_file():
            raise FileNotFoundError(
                "feature scaler not found: "
                f"{self.scaler_path}; expected beside the M2 checkpoint"
            )

    def describe(self) -> str:
        """Return a short path summary."""
        return (
            f"processed={self.processed_dir}, "
            f"ckpt={self.checkpoint.name}, "
            f"out={self.artifact_dir}"
        )


@dataclass(frozen=True)
class SampleKnobs:
    """TimeFill sample count and solver steps.

    Public methods
    --------------
    describe
        Short log string.
    as_dict
        Serialize for debugging dumps.

    Instance variables
    ------------------
    k_samples : int
        Posterior sample count ``K``.
    solver_steps : int
        Reverse-SDE Euler steps.
    """

    k_samples: int
    solver_steps: int

    def describe(self) -> str:
        """Return a short sample/solver summary."""
        return f"K={self.k_samples}, solver_steps={self.solver_steps}"

    def as_dict(self) -> dict[str, int]:
        """Return knobs as a plain dict."""
        return {
            "k_samples": int(self.k_samples),
            "solver_steps": int(self.solver_steps),
        }


@dataclass(frozen=True)
class BenchSelection:
    """Split selection, smoke window cap, and enabled methods.

    Public methods
    --------------
    describe
        Short log string.
    method_tuple
        Enabled methods as a tuple.

    Instance variables
    ------------------
    split_name : str
        Report split (``test``) or smoke split (``val``).
    max_windows : int or None
        Optional leading-window cap.
    n_features : int
        Expected feature count ``F``.
    methods : tuple of str
        Enabled method names in report order.
    """

    split_name: str
    max_windows: int | None
    n_features: int
    methods: tuple[str, ...]

    def describe(self) -> str:
        """Return a short selection summary."""
        cap = "all" if self.max_windows is None else str(self.max_windows)
        return f"split={self.split_name}, max_windows={cap}"

    def method_tuple(self) -> tuple[str, ...]:
        """Return enabled methods."""
        return self.methods


@dataclass(frozen=True)
class BenchRunKnobs:
    """Sample, seed, and split/method selection for one bench run.

    Public methods
    --------------
    describe
        Short log string combining selection, sample, and seeds.
    method_tuple
        Enabled methods in report order.

    Instance variables
    ------------------
    sample : SampleKnobs
        ``K`` and solver steps.
    seeds : SeedPair
        Mask and sampler seeds.
    selection : BenchSelection
        Split / methods / feature count.
    """

    sample: SampleKnobs
    seeds: SeedPair
    selection: BenchSelection

    def describe(self) -> str:
        """Return a short run-knob summary."""
        return (
            f"{self.selection.describe()}; {self.sample.describe()}; "
            f"{self.seeds.describe()}"
        )

    def method_tuple(self) -> tuple[str, ...]:
        """Return enabled methods."""
        return self.selection.method_tuple()


class BenchM3Plan:
    """Resolved bench config: paths, run knobs, and protocol grid.

    Public methods
    --------------
    require_inputs
        Delegate existence checks to ``paths``.
    describe
        One-line run description.
    protocol_grid
        Expand YAML protocols into ``ProtocolKey`` values.
    to_run_grid
        Build a ``RunGrid`` for the benchmark runner.

    Instance variables
    ------------------
    paths : BenchPaths
        Filesystem inputs and artifact root.
    knobs : BenchRunKnobs
        Sample, seeds, and selection.
    protocols_cfg : dict
        Raw ``protocols`` mapping from YAML.
    """

    def __init__(
            self,
            paths: BenchPaths,
            knobs: BenchRunKnobs,
            protocols_cfg: dict[str, Any],
    ) -> None:
        """Store composed bench plan collaborators.

        Parameters
        ----------
        paths : BenchPaths
            Absolute input / output paths.
        knobs : BenchRunKnobs
            Sample, seeds, and split/method selection.
        protocols_cfg : dict
            Raw protocol grid from YAML.
        """
        self.paths = paths
        self.knobs = knobs
        self.protocols_cfg = protocols_cfg

    @property
    def sample(self) -> SampleKnobs:
        """Return sample knobs (``K``, solver steps)."""
        return self.knobs.sample

    @property
    def seeds(self) -> SeedPair:
        """Return mask and sampler seeds."""
        return self.knobs.seeds

    @property
    def selection(self) -> BenchSelection:
        """Return split / method selection."""
        return self.knobs.selection

    def require_inputs(self) -> None:
        """Fail fast if M2 processed data / checkpoint / scaler is missing."""
        self.paths.require_exist()

    def describe(self) -> str:
        """Return a one-line run description."""
        return self.knobs.describe()

    def protocol_grid(self):
        """Expand the YAML protocol section."""
        return build_protocol_grid(self.protocols_cfg)

    def to_run_grid(self) -> RunGrid:
        """Build the runner grid from this plan."""
        return RunGrid(
            protocols=self.protocol_grid(),
            methods=self.knobs.method_tuple(),
            seeds=self.knobs.seeds,
            k_samples=self.knobs.sample.k_samples,
            artifact_dir=self.paths.artifact_dir,
        )


class BenchM3ConfigLoader:
    """Parse ``bench_m3.yaml`` and apply optional smoke overrides.

    Public methods
    --------------
    load
        Read YAML and return the root mapping.
    resolve
        Build a ``BenchM3Plan`` with or without smoke knobs.

    Instance variables
    ------------------
    repo_root : pathlib.Path
        Project root for resolving relative paths.
    """

    def __init__(self, repo_root: Path) -> None:
        """Store the repository root.

        Parameters
        ----------
        repo_root : pathlib.Path
            Project root containing ``configs/`` and ``data/``.
        """
        self.repo_root = Path(repo_root)

    def load(self, config_path: Path) -> dict[str, Any]:
        """Load a bench YAML mapping.

        Parameters
        ----------
        config_path : pathlib.Path
            Path to ``bench_m3.yaml``.

        Returns
        -------
        dict
            Raw YAML root mapping.
        """
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"bench config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("bench config root must be a mapping")
        return dict(raw)

    def resolve(self, raw: Mapping[str, Any], smoke: bool) -> BenchM3Plan:
        """Resolve paths and effective sample knobs.

        Parameters
        ----------
        raw : Mapping
            Parsed ``bench_m3.yaml``.
        smoke : bool
            When True, apply ``smoke.*`` overrides only.

        Returns
        -------
        BenchM3Plan
            Composed plan ready for asset loading.
        """
        data = raw["data"]
        sample_raw = raw["sample"]
        seeds_raw = raw["seeds"]
        smoke_cfg = raw.get("smoke") or {}
        split_name = str(data["split"])
        max_windows = smoke_cfg.get("max_windows")
        k_samples = int(sample_raw["K_samples"])
        solver_steps = int(sample_raw["solver_steps"])
        if smoke:
            split_name = str(smoke_cfg.get("split", split_name))
            max_windows = smoke_cfg.get("max_windows")
            k_samples = int(smoke_cfg.get("K_samples", k_samples))
            solver_steps = int(smoke_cfg.get("solver_steps", solver_steps))
        paths = BenchPaths(
            processed_dir=self._abs(data["processed_dir"]),
            checkpoint=self._abs(data["checkpoint"]),
            scaler_path=self._abs(data["scaler"]),
            artifact_dir=self._abs(raw["artifact_dir"]),
        )
        selection = BenchSelection(
            split_name=split_name,
            max_windows=(
                None if max_windows is None else int(max_windows)
            ),
            n_features=int(data["n_features"]),
            methods=tuple(str(name) for name in raw["methods"]),
        )
        knobs = BenchRunKnobs(
            sample=SampleKnobs(
                k_samples=k_samples,
                solver_steps=solver_steps,
            ),
            seeds=SeedPair(
                mask=int(seeds_raw["mask"]),
                sample=int(seeds_raw["sample"]),
            ),
            selection=selection,
        )
        return BenchM3Plan(
            paths=paths,
            knobs=knobs,
            protocols_cfg=dict(raw["protocols"]),
        )

    def _abs(self, path_str: str) -> Path:
        """Resolve a repo-relative path."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.repo_root / path


def load_data_bundle(plan: BenchM3Plan) -> DataBundle:
    """Load scaler and normalized report-split windows.

    Parameters
    ----------
    plan : BenchM3Plan
        Resolved bench plan.

    Returns
    -------
    DataBundle
        Windows + scaler with feature-count check applied.
    """
    plan.require_inputs()
    scaler = FeatureScaler.load(plan.paths.scaler_path)
    windows = load_split_window_bundle(
        SplitLoadRequest(
            processed_dir=plan.paths.processed_dir,
            scaler=scaler,
            split_name=plan.selection.split_name,
            max_windows=plan.selection.max_windows,
        )
    )
    bundle = DataBundle(windows=windows, scaler=scaler)
    bundle.require_feature_count(plan.selection.n_features)
    return bundle


def load_impute_context(
        plan: BenchM3Plan,
        real_config_path: Path,
):
    """Load the score-net checkpoint and build an impute context.

    Parameters
    ----------
    plan : BenchM3Plan
        Resolved bench plan (``K`` / solver steps / ``F``).
    real_config_path : pathlib.Path
        ``real_m2.yaml`` for SDE / ``t_eps``.

    Returns
    -------
    ImputeContext
        Ready for ``ReverseSDEImputer``.
    """
    real_cfg = load_real_m2_config(real_config_path)
    key = jax.random.PRNGKey(real_cfg.train.protocol.seed)
    template = ConditionalScoreNet(
        key,
        default_score_net_config(plan.selection.n_features),
    )
    model = load_checkpoint(plan.paths.checkpoint, template)
    return ImputeContextRequest(
        model=model,
        sde_cfg=real_cfg.sde,
        t_eps=float(real_cfg.train.protocol.t_eps),
        solver_steps=plan.sample.solver_steps,
    ).build()


def main(argv: list[str] | None = None) -> list[dict]:
    """CLI entry: run M3 benchmark and return row dicts.

    Parameters
    ----------
    argv : list of str or None, optional
        Optional CLI args for tests.

    Returns
    -------
    list of dict
        Long-form metric rows as dicts.
    """
    args = parse_args(argv)
    loader = BenchM3ConfigLoader(_REPO_ROOT)
    plan = loader.resolve(loader.load(args.config), smoke=bool(args.smoke))
    print(plan.describe())
    data = load_data_bundle(plan)
    context = load_impute_context(plan, args.real_config)
    runner = M3BenchmarkRunner(data, context, plan.to_run_grid())
    rows = runner.run()
    print(f"Wrote {runner.paths.metrics_csv()}")
    print(f"Wrote {runner.paths.metrics_json()}")
    print(f"Wrote {runner.paths.comparison_panel()}")
    return [row.as_dict() for row in rows]


if __name__ == "__main__":
    main()
