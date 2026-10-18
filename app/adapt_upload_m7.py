"""M7 upload-adapt CLI pipeline (ingest → adapt → impute → export).

Classes
-------
AdaptUploadPaths
    CSV / config / artifact filesystem paths for one run.
AdaptUploadFlags
    Mode, smoke-eval, and multi-sample export flags.
AdaptUploadPipelineResult
    Session artifacts plus written CSV paths.

Functions
---------
run_adapt_upload_pipeline
    Full library entry: Flexible ingest → capped adapt → mean-fill CSV.
parse_adapt_upload_args
    argparse flags shared with ``scripts/adapt_upload_m7.py``.
adapt_upload_main
    CLI main with clear exit codes (no stack-trace-only failures).

Exit codes
----------
0
    Success (adapt + mean-fill export).
1
    Unexpected / unclassified failure.
2
    Ingest / too-little-data / policy reject (no adapt start).
3
    Adapt abort: non-finite loss (no promote).
4
    Adapt abort: wall-clock timeout (no promote).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app.demo_eval_smoke import DemoEvalSmokeRequest, run_demo_eval_smoke
from app.export_frames import (
    MeanFillExportRequest,
    SampleExportRequest,
    mean_fill_dataframe,
    samples_long_dataframe,
)
from app.flexible_ingest import (
    ColumnRoleMap,
    column_role_map_from_mapping,
    ingest_flexible_csv,
)
from app.m4_config import AppM4SampleKnobs
from app.preview_ingest import (
    MODE_FLEXIBLE,
    FlexiblePreviewKnobs,
    try_adapt_flexible_windows,
)
from app.session_impute import (
    DemoMaskRequest,
    SessionAssetRequest,
    SessionImputeRequest,
    SessionImputer,
)
from app.window_adapter import AdaptedWindows
from src.data.adapt_config import (
    AdaptCliOverrides,
    AdaptLoopCliOverrides,
    AdaptM7Config,
    AdaptWindowCliOverrides,
    apply_adapt_cli_overrides,
    load_adapt_m7_config,
)
from src.train.adapt_m7 import (
    AdaptInitPolicy,
    AdaptSessionRequest,
    AdaptSessionResult,
    AdaptWindowTensors,
    run_adapt_session,
)

_EXIT_OK = 0
_EXIT_UNEXPECTED = 1
_EXIT_INGEST = 2
_EXIT_NAN = 3
_EXIT_TIMEOUT = 4
_MEAN_FILL_NAME = "mean_fill.csv"
_SAMPLES_NAME = "samples_long.csv"
_MODE_NONE = "none"
_MODE_SCRATCH = "scratch"
_NAN_TOKEN = "non-finite"
_TIMEOUT_TOKEN = "timeout"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "adapt_m7.yaml"


@dataclass(frozen=True)
class AdaptUploadPaths:
    """Filesystem paths for one CLI run.

    Public methods
    --------------
    resolve_artifact_root
        Absolute artifact root (optional override).
    has_role_map
        Whether a column-map YAML was supplied.

    Instance variables
    ------------------
    csv_path : pathlib.Path
        Input CSV.
    config_path : pathlib.Path
        ``adapt_m7.yaml`` path.
    map_path : pathlib.Path or None
        Optional Flexible role map.
    artifact_root : pathlib.Path or None
        Optional override for ``adapt.artifact_dir``.
    """

    csv_path: Path
    config_path: Path
    map_path: Path | None
    artifact_root: Path | None

    def resolve_artifact_root(self) -> Path | None:
        """Return absolute artifact root when overridden.

        Returns
        -------
        pathlib.Path or None
            Absolute root, or ``None`` to use YAML ``artifact_dir``.
        """
        if self.artifact_root is None:
            return None
        root = Path(self.artifact_root)
        if root.is_absolute():
            return root
        return _REPO_ROOT / root

    def has_role_map(self) -> bool:
        """Return whether an explicit role map path was set.

        Returns
        -------
        bool
            ``True`` when ``map_path`` is not ``None``.
        """
        return self.map_path is not None


@dataclass(frozen=True)
class AdaptUploadFlags:
    """Boolean / mode CLI product flags.

    Public methods
    --------------
    want_smoke_eval
        Whether to print file-local smoke MAE/RMSE.
    want_samples_export
        Whether to write multi-sample long CSV.

    Instance variables
    ------------------
    smoke_eval : bool
        Run held-out artificial-gap smoke metrics.
    export_samples : bool
        Write ``samples_long.csv``.
    mode : str
        ``scratch`` or ``fine_tune``.
    """

    smoke_eval: bool
    export_samples: bool
    mode: str

    def want_smoke_eval(self) -> bool:
        """Return whether smoke metrics should run.

        Returns
        -------
        bool
            ``smoke_eval`` flag.
        """
        return bool(self.smoke_eval)

    def want_samples_export(self) -> bool:
        """Return whether multi-sample CSV should be written.

        Returns
        -------
        bool
            ``export_samples`` flag.
        """
        return bool(self.export_samples)


@dataclass(frozen=True)
class AdaptUploadPipelineResult:
    """Successful CLI outputs for logging / tests.

    Public methods
    --------------
    mean_fill_path
        Path to written mean-fill CSV.
    run_dir
        Session artifact directory.

    Instance variables
    ------------------
    session : AdaptSessionResult
        Promoted adapt session.
    mean_fill : pathlib.Path
        Mean-fill CSV path.
    samples_path : pathlib.Path or None
        Optional samples CSV.
    """

    session: AdaptSessionResult
    mean_fill: Path
    samples_path: Path | None

    def mean_fill_path(self) -> Path:
        """Return the mean-fill CSV path.

        Returns
        -------
        pathlib.Path
            Downloadable fill path.
        """
        return self.mean_fill

    def run_dir(self) -> Path:
        """Return the session run directory.

        Returns
        -------
        pathlib.Path
            ``artifacts/m7/<run_id>/`` (or override root).
        """
        return self.session.run_dir


def parse_adapt_upload_args(
        argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI flags for M7 upload-adapt E2E.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed flags.
    """
    parser = argparse.ArgumentParser(
        description=(
            "M7 CLI: Flexible ingest → upload-adapt → impute → "
            "downloadable mean-fill (file-local smoke only)."
        )
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Override adapt.artifact_dir (never artifacts/m2)",
    )
    parser.add_argument(
        "--mode",
        choices=("scratch", "fine_tune"),
        default=None,
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--k-samples", type=int, default=None)
    parser.add_argument("--solver-steps", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--smoke-eval",
        action="store_true",
        help="Print file-local smoke MAE/RMSE (not an M3 claim)",
    )
    parser.add_argument(
        "--export-samples",
        action="store_true",
        help="Also write samples_long.csv under the run dir",
    )
    return parser.parse_args(argv)


def adapt_upload_main(argv: list[str] | None = None) -> int:
    """Run ingest → adapt → impute → export with clear exit codes.

    Parameters
    ----------
    argv : list of str or None, optional
        Optional CLI args.

    Returns
    -------
    int
        Process exit code (see module docstring).
    """
    args = parse_adapt_upload_args(argv)
    paths = AdaptUploadPaths(
        csv_path=Path(args.csv),
        config_path=Path(args.config),
        map_path=None if args.map is None else Path(args.map),
        artifact_root=args.artifact_dir,
    )
    mode = _MODE_SCRATCH if args.mode is None else str(args.mode)
    flags = AdaptUploadFlags(
        smoke_eval=bool(args.smoke_eval),
        export_samples=bool(args.export_samples),
        mode=mode,
    )
    overrides = AdaptCliOverrides(
        window=AdaptWindowCliOverrides(
            seq_len=args.seq_len,
            max_windows=args.max_windows,
            batch_size=args.batch_size,
        ),
        loop=AdaptLoopCliOverrides(
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            k_samples=args.k_samples,
            solver_steps=args.solver_steps,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    try:
        result = run_adapt_upload_pipeline(paths, flags, overrides)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_INGEST
    except RuntimeError as exc:
        return _exit_for_runtime(str(exc))
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_UNEXPECTED
    _print_success(result, flags)
    return _EXIT_OK


def run_adapt_upload_pipeline(
        paths: AdaptUploadPaths,
        flags: AdaptUploadFlags,
        overrides: AdaptCliOverrides,
) -> AdaptUploadPipelineResult:
    """Execute the full CLI pipeline (library entry for tests).

    Parameters
    ----------
    paths : AdaptUploadPaths
        CSV / config / artifact paths.
    flags : AdaptUploadFlags
        Smoke-eval / export / mode flags.
    overrides : AdaptCliOverrides
        Optional smoke-cap overrides.

    Returns
    -------
    AdaptUploadPipelineResult
        Session paths and export locations.

    Raises
    ------
    ValueError
        Ingest reject, too little data, or fine-tune policy refusal.
    RuntimeError
        NaN loss or adapt timeout.
    """
    config = apply_adapt_cli_overrides(
        load_adapt_m7_config(paths.config_path),
        overrides,
    )
    _print_caps(config, flags)
    windows = _ingest_flexible_windows(paths, config)
    session = run_adapt_session(
        AdaptSessionRequest(
            windows=_to_adapt_tensors(windows),
            config=config,
            policy=AdaptInitPolicy(mode=flags.mode),
            artifact_root=paths.resolve_artifact_root(),
        )
    )
    imputer = _session_imputer(session, config)
    fill_result = imputer.impute(
        SessionImputeRequest(
            windows=windows,
            demo_mask=DemoMaskRequest(mode=_MODE_NONE),
            k_samples=config.sample.k_samples,
            solver_steps=config.sample.solver_steps,
        )
    )
    mean_path = _write_mean_fill(session, windows, fill_result.context.mean_fill)
    samples_path = None
    if flags.want_samples_export():
        samples_path = _write_samples(
            session,
            windows,
            fill_result.tensors.samples,
        )
    if flags.want_smoke_eval():
        _run_file_local_smoke(imputer, windows, session, config)
    return AdaptUploadPipelineResult(
        session=session,
        mean_fill=mean_path,
        samples_path=samples_path,
    )


def _exit_for_runtime(message: str) -> int:
    """Map adapt RuntimeError messages to exit codes.

    Parameters
    ----------
    message : str
        Exception text.

    Returns
    -------
    int
        ``3`` for NaN, ``4`` for timeout, else ``1``.
    """
    print(f"error: {message}", file=sys.stderr)
    lowered = message.lower()
    if _NAN_TOKEN in lowered:
        return _EXIT_NAN
    if _TIMEOUT_TOKEN in lowered:
        return _EXIT_TIMEOUT
    return _EXIT_UNEXPECTED


def _ingest_flexible_windows(
        paths: AdaptUploadPaths,
        config: AdaptM7Config,
) -> AdaptedWindows:
    """Load CSV under M6 Flexible rules and adapt windows.

    Parameters
    ----------
    paths : AdaptUploadPaths
        CSV and optional role map.
    config : AdaptM7Config
        Effective ``seq_len`` / ``max_windows``.

    Returns
    -------
    AdaptedWindows
        Non-empty adapted batch.

    Raises
    ------
    ValueError
        When Flexible adapt rejects or ``B`` is too small.
    """
    role_map = None
    if paths.has_role_map():
        role_map = _load_role_map(paths.map_path)
    frame = ingest_flexible_csv(paths.csv_path, role_map=role_map)
    knobs = FlexiblePreviewKnobs(
        seq_len=int(config.geometry.seq_len),
        max_windows=config.geometry.max_windows,
    )
    windows, adapt_error = try_adapt_flexible_windows(frame, knobs)
    if windows is None:
        detail = adapt_error or "flexible window adapt failed"
        raise ValueError(detail)
    if windows.n_windows() < int(config.caps.min_windows):
        raise ValueError(
            f"too few windows for adapt: n_windows={windows.n_windows()} "
            f"< min_windows={config.caps.min_windows} "
            f"(ingest_mode={MODE_FLEXIBLE})"
        )
    return windows


def _to_adapt_tensors(windows: AdaptedWindows) -> AdaptWindowTensors:
    """Convert M6 ``AdaptedWindows`` into M7 adapt tensors.

    Parameters
    ----------
    windows : AdaptedWindows
        Ingested windows.

    Returns
    -------
    AdaptWindowTensors
        Values / native masks / names for ``run_adapt_session``.
    """
    return AdaptWindowTensors(
        values=np.asarray(windows.values, dtype=np.float32),
        native_masks=np.asarray(windows.native_mask, dtype=np.float32),
        feature_names=tuple(windows.feature_names),
    )


def _session_imputer(
        session: AdaptSessionResult,
        config: AdaptM7Config,
) -> SessionImputer:
    """Build an imputer bound to the session model and scaler.

    Parameters
    ----------
    session : AdaptSessionResult
        Successful adapt outputs.
    config : AdaptM7Config
        SDE / ``t_eps`` / sample knobs.

    Returns
    -------
    SessionImputer
        Ready for reverse-SDE impute in original units.
    """
    sample = AppM4SampleKnobs(
        k_samples=int(config.sample.k_samples),
        solver_steps=int(config.sample.solver_steps),
        seed_mask=int(config.sample.seed_mask),
        seed_sample=int(config.sample.seed_sample),
    )
    return SessionImputer.from_session_assets(
        SessionAssetRequest(
            model=session.model,
            scaler=session.scaler,
            sde_cfg=config.sde,
            t_eps=float(config.train.protocol.t_eps),
            sample=sample,
        )
    )


def _write_mean_fill(
        session: AdaptSessionResult,
        windows: AdaptedWindows,
        mean_fill: np.ndarray,
) -> Path:
    """Write mean-fill CSV under the session run directory.

    Parameters
    ----------
    session : AdaptSessionResult
        Promoted session (provides ``run_dir``).
    windows : AdaptedWindows
        Feature names for columns.
    mean_fill : numpy.ndarray
        ``(B, T, F)`` original-unit mean fill.

    Returns
    -------
    pathlib.Path
        Written CSV path.
    """
    mean_path = session.run_dir / _MEAN_FILL_NAME
    mean_fill_dataframe(
        MeanFillExportRequest(
            mean_fill=mean_fill,
            feature_names=tuple(windows.feature_names),
        )
    ).to_csv(mean_path, index=False)
    return mean_path


def _write_samples(
        session: AdaptSessionResult,
        windows: AdaptedWindows,
        samples: np.ndarray,
) -> Path:
    """Write multi-sample long CSV under the session run directory.

    Parameters
    ----------
    session : AdaptSessionResult
        Promoted session (provides ``run_dir``).
    windows : AdaptedWindows
        Feature names for columns.
    samples : numpy.ndarray
        ``(B, K, T, F)`` original-unit samples.

    Returns
    -------
    pathlib.Path
        Written CSV path.
    """
    samples_path = session.run_dir / _SAMPLES_NAME
    samples_long_dataframe(
        SampleExportRequest(
            samples=samples,
            feature_names=tuple(windows.feature_names),
        )
    ).to_csv(samples_path, index=False)
    return samples_path


def _run_file_local_smoke(
        imputer: SessionImputer,
        windows: AdaptedWindows,
        session: AdaptSessionResult,
        config: AdaptM7Config,
) -> None:
    """Print file-local smoke metrics on held-out windows only.

    Parameters
    ----------
    imputer : SessionImputer
        Session-bound imputer.
    windows : AdaptedWindows
        Full adapted batch.
    session : AdaptSessionResult
        Provides smoke index split.
    config : AdaptM7Config
        MCAR rate and sample knobs.
    """
    smoke_idx = session.split.smoke_idx
    smoke_windows = AdaptedWindows(
        values=np.asarray(windows.values[smoke_idx], dtype=np.float32),
        native_mask=np.asarray(
            windows.native_mask[smoke_idx], dtype=np.float32
        ),
        feature_names=tuple(windows.feature_names),
        entity_ids=np.asarray(windows.entity_ids[smoke_idx]),
    )
    smoke = run_demo_eval_smoke(
        imputer,
        DemoEvalSmokeRequest(
            windows=smoke_windows,
            mcar_rate=float(config.caps.smoke_mcar_rate),
            k_samples=int(config.sample.k_samples),
            solver_steps=int(config.sample.solver_steps),
            mask_seed=int(config.sample.seed_mask),
            block_len=int(config.geometry.block_len),
        ),
    )
    print(
        "file-local smoke (this upload only; not an M3/Beijing claim): "
        f"{smoke.caption()}"
    )
    for name, value in smoke.as_rows():
        print(f"  {name}={value}")


def _load_role_map(path: Path | None) -> ColumnRoleMap:
    """Load a YAML column role map.

    Parameters
    ----------
    path : pathlib.Path or None
        Map YAML path (must not be ``None`` when called).

    Returns
    -------
    ColumnRoleMap
        Validated roles.
    """
    if path is None:
        raise ValueError("role map path is required when loading a map")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"map YAML must be a mapping: {path}")
    return column_role_map_from_mapping(raw)


def _print_caps(config: AdaptM7Config, flags: AdaptUploadFlags) -> None:
    """Print effective smoke caps before adapt.

    Parameters
    ----------
    config : AdaptM7Config
        Effective config after overrides.
    flags : AdaptUploadFlags
        Mode / smoke flags.
    """
    print(
        "adapt_caps: "
        f"mode={flags.mode} "
        f"epochs={config.train.schedule.epochs} "
        f"steps_per_epoch={config.train.schedule.steps_per_epoch} "
        f"batch_size={config.geometry.batch_size} "
        f"seq_len={config.geometry.seq_len} "
        f"K={config.sample.k_samples} "
        f"solver_steps={config.sample.solver_steps} "
        f"timeout_s={config.caps.timeout_seconds} "
        f"min_windows={config.caps.min_windows}"
    )


def _print_success(
        result: AdaptUploadPipelineResult,
        flags: AdaptUploadFlags,
) -> None:
    """Print a short success summary.

    Parameters
    ----------
    result : AdaptUploadPipelineResult
        Pipeline outputs.
    flags : AdaptUploadFlags
        Export flags.
    """
    session = result.session
    print(
        f"run_id={session.run_id} "
        f"run_dir={session.run_dir} "
        f"F={session.meta.get('F')} "
        f"n_windows={session.meta.get('n_windows')} "
        f"mean_fill={result.mean_fill_path()}"
    )
    if flags.want_samples_export() and result.samples_path is not None:
        print(f"samples={result.samples_path}")
