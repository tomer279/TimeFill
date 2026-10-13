"""Load and validate Milestone 4 app YAML (``configs/app_m4.yaml``).

Classes
-------
AppM4ArtifactPaths
    Resolved checkpoint, scaler, sample CSV, and real-m2 config paths.
AppM4Schema
    Frozen ``F``, feature names, and ``seq_len``.
AppM4SampleKnobs
    ``K``, solver steps, and PRNG seeds.
AppM4SmokeCaps
    UI / smoke caps (``max_windows``, demo-eval slice).
AppM4Config
    Root product config facade.

Functions
---------
load_app_m4_config
    Parse and validate an ``app_m4.yaml`` file.
resolve_repo_path
    Resolve a path relative to the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from app.window_adapter import WindowAdaptRequest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_ROOT = ("data", "sample", "seeds")
_REQUIRED_DATA = (
    "checkpoint",
    "scaler",
    "sample_csv",
    "n_features",
    "feature_names",
    "seq_len",
)
_DEFAULT_REAL_CONFIG = "configs/real_m2.yaml"
_DEFAULT_MAX_WINDOWS = 4
_DEFAULT_SMOKE_K = 2
_DEFAULT_SMOKE_STEPS = 10
_DEFAULT_BLOCK_LEN = 8
_DEFAULT_MCAR_RATE = 0.30


@dataclass(frozen=True)
class AppM4ArtifactPaths:
    """Filesystem paths required for the pretrained M4 impute path.

    Public methods
    --------------
    require_exist
        Fail fast if checkpoint or scaler is missing.
    describe
        Short one-line summary for logs / CLI.

    Instance variables
    ------------------
    checkpoint : pathlib.Path
        Score-net ``.eqx`` (typically M2 ``best_score_net.eqx``).
    scaler : pathlib.Path
        ``feature_scaler.npz``.
    sample_csv : pathlib.Path
        Bundled demo CSV.
    real_config : pathlib.Path
        ``real_m2.yaml`` for SDE / ``t_eps`` / architecture parity.
    """

    checkpoint: Path
    scaler: Path
    sample_csv: Path
    real_config: Path

    def require_exist(self) -> None:
        """Raise ``FileNotFoundError`` if pretrained artifacts are absent."""
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"M2 checkpoint missing: {self.checkpoint}. "
                "Train or copy artifacts/m2/checkpoints/best_score_net.eqx"
            )
        if not self.scaler.is_file():
            raise FileNotFoundError(
                f"Feature scaler missing: {self.scaler}. "
                "Expected feature_scaler.npz beside the checkpoint"
            )
        if not self.real_config.is_file():
            raise FileNotFoundError(
                f"real_m2 config missing: {self.real_config}"
            )

    def describe(self) -> str:
        """Return a short path summary.

        Returns
        -------
        str
            Checkpoint / scaler / sample paths.
        """
        return (
            f"ckpt={self.checkpoint}, scaler={self.scaler}, "
            f"sample={self.sample_csv}"
        )


@dataclass(frozen=True)
class AppM4Schema:
    """Feature / window geometry locked for the M4 product path.

    Public methods
    --------------
    validate
        Check ``F``, names, and ``seq_len``.
    window_request
        Build a ``WindowAdaptRequest`` with an optional cap.

    Instance variables
    ------------------
    n_features : int
        Expected ``F`` (Beijing = 6).
    feature_names : tuple of str
        Frozen EVAL order.
    seq_len : int
        Window length ``T``.
    """

    n_features: int
    feature_names: tuple[str, ...]
    seq_len: int

    def validate(self) -> None:
        """Raise ``ValueError`` if schema knobs are inconsistent."""
        if self.n_features <= 0:
            raise ValueError(
                f"n_features must be positive, got {self.n_features}"
            )
        if len(self.feature_names) != self.n_features:
            raise ValueError(
                "feature_names length must equal n_features; "
                f"got {len(self.feature_names)} vs {self.n_features}"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(
                f"feature_names must be unique; got {self.feature_names!r}"
            )
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")

    def window_request(self, max_windows: int | None):
        """Return a ``WindowAdaptRequest`` for WP1 adaptation.

        Parameters
        ----------
        max_windows : int or None
            Optional ``smoke.max_windows`` cap.

        Returns
        -------
        WindowAdaptRequest
            ``seq_len`` / ``max_windows`` bundle.
        """

        return WindowAdaptRequest(
            seq_len=self.seq_len,
            max_windows=max_windows,
        )


@dataclass(frozen=True)
class AppM4SampleKnobs:
    """Latency-first reverse-SDE sample defaults.

    Public methods
    --------------
    validate
        Check positivity of ``K`` / steps.
    effective
        Return knobs after optional CLI overrides.

    Instance variables
    ------------------
    k_samples : int
        Posterior sample count ``K``.
    solver_steps : int
        Euler–Maruyama steps.
    seed_mask : int
        PRNG seed for optional demo corruption.
    seed_sample : int
        PRNG seed for reverse-SDE sampling.
    """

    k_samples: int
    solver_steps: int
    seed_mask: int
    seed_sample: int

    def validate(self) -> None:
        """Raise ``ValueError`` if sample knobs are invalid."""
        if self.k_samples <= 0:
            raise ValueError(
                f"K_samples must be positive, got {self.k_samples}"
            )
        if self.solver_steps <= 0:
            raise ValueError(
                f"solver_steps must be positive, got {self.solver_steps}"
            )

    def effective(
            self,
            k_samples: int | None,
            solver_steps: int | None,
    ) -> AppM4SampleKnobs:
        """Return a copy with optional overrides.

        Parameters
        ----------
        k_samples : int or None
            Override ``K`` when set.
        solver_steps : int or None
            Override solver steps when set.

        Returns
        -------
        AppM4SampleKnobs
            Validated knobs.
        """
        knobs = AppM4SampleKnobs(
            k_samples=self.k_samples if k_samples is None else int(k_samples),
            solver_steps=(
                self.solver_steps
                if solver_steps is None
                else int(solver_steps)
            ),
            seed_mask=self.seed_mask,
            seed_sample=self.seed_sample,
        )
        knobs.validate()
        return knobs


@dataclass(frozen=True)
class AppM4SmokeCaps:
    """Optional UI / smoke caps (not the full M3 grid).

    Public methods
    --------------
    validate
        Check cap positivity when set.
    default_demo_rate
        First MCAR rate, or a locked default.

    Instance variables
    ------------------
    max_windows : int or None
        Cap on adapted windows.
    k_samples : int
        Tiny ``K`` for demo-eval button (WP3+).
    solver_steps : int
        Tiny solver steps for demo-eval.
    mcar_rates : tuple of float
        Demo-eval / optional corruption rates.
    block_len : int
        Contiguous block length for block demo masks.
    """

    max_windows: int | None
    k_samples: int
    solver_steps: int
    mcar_rates: tuple[float, ...]
    block_len: int

    def validate(self) -> None:
        """Raise ``ValueError`` if smoke caps are invalid."""
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError(
                f"max_windows must be positive when set, got {self.max_windows}"
            )
        if self.block_len <= 0:
            raise ValueError(
                f"block_len must be positive, got {self.block_len}"
            )
        for rate in self.mcar_rates:
            if not 0.0 <= float(rate) <= 1.0:
                raise ValueError(f"mcar rate must be in [0, 1], got {rate}")

    def default_demo_rate(self) -> float:
        """Return the first configured MCAR rate (or a default).

        Returns
        -------
        float
            Rate in ``[0, 1]``.
        """
        if self.mcar_rates:
            return float(self.mcar_rates[0])
        return _DEFAULT_MCAR_RATE


@dataclass(frozen=True)
class AppM4Config:
    """Root M4 product config loaded from YAML.

    Public methods
    --------------
    validate
        Validate nested sections.
    artifact_paths
        Return resolved filesystem paths.

    Instance variables
    ------------------
    paths : AppM4ArtifactPaths
        Checkpoint / scaler / sample / real config.
    schema : AppM4Schema
        ``F`` / names / ``T``.
    sample : AppM4SampleKnobs
        Impute defaults.
    smoke : AppM4SmokeCaps
        Latency / demo-eval caps.
    artifact_dir : pathlib.Path
        Optional ``artifacts/m4`` root.
    """

    paths: AppM4ArtifactPaths
    schema: AppM4Schema
    sample: AppM4SampleKnobs
    smoke: AppM4SmokeCaps
    artifact_dir: Path

    def validate(self) -> None:
        """Validate nested config sections."""
        self.schema.validate()
        self.sample.validate()
        self.smoke.validate()

    def artifact_paths(self) -> AppM4ArtifactPaths:
        """Return resolved artifact paths.

        Returns
        -------
        AppM4ArtifactPaths
            Same as ``self.paths``.
        """
        return self.paths


def resolve_repo_path(path_str: str) -> Path:
    """Resolve ``path_str`` absolute or relative to the repo root.

    Parameters
    ----------
    path_str : str
        Absolute or repo-relative path.

    Returns
    -------
    pathlib.Path
        Absolute path.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def load_app_m4_config(path: Path | str) -> AppM4Config:
    """Parse and validate an M4 app YAML file.

    Parameters
    ----------
    path : path-like
        Typically ``configs/app_m4.yaml``.

    Returns
    -------
    AppM4Config
        Validated product config.

    Raises
    ------
    ValueError
        If required keys are missing or invalid.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"app config root must be a mapping: {config_path}")
    _require_keys(raw, _REQUIRED_ROOT, "root")
    data = raw["data"]
    if not isinstance(data, Mapping):
        raise ValueError("data section must be a mapping")
    _require_keys(data, _REQUIRED_DATA, "data")
    sample = raw["sample"]
    seeds = raw["seeds"]
    if not isinstance(sample, Mapping) or not isinstance(seeds, Mapping):
        raise ValueError("sample and seeds must be mappings")
    smoke_raw = raw.get("smoke", {})
    if smoke_raw is None:
        smoke_raw = {}
    if not isinstance(smoke_raw, Mapping):
        raise ValueError("smoke section must be a mapping when present")
    paths = AppM4ArtifactPaths(
        checkpoint=resolve_repo_path(str(data["checkpoint"])),
        scaler=resolve_repo_path(str(data["scaler"])),
        sample_csv=resolve_repo_path(str(data["sample_csv"])),
        real_config=resolve_repo_path(
            str(raw.get("real_config", _DEFAULT_REAL_CONFIG))
        ),
    )
    schema = AppM4Schema(
        n_features=int(data["n_features"]),
        feature_names=tuple(str(name) for name in data["feature_names"]),
        seq_len=int(data["seq_len"]),
    )
    knobs = AppM4SampleKnobs(
        k_samples=int(sample["K_samples"]),
        solver_steps=int(sample["solver_steps"]),
        seed_mask=int(seeds["mask"]),
        seed_sample=int(seeds["sample"]),
    )
    smoke = _parse_smoke(smoke_raw)
    config = AppM4Config(
        paths=paths,
        schema=schema,
        sample=knobs,
        smoke=smoke,
        artifact_dir=resolve_repo_path(
            str(raw.get("artifact_dir", "artifacts/m4"))
        ),
    )
    config.validate()
    return config


def _parse_smoke(mapping: Mapping[str, Any]) -> AppM4SmokeCaps:
    """Build ``AppM4SmokeCaps`` from a YAML ``smoke`` mapping.

    Parameters
    ----------
    mapping : Mapping
        Optional smoke section.

    Returns
    -------
    AppM4SmokeCaps
        Validated caps.
    """
    max_windows = mapping.get("max_windows", _DEFAULT_MAX_WINDOWS)
    if max_windows is not None:
        max_windows = int(max_windows)
    rates_raw = mapping.get("mcar_rates", [_DEFAULT_MCAR_RATE])
    rates = tuple(float(rate) for rate in rates_raw)
    caps = AppM4SmokeCaps(
        max_windows=max_windows,
        k_samples=int(mapping.get("K_samples", _DEFAULT_SMOKE_K)),
        solver_steps=int(mapping.get("solver_steps", _DEFAULT_SMOKE_STEPS)),
        mcar_rates=rates,
        block_len=int(mapping.get("block_len", _DEFAULT_BLOCK_LEN)),
    )
    caps.validate()
    return caps


def _require_keys(
        mapping: Mapping[str, Any],
        keys: tuple[str, ...],
        section: str,
) -> None:
    """Raise ``ValueError`` if any required key is missing.

    Parameters
    ----------
    mapping : Mapping
        YAML section.
    keys : tuple of str
        Required keys.
    section : str
        Section label for the error message.
    """
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(
            f"app config {section} missing required key(s): {missing}"
        )
