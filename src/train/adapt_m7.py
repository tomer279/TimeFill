"""Upload-adapt session API for Milestone 7 (train-only scaler + save).

Classes
-------
AdaptWindowTensors
    ``(B, T, F)`` values / native masks plus feature names.
AdaptWindowSplit
    Deterministic train / smoke window indices.
AdaptInitPolicy
    Scratch vs fine-tune-from-base policy knobs.
AdaptSessionRequest
    Bundled inputs for one adapt invocation.
AdaptSessionResult
    Run id, artifact paths, scaler, and trained score net.
AdaptSessionRunner
    Orchestrate split → scaler → model → capped DSM → save.

Functions
---------
run_adapt_session
    Convenience wrapper around ``AdaptSessionRunner.run``.
split_adapt_windows
    Deterministic train/smoke window-index split.
make_adapt_run_id
    UTC stamp + short suffix for ``artifacts/m7/<run_id>/``.

Re-exports
----------
AdaptCaps, AdaptModeSection, AdaptGeometry, AdaptSampleSection
    From ``src.data.adapt_config``.
AdaptCliOverrides, AdaptWindowCliOverrides, AdaptLoopCliOverrides
    From ``src.data.adapt_config``.
AdaptM7Config, load_adapt_m7_config, apply_adapt_cli_overrides
    From ``src.data.adapt_config``.

Notes
-----
Does **not** write into ``artifacts/m2/``. Capped DSM lives in
``src.train.adapt_loop``. CLI E2E lives in ``app.adapt_upload_m7``;
Streamlit adapt panel is a later WP.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from src.data.adapt_config import (
    AdaptCaps,
    AdaptCliOverrides,
    AdaptGeometry,
    AdaptLoopCliOverrides,
    AdaptM7Config,
    AdaptModeSection,
    AdaptSampleSection,
    AdaptWindowCliOverrides,
    allowed_adapt_modes,
    apply_adapt_cli_overrides,
    default_adapt_mode,
    load_adapt_m7_config,
)
from src.data.normalize import FeatureScaler
from src.data.real.specs import default_beijing_features
from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_OBSERVED,
    N_AXES,
    TIME_AXIS,
)
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.train.adapt_loop import run_capped_adapt_dsm
from src.train.artifacts import load_checkpoint, save_checkpoint
from src.train.real_pool import RealBatchPool

_MODE_SCRATCH = default_adapt_mode()
_MODE_FINE_TUNE = "fine_tune"
_ALLOWED_MODES = allowed_adapt_modes()
_DEFAULT_BASE_FEATURES = default_beijing_features()
_MASK_TOL = 1e-6
_BEST_CKPT_NAME = "best_score_net.eqx"
_SCALER_NAME = "feature_scaler.npz"
_META_NAME = "adapt_meta.json"
_STATUS_SUCCESS = "success"
_RUN_ID_SUFFIX_LEN = 6
_REPO_ROOT = Path(__file__).resolve().parents[2]
_M2_ARTIFACT_MARKER = "artifacts/m2"

__all__ = [
    "AdaptCaps",
    "AdaptCliOverrides",
    "AdaptGeometry",
    "AdaptInitPolicy",
    "AdaptLoopCliOverrides",
    "AdaptM7Config",
    "AdaptModeSection",
    "AdaptSampleSection",
    "AdaptSessionRequest",
    "AdaptSessionResult",
    "AdaptSessionRunner",
    "AdaptWindowCliOverrides",
    "AdaptWindowSplit",
    "AdaptWindowTensors",
    "apply_adapt_cli_overrides",
    "load_adapt_m7_config",
    "make_adapt_run_id",
    "run_adapt_session",
    "split_adapt_windows",
]


@dataclass(frozen=True)
class AdaptWindowTensors:
    """Upload windows under the M1/M2 tensor / mask contract.

    Public methods
    --------------
    n_windows
        Batch size ``B``.
    validate
        Check ranks, shapes, and feature-name length.

    Instance variables
    ------------------
    values : numpy.ndarray
        Float32 ``(B, T, F)``; missing cells use sentinel ``0.0``.
    native_masks : numpy.ndarray
        Float32 ``(B, T, F)``; ``1`` observed, ``0`` missing / pad.
    feature_names : tuple of str
        Ordered feature labels (length ``F``).
    """

    values: np.ndarray
    native_masks: np.ndarray
    feature_names: tuple[str, ...]

    def n_windows(self) -> int:
        """Return the number of windows ``B``.

        Returns
        -------
        int
            ``values.shape[BATCH_AXIS]``.
        """
        return int(self.values.shape[BATCH_AXIS])

    def validate(self) -> None:
        """Raise ``ValueError`` if tensors or names are inconsistent."""
        values = np.asarray(self.values)
        masks = np.asarray(self.native_masks)
        if values.ndim != N_AXES or masks.ndim != N_AXES:
            raise ValueError(
                "values and native_masks must have rank 3 (B, T, F); "
                f"got {values.ndim}, {masks.ndim}"
            )
        if values.shape != masks.shape:
            raise ValueError(
                "values and native_masks shapes must match; "
                f"got {values.shape} vs {masks.shape}"
            )
        n_features = int(values.shape[FEATURE_AXIS])
        if len(self.feature_names) != n_features:
            raise ValueError(
                "len(feature_names) must equal F; "
                f"got {len(self.feature_names)} vs F={n_features}"
            )


@dataclass(frozen=True)
class AdaptWindowSplit:
    """Deterministic train / smoke split by window index.

    Public methods
    --------------
    n_train
        Number of train windows.
    n_smoke
        Number of smoke-held-out windows.

    Instance variables
    ------------------
    train_idx : numpy.ndarray
        1-D int64 train indices (adapt order).
    smoke_idx : numpy.ndarray
        1-D int64 smoke indices (adapt order).
    """

    train_idx: np.ndarray
    smoke_idx: np.ndarray

    def n_train(self) -> int:
        """Return train window count.

        Returns
        -------
        int
            ``train_idx.size``.
        """
        return int(self.train_idx.size)

    def n_smoke(self) -> int:
        """Return smoke window count.

        Returns
        -------
        int
            ``smoke_idx.size``.
        """
        return int(self.smoke_idx.size)


@dataclass(frozen=True)
class AdaptInitPolicy:
    """Scratch vs fine-tune initialization policy.

    Public methods
    --------------
    validate
        Check mode string and fine-tune path knobs.
    resolved_mode
        Normalized lowercase mode.

    Instance variables
    ------------------
    mode : str
        ``scratch`` (default) or ``fine_tune``.
    base_checkpoint : pathlib.Path or None
        Base ``.eqx`` path when fine-tuning.
    base_feature_names : tuple of str
        Base schema names (default Beijing ``F = 6``).
    require_feature_name_match : bool
        If true, names must match unless confirmed.
    confirm_same_semantics : bool
        User override when names differ but ``F`` matches.
    """

    mode: str = _MODE_SCRATCH
    base_checkpoint: Path | None = None
    base_feature_names: tuple[str, ...] = _DEFAULT_BASE_FEATURES
    require_feature_name_match: bool = True
    confirm_same_semantics: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` if policy knobs are invalid."""
        mode = self.resolved_mode()
        if mode not in _ALLOWED_MODES:
            raise ValueError(
                f"adapt mode must be one of {sorted(_ALLOWED_MODES)}, "
                f"got {self.mode!r}"
            )
        if mode == _MODE_FINE_TUNE and self.base_checkpoint is None:
            raise ValueError("fine_tune requires base_checkpoint to be set")

    def resolved_mode(self) -> str:
        """Return normalized mode string.

        Returns
        -------
        str
            Lowercased mode.
        """
        return str(self.mode).strip().lower()


@dataclass(frozen=True)
class AdaptSessionRequest:
    """All inputs for one upload-adapt session.

    Public methods
    --------------
    validate
        Validate windows, config, and init policy.
    effective_policy
        Merge YAML defaults into the init policy.

    Instance variables
    ------------------
    windows : AdaptWindowTensors
        Native upload windows.
    config : AdaptM7Config
        Smoke caps and paths.
    policy : AdaptInitPolicy
        Scratch / fine-tune policy (may be partially filled).
    run_id : str or None
        Optional fixed run id (tests); else auto-generated.
    artifact_root : pathlib.Path or None
        Optional override for ``config.caps.artifact_dir`` (tests).
    """

    windows: AdaptWindowTensors
    config: AdaptM7Config
    policy: AdaptInitPolicy
    run_id: str | None = None
    artifact_root: Path | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` if the request cannot start adapt."""
        self.windows.validate()
        self.config.validate()
        self.effective_policy().validate()
        n_windows = self.windows.n_windows()
        if n_windows < int(self.config.caps.min_windows):
            raise ValueError(
                f"too few windows for adapt: n_windows={n_windows} "
                f"< min_windows={self.config.caps.min_windows}"
            )

    def effective_policy(self) -> AdaptInitPolicy:
        """Fill mode / base path from config defaults.

        Returns
        -------
        AdaptInitPolicy
            Policy used for model init.
        """
        mode = self.policy.resolved_mode()
        if not mode:
            mode = str(self.config.mode.default_mode).strip().lower()
        base = self.policy.base_checkpoint
        if base is None and mode == _MODE_FINE_TUNE:
            base = _resolve_repo_path(self.config.mode.base_checkpoint)
        return AdaptInitPolicy(
            mode=mode,
            base_checkpoint=base,
            base_feature_names=self.policy.base_feature_names,
            require_feature_name_match=(
                self.config.mode.require_feature_name_match_for_fine_tune
            ),
            confirm_same_semantics=self.policy.confirm_same_semantics,
        )


@dataclass(frozen=True)
class AdaptSessionResult:
    """Successful adapt outputs under ``artifacts/m7/<run_id>/``.

    Public methods
    --------------
    checkpoint_path
        Path to ``best_score_net.eqx``.
    scaler_path
        Path to ``feature_scaler.npz``.

    Instance variables
    ------------------
    run_id : str
        Opaque run directory name.
    run_dir : pathlib.Path
        Absolute run directory.
    model : ConditionalScoreNet
        Best-val score net from the capped run.
    scaler : FeatureScaler
        Train-only fitted scaler.
    split : AdaptWindowSplit
        Train / smoke indices used.
    meta : dict of str to Any
        Contents written to ``adapt_meta.json``.
    """

    run_id: str
    run_dir: Path
    model: ConditionalScoreNet
    scaler: FeatureScaler
    split: AdaptWindowSplit
    meta: dict[str, Any]

    def checkpoint_path(self) -> Path:
        """Return the session checkpoint path.

        Returns
        -------
        pathlib.Path
            ``run_dir / best_score_net.eqx``.
        """
        return self.run_dir / _BEST_CKPT_NAME

    def scaler_path(self) -> Path:
        """Return the session scaler path.

        Returns
        -------
        pathlib.Path
            ``run_dir / feature_scaler.npz``.
        """
        return self.run_dir / _SCALER_NAME


class AdaptSessionRunner:
    """Run train-only scaler fit + capped DSM adapt for one upload.

    Public methods
    --------------
    run
        Full adapt → session artifacts (or abort with no promote).
    build_scaled_pool
        Fit scaler on train; return normalized ``RealBatchPool``.
    """

    def run(self, request: AdaptSessionRequest) -> AdaptSessionResult:
        """Execute adapt and promote artifacts only on clean success.

        Parameters
        ----------
        request : AdaptSessionRequest
            Windows, config, and init policy.

        Returns
        -------
        AdaptSessionResult
            Paths, model, scaler, and metadata.

        Raises
        ------
        ValueError
            Too little data, invalid tensors, or fine-tune refused.
        RuntimeError
            Non-finite loss or wall-clock timeout.
        """
        request.validate()
        split = split_adapt_windows(
            request.windows.n_windows(),
            request.config.caps.train_fraction,
        )
        scaler, pool = self.build_scaled_pool(request.windows, split)
        policy = request.effective_policy()
        key_model, key_train = jax.random.split(
            jax.random.PRNGKey(request.config.train.protocol.seed)
        )
        model = _build_adapt_model(
            key_model,
            request.windows,
            policy,
            request.config.mode.allow_fine_tune_when_f_matches,
        )
        best_model = run_capped_adapt_dsm(
            model,
            pool,
            request.config,
            key_train,
        )
        run_id = request.run_id or make_adapt_run_id()
        run_dir = _resolve_run_dir(request, run_id)
        meta = _build_success_meta(request, policy, split, run_id)
        _promote_session_artifacts(run_dir, best_model, scaler, meta)
        return AdaptSessionResult(
            run_id=run_id,
            run_dir=run_dir,
            model=best_model,
            scaler=scaler,
            split=split,
            meta=meta,
        )

    def build_scaled_pool(
            self,
            windows: AdaptWindowTensors,
            split: AdaptWindowSplit,
    ) -> tuple[FeatureScaler, RealBatchPool]:
        """Fit train-only scaler and build normalized train/smoke pools.

        Parameters
        ----------
        windows : AdaptWindowTensors
            Native upload windows.
        split : AdaptWindowSplit
            Train / smoke indices.

        Returns
        -------
        scaler : FeatureScaler
            Fitted on train native-observed cells only.
        pool : RealBatchPool
            Normalized train (DSM) and smoke (val) pools.
        """
        values = np.asarray(windows.values, dtype=np.float32)
        masks = np.asarray(windows.native_masks, dtype=np.float32)
        scaler = FeatureScaler().fit(
            values[split.train_idx],
            masks[split.train_idx],
        )
        values_n = np.asarray(
            scaler.transform(values, masks),
            dtype=np.float32,
        )
        _assert_observed_finite(values_n, masks)
        pool = RealBatchPool(
            train_values=jnp.asarray(
                values_n[split.train_idx], dtype=jnp.float32
            ),
            train_masks=jnp.asarray(
                masks[split.train_idx], dtype=jnp.float32
            ),
            val_values=jnp.asarray(
                values_n[split.smoke_idx], dtype=jnp.float32
            ),
            val_masks=jnp.asarray(
                masks[split.smoke_idx], dtype=jnp.float32
            ),
        )
        return scaler, pool


def run_adapt_session(request: AdaptSessionRequest) -> AdaptSessionResult:
    """Run one adapt session (library entrypoint).

    Parameters
    ----------
    request : AdaptSessionRequest
        Bundled adapt inputs.

    Returns
    -------
    AdaptSessionResult
        Successful session artifacts and in-memory model / scaler.
    """
    return AdaptSessionRunner().run(request)


def split_adapt_windows(
        n_windows: int,
        train_fraction: float,
) -> AdaptWindowSplit:
    """Split window indices into train / smoke by adapt order.

    Parameters
    ----------
    n_windows : int
        Total windows ``B`` (must be ``>= 2``).
    train_fraction : float
        Fraction assigned to train (ASSUMPTIONS default ``0.80``).

    Returns
    -------
    AdaptWindowSplit
        Contiguous index ranges ``[0, n_train)`` / ``[n_train, B)``.
    """
    if n_windows < 2:
        raise ValueError(
            f"n_windows must be >= 2 for adapt split, got {n_windows}"
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be in (0, 1), got {train_fraction}"
        )
    n_train = int(n_windows * train_fraction)
    n_train = max(1, min(n_train, n_windows - 1))
    return AdaptWindowSplit(
        train_idx=np.arange(n_train, dtype=np.int64),
        smoke_idx=np.arange(n_train, n_windows, dtype=np.int64),
    )


def make_adapt_run_id() -> str:
    """Build an opaque UTC run id with a short random suffix.

    Returns
    -------
    str
        e.g. ``20260322T101500Z_a1b2c3``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:_RUN_ID_SUFFIX_LEN]
    return f"{stamp}_{suffix}"


def _build_adapt_model(
        key: jax.Array,
        windows: AdaptWindowTensors,
        policy: AdaptInitPolicy,
        allow_fine_tune: bool,
) -> ConditionalScoreNet:
    """Init scratch net or load base weights when policy allows.

    Parameters
    ----------
    key : jax.Array
        PRNG key for scratch init.
    windows : AdaptWindowTensors
        Upload windows (defines ``F`` and names).
    policy : AdaptInitPolicy
        Resolved init policy.
    allow_fine_tune : bool
        Config gate for fine-tune path.

    Returns
    -------
    ConditionalScoreNet
        Model ready for capped DSM.
    """
    n_features = int(windows.values.shape[FEATURE_AXIS])
    template = ConditionalScoreNet(
        key,
        default_score_net_config(n_features),
    )
    if policy.resolved_mode() != _MODE_FINE_TUNE:
        return template
    _assert_fine_tune_allowed(windows, policy, allow_fine_tune, n_features)
    base_checkpoint = policy.base_checkpoint
    if base_checkpoint is None:
        raise ValueError("fine_tune requires base_checkpoint to be set")
    base_path = Path(base_checkpoint)
    if not base_path.is_file():
        raise ValueError(f"base checkpoint not found: {base_path}")
    return load_checkpoint(base_path, like=template)


def _assert_fine_tune_allowed(
        windows: AdaptWindowTensors,
        policy: AdaptInitPolicy,
        allow_fine_tune: bool,
        n_features: int,
) -> None:
    """Raise ``ValueError`` when fine-tune is refused by M7 policy.

    Parameters
    ----------
    windows : AdaptWindowTensors
        Upload windows.
    policy : AdaptInitPolicy
        Resolved init policy.
    allow_fine_tune : bool
        Config gate for fine-tune path.
    n_features : int
        Upload feature count ``F``.
    """
    if not allow_fine_tune:
        raise ValueError(
            "fine_tune disabled by adapt config "
            "(allow_fine_tune_when_f_matches=false); use mode='scratch'"
        )
    base_names = tuple(policy.base_feature_names)
    base_f = len(base_names)
    if n_features != base_f:
        raise ValueError(
            "Fine-tune refused: upload F="
            f"{n_features} does not match base F={base_f}. "
            "ConditionalScoreNet.n_features is static; use mode='scratch'."
        )
    names_match = tuple(windows.feature_names) == base_names
    if (
            policy.require_feature_name_match
            and not names_match
            and not policy.confirm_same_semantics
    ):
        raise ValueError(
            "Fine-tune refused: feature_names do not match the base "
            "schema. Confirm same semantics or use mode='scratch'. "
            "Silent remap to Beijing names is banned."
        )


def _promote_session_artifacts(
        run_dir: Path,
        model: ConditionalScoreNet,
        scaler: FeatureScaler,
        meta: Mapping[str, Any],
) -> None:
    """Write checkpoint, scaler, and meta only after a successful run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Session run directory.
    model : ConditionalScoreNet
        Best-val score network.
    scaler : FeatureScaler
        Train-only fitted scaler.
    meta : Mapping[str, Any]
        Metadata payload for ``adapt_meta.json``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = str(run_dir.resolve()).replace("\\", "/")
    if _M2_ARTIFACT_MARKER in resolved:
        raise RuntimeError(
            "refusing to promote adapt artifacts under artifacts/m2"
        )
    save_checkpoint(run_dir / _BEST_CKPT_NAME, model)
    scaler.save(run_dir / _SCALER_NAME)
    (run_dir / _META_NAME).write_text(
        json.dumps(dict(meta), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_success_meta(
        request: AdaptSessionRequest,
        policy: AdaptInitPolicy,
        split: AdaptWindowSplit,
        run_id: str,
) -> dict[str, Any]:
    """Build ``adapt_meta.json`` payload for a successful run.

    Parameters
    ----------
    request : AdaptSessionRequest
        Adapt request.
    policy : AdaptInitPolicy
        Effective init policy.
    split : AdaptWindowSplit
        Train / smoke indices.
    run_id : str
        Opaque run id.

    Returns
    -------
    dict of str to Any
        Metadata written beside the session checkpoint.
    """
    values = request.windows.values
    return {
        "status": _STATUS_SUCCESS,
        "run_id": run_id,
        "mode": policy.resolved_mode(),
        "feature_names": list(request.windows.feature_names),
        "F": int(values.shape[FEATURE_AXIS]),
        "T": int(values.shape[TIME_AXIS]),
        "n_windows": int(values.shape[BATCH_AXIS]),
        "n_train_windows": split.n_train(),
        "n_smoke_windows": split.n_smoke(),
        "train_fraction": float(request.config.caps.train_fraction),
        "seed": int(request.config.train.protocol.seed),
        "epochs": int(request.config.train.schedule.epochs),
        "steps_per_epoch": int(request.config.train.schedule.steps_per_epoch),
        "batch_size": int(request.config.geometry.batch_size),
        "timeout_seconds": float(request.config.caps.timeout_seconds),
        "claim_discipline": (
            "file-local adapt smoke; not a Beijing/M3 claim expansion"
        ),
    }


def _resolve_run_dir(request: AdaptSessionRequest, run_id: str) -> Path:
    """Resolve ``<artifact_root>/<run_id>``.

    Parameters
    ----------
    request : AdaptSessionRequest
        Adapt request (optional artifact root override).
    run_id : str
        Opaque run id.

    Returns
    -------
    pathlib.Path
        Absolute run directory path.
    """
    if request.artifact_root is not None:
        root = Path(request.artifact_root)
    else:
        root = _resolve_repo_path(request.config.caps.artifact_dir)
    return root / run_id


def _resolve_repo_path(path_str: str) -> Path:
    """Resolve a path against the repo root when relative.

    Parameters
    ----------
    path_str : str
        Absolute or repo-relative path string.

    Returns
    -------
    pathlib.Path
        Absolute path.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def _assert_observed_finite(values: np.ndarray, masks: np.ndarray) -> None:
    """Raise if any natively observed cell is non-finite.

    Parameters
    ----------
    values : numpy.ndarray
        Window values (typically normalized).
    masks : numpy.ndarray
        Native masks.
    """
    observed = np.abs(masks - MASK_OBSERVED) < _MASK_TOL
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError(
            "non-finite values found on natively observed cells"
        )
