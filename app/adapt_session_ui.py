"""Hermetic M7 adapt helpers for Streamlit (no Streamlit import).

Classes
-------
AdaptLoopCapsView
    Train / sample smoke caps for UI captions.
AdaptCapsDisplay
    Nested smoke caps / timeout / ETA note for the adapt panel.
AdaptRunRequest
    Windows + config + mode for one UI adapt invocation.
AdaptFailureInfo
    User-facing failure kind and message (no stack-trace-only dead end).
BoundAdaptSession
    Successful session imprint: windows, imputer, run paths, fingerprint.
AdaptUiOutcome
    Either a bound session or a failure info object.

Functions
---------
make_upload_fingerprint
    Stable key for session invalidation when the upload changes.
caps_display_from_config
    Build ``AdaptCapsDisplay`` from ``AdaptM7Config``.
prepare_adapt_windows
    Re-adapt Flexible frame at an effective ``seq_len`` (smoke override).
run_ui_adapt
    Call ``run_adapt_session`` and map errors to ``AdaptUiOutcome``.
imputer_from_adapt_result
    Bind ``SessionImputer`` to promoted session assets.
format_adapt_failure
    Map ``ValueError`` / ``RuntimeError`` to user copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.m4_config import AppM4SampleKnobs
from app.preview_ingest import FlexiblePreviewKnobs, try_adapt_flexible_windows
from app.schema_validate import ValidatedFrame
from app.session_impute import SessionAssetRequest, SessionImputer
from app.window_adapter import AdaptedWindows
from src.data.adapt_config import AdaptM7Config
from src.train.adapt_m7 import (
    AdaptInitPolicy,
    AdaptSessionRequest,
    AdaptSessionResult,
    AdaptWindowTensors,
    run_adapt_session,
)

_MODE_SCRATCH = "scratch"
_MODE_FINE_TUNE = "fine_tune"
_ALLOWED_MODES = frozenset({_MODE_SCRATCH, _MODE_FINE_TUNE})
_KIND_TOO_LITTLE = "too_little_data"
_KIND_NAN = "nan_loss"
_KIND_TIMEOUT = "timeout"
_KIND_POLICY = "policy"
_KIND_UNEXPECTED = "unexpected"
_NAN_TOKEN = "non-finite"
_TIMEOUT_TOKEN = "timeout"
_SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class AdaptLoopCapsView:
    """Train / sample smoke caps for UI captions.

    Public methods
    --------------
    as_lines
        Short train/sample lines.
    describe
        One-line ``epochs×steps`` summary.

    Instance variables
    ------------------
    epochs : int
        Capped train epochs.
    steps_per_epoch : int
        Steps per epoch.
    batch_size : int
        DSM batch size.
    k_samples : int
        Post-adapt ``K``.
    solver_steps : int
        Reverse-SDE steps.
    """

    epochs: int
    steps_per_epoch: int
    batch_size: int
    k_samples: int
    solver_steps: int

    def as_lines(self) -> tuple[str, ...]:
        """Return train/sample cap lines.

        Returns
        -------
        tuple of str
            Human-readable lines.
        """
        return (
            f"epochs={self.epochs}",
            f"steps/epoch={self.steps_per_epoch}",
            f"batch_size={self.batch_size}",
            f"K={self.k_samples}",
            f"solver_steps={self.solver_steps}",
        )

    def describe(self) -> str:
        """Return a one-line train/sample summary.

        Returns
        -------
        str
            Compact cap string.
        """
        return (
            f"epochs={self.epochs} steps/epoch={self.steps_per_epoch} "
            f"batch={self.batch_size} K={self.k_samples} "
            f"steps={self.solver_steps}"
        )


@dataclass(frozen=True)
class AdaptCapsDisplay:
    """Visible smoke caps for the adapt panel.

    Public methods
    --------------
    as_lines
        Bullet-style caption lines for the UI.
    eta_note
        Soft ETA / timeout note (not a hard progress bar).

    Instance variables
    ------------------
    loop : AdaptLoopCapsView
        Epochs / steps / batch / ``K`` / solver steps.
    timeout_seconds : float
        Wall-clock abort budget.
    min_windows : int
        Minimum ``B`` before adapt starts.
    seq_len : int
        Effective window length ``T``.
    mode : str
        ``scratch`` or ``fine_tune``.
    """

    loop: AdaptLoopCapsView
    timeout_seconds: float
    min_windows: int
    seq_len: int
    mode: str

    def as_lines(self) -> tuple[str, ...]:
        """Return short cap lines for captions / expanders.

        Returns
        -------
        tuple of str
            Human-readable cap summaries.
        """
        return (
            f"mode={self.mode}",
            *self.loop.as_lines(),
            f"seq_len={self.seq_len}",
            f"timeout={self.timeout_seconds:.0f}s",
            f"min_windows={self.min_windows}",
        )

    def eta_note(self) -> str:
        """Return a soft ETA note anchored on the timeout budget.

        Returns
        -------
        str
            User-facing timing expectation.
        """
        minutes = self.timeout_seconds / _SECONDS_PER_MINUTE
        return (
            "Smoke-capped adapt on CPU often finishes in a few minutes; "
            f"hard stop at {minutes:.0f} min (timeout). "
            "This is not a full M2 train and not a universal SOTA claim."
        )


@dataclass(frozen=True)
class AdaptRunRequest:
    """Inputs for one UI adapt invocation.

    Public methods
    --------------
    validate
        Check windows and mode basics.
    to_window_tensors
        Convert ``AdaptedWindows`` to ``AdaptWindowTensors``.

    Instance variables
    ------------------
    windows : AdaptedWindows
        Windows used for DSM (may use smoke ``seq_len``).
    config : AdaptM7Config
        Effective caps (after optional overrides).
    mode : str
        ``scratch`` (WP3 default) or ``fine_tune``.
    fingerprint : str
        Upload fingerprint for session binding.
    artifact_root : pathlib.Path or None
        Optional override (tests); default YAML ``artifacts/m7``.
    """

    windows: AdaptedWindows
    config: AdaptM7Config
    mode: str
    fingerprint: str
    artifact_root: Path | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` if the request cannot start."""
        if self.windows.n_windows() <= 0:
            raise ValueError("adapt requires at least one window")
        mode = str(self.mode).strip().lower()
        if mode not in _ALLOWED_MODES:
            raise ValueError(f"unknown adapt mode: {self.mode!r}")

    def to_window_tensors(self) -> AdaptWindowTensors:
        """Convert adapted windows into adapt tensors.

        Returns
        -------
        AdaptWindowTensors
            Values / native masks / names.
        """
        return AdaptWindowTensors(
            values=np.asarray(self.windows.values, dtype=np.float32),
            native_masks=np.asarray(
                self.windows.native_mask, dtype=np.float32
            ),
            feature_names=tuple(self.windows.feature_names),
        )


@dataclass(frozen=True)
class AdaptFailureInfo:
    """User-facing adapt failure (no promote / no gate lift).

    Public methods
    --------------
    is_timeout
        Whether this failure is a wall-clock timeout.
    as_markdown
        Short markdown block for Streamlit.

    Instance variables
    ------------------
    kind : str
        ``too_little_data``, ``nan_loss``, ``timeout``, ``policy``, or
        ``unexpected``.
    message : str
        User-facing explanation.
    """

    kind: str
    message: str

    def is_timeout(self) -> bool:
        """Return whether the failure is a timeout abort.

        Returns
        -------
        bool
            ``True`` when ``kind`` is ``timeout``.
        """
        return self.kind == _KIND_TIMEOUT

    def as_markdown(self) -> str:
        """Return markdown-friendly failure copy.

        Returns
        -------
        str
            Title + message.
        """
        titles = {
            _KIND_TOO_LITTLE: "Too little data for adapt",
            _KIND_NAN: "Adapt stopped (non-finite loss)",
            _KIND_TIMEOUT: "Adapt timed out",
            _KIND_POLICY: "Adapt policy refused this run",
            _KIND_UNEXPECTED: "Adapt failed",
        }
        title = titles.get(self.kind, titles[_KIND_UNEXPECTED])
        return f"**{title}**\n\n{self.message}"


@dataclass(frozen=True)
class BoundAdaptSession:
    """Successful adapt bound to one upload fingerprint.

    Public methods
    --------------
    matches
        Whether ``fingerprint`` still matches this bind.
    describe
        Short success summary for the UI.

    Instance variables
    ------------------
    fingerprint : str
        Upload fingerprint at bind time.
    windows : AdaptedWindows
        Windows used for adapt / post-adapt impute.
    imputer : SessionImputer
        Session-bound reverse-SDE imputer.
    run_id : str
        Opaque run id.
    run_dir : pathlib.Path
        ``artifacts/m7/<run_id>/`` (or override root).
    feature_names : tuple of str
        Session schema names (gate unlock key).
    """

    fingerprint: str
    windows: AdaptedWindows
    imputer: SessionImputer
    run_id: str
    run_dir: Path
    feature_names: tuple[str, ...]

    def matches(self, fingerprint: str) -> bool:
        """Return whether this bind still applies.

        Parameters
        ----------
        fingerprint : str
            Current upload fingerprint.

        Returns
        -------
        bool
            ``True`` on exact string match.
        """
        return self.fingerprint == fingerprint

    def describe(self) -> str:
        """Return a short success summary.

        Returns
        -------
        str
            Run id / ``F`` / window count note.
        """
        n_features = len(self.feature_names)
        return (
            f"Session adapt ready (run_id={self.run_id}, "
            f"F={n_features}, B={self.windows.n_windows()}). "
            f"Artifacts: {self.run_dir}. "
            "File-local only — not a Beijing/M3 claim."
        )


@dataclass(frozen=True)
class AdaptUiOutcome:
    """Result of a UI adapt attempt.

    Public methods
    --------------
    ok
        Whether adapt promoted a session.
    failure_markdown
        Failure markdown, or empty when ok.

    Instance variables
    ------------------
    session : BoundAdaptSession or None
        Bound session on success.
    failure : AdaptFailureInfo or None
        Failure info on abort / reject.
    """

    session: BoundAdaptSession | None
    failure: AdaptFailureInfo | None

    def ok(self) -> bool:
        """Return whether adapt succeeded.

        Returns
        -------
        bool
            ``True`` when ``session`` is set.
        """
        return self.session is not None

    def failure_markdown(self) -> str:
        """Return failure markdown or empty string.

        Returns
        -------
        str
            User-facing failure block.
        """
        if self.failure is None:
            return ""
        return self.failure.as_markdown()


def make_upload_fingerprint(
        source_label: str,
        ingest_mode: str,
        feature_names: tuple[str, ...],
) -> str:
    """Build a stable fingerprint for session invalidation.

    Parameters
    ----------
    source_label : str
        Bundled label or upload filename.
    ingest_mode : str
        ``exact_beijing`` or ``flexible``.
    feature_names : tuple of str
        Mapped feature names.

    Returns
    -------
    str
        Opaque fingerprint string.
    """
    joined = "|".join(feature_names)
    return f"{source_label}::{ingest_mode}::{joined}"


def caps_display_from_config(
        config: AdaptM7Config,
        mode: str,
) -> AdaptCapsDisplay:
    """Build visible caps from an effective adapt config.

    Parameters
    ----------
    config : AdaptM7Config
        Caps after optional UI overrides.
    mode : str
        Active adapt mode.

    Returns
    -------
    AdaptCapsDisplay
        Caption payload.
    """
    resolved = str(mode).strip().lower() or _MODE_SCRATCH
    return AdaptCapsDisplay(
        loop=AdaptLoopCapsView(
            epochs=int(config.train.schedule.epochs),
            steps_per_epoch=int(config.train.schedule.steps_per_epoch),
            batch_size=int(config.geometry.batch_size),
            k_samples=int(config.sample.k_samples),
            solver_steps=int(config.sample.solver_steps),
        ),
        timeout_seconds=float(config.caps.timeout_seconds),
        min_windows=int(config.caps.min_windows),
        seq_len=int(config.geometry.seq_len),
        mode=resolved,
    )


def prepare_adapt_windows(
        frame: ValidatedFrame,
        seq_len: int,
        max_windows: int | None,
) -> tuple[AdaptedWindows | None, str | None]:
    """Adapt a Flexible (or any) frame at the effective smoke ``seq_len``.

    Parameters
    ----------
    frame : ValidatedFrame
        Current upload frame.
    seq_len : int
        Window length (may be a UI smoke override).
    max_windows : int or None
        Optional ``B`` cap.

    Returns
    -------
    windows : AdaptedWindows or None
        Adapted batch on success.
    error : str or None
        User-facing reject text.
    """
    knobs = FlexiblePreviewKnobs(
        seq_len=int(seq_len),
        max_windows=max_windows,
    )
    return try_adapt_flexible_windows(frame, knobs)


def imputer_from_adapt_result(
        session: AdaptSessionResult,
        config: AdaptM7Config,
) -> SessionImputer:
    """Build a session imputer from a promoted adapt result.

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


def format_adapt_failure(exc: BaseException) -> AdaptFailureInfo:
    """Map adapt exceptions to user-facing failure info.

    Parameters
    ----------
    exc : BaseException
        Raised error from prepare / ``run_adapt_session``.

    Returns
    -------
    AdaptFailureInfo
        Kind + message (no traceback required for the UI path).
    """
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    kind = _KIND_UNEXPECTED
    text = message
    if isinstance(exc, ValueError):
        kind, text = _value_error_failure(message, lowered)
    elif isinstance(exc, RuntimeError):
        kind, text = _runtime_error_failure(message, lowered)
    elif isinstance(exc, MemoryError):
        text = (
            "Out of memory during adapt. Lower batch size / windows / "
            "K, or use the CLI with smaller caps. No checkpoint promoted."
        )
    return AdaptFailureInfo(kind=kind, message=text)


def _value_error_failure(
        message: str,
        lowered: str,
) -> tuple[str, str]:
    """Classify a ``ValueError`` into failure kind + user copy.

    Parameters
    ----------
    message : str
        Original exception text.
    lowered : str
        Lowercased ``message``.

    Returns
    -------
    kind : str
        ``too_little_data`` or ``policy``.
    text : str
        User-facing message.
    """
    if "too few windows" in lowered or "too short" in lowered:
        return (
            _KIND_TOO_LITTLE,
            (
                f"{message} Upload more rows, lower seq_len in Advanced "
                "smoke knobs, or fix missingness so M6 rejects pass."
            ),
        )
    return _KIND_POLICY, message


def _runtime_error_failure(
        message: str,
        lowered: str,
) -> tuple[str, str]:
    """Classify a ``RuntimeError`` into failure kind + user copy.

    Parameters
    ----------
    message : str
        Original exception text.
    lowered : str
        Lowercased ``message``.

    Returns
    -------
    kind : str
        ``nan_loss``, ``timeout``, or ``unexpected``.
    text : str
        User-facing message.
    """
    if _NAN_TOKEN in lowered:
        return (
            _KIND_NAN,
            (
                f"{message} No session checkpoint was promoted. "
                "Try scratch mode, fewer steps, or a denser series."
            ),
        )
    if _TIMEOUT_TOKEN in lowered:
        return (
            _KIND_TIMEOUT,
            (
                f"{message} Raise timeout only via config for longer "
                "CLI runs; UI defaults stay smoke-capped."
            ),
        )
    return _KIND_UNEXPECTED, message


def run_ui_adapt(request: AdaptRunRequest) -> AdaptUiOutcome:
    """Run capped adapt and bind a session imputer on success.

    Parameters
    ----------
    request : AdaptRunRequest
        Windows, config, mode, fingerprint.

    Returns
    -------
    AdaptUiOutcome
        Bound session or user-facing failure (never half-binds).
    """
    try:
        return _run_ui_adapt_checked(request)
    except (ValueError, RuntimeError, MemoryError, OSError) as exc:
        return AdaptUiOutcome(
            session=None,
            failure=format_adapt_failure(exc),
        )


def _run_ui_adapt_checked(request: AdaptRunRequest) -> AdaptUiOutcome:
    """Validate and run adapt; raise typed errors for ``run_ui_adapt``.

    Parameters
    ----------
    request : AdaptRunRequest
        Windows, config, mode, fingerprint.

    Returns
    -------
    AdaptUiOutcome
        Bound session on success.

    Raises
    ------
    ValueError
        Too little data or invalid request.
    RuntimeError
        NaN loss or timeout from the adapt loop.
    """
    request.validate()
    min_windows = int(request.config.caps.min_windows)
    n_windows = request.windows.n_windows()
    if n_windows < min_windows:
        raise ValueError(
            f"too few windows for adapt: n_windows={n_windows} "
            f"< min_windows={min_windows}"
        )
    result = run_adapt_session(
        AdaptSessionRequest(
            windows=request.to_window_tensors(),
            config=request.config,
            policy=AdaptInitPolicy(mode=request.mode),
            artifact_root=request.artifact_root,
        )
    )
    imputer = imputer_from_adapt_result(result, request.config)
    bound = BoundAdaptSession(
        fingerprint=request.fingerprint,
        windows=request.windows,
        imputer=imputer,
        run_id=result.run_id,
        run_dir=result.run_dir,
        feature_names=tuple(request.windows.feature_names),
    )
    return AdaptUiOutcome(session=bound, failure=None)
