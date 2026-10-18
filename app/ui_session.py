"""Non-Streamlit session types and CSV → preview payload assembly (M4/M6/M7).

Keeps ``streamlit_app`` under the module line limit by owning sidebar
control dataclasses, upload/bundled payload builders, and impute-section
context. No Streamlit imports.

Classes
-------
UiDemoMaskControls
    Demo MCAR / block sidebar knobs.
UiSolverControls
    ``K`` / steps / quantile sidebar knobs.
UiSidebarState
    Nested sidebar state collected once per rerun.
PayloadIdentity
    Source label, ingest mode, and checkpoint feature names.
AdaptOutcome
    Windows from Exact/Flexible adapt, or a reject message.
UiSessionPayload
    Validated frame, optional windows, preview, and impute gate.
ImputeSectionContext
    Bundle for gated impute UI rendering (pretrained or M7 session).

Functions
---------
build_bundled_payload
    Exact Beijing sample CSV → ``UiSessionPayload``.
build_upload_payload_from_bytes
    Upload bytes → Exact / Flexible preview payload.
refresh_impute_gate
    Re-evaluate gate with optional M7 session feature unlock.
write_temp_csv
    Persist upload bytes to a temp path (caller unlinks).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd  # type: ignore  # pylint: disable=import-error

from app.adapt_session_ui import BoundAdaptSession
from app.flexible_ingest import ingest_flexible_csv
from app.m4_config import AppM4Config
from app.preview_ingest import (
    MODE_EXACT_BEIJING,
    FlexiblePreviewKnobs,
    ImputeGateResult,
    IngestPreviewSummary,
    PreviewBuildInputs,
    build_ingest_preview,
    evaluate_impute_gate,
    read_csv_headers,
    select_ingest_mode,
    try_adapt_flexible_windows,
)
from app.schema_validate import (
    ValidatedFrame,
    default_app_schema,
    summarize_missingness,
    validate_upload_csv,
)
from app.session_impute import (
    DemoMaskRequest,
    SessionImputeRequest,
    SessionImputer,
)
from app.window_adapter import AdaptedWindows, adapt_validated_frame

_MASK_NONE = "none"
_SOURCE_BUNDLED = "Try the sample dataset"


@dataclass(frozen=True)
class UiDemoMaskControls:
    """Demo corruption sidebar knobs.

    Public methods
    --------------
    as_request
        Build a ``DemoMaskRequest`` with a seed.
    is_active
        True when mode is not ``none``.

    Instance variables
    ------------------
    mode : str
        ``none``, ``random``, or ``block``.
    rate : float
        MCAR rate when mode is ``random``.
    block_len : int
        Block length when mode is ``block``.
    """

    mode: str
    rate: float
    block_len: int

    def as_request(self, seed: int) -> DemoMaskRequest:
        """Build the demo corruption request.

        Parameters
        ----------
        seed : int
            ``seeds.mask`` from the app config.

        Returns
        -------
        DemoMaskRequest
            Sidebar-composed mask knobs.
        """
        return DemoMaskRequest(
            mode=self.mode,
            rate=self.rate,
            block_len=self.block_len,
            seed=seed,
        )

    def is_active(self) -> bool:
        """Return whether artificial demo masking is enabled.

        Returns
        -------
        bool
            ``True`` when mode is not ``none``.
        """
        return self.mode != _MASK_NONE


@dataclass(frozen=True)
class UiSolverControls:
    """Sampler / predictive-band sidebar knobs.

    Public methods
    --------------
    describe
        Short ``K`` / steps summary.
    apply_to_request
        Build a ``SessionImputeRequest`` for adapted windows.

    Instance variables
    ------------------
    k_samples : int
        Posterior sample count ``K``.
    solver_steps : int
        Reverse-SDE solver steps.
    quantile_low : float
        Lower predictive quantile.
    quantile_high : float
        Upper predictive quantile.
    """

    k_samples: int
    solver_steps: int
    quantile_low: float
    quantile_high: float

    def describe(self) -> str:
        """Return a short knob summary.

        Returns
        -------
        str
            ``K=... steps=...``.
        """
        return f"K={self.k_samples} steps={self.solver_steps}"

    def apply_to_request(
            self,
            windows: AdaptedWindows,
            demo_mask: DemoMaskRequest,
    ) -> SessionImputeRequest:
        """Build the WP2 impute request.

        Parameters
        ----------
        windows : AdaptedWindows
            Adapted upload / sample windows.
        demo_mask : DemoMaskRequest
            Composed demo corruption knobs.

        Returns
        -------
        SessionImputeRequest
            Ready for ``SessionImputer.impute``.
        """
        return SessionImputeRequest(
            windows=windows,
            demo_mask=demo_mask,
            k_samples=self.k_samples,
            solver_steps=self.solver_steps,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
        )


@dataclass(frozen=True)
class UiSidebarState:
    """Nested sidebar controls for one Streamlit rerun.

    Public methods
    --------------
    impute_request
        Build a ``SessionImputeRequest`` for adapted windows.
    wants_sample_overlay
        Whether to overlay posterior draws on the impute chart.

    Instance variables
    ------------------
    demo_mask : UiDemoMaskControls
        Demo corruption knobs.
    solver : UiSolverControls
        Sampler / quantile knobs.
    overlay_samples : bool
        Whether to overlay posterior draws on the impute chart.
    """

    demo_mask: UiDemoMaskControls
    solver: UiSolverControls
    overlay_samples: bool

    def impute_request(
            self,
            windows: AdaptedWindows,
            seed: int,
    ) -> SessionImputeRequest:
        """Build the WP2 impute request.

        Parameters
        ----------
        windows : AdaptedWindows
            Adapted upload / sample windows.
        seed : int
            Mask seed from config.

        Returns
        -------
        SessionImputeRequest
            Ready for ``SessionImputer.impute``.
        """
        return self.solver.apply_to_request(
            windows,
            self.demo_mask.as_request(seed),
        )

    def wants_sample_overlay(self) -> bool:
        """Return whether posterior sample overlay is enabled.

        Returns
        -------
        bool
            ``overlay_samples``.
        """
        return self.overlay_samples


@dataclass(frozen=True)
class PayloadIdentity:
    """Source label, ingest mode, and checkpoint feature names.

    Public methods
    --------------
    source_label
        Display label for the loaded CSV / sample.
    ingest_mode
        ``exact_beijing`` or ``flexible``.

    Instance variables
    ------------------
    label : str
        Sample-dataset label or upload filename.
    mode : str
        Ingest mode string.
    checkpoint_features : tuple of str
        Loaded app / checkpoint feature names for the impute gate.
    """

    label: str
    mode: str
    checkpoint_features: tuple[str, ...]

    def source_label(self) -> str:
        """Return the UI source label.

        Returns
        -------
        str
            ``label``.
        """
        return self.label

    def ingest_mode(self) -> str:
        """Return the ingest mode string.

        Returns
        -------
        str
            ``mode``.
        """
        return self.mode


@dataclass(frozen=True)
class AdaptOutcome:
    """Windows from Exact/Flexible adapt, or a reject message.

    Public methods
    --------------
    has_windows
        Whether adaptation produced windows.
    error_text
        Reject message when windows are absent.

    Instance variables
    ------------------
    windows : AdaptedWindows or None
        Adapted batch when successful.
    adapt_error : str or None
        User-facing reject text when adapt failed.
    """

    windows: AdaptedWindows | None
    adapt_error: str | None

    def has_windows(self) -> bool:
        """Return whether adapted windows are available.

        Returns
        -------
        bool
            ``True`` when ``windows`` is not ``None``.
        """
        return self.windows is not None

    def error_text(self) -> str | None:
        """Return the adapt reject message, if any.

        Returns
        -------
        str or None
            ``adapt_error``.
        """
        return self.adapt_error


@dataclass(frozen=True)
class UiSessionPayload:
    """Loaded CSV + preview / gate state for the current selection.

    Public methods
    --------------
    n_windows
        Adapted batch size ``B`` (0 when adapt rejected).
    missingness_table
        Per-feature percent-missing dataframe for the UI.
    can_impute
        True when the pretrained gate allows and windows exist.

    Instance variables
    ------------------
    frame : ValidatedFrame
        Schema-validated table.
    windows : AdaptedWindows or None
        Pad / cap windows when adapt succeeded.
    source_label : str
        Sample-dataset label or upload filename.
    ingest_mode : str
        ``exact_beijing`` or ``flexible``.
    preview : IngestPreviewSummary
        Schema / missingness / window preview.
    impute_gate : ImputeGateResult
        Pretrained impute allow / block decision.
    """

    frame: ValidatedFrame
    windows: AdaptedWindows | None
    source_label: str
    ingest_mode: str
    preview: IngestPreviewSummary
    impute_gate: ImputeGateResult

    def n_windows(self) -> int:
        """Return adapted window count.

        Returns
        -------
        int
            ``B``, or ``0`` when adapt failed.
        """
        if self.windows is None:
            return 0
        return self.windows.n_windows()

    def missingness_table(self) -> pd.DataFrame:
        """Return a small missingness summary table.

        Returns
        -------
        pandas.DataFrame
            Feature name and percent missing.
        """
        summary = summarize_missingness(self.frame)
        percents = summary.per_feature_percent()
        return pd.DataFrame(
            {
                "Feature": list(percents.keys()),
                "% missing": list(percents.values()),
            }
        )

    def can_impute(self) -> bool:
        """Return whether pretrained reverse-SDE impute may run.

        Returns
        -------
        bool
            Gate allowed and windows present.
        """
        return self.impute_gate.is_allowed() and self.windows is not None


@dataclass(frozen=True)
class ImputeSectionContext:
    """Bundle for gated impute UI rendering (keeps call arity low).

    Public methods
    --------------
    seed_mask
        Demo-mask seed from config.
    overlay_enabled
        Whether posterior sample overlay is requested.
    active_imputer
        Session imputer when bound; else pretrained.
    active_windows
        Session windows when bound; else payload windows.

    Instance variables
    ------------------
    payload : UiSessionPayload
        Current data + gate.
    sidebar : UiSidebarState
        Sampler knobs.
    imputer : SessionImputer
        Default pretrained imputer (Exact path).
    config : AppM4Config
        Seeds and schema.
    bound_session : BoundAdaptSession or None
        Successful M7 adapt bind for this upload, if any.
    """

    payload: UiSessionPayload
    sidebar: UiSidebarState
    imputer: SessionImputer
    config: AppM4Config
    bound_session: BoundAdaptSession | None = None

    def seed_mask(self) -> int:
        """Return the demo-mask PRNG seed.

        Returns
        -------
        int
            ``config.sample.seed_mask``.
        """
        return self.config.sample.seed_mask

    def overlay_enabled(self) -> bool:
        """Return whether sample overlay is enabled.

        Returns
        -------
        bool
            Sidebar overlay flag.
        """
        return self.sidebar.wants_sample_overlay()

    def active_imputer(self) -> SessionImputer:
        """Return the session imputer when bound; else pretrained.

        Returns
        -------
        SessionImputer
            Imputer used for Fill missing values.
        """
        if self.bound_session is not None:
            return self.bound_session.imputer
        return self.imputer

    def active_windows(self) -> AdaptedWindows | None:
        """Return session windows when bound; else payload windows.

        Returns
        -------
        AdaptedWindows or None
            Windows for impute, or ``None`` when unavailable.
        """
        if self.bound_session is not None:
            return self.bound_session.windows
        return self.payload.windows


def refresh_impute_gate(
        payload: UiSessionPayload,
        checkpoint_features: tuple[str, ...],
        session_features: tuple[str, ...] | None,
) -> UiSessionPayload:
    """Return a copy of ``payload`` with gate re-evaluated (M7 unlock).

    Parameters
    ----------
    payload : UiSessionPayload
        Current preview payload.
    checkpoint_features : tuple of str
        Pretrained app / M2 feature names.
    session_features : tuple of str or None
        Bound adapt feature names, or ``None``.

    Returns
    -------
    UiSessionPayload
        Same frame/windows/preview; updated ``impute_gate``.
    """
    gate = evaluate_impute_gate(
        payload.frame.feature_names,
        checkpoint_features,
        payload.ingest_mode,
        session_features,
    )
    return replace(payload, impute_gate=gate)


def build_bundled_payload(config: AppM4Config) -> UiSessionPayload | None:
    """Load the bundled Beijing sample (Exact pretrained path).

    Parameters
    ----------
    config : AppM4Config
        Sample path and window knobs.

    Returns
    -------
    UiSessionPayload or None
        Payload, or ``None`` if the sample file is missing.
    """
    if not config.paths.sample_csv.is_file():
        return None
    frame = validate_upload_csv(config.paths.sample_csv, default_app_schema())
    outcome = AdaptOutcome(
        windows=adapt_validated_frame(
            frame,
            config.schema.window_request(config.smoke.max_windows),
        ),
        adapt_error=None,
    )
    identity = PayloadIdentity(
        label=_SOURCE_BUNDLED,
        mode=MODE_EXACT_BEIJING,
        checkpoint_features=config.schema.feature_names,
    )
    return payload_from_parts(frame, outcome, identity)


def build_upload_payload_from_bytes(
        payload: bytes,
        label: str,
        config: AppM4Config,
) -> UiSessionPayload:
    """Parse upload bytes into Exact or Flexible preview state.

    Parameters
    ----------
    payload : bytes
        Raw CSV bytes.
    label : str
        Upload filename for the UI.
    config : AppM4Config
        Checkpoint feature names and window knobs.

    Returns
    -------
    UiSessionPayload
        Preview + gate payload.
    """
    temp_path = write_temp_csv(payload)
    try:
        headers = read_csv_headers(temp_path)
        mode = select_ingest_mode(headers, default_app_schema())
        frame, outcome = validate_and_adapt_upload(temp_path, mode, config)
        identity = PayloadIdentity(
            label=label,
            mode=mode,
            checkpoint_features=config.schema.feature_names,
        )
        return payload_from_parts(frame, outcome, identity)
    finally:
        temp_path.unlink(missing_ok=True)


def validate_and_adapt_upload(
        path: Path,
        mode: str,
        config: AppM4Config,
) -> tuple[ValidatedFrame, AdaptOutcome]:
    """Validate a CSV path and adapt windows for the selected mode.

    Parameters
    ----------
    path : pathlib.Path
        Temp or on-disk CSV path.
    mode : str
        ``exact_beijing`` or ``flexible``.
    config : AppM4Config
        Window length and ``max_windows`` cap.

    Returns
    -------
    frame : ValidatedFrame
        Parsed table.
    outcome : AdaptOutcome
        Windows or adapt reject text.
    """
    if mode == MODE_EXACT_BEIJING:
        frame = validate_upload_csv(path, default_app_schema())
        windows = adapt_validated_frame(
            frame,
            config.schema.window_request(config.smoke.max_windows),
        )
        return frame, AdaptOutcome(windows=windows, adapt_error=None)
    frame = ingest_flexible_csv(path)
    knobs = FlexiblePreviewKnobs(
        seq_len=config.schema.seq_len,
        max_windows=config.smoke.max_windows,
    )
    windows, adapt_error = try_adapt_flexible_windows(frame, knobs)
    return frame, AdaptOutcome(windows=windows, adapt_error=adapt_error)


def payload_from_parts(
        frame: ValidatedFrame,
        outcome: AdaptOutcome,
        identity: PayloadIdentity,
) -> UiSessionPayload:
    """Assemble preview + impute gate for a validated frame.

    Parameters
    ----------
    frame : ValidatedFrame
        Parsed table.
    outcome : AdaptOutcome
        Adapted windows or reject text.
    identity : PayloadIdentity
        Source label, mode, and checkpoint feature names.

    Returns
    -------
    UiSessionPayload
        UI session payload.
    """
    preview = build_ingest_preview(
        PreviewBuildInputs(
            frame=frame,
            ingest_mode=identity.ingest_mode(),
            windows=outcome.windows,
            adapt_error=outcome.error_text(),
        )
    )
    gate = evaluate_impute_gate(
        frame.feature_names,
        identity.checkpoint_features,
        identity.ingest_mode(),
    )
    return UiSessionPayload(
        frame=frame,
        windows=outcome.windows,
        source_label=identity.source_label(),
        ingest_mode=identity.ingest_mode(),
        preview=preview,
        impute_gate=gate,
    )


def write_temp_csv(payload: bytes) -> Path:
    """Write upload bytes to a temporary CSV path.

    Parameters
    ----------
    payload : bytes
        Raw CSV bytes from Streamlit.

    Returns
    -------
    pathlib.Path
        Temp path (caller must unlink).
    """
    with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
    ) as handle:
        handle.write(payload)
        return Path(handle.name)
