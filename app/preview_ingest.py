"""Preview summaries and pretrained impute gate for M6 flexible ingest.

Hermetic helpers for Streamlit / CLI. No training; no Streamlit dependency.

Classes
-------
FlexiblePreviewKnobs
    ``seq_len`` / ``max_windows`` / hourly-bin knobs for flexible adapt.
PreviewBuildInputs
    Frame + mode + optional windows / adapt error for preview assembly.
PreviewAdaptInfo
    Window count and adapt reject / success message.
IngestPreviewSummary
    Schema / missingness / window (or reject) summary before impute.
ImputeGateResult
    Whether pretrained impute is allowed and the user-facing reason.

Functions
---------
MODE_EXACT_BEIJING / MODE_FLEXIBLE
    Ingest mode string constants.
select_ingest_mode
    Auto-detect Exact vs Flexible from CSV headers.
features_match_checkpoint
    Exact name + count (+ order) match vs loaded schema names.
evaluate_impute_gate
    Allow Exact path, or Flexible only on checkpoint feature match.
try_adapt_flexible_windows
    Adapt with M6 reject policy; return windows or reject message.
build_ingest_preview
    Assemble an ``IngestPreviewSummary``.
read_csv_headers
    Header tuple from a CSV path (no Beijing rename).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.schema_validate import (
    AppSchemaSpec,
    ValidatedFrame,
    summarize_missingness,
)
from app.window_adapter import (
    AdaptedWindows,
    FlexibleAdaptRequest,
    IngestRejectPolicy,
    WindowAdaptRequest,
    adapt_flexible_frame,
    default_ingest_reject_policy,
)

MODE_EXACT_BEIJING = "exact_beijing"
MODE_FLEXIBLE = "flexible"

_GATE_BLOCK_MESSAGE = (
    "Pretrained imputation is blocked: mapped features do not match the "
    "loaded checkpoint/scaler names and count. Flexible ingest in M6 is "
    "preview-only - use Milestone 7 (train-on-upload / adapt) or load a "
    "compatible checkpoint and scaler. Arbitrary CSVs cannot be imputed "
    "with the M2 Beijing weights."
)
_GATE_ALLOW_EXACT = (
    "Exact Beijing schema: pretrained impute is allowed with the loaded "
    "checkpoint/scaler."
)
_GATE_ALLOW_FLEXIBLE_MATCH = (
    "Mapped features exactly match the loaded checkpoint/scaler; "
    "pretrained impute is allowed."
)


@dataclass(frozen=True)
class FlexiblePreviewKnobs:
    """Window / reject knobs for flexible preview adapt.

    Public methods
    --------------
    as_adapt_request
        Build a ``FlexibleAdaptRequest`` with ASSUMPTIONS reject defaults.
    reject_policy
        Return the locked ``IngestRejectPolicy`` for ``seq_len``.

    Instance variables
    ------------------
    seq_len : int
        Window length ``T``.
    max_windows : int or None
        Optional UI / preview cap.
    apply_hourly_bin : bool
        Hourly last-value-in-bin when timestamps exist.
    """

    seq_len: int
    max_windows: int | None = None
    apply_hourly_bin: bool = True

    def reject_policy(self) -> IngestRejectPolicy:
        """Return ASSUMPTIONS M6 reject defaults for ``seq_len``.

        Returns
        -------
        IngestRejectPolicy
            ``min_rows`` / observed-rate floors.
        """
        return default_ingest_reject_policy(self.seq_len)

    def as_adapt_request(self) -> FlexibleAdaptRequest:
        """Return the flexible adapt request used for preview windows.

        Returns
        -------
        FlexibleAdaptRequest
            Shared pad path plus flexible reject / bin knobs.
        """
        return FlexibleAdaptRequest(
            window=WindowAdaptRequest(
                seq_len=self.seq_len,
                max_windows=self.max_windows,
            ),
            reject=self.reject_policy(),
            apply_hourly_bin=self.apply_hourly_bin,
        )


@dataclass(frozen=True)
class PreviewBuildInputs:
    """Inputs for assembling an ingest preview summary.

    Public methods
    --------------
    has_windows
        Whether adaptation produced windows.
    window_count
        ``B`` when windows exist; otherwise ``None``.

    Instance variables
    ------------------
    frame : ValidatedFrame
        Parsed Exact or Flexible frame.
    ingest_mode : str
        ``exact_beijing`` or ``flexible``.
    windows : AdaptedWindows or None
        Adapted batch, or ``None`` when adapt rejected / skipped.
    adapt_error : str or None
        User-facing reject text when ``windows`` is ``None``.
    """

    frame: ValidatedFrame
    ingest_mode: str
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

    def window_count(self) -> int | None:
        """Return adapted window count ``B``, if any.

        Returns
        -------
        int or None
            ``n_windows()`` or ``None``.
        """
        if self.windows is None:
            return None
        return self.windows.n_windows()


@dataclass(frozen=True)
class PreviewAdaptInfo:
    """Window adapt outcome for the ingest preview.

    Public methods
    --------------
    has_windows
        Whether ``n_windows`` is set.
    describe
        Short adapt note for CLI / UI.

    Instance variables
    ------------------
    n_windows : int or None
        Adapted ``B``, or ``None`` when adapt failed.
    adapt_message : str or None
        Reject / success text for the window step.
    """

    n_windows: int | None
    adapt_message: str | None

    def has_windows(self) -> bool:
        """Return whether adapted windows are available.

        Returns
        -------
        bool
            ``True`` when ``n_windows`` is not ``None``.
        """
        return self.n_windows is not None

    def describe(self) -> str:
        """Return a short adapt note.

        Returns
        -------
        str
            ``adapt_message``, or a fallback when absent.
        """
        if self.adapt_message:
            return self.adapt_message
        if self.n_windows is None:
            return "no windows"
        return f"adapted {self.n_windows} non-overlapping window(s)"


@dataclass(frozen=True)
class IngestPreviewSummary:
    """Schema / missingness / window preview before impute.

    Public methods
    --------------
    feature_count
        Return ``F``.
    as_text_lines
        CLI-friendly summary lines.

    Instance variables
    ------------------
    ingest_mode : str
        ``exact_beijing`` or ``flexible``.
    feature_names : tuple of str
        Mapped / Exact feature names (never silently remapped).
    n_rows : int
        Usable frame length ``L`` (pre-window; post-parse / pre-pad).
    per_feature_missing_percent : dict of str to float
        Native percent missing in ``[0, 100]`` per feature.
    adapt : PreviewAdaptInfo
        Window count and adapt reject / success message.
    entity_id : str or None
        Optional entity label.
    has_timestamps : bool
        Whether wall-clock timestamps were present on the frame.
    """

    ingest_mode: str
    feature_names: tuple[str, ...]
    n_rows: int
    per_feature_missing_percent: dict[str, float]
    adapt: PreviewAdaptInfo
    entity_id: str | None
    has_timestamps: bool

    def feature_count(self) -> int:
        """Return the feature count ``F``.

        Returns
        -------
        int
            ``len(feature_names)``.
        """
        return len(self.feature_names)

    def as_text_lines(self) -> list[str]:
        """Return human-readable preview lines for CLI / logs.

        Returns
        -------
        list of str
            Multi-line summary (no claim that arbitrary CSVs impute).
        """
        names = ", ".join(self.feature_names)
        n_features = self.feature_count()
        lines = [
            f"mode={self.ingest_mode}",
            f"features (F={n_features}): {names}",
            f"length L={self.n_rows}",
            f"timestamps={'yes' if self.has_timestamps else 'no'}",
            f"entity={self.entity_id!r}",
        ]
        for name, percent in self.per_feature_missing_percent.items():
            lines.append(f"missing% {name}={percent:.2f}")
        if self.adapt.n_windows is not None:
            lines.append(f"windows B={self.adapt.n_windows}")
        if self.adapt.adapt_message:
            lines.append(f"adapt: {self.adapt.adapt_message}")
        return lines


@dataclass(frozen=True)
class ImputeGateResult:
    """Pretrained impute allow / block decision for M6.

    Public methods
    --------------
    is_allowed
        Whether reverse-SDE impute may run.
    block_message
        User-facing block text (empty when allowed).

    Instance variables
    ------------------
    allowed : bool
        Gate outcome.
    reason : str
        Allow note or block copy (always set).
    """

    allowed: bool
    reason: str

    def is_allowed(self) -> bool:
        """Return whether pretrained impute is permitted.

        Returns
        -------
        bool
            ``allowed``.
        """
        return self.allowed

    def block_message(self) -> str:
        """Return the block message, or empty string when allowed.

        Returns
        -------
        str
            User-facing gate copy when blocked.
        """
        if self.allowed:
            return ""
        return self.reason


def select_ingest_mode(
        headers: tuple[str, ...],
        exact_schema: AppSchemaSpec,
) -> str:
    """Auto-detect Exact Beijing vs Flexible from CSV headers.

    Exact when every required Beijing feature name is present. No column
    is renamed to ``PM2.5``…``O3``.

    Parameters
    ----------
    headers : tuple of str
        CSV header names.
    exact_schema : AppSchemaSpec
        Locked Exact Beijing feature contract.

    Returns
    -------
    str
        ``exact_beijing`` or ``flexible``.
    """
    exact_schema.validate()
    header_set = set(headers)
    if all(name in header_set for name in exact_schema.feature_names):
        return MODE_EXACT_BEIJING
    return MODE_FLEXIBLE


def features_match_checkpoint(
        mapped_features: tuple[str, ...],
        checkpoint_features: tuple[str, ...],
) -> bool:
    """Return whether mapped features exactly match checkpoint names.

    Match requires identical length, names, and order (EVAL / scaler
    contract). Scaler NPZ stores ``F`` only; names come from app schema.

    Parameters
    ----------
    mapped_features : tuple of str
        Features from Exact validation or Flexible map/infer.
    checkpoint_features : tuple of str
        Loaded checkpoint / app schema feature names.

    Returns
    -------
    bool
        ``True`` on exact tuple equality.
    """
    return mapped_features == checkpoint_features


def evaluate_impute_gate(
        mapped_features: tuple[str, ...],
        checkpoint_features: tuple[str, ...],
        ingest_mode: str,
) -> ImputeGateResult:
    """Decide whether pretrained impute may run for this ingest.

    Exact Beijing mode is allowed (upload already validated against the
    Exact schema). Flexible mode is allowed only when mapped features
    exactly match the loaded checkpoint/scaler names and count.

    Parameters
    ----------
    mapped_features : tuple of str
        Features on the validated frame.
    checkpoint_features : tuple of str
        Names tied to the loaded M2 checkpoint / scaler.
    ingest_mode : str
        ``exact_beijing`` or ``flexible``.

    Returns
    -------
    ImputeGateResult
        Allow / block with user-facing reason.

    Raises
    ------
    ValueError
        If ``ingest_mode`` is unknown.
    """
    if ingest_mode == MODE_EXACT_BEIJING:
        return ImputeGateResult(allowed=True, reason=_GATE_ALLOW_EXACT)
    if ingest_mode != MODE_FLEXIBLE:
        raise ValueError(f"unknown ingest_mode: {ingest_mode!r}")
    if features_match_checkpoint(mapped_features, checkpoint_features):
        return ImputeGateResult(
            allowed=True,
            reason=_GATE_ALLOW_FLEXIBLE_MATCH,
        )
    return ImputeGateResult(allowed=False, reason=_GATE_BLOCK_MESSAGE)


def try_adapt_flexible_windows(
        frame: ValidatedFrame,
        knobs: FlexiblePreviewKnobs,
) -> tuple[AdaptedWindows | None, str | None]:
    """Adapt a flexible frame; return windows or a reject message.

    Parameters
    ----------
    frame : ValidatedFrame
        Flexible (or any) validated table.
    knobs : FlexiblePreviewKnobs
        ``T``, cap, and bin flag.

    Returns
    -------
    windows : AdaptedWindows or None
        Adapted batch on success.
    adapt_error : str or None
        User-facing reject text on failure.
    """
    try:
        windows = adapt_flexible_frame(frame, knobs.as_adapt_request())
    except ValueError as exc:
        return None, str(exc)
    return windows, None


def build_ingest_preview(inputs: PreviewBuildInputs) -> IngestPreviewSummary:
    """Build a schema / missingness / window preview summary.

    Parameters
    ----------
    inputs : PreviewBuildInputs
        Frame, mode, and optional adapt outcome.

    Returns
    -------
    IngestPreviewSummary
        Preview payload for UI / CLI (before impute).
    """
    frame = inputs.frame
    missing = summarize_missingness(frame).per_feature_percent()
    adapt_message = inputs.adapt_error
    if adapt_message is None and inputs.has_windows():
        adapt_message = (
            f"adapted {inputs.window_count()} non-overlapping "
            f"window(s)"
        )
    return IngestPreviewSummary(
        ingest_mode=inputs.ingest_mode,
        feature_names=frame.feature_names,
        n_rows=frame.n_rows(),
        per_feature_missing_percent=missing,
        adapt=PreviewAdaptInfo(
            n_windows=inputs.window_count(),
            adapt_message=adapt_message,
        ),
        entity_id=frame.entity_id,
        has_timestamps=frame.timestamps is not None,
    )


def read_csv_headers(path: Path) -> tuple[str, ...]:
    """Return non-empty CSV header names in file order.

    Parameters
    ----------
    path : pathlib.Path
        CSV path.

    Returns
    -------
    tuple of str
        Header field names.

    Raises
    ------
    ValueError
        If the file is missing, empty, or has no header columns.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"CSV path does not exist: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        headers = tuple(
            name for name in reader.fieldnames
            if name is not None and str(name).strip() != ""
        )
    if not headers:
        raise ValueError(f"CSV has no header columns: {csv_path}")
    return headers
