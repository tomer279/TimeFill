"""Milestone 4 / 6 / 7 Streamlit product UI.

Upload or bundled sample → preview → optional M7 adapt → Fill → charts /
honest CSV download. Session types live in ``app.ui_session``.

Launch: ``streamlit run app/streamlit_app.py``.

Functions
---------
main
    Streamlit entrypoint.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd  # type: ignore # pylint: disable = import-error
import streamlit as st  # type: ignore # pylint: disable = import-error

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local package imports follow path bootstrap (Streamlit script entry).
# pylint: disable=wrong-import-position
from app.adapt_session_ui import (
    SESSION_ADAPT_KEY,
    SESSION_FEATURES_KEY,
    SESSION_RESULT_DIGEST_KEY,
    SESSION_RESULT_KEY,
    SESSION_RESULT_WINDOWS_KEY,
    AdaptRunRequest,
    BoundAdaptSession,
    caps_display_from_config,
    content_digest_from_bytes,
    impute_result_is_current,
    make_upload_fingerprint,
    prepare_adapt_windows,
    reset_session_binding,
    run_ui_adapt,
)
from app.demo_eval_smoke import DemoEvalSmokeRequest, run_demo_eval_smoke
from app.export_frames import (
    covered_row_range,
    dataframe_to_csv_bytes,
    mean_fill_dataframe,
    mean_fill_download_filename,
    mean_fill_request_from_windows,
    sample_request_from_windows,
    samples_long_dataframe,
    samples_long_download_filename,
    truncation_warning_message,
)
from app.window_adapter import AdaptedWindows
from app.m4_config import AppM4Config, load_app_m4_config
from app.plot_impute import (
    SeriesPlotRequest,
    figure_imputed_mean_bands,
    figure_original_with_gaps,
)
from app.preview_ingest import MODE_EXACT_BEIJING, MODE_FLEXIBLE
from app.session_impute import SessionImputeResult, SessionImputer
from app.ui_session import (
    ImputeSectionContext,
    UiDemoMaskControls,
    UiSessionPayload,
    UiSidebarState,
    UiSolverControls,
    build_bundled_payload,
    build_upload_payload_from_bytes,
    refresh_impute_gate,
)
from src.data.adapt_config import (
    AdaptCliOverrides,
    AdaptLoopCliOverrides,
    AdaptM7Config,
    AdaptWindowCliOverrides,
    apply_adapt_cli_overrides,
    load_adapt_m7_config,
)
# pylint: enable=wrong-import-position

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "app_m4.yaml"
_DEFAULT_ADAPT_CONFIG = _REPO_ROOT / "configs" / "adapt_m7.yaml"
_Q_LOW_INDEX = 0
_Q_HIGH_INDEX = 1
_DEFAULT_QUANTILE_LOW = 0.1
_DEFAULT_QUANTILE_HIGH = 0.9
_QUANTILE_LOW_MAX = 0.49
_QUANTILE_HIGH_MIN = 0.51
_SOURCE_BUNDLED = "Try the sample dataset"
_SOURCE_UPLOAD = "Upload your CSV"
_MASK_NONE = "none"
_MASK_RANDOM = "random"
_MASK_BLOCK = "block"
_MASK_MODE_LABELS = {
    _MASK_NONE: "Off",
    _MASK_RANDOM: "Random",
    _MASK_BLOCK: "Contiguous block",
}
_UCI_BEIJING_URL = (
    "https://archive.ics.uci.edu/dataset/501/"
    "beijing+multi+site+air+quality+data"
)
_PRSA_ZIP_NAME = "PRSA2017_Data_20130301-20170228.zip"
_SESSION_RESULT_KEY = SESSION_RESULT_KEY
_SESSION_FEATURES_KEY = SESSION_FEATURES_KEY
_SESSION_ADAPT_KEY = SESSION_ADAPT_KEY
_SESSION_RESULT_DIGEST_KEY = SESSION_RESULT_DIGEST_KEY
_SESSION_RESULT_WINDOWS_KEY = SESSION_RESULT_WINDOWS_KEY
_SESSION_UPLOAD_DIGEST_KEY = "m7_upload_content_digest"
_CAP_KNOB_SMOKE = "smoke.max_windows"
_CAP_KNOB_ADAPT = "window.max_windows / --max-windows"
_MODE_SCRATCH = "scratch"
_MIN_SEQ_LEN = 2
_DIGEST_DISPLAY_LEN = 8


@st.cache_resource
def _load_config(path_str: str) -> AppM4Config:
    """Load and cache the M4 app YAML."""
    return load_app_m4_config(path_str)


@st.cache_resource
def _load_imputer(path_str: str) -> SessionImputer:
    """Load and cache the pretrained SessionImputer."""
    return SessionImputer.from_app_config(path_str)


@st.cache_resource
def _load_adapt_config(path_str: str) -> AdaptM7Config:
    """Load and cache the M7 adapt YAML."""
    return load_adapt_m7_config(path_str)


def _clear_impute_session_keys() -> None:
    """Drop prior impute chart / download session state."""
    st.session_state.pop(_SESSION_RESULT_KEY, None)
    st.session_state.pop(_SESSION_FEATURES_KEY, None)
    st.session_state.pop(_SESSION_RESULT_DIGEST_KEY, None)
    st.session_state.pop(_SESSION_RESULT_WINDOWS_KEY, None)


def _resolve_bound_session(fingerprint: str) -> BoundAdaptSession | None:
    """Return a fingerprint-matching bound adapt session, if any.

    Parameters
    ----------
    fingerprint : str
        Current upload fingerprint.

    Returns
    -------
    BoundAdaptSession or None
        Valid bind, or ``None`` after clearing a stale bind.
    """
    bound = st.session_state.get(_SESSION_ADAPT_KEY)
    if bound is None:
        return None
    if not isinstance(bound, BoundAdaptSession) or not bound.matches(
            fingerprint
    ):
        reset_session_binding(st.session_state)
        return None
    return bound


def _render_sidebar(config: AppM4Config) -> UiSidebarState:
    """Draw sidebar knobs and return a frozen state object.

    Parameters
    ----------
    config : AppM4Config
        Defaults from ``app_m4.yaml``.

    Returns
    -------
    UiSidebarState
        Sidebar selections for this rerun.
    """
    st.sidebar.header("Settings")
    st.sidebar.caption(
        "Exact Beijing uses the pretrained model (no training). Flexible "
        "schemas need Run adapt first. Raising draws or quality steps "
        "slows things down, especially on CPU."
    )
    with st.sidebar.expander("Advanced (research names)", expanded=False):
        st.caption(
            "Internally: reverse-SDE impute, K draws, solver steps, "
            "MCAR / block demo masks. Config: configs/app_m4.yaml."
        )
    mask_mode = st.sidebar.selectbox(
        "Simulate extra gaps",
        options=[_MASK_NONE, _MASK_RANDOM, _MASK_BLOCK],
        index=0,
        format_func=lambda mode: _MASK_MODE_LABELS[mode],
        help="Hide some known values to demo filling. Off keeps native gaps only.",
    )
    mask_rate = st.sidebar.slider(
        "Fraction of values to hide",
        min_value=0.0,
        max_value=1.0,
        value=float(config.smoke.default_demo_rate()),
        step=0.05,
        disabled=mask_mode != _MASK_RANDOM,
        help="Used when Simulate extra gaps is Random.",
    )
    block_len = st.sidebar.number_input(
        "Gap length (time steps)",
        min_value=1,
        value=int(config.smoke.block_len),
        step=1,
        disabled=mask_mode != _MASK_BLOCK,
        help="Used when Simulate extra gaps is Contiguous block.",
    )
    k_samples = st.sidebar.number_input(
        "Number of draws",
        min_value=1,
        value=int(config.sample.k_samples),
        step=1,
        help="Higher = smoother uncertainty bands, slower runtime (K).",
    )
    solver_steps = st.sidebar.number_input(
        "Quality steps",
        min_value=1,
        value=int(config.sample.solver_steps),
        step=1,
        help="Higher usually improves fill quality but takes longer.",
    )
    quantile_low = st.sidebar.slider(
        "Band lower percentile",
        min_value=0.0,
        max_value=_QUANTILE_LOW_MAX,
        value=_DEFAULT_QUANTILE_LOW,
        step=0.05,
        help="Lower edge of the uncertainty band (e.g. 0.1 ≈ 10th percentile).",
    )
    quantile_high = st.sidebar.slider(
        "Band upper percentile",
        min_value=_QUANTILE_HIGH_MIN,
        max_value=1.0,
        value=_DEFAULT_QUANTILE_HIGH,
        step=0.05,
        help="Upper edge of the uncertainty band (e.g. 0.9 ≈ 90th percentile).",
    )
    overlay_samples = st.sidebar.checkbox(
        "Show individual draws",
        value=False,
        help="Overlay a few raw model draws on the filled series chart.",
    )
    return UiSidebarState(
        demo_mask=UiDemoMaskControls(
            mode=str(mask_mode),
            rate=float(mask_rate),
            block_len=int(block_len),
        ),
        solver=UiSolverControls(
            k_samples=int(k_samples),
            solver_steps=int(solver_steps),
            quantile_low=float(quantile_low),
            quantile_high=float(quantile_high),
        ),
        overlay_samples=bool(overlay_samples),
    )


def _load_payload(
        config: AppM4Config,
) -> tuple[UiSessionPayload | None, str | None]:
    """Resolve bundled sample or upload into validated preview state.

    Parameters
    ----------
    config : AppM4Config
        Paths and window policy.

    Returns
    -------
    payload : UiSessionPayload or None
        Loaded payload, or ``None`` when upload is missing.
    content_digest : str or None
        SHA-256 of the loaded bytes, or ``None`` when no payload.
    """
    st.subheader("Your data")
    source = st.radio(
        "Choose a dataset",
        options=[_SOURCE_BUNDLED, _SOURCE_UPLOAD],
        horizontal=True,
    )
    if source == _SOURCE_BUNDLED:
        return _load_bundled_payload(config)
    return _load_upload_payload(config)


def _load_bundled_payload(
        config: AppM4Config,
) -> tuple[UiSessionPayload | None, str | None]:
    """Load the bundled Beijing sample (Exact pretrained path).

    Parameters
    ----------
    config : AppM4Config
        Sample path and window knobs.

    Returns
    -------
    payload : UiSessionPayload or None
        Payload, or ``None`` if the sample file is missing.
    content_digest : str or None
        SHA-256 of sample file bytes when loaded.
    """
    st.caption(
        "Sample matches the Beijing air-quality schema this demo was "
        "trained on (known-good pretrained path)."
    )
    if not config.paths.sample_csv.is_file():
        st.error(
            "Sample file is missing. Expected "
            f"{config.paths.sample_csv}."
        )
        return None, None
    raw = config.paths.sample_csv.read_bytes()
    payload = build_bundled_payload(config)
    if payload is None:
        return None, None
    return payload, content_digest_from_bytes(raw)


def _load_upload_payload(
        config: AppM4Config,
) -> tuple[UiSessionPayload | None, str | None]:
    """Load an uploaded CSV with Exact / Flexible auto-detect.

    Parameters
    ----------
    config : AppM4Config
        Schema names and window policy.

    Returns
    -------
    payload : UiSessionPayload or None
        Payload, or ``None`` when no file is uploaded yet.
    content_digest : str or None
        SHA-256 of upload bytes when loaded.
    """
    _render_upload_guidance(config)
    uploaded = st.file_uploader(
        "Upload a CSV (Exact Beijing names → impute; else Flexible preview)",
        type="csv",
        help=(
            "Exact pollutant names enable pretrained impute. Other schemas "
            "preview first; use Run adapt (M7) to unlock session impute. "
            "Columns are never silently remapped to PM2.5…O3."
        ),
    )
    if uploaded is None:
        st.info(
            "Upload a CSV, or switch to the sample dataset for a "
            "known-good Beijing demo."
        )
        return None, None
    raw = uploaded.getvalue()
    payload = build_upload_payload_from_bytes(raw, uploaded.name, config)
    return payload, content_digest_from_bytes(raw)


def _render_upload_guidance(config: AppM4Config) -> None:
    """Show Exact vs Flexible upload guidance.

    Parameters
    ----------
    config : AppM4Config
        Schema names and window length for the checklist.
    """
    feature_list = ", ".join(config.schema.feature_names)
    seq_len = config.schema.seq_len
    st.warning(
        "This demo’s **pretrained** imputer is trained on **UCI Beijing "
        "Multi-Site Air-Quality (PRSA)**. Exact-schema uploads (pollutant "
        "names below) can be imputed. Other CSVs use **Flexible ingest**: "
        "schema / missingness preview only unless features **exactly** "
        "match the loaded checkpoint. Otherwise use **Run adapt** below "
        "for a file-local session model. Columns are **never** silently "
        f"renamed to `{feature_list}`. One-file adapt is not universal SOTA."
    )
    with st.expander(
            "Exact Beijing CSV checklist (pretrained path)",
            expanded=False,
    ):
        st.markdown(
            f"**Best-match data source**\n\n"
            f"- Download **{_PRSA_ZIP_NAME}** from "
            f"[UCI Beijing Multi-Site Air-Quality]"
            f"({_UCI_BEIJING_URL}) "
            f"(same data used for M2 training).\n"
            f"- Use a per-site `PRSA_Data_*.csv` slice; keep pollutant columns "
            f"`{feature_list}`.\n"
            f"- Raw PRSA has `year, month, day, hour` — build a single "
            f"`timestamp` if needed; map `station` → optional `entity` / "
            f"`site`; drop meteo columns (TEMP, PRES, …).\n\n"
            f"**CSV shape for Exact impute**\n\n"
            f"- **Required columns (exact spelling):** `{feature_list}`\n"
            f"- **Recommended:** `timestamp` (hourly-style wall clock)\n"
            f"- **Optional:** `entity` or `site` (display / grouping only)\n"
            f"- **Missing values:** empty cells or NaN "
            f"(do not rely on zeros to mean missing)\n"
            f"- **Shape:** contiguous rows; the app cuts segments of length "
            f"**{seq_len}** (short tails are padded)\n"
            f"- **Flexible uploads:** preview schema / missingness; then "
            f"**Run adapt** to unlock session impute (M7)"
        )


def _render_preview_panel(payload: UiSessionPayload) -> None:
    """Show schema / missingness / window preview before impute.

    Parameters
    ----------
    payload : UiSessionPayload
        Current session payload.
    """
    preview = payload.preview
    mode_label = (
        "Exact Beijing (pretrained path)"
        if payload.ingest_mode == MODE_EXACT_BEIJING
        else "Flexible schema (preview; impute gated)"
    )
    window_text = (
        str(preview.adapt.n_windows)
        if preview.adapt.n_windows is not None
        else "—"
    )
    st.subheader("Ingest preview (before fill)")
    st.write(
        f"**{payload.source_label}** · mode: **{mode_label}** · "
        f"F={preview.feature_count()} · L={preview.n_rows} · "
        f"windows B={window_text}"
    )
    st.caption(
        "Mapped features: " + ", ".join(preview.feature_names)
    )
    if preview.adapt.adapt_message:
        if preview.adapt.n_windows is None:
            st.error(preview.adapt.adapt_message)
        else:
            st.info(preview.adapt.adapt_message)
    st.caption("Native missingness in your file (before simulated gaps):")
    st.dataframe(payload.missingness_table(), width="stretch")


def _render_charts(
        result: SessionImputeResult,
        feature_names: tuple[str, ...],
        overlay: bool,
) -> None:
    """Render gap + mean±band charts for one window / feature.

    Parameters
    ----------
    result : SessionImputeResult
        Original-unit impute outputs.
    feature_names : tuple of str
        Frozen feature labels.
    overlay : bool
        Whether to overlay posterior samples.
    """
    st.subheader("Charts")
    window_idx = st.number_input(
        "Segment",
        min_value=0,
        max_value=max(result.n_windows() - 1, 0),
        value=0,
        step=1,
        help="Data is split into fixed-length segments for the model.",
    )
    feature_name = st.selectbox(
        "Feature",
        options=list(feature_names),
    )
    feat_idx = feature_names.index(feature_name)
    win = int(window_idx)
    samples = None
    if overlay:
        samples = result.tensors.samples[win, :, :, feat_idx]
    request = SeriesPlotRequest(
        observations=result.context.observations[win, :, feat_idx],
        mask=result.context.m_eval[win, :, feat_idx],
        mean=result.tensors.mean[win, :, feat_idx],
        quantile_low=result.tensors.quantiles[win, :, feat_idx, _Q_LOW_INDEX],
        quantile_high=result.tensors.quantiles[win, :, feat_idx, _Q_HIGH_INDEX],
        samples=samples,
        feature_name=feature_name,
    )
    st.plotly_chart(figure_original_with_gaps(request), width="stretch")
    st.plotly_chart(figure_imputed_mean_bands(request), width="stretch")


def _render_downloads(
        result: SessionImputeResult,
        windows: AdaptedWindows,
        cap_knob: str,
) -> None:
    """Offer mean-fill and optional multi-sample CSV downloads.

    Parameters
    ----------
    result : SessionImputeResult
        Impute outputs in original units.
    windows : AdaptedWindows
        Windows used for Fill (pad provenance for honest export).
    cap_knob : str
        Window-cap knob name for the truncation warning.
    """
    st.subheader("Downloads")
    if windows.is_partial() and windows.valid_len is not None:
        st.warning(
            truncation_warning_message(
                int(windows.valid_len.sum()),
                int(windows.source_n_rows or 0),
                cap_knob,
            )
        )
    mean_frame = mean_fill_dataframe(
        mean_fill_request_from_windows(
            result.context.mean_fill,
            windows,
        )
    )
    row_start, row_end = (0, 0)
    if windows.is_partial() and windows.row_index is not None:
        row_start, row_end = covered_row_range(windows)
    mean_name = mean_fill_download_filename(
        windows.is_partial(),
        row_start,
        row_end,
    )
    st.download_button(
        "Download completed CSV (best estimate)",
        data=dataframe_to_csv_bytes(mean_frame),
        file_name=mean_name,
        mime="text/csv",
        help=(
            "Native observed values kept; only native gaps filled. "
            "Simulate extra gaps affects charts/metrics only, not this file."
        ),
    )
    long_frame = samples_long_dataframe(
        sample_request_from_windows(
            result.tensors.samples,
            windows,
        )
    )
    long_name = samples_long_download_filename(
        windows.is_partial(),
        row_start,
        row_end,
    )
    st.download_button(
        "Download all draws (long format, optional)",
        data=dataframe_to_csv_bytes(long_frame),
        file_name=long_name,
        mime="text/csv",
        help="One row per draw × time × feature; pad rows omitted.",
    )


def _render_demo_eval(
        imputer: SessionImputer,
        config: AppM4Config,
        windows: AdaptedWindows,
) -> None:
    """Optional smoke metrics button (not the M3 report).

    Parameters
    ----------
    imputer : SessionImputer
        Pretrained imputer.
    config : AppM4Config
        Smoke caps.
    windows : AdaptedWindows
        Current adapted windows used as GT where native.
    """
    st.subheader("Quick quality check")
    st.caption(
        "Demo check only — not the full benchmark report. "
        "For the full tables, run scripts/run_m3_benchmark.py."
    )
    if st.button("Run a quick quality check"):
        with st.spinner("Running a small quality check…"):
            smoke = run_demo_eval_smoke(
                imputer,
                DemoEvalSmokeRequest(
                    windows=windows,
                    mcar_rate=config.smoke.default_demo_rate(),
                    k_samples=config.smoke.k_samples,
                    solver_steps=config.smoke.solver_steps,
                    mask_seed=config.sample.seed_mask,
                    block_len=config.smoke.block_len,
                ),
            )
        rows = smoke.as_rows()
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric", "Value"]),
            width="stretch",
        )
        st.caption(smoke.caption())


def _render_about_this_model(config: AppM4Config) -> None:
    """Show a short plain-language note about the pretrained model.

    Parameters
    ----------
    config : AppM4Config
        Feature names and window length for the note.
    """
    feature_list = ", ".join(config.schema.feature_names)
    seq_len = config.schema.seq_len
    with st.expander("About this model", expanded=False):
        st.markdown(
            f"TimeFill here uses a **pretrained** imputer trained on the UCI "
            f"**[PRSA Beijing Multi-Site Air-Quality]"
            f"({_UCI_BEIJING_URL})** archive "
            f"(hourly data from 12 sites, 2013–2017; zip "
            f"`{_PRSA_ZIP_NAME}`). "
            f"It fills the six pollutants **{feature_list}** on fixed "
            f"**{seq_len}-step** windows after hourly resampling.\n\n"
            f"**Exact Beijing** runs gap-filling with pretrained weights "
            f"(no training). **Flexible** schemas preview first; **Run "
            f"adapt** trains a smoke-capped **session** model under "
            f"`artifacts/m7/` (does not overwrite M2 claim weights). "
            f"File-local adapt is not a cross-dataset SOTA claim."
        )


def _effective_adapt_config(
        base: AdaptM7Config,
        seq_len: int,
) -> AdaptM7Config:
    """Apply UI ``seq_len`` override without raising other caps.

    Parameters
    ----------
    base : AdaptM7Config
        YAML defaults.
    seq_len : int
        Effective window length from Advanced smoke knobs.

    Returns
    -------
    AdaptM7Config
        Config used for prepare + ``run_ui_adapt``.
    """
    return apply_adapt_cli_overrides(
        base,
        AdaptCliOverrides(
            window=AdaptWindowCliOverrides(seq_len=int(seq_len)),
            loop=AdaptLoopCliOverrides(),
        ),
    )


def _run_adapt_button(
        payload: UiSessionPayload,
        fingerprint: str,
        config: AdaptM7Config,
) -> None:
    """Execute Run adapt and bind or surface a user-facing failure.

    Parameters
    ----------
    payload : UiSessionPayload
        Current upload / sample payload.
    fingerprint : str
        Upload fingerprint for session binding.
    config : AdaptM7Config
        Effective smoke-cap config.
    """
    # Clear first so any crash / uncaught error cannot leave a stale bind.
    reset_session_binding(st.session_state)
    windows, adapt_error = prepare_adapt_windows(
        payload.frame,
        int(config.geometry.seq_len),
        config.geometry.max_windows,
    )
    if windows is None:
        detail = adapt_error or "window adapt failed"
        st.error(
            f"**Too little data for adapt**\n\n{detail} "
            "Lower seq_len in Advanced smoke knobs or upload a longer "
            "series. No session model is bound."
        )
        return
    with st.spinner(
            "Running smoke-capped adapt… may take several minutes on CPU. "
            f"Hard stop at {config.caps.timeout_seconds:.0f}s."
    ):
        outcome = run_ui_adapt(
            AdaptRunRequest(
                windows=windows,
                config=config,
                mode=_MODE_SCRATCH,
                fingerprint=fingerprint,
            )
        )
    if not outcome.ok() or outcome.session is None:
        st.error(
            outcome.failure_markdown()
            + "\n\nNo session model is bound."
        )
        return
    st.session_state[_SESSION_ADAPT_KEY] = outcome.session
    _clear_impute_session_keys()
    st.success(outcome.session.describe())


def _adapt_panel_body(
        payload: UiSessionPayload,
        fingerprint: str,
        base_config: AdaptM7Config,
) -> BoundAdaptSession | None:
    """Render caps, Advanced seq_len, and Run adapt.

    Parameters
    ----------
    payload : UiSessionPayload
        Current payload.
    fingerprint : str
        Upload fingerprint.
    base_config : AdaptM7Config
        YAML smoke caps.

    Returns
    -------
    BoundAdaptSession or None
        Bound session when fingerprint still matches.
    """
    default_seq = int(base_config.geometry.seq_len)
    with st.expander("Advanced smoke knobs", expanded=False):
        st.caption(
            "Short fixtures may need a smaller seq_len than the M6 preview "
            "default (e.g. toy non-Beijing CSV). Does not write to "
            "artifacts/m2/."
        )
        seq_len = st.number_input(
            "Adapt window length (seq_len)",
            min_value=_MIN_SEQ_LEN,
            value=default_seq,
            step=1,
            help="Used for adapt windows; may differ from preview seq_len.",
        )
    effective = _effective_adapt_config(base_config, int(seq_len))
    caps = caps_display_from_config(effective, _MODE_SCRATCH)
    st.caption("Smoke caps: " + " · ".join(caps.as_lines()))
    st.caption(caps.eta_note())
    if st.button("Run adapt", type="secondary"):
        _run_adapt_button(payload, fingerprint, effective)
    return _resolve_bound_session(fingerprint)


def _show_bound_adapt_info(bound: BoundAdaptSession | None) -> None:
    """Show bind status after the Run adapt control has been handled.

    Parameters
    ----------
    bound : BoundAdaptSession or None
        Current fingerprint-matching bind, if any.
    """
    if bound is not None:
        st.info(bound.describe())


def _render_adapt_panel(
        payload: UiSessionPayload,
        fingerprint: str,
) -> BoundAdaptSession | None:
    """Show Run adapt after preview; return bound session if present.

    Parameters
    ----------
    payload : UiSessionPayload
        Current ingest payload.
    fingerprint : str
        Upload fingerprint.

    Returns
    -------
    BoundAdaptSession or None
        Session bind for gate lift / impute.
    """
    try:
        base_config = _load_adapt_config(str(_DEFAULT_ADAPT_CONFIG))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.warning(f"Adapt config unavailable: {exc}")
        return _resolve_bound_session(fingerprint)
    st.subheader("Adapt on this upload (M7)")
    st.caption(
        "Train a **session** model under smoke caps, then impute. "
        "Artifacts go under ``artifacts/m7/`` — never overwrites Beijing "
        "M2 claim weights. File-local only; not universal SOTA."
    )
    if payload.ingest_mode == MODE_EXACT_BEIJING:
        with st.expander(
                "Optional: adapt instead of pretrained",
                expanded=False,
        ):
            bound = _adapt_panel_body(payload, fingerprint, base_config)
        _show_bound_adapt_info(bound)
        return bound
    if payload.ingest_mode == MODE_FLEXIBLE:
        bound = _adapt_panel_body(payload, fingerprint, base_config)
        _show_bound_adapt_info(bound)
        return bound
    return _resolve_bound_session(fingerprint)


def _render_impute_section(context: ImputeSectionContext) -> None:
    """Render gated impute controls and prior results.

    Parameters
    ----------
    context : ImputeSectionContext
        Payload, sidebar, imputer, and config.
    """
    payload = context.payload
    windows = context.active_windows()
    content_digest = str(
        st.session_state.get(_SESSION_UPLOAD_DIGEST_KEY, "")
    )
    if not payload.impute_gate.is_allowed() or windows is None:
        _clear_impute_session_keys()
        st.warning(payload.impute_gate.reason)
        if (
                context.bound_session is None
                and payload.ingest_mode == MODE_FLEXIBLE
        ):
            st.info("Run adapt above to unlock impute for this schema.")
        return
    model_ref = (
        context.bound_session.run_id
        if context.bound_session is not None
        else "pretrained"
    )
    short_digest = content_digest[:_DIGEST_DISPLAY_LEN]
    st.caption(f"Fill uses: {model_ref} · data {short_digest}")
    cap_knob = (
        _CAP_KNOB_ADAPT
        if context.bound_session is not None
        else _CAP_KNOB_SMOKE
    )
    if windows.is_partial() and windows.valid_len is not None:
        st.warning(
            truncation_warning_message(
                int(windows.valid_len.sum()),
                int(windows.source_n_rows or 0),
                cap_knob,
            )
        )
    if st.button("Fill missing values", type="primary"):
        with st.spinner("Filling gaps… this can take a few minutes on CPU."):
            result = context.active_imputer().impute(
                context.sidebar.impute_request(
                    windows,
                    context.seed_mask(),
                )
            )
        st.session_state[_SESSION_RESULT_KEY] = result
        st.session_state[_SESSION_FEATURES_KEY] = windows.feature_names
        st.session_state[_SESSION_RESULT_DIGEST_KEY] = content_digest
        st.session_state[_SESSION_RESULT_WINDOWS_KEY] = windows
    if not impute_result_is_current(
            st.session_state.get(_SESSION_RESULT_DIGEST_KEY),
            content_digest,
    ):
        _clear_impute_session_keys()
        return
    result = st.session_state.get(_SESSION_RESULT_KEY)
    feature_names = st.session_state.get(_SESSION_FEATURES_KEY)
    result_windows = st.session_state.get(_SESSION_RESULT_WINDOWS_KEY)
    if (
            result is not None
            and feature_names is not None
            and result_windows is not None
    ):
        st.success(
            f"Filled {result.n_windows()} segment(s) "
            f"({result.meta.describe()}); "
            f"kept {result.observed_cell_count()} observed values."
        )
        _render_charts(
            result,
            tuple(feature_names),
            context.overlay_enabled(),
        )
        _render_downloads(result, result_windows, cap_knob)


def main() -> None:
    """Run the TimeFill Streamlit app (Exact / Flexible / M7 adapt)."""
    st.set_page_config(page_title="TimeFill", layout="wide")
    st.title("TimeFill — fill gaps in air-quality time series")
    st.caption(
        "Upload sensor data (or try the Beijing sample), preview schema / "
        "missingness, optionally Run adapt for Flexible schemas, then fill "
        "gaps. Exact Beijing pretrained path needs no training."
    )
    config_path = str(_DEFAULT_CONFIG)
    try:
        config = _load_config(config_path)
        config.paths.require_exist()
        imputer = _load_imputer(config_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
        return
    _render_about_this_model(config)
    sidebar = _render_sidebar(config)
    try:
        payload, content_digest = _load_payload(config)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
        return
    if payload is None or content_digest is None:
        st.stop()
        return
    prev_digest = st.session_state.get(_SESSION_UPLOAD_DIGEST_KEY)
    if prev_digest is not None and prev_digest != content_digest:
        reset_session_binding(st.session_state)
    st.session_state[_SESSION_UPLOAD_DIGEST_KEY] = content_digest
    fingerprint = make_upload_fingerprint(
        payload.source_label,
        payload.ingest_mode,
        payload.frame.feature_names,
        content_digest,
    )
    _render_preview_panel(payload)
    bound = _render_adapt_panel(payload, fingerprint)
    session_features = None if bound is None else bound.feature_names
    payload = refresh_impute_gate(
        payload,
        config.schema.feature_names,
        session_features,
    )
    if bound is not None:
        payload = replace(payload, windows=bound.windows)
    _render_impute_section(
        ImputeSectionContext(
            payload=payload,
            sidebar=sidebar,
            imputer=imputer,
            config=config,
            bound_session=bound,
        )
    )
    active_imputer = (
        bound.imputer if bound is not None else imputer
    )
    active_windows = (
        bound.windows if bound is not None else payload.windows
    )
    if payload.can_impute() and active_windows is not None:
        _render_demo_eval(active_imputer, config, active_windows)


if __name__ == "__main__":
    main()
