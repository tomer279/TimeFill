"""Milestone 4 / 6 Streamlit product UI (Phase 5 + M6 preview).

Sidebar demo-mask / ``K`` / solver / quantile knobs; main pane upload or
bundled sample → schema / missingness preview → reverse-SDE impute when
the pretrained gate allows → Plotly charts → mean-fill download.

Exact Beijing (bundled sample + exact-schema upload) remains the
pretrained impute path. Flexible uploads preview schema / missingness /
windows (or reject) and **block impute** unless mapped features exactly
match the loaded checkpoint/scaler (ASSUMPTIONS M6). Fine-tune controls
are omitted (M7). Irregular wall-clock SDE dynamics remain deferred.

Session types and CSV → payload assembly live in ``app.ui_session``.

Launch
------
streamlit run app/streamlit_app.py

Functions
---------
main
    Streamlit entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # type: ignore # pylint: disable = import-error
import streamlit as st  # type: ignore # pylint: disable = import-error

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local package imports follow path bootstrap (Streamlit script entry).
# pylint: disable=wrong-import-position
from app.demo_eval_smoke import DemoEvalSmokeRequest, run_demo_eval_smoke
from app.export_frames import (
    MeanFillExportRequest,
    SampleExportRequest,
    dataframe_to_csv_bytes,
    mean_fill_dataframe,
    samples_long_dataframe,
)
from app.m4_config import AppM4Config, load_app_m4_config
from app.plot_impute import (
    SeriesPlotRequest,
    figure_imputed_mean_bands,
    figure_original_with_gaps,
)
from app.preview_ingest import MODE_EXACT_BEIJING
from app.session_impute import SessionImputeResult, SessionImputer
from app.ui_session import (
    ImputeSectionContext,
    UiDemoMaskControls,
    UiSessionPayload,
    UiSidebarState,
    UiSolverControls,
    build_bundled_payload,
    build_upload_payload_from_bytes,
)
from app.window_adapter import AdaptedWindows
# pylint: enable=wrong-import-position

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "app_m4.yaml"
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
_SESSION_RESULT_KEY = "m4_impute_result"
_SESSION_FEATURES_KEY = "m4_feature_names"


@st.cache_resource
def _load_config(path_str: str) -> AppM4Config:
    """Load and cache the M4 app YAML.

    Parameters
    ----------
    path_str : str
        Path to ``configs/app_m4.yaml``.

    Returns
    -------
    AppM4Config
        Validated product config.
    """
    return load_app_m4_config(path_str)


@st.cache_resource
def _load_imputer(path_str: str) -> SessionImputer:
    """Load and cache the pretrained SessionImputer.

    Parameters
    ----------
    path_str : str
        Path to ``configs/app_m4.yaml``.

    Returns
    -------
    SessionImputer
        Ready for reverse-SDE impute.
    """
    return SessionImputer.from_app_config(path_str)


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
        "Uses a trained model (no fine-tuning here). Defaults aim for a few "
        "minutes; raising draws or quality steps slows things down, "
        "especially on CPU."
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


def _load_payload(config: AppM4Config) -> UiSessionPayload | None:
    """Resolve bundled sample or upload into validated preview state.

    Parameters
    ----------
    config : AppM4Config
        Paths and window policy.

    Returns
    -------
    UiSessionPayload or None
        Loaded payload, or ``None`` when upload is missing.
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


def _load_bundled_payload(config: AppM4Config) -> UiSessionPayload | None:
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
    st.caption(
        "Sample matches the Beijing air-quality schema this demo was "
        "trained on (known-good pretrained path)."
    )
    payload = build_bundled_payload(config)
    if payload is None:
        st.error(
            "Sample file is missing. Expected "
            f"{config.paths.sample_csv}."
        )
    return payload


def _load_upload_payload(config: AppM4Config) -> UiSessionPayload | None:
    """Load an uploaded CSV with Exact / Flexible auto-detect.

    Parameters
    ----------
    config : AppM4Config
        Schema names and window policy.

    Returns
    -------
    UiSessionPayload or None
        Payload, or ``None`` when no file is uploaded yet.
    """
    _render_upload_guidance(config)
    uploaded = st.file_uploader(
        "Upload a CSV (Exact Beijing names → impute; else Flexible preview)",
        type="csv",
        help=(
            "Exact pollutant names enable pretrained impute. Other schemas "
            "preview only unless features match the loaded checkpoint. "
            "Columns are never silently remapped to PM2.5…O3."
        ),
    )
    if uploaded is None:
        st.info(
            "Upload a CSV, or switch to the sample dataset for a "
            "known-good Beijing demo."
        )
        return None
    return build_upload_payload_from_bytes(
        uploaded.getvalue(),
        uploaded.name,
        config,
    )


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
        "match the loaded checkpoint — then you still need matching "
        "weights. Columns are **never** silently renamed to "
        f"`{feature_list}`. M6 is not a universal imputer (see M7)."
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
            f"- **Flexible uploads:** preview schema / missingness; impute "
            f"stays gated unless features match the checkpoint (or use M7)"
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
        feature_names: tuple[str, ...],
) -> None:
    """Offer mean-fill and optional multi-sample CSV downloads.

    Parameters
    ----------
    result : SessionImputeResult
        Impute outputs in original units.
    feature_names : tuple of str
        Feature axis labels.
    """
    st.subheader("Downloads")
    mean_frame = mean_fill_dataframe(
        MeanFillExportRequest(
            mean_fill=result.context.mean_fill,
            feature_names=feature_names,
        )
    )
    st.download_button(
        "Download completed CSV (best estimate)",
        data=dataframe_to_csv_bytes(mean_frame),
        file_name="timefill_mean_fill.csv",
        mime="text/csv",
        help="Observed values kept; gaps filled with the model mean.",
    )
    long_frame = samples_long_dataframe(
        SampleExportRequest(
            samples=result.tensors.samples,
            feature_names=feature_names,
        )
    )
    st.download_button(
        "Download all draws (long format, optional)",
        data=dataframe_to_csv_bytes(long_frame),
        file_name="timefill_samples_long.csv",
        mime="text/csv",
        help="One row per draw × time × feature for uncertainty analysis.",
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
            f"**Uploading a CSV does not retrain the model** — Exact Beijing "
            f"schema runs gap-filling (inference). Flexible schemas preview "
            f"missingness only unless features match this checkpoint; "
            f"otherwise use Milestone 7 train-on-upload / adapt or a "
            f"compatible checkpoint. This demo does not fine-tune on your "
            f"upload."
        )


def _render_impute_section(context: ImputeSectionContext) -> None:
    """Render gated impute controls and prior results.

    Parameters
    ----------
    context : ImputeSectionContext
        Payload, sidebar, imputer, and config.
    """
    payload = context.payload
    if not payload.can_impute():
        st.session_state.pop(_SESSION_RESULT_KEY, None)
        st.session_state.pop(_SESSION_FEATURES_KEY, None)
        st.warning(payload.impute_gate.reason)
        return
    windows = payload.windows
    if windows is None:
        st.warning(payload.impute_gate.reason)
        return
    if st.button("Fill missing values", type="primary"):
        with st.spinner("Filling gaps… this can take a few minutes on CPU."):
            result = context.imputer.impute(
                context.sidebar.impute_request(
                    windows,
                    context.seed_mask(),
                )
            )
        st.session_state[_SESSION_RESULT_KEY] = result
        st.session_state[_SESSION_FEATURES_KEY] = windows.feature_names
    result = st.session_state.get(_SESSION_RESULT_KEY)
    feature_names = st.session_state.get(_SESSION_FEATURES_KEY)
    if result is not None and feature_names is not None:
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
        _render_downloads(result, tuple(feature_names))


def main() -> None:
    """Run the TimeFill Streamlit app (M4 Exact + M6 Flexible preview)."""
    st.set_page_config(page_title="TimeFill", layout="wide")
    st.title("TimeFill — fill gaps in air-quality time series")
    st.caption(
        "Upload sensor data (or try the Beijing sample), preview schema / "
        "missingness, then fill gaps when the pretrained checkpoint "
        "matches. Flexible CSVs preview first; they are not a universal "
        "imputer."
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
        payload = _load_payload(config)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
        return
    if payload is None:
        st.stop()
        return
    _render_preview_panel(payload)
    _render_impute_section(
        ImputeSectionContext(
            payload=payload,
            sidebar=sidebar,
            imputer=imputer,
            config=config,
        )
    )
    if payload.can_impute() and payload.windows is not None:
        _render_demo_eval(imputer, config, payload.windows)


if __name__ == "__main__":
    main()
