"""Milestone 4 Streamlit product UI (Phase 5).

Sidebar demo-mask / ``K`` / solver / quantile knobs; main pane upload or
bundled sample → missingness → reverse-SDE impute (WP2) → Plotly charts →
mean-fill download. Optional demo-eval smoke (not the M3 report).

Fine-tune controls are omitted (ASSUMPTIONS M4: pretrained-only). Irregular
wall-clock SDE dynamics remain deferred.

Launch
------
streamlit run app/streamlit_app.py

Classes
-------
UiDemoMaskControls
    Demo MCAR / block sidebar knobs.
UiSolverControls
    ``K`` / steps / quantile sidebar knobs.
UiSidebarState
    Nested sidebar state collected once per rerun.
UiSessionPayload
    Validated frame, adapted windows, and optional impute result.

Functions
---------
main
    Streamlit entrypoint.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # pylint: disable = import-error
import streamlit as st  # pylint: disable = import-error

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
from app.schema_validate import (
    AppSchemaSpec,
    ValidatedFrame,
    default_app_schema,
    summarize_missingness,
    validate_upload_csv,
)
from app.session_impute import (
    DemoMaskRequest,
    SessionImputeRequest,
    SessionImputeResult,
    SessionImputer,
)
from app.window_adapter import AdaptedWindows, adapt_validated_frame
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
class UiSessionPayload:
    """Loaded CSV + windows for the current session selection.

    Public methods
    --------------
    n_windows
        Adapted batch size ``B``.
    missingness_table
        Per-feature percent-missing dataframe for the UI.

    Instance variables
    ------------------
    frame : ValidatedFrame
        Schema-validated table.
    windows : AdaptedWindows
        Pad / cap windows ready to impute.
    source_label : str
        Sample-dataset label or upload filename.
    """

    frame: ValidatedFrame
    windows: AdaptedWindows
    source_label: str

    def n_windows(self) -> int:
        """Return adapted window count.

        Returns
        -------
        int
            ``B``.
        """
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
                "Pollutant": list(percents.keys()),
                "% missing": list(percents.values()),
            }
        )


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
    """Resolve bundled sample or upload into validated windows.

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
    schema = default_app_schema()
    if source == _SOURCE_BUNDLED:
        st.caption(
            "Sample matches the Beijing air-quality schema this demo was "
            "trained on (known-good path)."
        )
        if not config.paths.sample_csv.is_file():
            st.error(
                "Sample file is missing. Expected "
                f"{config.paths.sample_csv}."
            )
            return None
        frame = validate_upload_csv(config.paths.sample_csv, schema)
        label = _SOURCE_BUNDLED
    else:
        _render_upload_guidance(config)
        uploaded = st.file_uploader(
            "Upload a CSV (exact pollutant column names required)",
            type="csv",
            help="Column names must match exactly; aliases are not remapped.",
        )
        if uploaded is None:
            st.info(
                "Upload a CSV that matches the checklist above, or switch to "
                "the sample dataset for a known-good demo."
            )
            return None
        frame = _validate_uploaded_bytes(uploaded.getvalue(), schema)
        label = uploaded.name
    windows = adapt_validated_frame(
        frame,
        config.schema.window_request(config.smoke.max_windows),
    )
    return UiSessionPayload(frame=frame, windows=windows, source_label=label)


def _render_upload_guidance(config: AppM4Config) -> None:
    """Show pretrained-domain warning and CSV checklist for uploads.

    Parameters
    ----------
    config : AppM4Config
        Schema names and window length for the checklist.
    """
    feature_list = ", ".join(config.schema.feature_names)
    seq_len = config.schema.seq_len
    st.warning(
        "This demo uses a model trained on **UCI Beijing Multi-Site "
        "Air-Quality (PRSA)** — see **About this model** above. Uploads "
        "work best from that same source and do **not** retrain the model. "
        "Your file must use the **exact** pollutant column names below. "
        "Other cities or unrelated sensors may parse after renaming but "
        "are unsupported and can fill poorly. Prefer the sample dataset "
        "if you only want to try the product."
    )
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
        f"**CSV shape for this app**\n\n"
        f"- **Required columns (exact spelling):** `{feature_list}`\n"
        f"- **Recommended:** `timestamp` (hourly-style wall clock)\n"
        f"- **Optional:** `entity` or `site` (display / grouping only)\n"
        f"- **Missing values:** empty cells or NaN "
        f"(do not rely on zeros to mean missing)\n"
        f"- **Shape:** contiguous rows; the app cuts segments of length "
        f"**{seq_len}** (short tails are padded)\n"
        f"- **Other air-quality tables:** not recommended unless they "
        f"already match this schema and units; column aliases are not "
        f"remapped"
    )


def _validate_uploaded_bytes(
        payload: bytes,
        schema: AppSchemaSpec,
) -> ValidatedFrame:
    """Write upload bytes to a temp file and validate.

    Parameters
    ----------
    payload : bytes
        Raw CSV bytes from Streamlit.
    schema : AppSchemaSpec
        Column contract for validation.

    Returns
    -------
    ValidatedFrame
        Parsed frame.
    """
    with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        return validate_upload_csv(temp_path, schema)
    finally:
        temp_path.unlink(missing_ok=True)


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
        "Pollutant",
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
        help="One row per draw × time × pollutant for uncertainty analysis.",
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
            f"**Uploading a CSV does not retrain the model** — it only runs "
            f"gap-filling (inference) on your file. PRSA Beijing uploads are "
            f"the best match; they are not automatically a held-out benchmark "
            f"(the model may have seen similar sites/times during training). "
            f"This demo does not fine-tune on your upload."
        )


def main() -> None:
    """Run the TimeFill M4 Streamlit app."""
    st.set_page_config(page_title="TimeFill", layout="wide")
    st.title("TimeFill — fill gaps in air-quality time series")
    st.caption(
        "Upload messy sensor data (or try the sample), get filled values with "
        "uncertainty bands, and download a completed CSV. Values are shown "
        "and exported in original measurement units."
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
    st.write(
        f"**{payload.source_label}** · "
        f"{payload.n_windows()} segment(s) · "
        f"{config.schema.seq_len} steps each · "
        f"{config.schema.n_features} pollutants"
    )
    st.caption("Missingness in your file (before simulated gaps):")
    st.dataframe(payload.missingness_table(), width="stretch")
    if st.button("Fill missing values", type="primary"):
        with st.spinner("Filling gaps… this can take a few minutes on CPU."):
            result = imputer.impute(
                sidebar.impute_request(
                    payload.windows,
                    config.sample.seed_mask,
                )
            )
        st.session_state["m4_impute_result"] = result
        st.session_state["m4_feature_names"] = payload.windows.feature_names
    result = st.session_state.get("m4_impute_result")
    feature_names = st.session_state.get("m4_feature_names")
    if result is not None and feature_names is not None:
        st.success(
            f"Filled {result.n_windows()} segment(s) "
            f"({result.meta.describe()}); "
            f"kept {result.observed_cell_count()} observed values."
        )
        _render_charts(
            result,
            tuple(feature_names),
            sidebar.wants_sample_overlay(),
        )
        _render_downloads(result, tuple(feature_names))
    _render_demo_eval(imputer, config, payload.windows)


if __name__ == "__main__":
    main()
