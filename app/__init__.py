"""Streamlit product surface for Milestone 4 / 6 (TimeFill).

Modules
-------
schema_validate
    Upload / sample CSV schema checks and missingness summaries.
flexible_ingest
    Infer / map arbitrary CSV columns into a ``ValidatedFrame`` (M6).
preview_ingest
    Schema / missingness preview helpers and pretrained impute gate (M6).
ui_session
    Non-Streamlit session types and CSV → preview payload assembly.
window_adapter
    CSV → ``(x, m)`` windows; Exact pad path + flexible bin/reject (M6).
m4_config
    Load / validate ``configs/app_m4.yaml``.
session_impute
    Non-UI pretrained reverse-SDE impute + mean-fill merge helpers.
export_frames
    Vectorized mean-fill / multi-sample download tables (CSV bytes).
plot_impute
    Plotly chart builders for gaps and mean ± bands.
demo_eval_smoke
    Tiny in-app MCAR smoke metrics (not the M3 report).
streamlit_app
    Streamlit entrypoint: Exact Beijing impute + Flexible preview / gate.
"""
