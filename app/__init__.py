"""Streamlit product surface for Milestone 4 (TimeFill).

Modules
-------
schema_validate
    Upload / sample CSV schema checks and missingness summaries.
window_adapter
    CSV → ``(x, m)`` windows under the M1/M2/M4 tensor contract.
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
    Phase 5 Streamlit entrypoint (``streamlit run app/streamlit_app.py``).
"""
