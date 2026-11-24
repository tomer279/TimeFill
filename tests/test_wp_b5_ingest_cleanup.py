"""WP-B5 red tests: flexible infer heuristics, preview drops, caption flatten.

Classes / fixtures in this module are hermetic helpers only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.adapt_session_ui as adapt_session_ui
from app.adapt_session_ui import caps_display_from_config
from app.flexible_ingest import (
    IngestInferConfig,
    infer_column_roles,
    ingest_flexible_rows,
    ingest_flexible_rows_noted,
)
from app.m4_config import (
    AppM4ArtifactPaths,
    AppM4Config,
    AppM4SampleKnobs,
    AppM4Schema,
    AppM4SmokeCaps,
)
from app.ui_session import build_upload_payload_from_bytes
from src.data.adapt_config import load_adapt_m7_config
from src.data.schema import MASK_MISSING, MISSING_VALUE_SENTINEL

_TEMP = "temp_c"
_YEAR = "year"
_HOUR = "hour"
_TIME = "time"
_TEMP_IDX = 0
_HUMIDITY_IDX = 1
_NA_ROW = 1
_DASH_ROW = 1
_N_STEP_ROWS = 4
_SEQ_LEN = 8


def _tiny_app_config(tmp_path: Path) -> AppM4Config:
    """Build a tiny app config (artifact paths unused for Flexible preview).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temp directory.

    Returns
    -------
    AppM4Config
        Smoke-sized Flexible preview config.
    """
    return AppM4Config(
        paths=AppM4ArtifactPaths(
            checkpoint=tmp_path / "unused.eqx",
            scaler=tmp_path / "unused.npz",
            sample_csv=tmp_path / "unused.csv",
            real_config=tmp_path / "unused.yaml",
        ),
        schema=AppM4Schema(
            n_features=1,
            feature_names=(_TEMP,),
            seq_len=_SEQ_LEN,
        ),
        sample=AppM4SampleKnobs(
            k_samples=2,
            solver_steps=2,
            seed_mask=0,
            seed_sample=1,
        ),
        smoke=AppM4SmokeCaps(
            max_windows=None,
            k_samples=2,
            solver_steps=2,
            mcar_rates=(0.3,),
            block_len=4,
        ),
        artifact_dir=tmp_path / "artifacts",
    )


def test_widened_na_tokens_keep_feature_and_parse_missing() -> None:
    """Odd NA tokens must not drop a numeric column; cells become missing."""
    headers = (_TEMP, "humidity_pct")
    rows = [
        {_TEMP: "1.0", "humidity_pct": "10.0"},
        {_TEMP: "n/a", "humidity_pct": "-"},
        {_TEMP: "3.0", "humidity_pct": "?"},
        {_TEMP: "4.0", "humidity_pct": "missing"},
    ]
    frame = ingest_flexible_rows(rows, headers)
    assert frame.feature_names == headers
    assert frame.native_mask[_NA_ROW, _TEMP_IDX] == MASK_MISSING
    assert frame.values[_NA_ROW, _TEMP_IDX] == MISSING_VALUE_SENTINEL
    assert frame.native_mask[_DASH_ROW, _HUMIDITY_IDX] == MASK_MISSING


def test_calendar_names_excluded_from_feature_infer() -> None:
    """year/hour stay out of inferred features unless the user maps them."""
    headers = (_YEAR, _HOUR, _TEMP)
    rows = [
        {_YEAR: "2020", _HOUR: "0", _TEMP: "20.0"},
        {_YEAR: "2020", _HOUR: "1", _TEMP: "21.0"},
    ]
    role_map = infer_column_roles(headers, rows, IngestInferConfig())
    assert role_map.features == (_TEMP,)
    assert _YEAR not in role_map.features
    assert _HOUR not in role_map.features


def test_step_index_time_not_timestamp() -> None:
    """Monotone integer ``time`` is neither timestamp nor a feature."""
    headers = (_TIME, _TEMP)
    rows = [
        {_TIME: str(step), _TEMP: "1.0"}
        for step in range(_N_STEP_ROWS)
    ]
    role_map = infer_column_roles(headers, rows, IngestInferConfig())
    assert role_map.timestamp is None
    assert _TIME not in role_map.features
    assert role_map.features == (_TEMP,)


def test_iso_time_still_inferred_as_timestamp() -> None:
    """Real datetime strings under a time hint remain the timestamp."""
    headers = (_TIME, _TEMP)
    rows = [
        {_TIME: "2020-01-01T00:00:00", _TEMP: "1.0"},
        {_TIME: "2020-01-01T01:00:00", _TEMP: "2.0"},
    ]
    role_map = infer_column_roles(headers, rows, IngestInferConfig())
    assert role_map.timestamp == _TIME
    assert role_map.features == (_TEMP,)


def test_timestamp_with_empty_cell_not_inferred() -> None:
    """Empty timestamp cells fail the parser rules → not selected."""
    headers = (_TIME, _TEMP)
    rows = [
        {_TIME: "2020-01-01T00:00:00", _TEMP: "1.0"},
        {_TIME: "", _TEMP: "2.0"},
    ]
    _, notes = ingest_flexible_rows_noted(rows, headers)
    role_map = infer_column_roles(headers, rows, IngestInferConfig())
    assert role_map.timestamp is None
    assert _TIME not in role_map.features
    joined = " ".join(notes).casefold()
    assert "empty" in joined
    assert "row 2" in joined


def test_timestamp_na_token_not_inferred() -> None:
    """NA token in a time column rejects timestamp (and feature) use."""
    headers = (_TIME, _TEMP)
    rows = [
        {_TIME: "2020-01-01T00:00:00", _TEMP: "1.0"},
        {_TIME: "n/a", _TEMP: "2.0"},
    ]
    role_map = infer_column_roles(headers, rows, IngestInferConfig())
    assert role_map.timestamp is None
    assert _TIME not in role_map.features


def test_timestamp_rejects_today_and_signed_digits() -> None:
    """Bare today/now and signed digit cells are not timestamps."""
    headers = (_TIME, _TEMP)
    today_rows = [
        {_TIME: "today", _TEMP: "1.0"},
        {_TIME: "now", _TEMP: "2.0"},
    ]
    signed_rows = [
        {_TIME: "-5", _TEMP: "1.0"},
        {_TIME: "+0005", _TEMP: "2.0"},
    ]
    for rows in (today_rows, signed_rows):
        role_map = infer_column_roles(headers, rows, IngestInferConfig())
        assert role_map.timestamp is None
        assert _TIME not in role_map.features


def test_zero_feature_error_includes_exclude_notes() -> None:
    """When every column is excluded, the error lists why."""
    headers = (_YEAR, _HOUR)
    rows = [
        {_YEAR: "2020", _HOUR: "0"},
        {_YEAR: "2020", _HOUR: "1"},
    ]
    with pytest.raises(ValueError, match="year") as caught:
        infer_column_roles(headers, rows, IngestInferConfig())
    message = str(caught.value).casefold()
    assert "zero feature" in message
    assert "exclud" in message


def test_preview_warnings_list_excluded_columns(tmp_path: Path) -> None:
    """Done-when: preview warnings name each excluded / dropped column."""
    payload_bytes = (
        f"{_YEAR},{_TEMP}\n"
        "2020,1.0\n"
        "2021,2.0\n"
    ).encode("utf-8")
    payload = build_upload_payload_from_bytes(
        payload_bytes,
        "wp_b5.csv",
        _tiny_app_config(tmp_path),
    )
    assert _YEAR not in payload.preview.feature_names
    assert _TEMP in payload.preview.feature_names
    joined = " ".join(payload.preview.adapt.warnings).casefold()
    assert _YEAR in joined
    assert "exclud" in joined or "drop" in joined


def test_adapt_caps_display_is_flat_strings() -> None:
    """Caption helper is flat strings; nested AdaptLoopCapsView is gone."""
    repo = Path(__file__).resolve().parents[1]
    cfg = load_adapt_m7_config(repo / "configs" / "adapt_m7.yaml")
    caps = caps_display_from_config(cfg, "scratch")
    assert not hasattr(adapt_session_ui, "AdaptLoopCapsView")
    assert not hasattr(caps, "loop")
    lines = caps.as_lines()
    assert any("epochs=" in line for line in lines)
    assert "timeout" in caps.eta_note().casefold()


def test_load_adapt_config_without_base_scaler(tmp_path: Path) -> None:
    """adapt.base_scaler is removed; YAML without it still loads."""
    yaml_text = """
adapt:
  default_mode: scratch
  allow_fine_tune_when_f_matches: true
  require_feature_name_match_for_fine_tune: true
  base_checkpoint: artifacts/m2/checkpoints/best_score_net.eqx
  artifact_dir: artifacts/m7
  timeout_seconds: 60
  min_windows: 2
  train_fraction: 0.80
  smoke_mcar_rate: 0.30
window:
  seq_len: 8
  max_windows: 2
train:
  lr: 0.0003
  epochs: 1
  steps_per_epoch: 1
  batch_size: 2
  weight_decay: 0.0
  seed: 0
  t_eps: 1.0e-5
  obs_loss_weight: 0.0
  grad_clip_norm: 1.0
  mask_mode: random
sde:
  beta_min: 0.1
  beta_max: 20.0
  T_diff: 1.0
sample:
  K_samples: 2
  solver_steps: 2
seeds:
  mask: 0
  sample: 1
"""
    path = tmp_path / "adapt_no_base_scaler.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    cfg = load_adapt_m7_config(path)
    assert cfg.mode.default_mode == "scratch"
    assert not hasattr(cfg.mode, "base_scaler")
