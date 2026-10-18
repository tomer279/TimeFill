"""Hermetic tests for M7 Streamlit adapt helpers (no Streamlit)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.adapt_session_ui import (
    AdaptCapsDisplay,
    AdaptLoopCapsView,
    AdaptRunRequest,
    caps_display_from_config,
    format_adapt_failure,
    make_upload_fingerprint,
    prepare_adapt_windows,
    run_ui_adapt,
)
from app.preview_ingest import MODE_FLEXIBLE, evaluate_impute_gate
from app.schema_validate import ValidatedFrame
from app.window_adapter import AdaptedWindows
from src.data.adapt_config import load_adapt_m7_config
from src.data.real.specs import default_beijing_features

REPO = Path(__file__).resolve().parents[1]
ADAPT_CFG = REPO / "configs" / "adapt_m7.yaml"
_FLEX = ("temp_c", "humidity_pct", "pressure_hpa")
_SEQ = 4
_FEAT = 3
_B = 2
_N_ROWS = 8


def _dense_frame(n_rows: int) -> ValidatedFrame:
    """Build a fully observed hermetic frame.

    Parameters
    ----------
    n_rows : int
        Grid length ``L``.

    Returns
    -------
    ValidatedFrame
        Dense float32 values / masks.
    """
    values = np.ones((n_rows, _FEAT), dtype=np.float32)
    mask = np.ones((n_rows, _FEAT), dtype=np.float32)
    return ValidatedFrame(
        values=values,
        native_mask=mask,
        feature_names=_FLEX,
        timestamps=None,
        entity_id="lab-A",
    )


def _windows() -> AdaptedWindows:
    """Build a tiny adapted batch for mocked adapt runs.

    Returns
    -------
    AdaptedWindows
        Synthetic windows.
    """
    return AdaptedWindows(
        values=np.ones((_B, _SEQ, _FEAT), dtype=np.float32),
        native_mask=np.ones((_B, _SEQ, _FEAT), dtype=np.float32),
        feature_names=_FLEX,
        entity_ids=np.array(["lab-A", "lab-A"], dtype=object),
    )


def test_fingerprint_changes_with_features() -> None:
    """Fingerprint must change when mapped feature names change."""
    first = make_upload_fingerprint("u.csv", MODE_FLEXIBLE, _FLEX)
    second = make_upload_fingerprint(
        "u.csv",
        MODE_FLEXIBLE,
        ("x", "y", "z"),
    )
    assert first != second


def test_caps_display_nested_under_attr_limit() -> None:
    """AdaptCapsDisplay nests loop caps and stays within attr budget."""
    cfg = load_adapt_m7_config(ADAPT_CFG)
    caps = caps_display_from_config(cfg, "scratch")
    assert isinstance(caps, AdaptCapsDisplay)
    assert isinstance(caps.loop, AdaptLoopCapsView)
    assert len(caps.__dataclass_fields__) <= 7
    assert len(caps.loop.__dataclass_fields__) <= 7
    lines = caps.as_lines()
    assert any(line.startswith("epochs=") for line in lines)
    assert "timeout" in caps.eta_note().lower()


def test_prepare_windows_with_small_seq_len() -> None:
    """Short dense frames adapt when seq_len is lowered for smoke."""
    frame = _dense_frame(_N_ROWS)
    windows, err = prepare_adapt_windows(
        frame,
        seq_len=_SEQ,
        max_windows=4,
    )
    assert err is None
    assert windows is not None
    assert windows.n_windows() >= 2


def test_format_too_little_and_timeout() -> None:
    """Failure mapper covers too-little-data, timeout, and NaN kinds."""
    too_little = format_adapt_failure(
        ValueError("too few windows for adapt: n_windows=1")
    )
    assert too_little.kind == "too_little_data"
    timed_out = format_adapt_failure(RuntimeError("adapt timeout exceeded"))
    assert timed_out.is_timeout()
    nan_loss = format_adapt_failure(RuntimeError("non-finite train loss"))
    assert nan_loss.kind == "nan_loss"


def test_run_ui_adapt_maps_value_error(tmp_path: Path) -> None:
    """UI adapt maps ValueError to a user-facing failure (no bind)."""
    cfg = load_adapt_m7_config(ADAPT_CFG)
    request = AdaptRunRequest(
        windows=_windows(),
        config=cfg,
        mode="scratch",
        fingerprint="fp",
        artifact_root=tmp_path / "m7",
    )
    with patch(
            "app.adapt_session_ui.run_adapt_session",
            side_effect=ValueError(
                "too few windows for adapt: n_windows=1"
            ),
    ):
        outcome = run_ui_adapt(request)
    assert not outcome.ok()
    assert outcome.failure is not None
    assert outcome.failure.kind == "too_little_data"


def test_gate_refresh_unlocks_flexible() -> None:
    """Session feature match unlocks Flexible impute in the gate helper."""
    gate = evaluate_impute_gate(
        _FLEX,
        default_beijing_features(),
        MODE_FLEXIBLE,
        session_features=_FLEX,
    )
    assert gate.is_allowed()
