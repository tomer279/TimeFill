"""WP-B3 honest-export Done-when tests (red until the fix lands).

Functions
---------
test_adapted_windows_exposes_pad_provenance
    ``valid_len`` / ``row_index`` distinguish pads from native gaps.
test_native_gap_on_last_real_row_keeps_row_index
    Last real row can be a native gap; pad slots use ``row_index == -1``.
test_max_windows_marks_partial_coverage
    Cap sets partial metadata so the UI can warn.
test_mean_fill_export_has_exactly_input_rows
    100-row input / ``seq_len=64`` → 100 download rows (no pads).
test_mean_fill_export_timestamps_entity_and_imputed_flags
    Timestamp / entity round-trip; ``*_imputed`` marks native gaps only.
test_samples_long_export_drops_pad_rows
    Long export drops pads, keeps ``row_index``, unique (row, sample, feature).
test_samples_long_freezes_native_observed_values
    Long export overwrites native-observed draws with window values.
test_impute_mean_fill_preserves_native_under_demo_mask
    Demo ``m_eval`` must not overwrite ``m_native == 1`` in the download.
test_partial_download_filename_and_warning
    Partial cap → filename mark + warning text naming filled vs total rows.
test_cli_partial_mean_fill_out_name
    CLI mean-fill filename marks partial coverage.
test_write_mean_fill_partial_file_and_stderr
    Capped CLI write: partial file on disk + stderr warning.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from app.export_frames import (
    MeanFillExportRequest,
    SampleExportRequest,
    mean_fill_dataframe,
    samples_long_dataframe,
)
from app.m4_config import AppM4SampleKnobs
from app.schema_validate import ValidatedFrame, default_app_schema
from app.session_impute import (
    DemoMaskRequest,
    ImputeTensors,
    SessionImputer,
    SessionImputeRequest,
)
from app.session_impute import _PreparedObservations
from app.window_adapter import (
    AdaptedWindows,
    WindowAdaptRequest,
    adapt_validated_frame,
)
from src.data.schema import MASK_MISSING, MASK_OBSERVED, MISSING_VALUE_SENTINEL

_FEATURES = default_app_schema().feature_names
_N_FEATURES = len(_FEATURES)
_SEQ_LEN_SHORT = 4
_SEQ_LEN_EXPORT = 64
_N_INPUT_ROWS = 100
_PAD_ROWS = _SEQ_LEN_EXPORT * 2 - _N_INPUT_ROWS
_ENTITY = "DemoSite"
_MASK_TOL = 1e-6
_DEMO_RATE = 0.5
_DEMO_SEED = 0
_FAKE_MEAN_OFFSET = 100.0
_N_FULL_WINDOWS = 3
_MAX_WINDOWS_CAP = 1
_EXPORT_FEAT_NAMES = ("PM2.5", "PM10")
_EXPORT_N_FEAT = len(_EXPORT_FEAT_NAMES)
_LONG_K = 2
_PAD_ROW_INDEX = -1


def _frame_from_matrix(
        values: np.ndarray,
        native_mask: np.ndarray,
        timestamps: np.ndarray | None = None,
        entity_id: str | None = _ENTITY,
) -> ValidatedFrame:
    """Build a ``ValidatedFrame`` around pre-made arrays.

    Parameters
    ----------
    values : numpy.ndarray
        Float32 ``(L, F)`` values.
    native_mask : numpy.ndarray
        Float32 ``(L, F)`` masks.
    timestamps : numpy.ndarray or None
        Optional length-``L`` timestamps.
    entity_id : str or None
        Optional entity label.

    Returns
    -------
    ValidatedFrame
        Frame ready for ``adapt_validated_frame``.
    """
    return ValidatedFrame(
        values=values.astype(np.float32),
        native_mask=native_mask.astype(np.float32),
        feature_names=_FEATURES,
        timestamps=timestamps,
        entity_id=entity_id,
    )


def _ones_frame(n_rows: int) -> ValidatedFrame:
    """Build an all-observed constant frame of length ``n_rows``.

    Parameters
    ----------
    n_rows : int
        Series length ``L``.

    Returns
    -------
    ValidatedFrame
        All-ones values / masks.
    """
    values = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    mask = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    return _frame_from_matrix(values, mask)


def test_adapted_windows_exposes_pad_provenance() -> None:
    """Short series: ``valid_len`` and ``row_index`` mark right-pad cells."""
    n_rows = 3
    adapted = adapt_validated_frame(
        _ones_frame(n_rows),
        WindowAdaptRequest(seq_len=_SEQ_LEN_SHORT, max_windows=None),
    )
    assert adapted.valid_len.shape == (1,)
    assert int(adapted.valid_len[0]) == n_rows
    assert adapted.row_index.shape == (1, _SEQ_LEN_SHORT)
    assert np.array_equal(
        adapted.row_index[0, :n_rows],
        np.arange(n_rows),
    )
    assert np.all(adapted.row_index[0, n_rows:] == _PAD_ROW_INDEX)


def test_native_gap_on_last_real_row_keeps_row_index() -> None:
    """Native gap on the last real row stays a real index; pads are ``-1``."""
    n_rows = 3
    values = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    mask = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    last_row = n_rows - 1
    values[last_row, 0] = MISSING_VALUE_SENTINEL
    mask[last_row, 0] = MASK_MISSING
    adapted = adapt_validated_frame(
        _frame_from_matrix(values, mask),
        WindowAdaptRequest(seq_len=_SEQ_LEN_SHORT, max_windows=None),
    )
    assert adapted.native_mask[0, last_row, 0] == MASK_MISSING
    assert int(adapted.row_index[0, last_row]) == last_row
    assert np.all(adapted.row_index[0, n_rows:] == _PAD_ROW_INDEX)


def test_max_windows_marks_partial_coverage() -> None:
    """``max_windows`` keeps leading windows and marks the export partial."""
    n_rows = _SEQ_LEN_SHORT * _N_FULL_WINDOWS
    adapted = adapt_validated_frame(
        _ones_frame(n_rows),
        WindowAdaptRequest(
            seq_len=_SEQ_LEN_SHORT,
            max_windows=_MAX_WINDOWS_CAP,
        ),
    )
    assert adapted.n_windows() == _MAX_WINDOWS_CAP
    assert adapted.is_partial() is True
    assert int(adapted.uncapped_n_windows) == _N_FULL_WINDOWS
    assert int(adapted.source_n_rows) == n_rows


def test_mean_fill_export_has_exactly_input_rows() -> None:
    """100-row Exact series / ``T=64`` downloads as 100 rows (pads dropped)."""
    adapted = adapt_validated_frame(
        _ones_frame(_N_INPUT_ROWS),
        WindowAdaptRequest(seq_len=_SEQ_LEN_EXPORT, max_windows=None),
    )
    assert adapted.n_windows() == 2
    assert int(adapted.valid_len.sum()) == _N_INPUT_ROWS
    frame = mean_fill_dataframe(
        MeanFillExportRequest(
            mean_fill=adapted.values,
            feature_names=adapted.feature_names,
            m_native=adapted.native_mask,
            valid_len=adapted.valid_len,
            row_index=adapted.row_index,
        )
    )
    assert len(frame) == _N_INPUT_ROWS
    assert len(frame) != adapted.n_windows() * _SEQ_LEN_EXPORT
    assert _PAD_ROWS > 0


def test_mean_fill_export_timestamps_entity_and_imputed_flags() -> None:
    """Download carries timestamp / entity and ``*_imputed`` for native gaps."""
    n_rows = 3
    values = np.arange(n_rows * _N_FEATURES, dtype=np.float32).reshape(
        n_rows,
        _N_FEATURES,
    )
    mask = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    gap_feat = 0
    values[1, gap_feat] = MISSING_VALUE_SENTINEL
    mask[1, gap_feat] = MASK_MISSING
    stamps = np.arange(
        "2020-01-01",
        "2020-01-01T03",
        dtype="datetime64[h]",
    )
    adapted = adapt_validated_frame(
        _frame_from_matrix(values, mask, timestamps=stamps),
        WindowAdaptRequest(seq_len=_SEQ_LEN_SHORT, max_windows=None),
    )
    filled = adapted.values.copy()
    filled[0, 1, gap_feat] = 42.0
    frame = mean_fill_dataframe(
        MeanFillExportRequest(
            mean_fill=filled,
            feature_names=adapted.feature_names,
            m_native=adapted.native_mask,
            valid_len=adapted.valid_len,
            row_index=adapted.row_index,
            timestamps=adapted.timestamps,
            entity=adapted.entity_ids[0],
        )
    )
    assert len(frame) == n_rows
    assert "timestamp" in frame.columns
    assert "entity" in frame.columns
    assert list(frame["entity"].unique()) == [_ENTITY]
    imputed_col = f"{_FEATURES[gap_feat]}_imputed"
    assert imputed_col in frame.columns
    assert bool(frame.loc[1, imputed_col]) is True
    assert bool(frame.loc[0, imputed_col]) is False
    assert np.array_equal(
        frame["timestamp"].to_numpy(dtype="datetime64[h]"),
        stamps,
    )


def test_samples_long_export_drops_pad_rows() -> None:
    """Long export omits pad cells and carries ``row_index``."""
    from app.export_frames import sample_request_from_windows

    n_rows = 3
    adapted = adapt_validated_frame(
        _ones_frame(n_rows),
        WindowAdaptRequest(seq_len=_SEQ_LEN_SHORT, max_windows=None),
    )
    samples = np.zeros(
        (adapted.n_windows(), _LONG_K, _SEQ_LEN_SHORT, _N_FEATURES),
        dtype=np.float32,
    )
    frame = samples_long_dataframe(
        sample_request_from_windows(samples, adapted)
    )
    n_real_cells = n_rows * _N_FEATURES * _LONG_K
    assert len(frame) == n_real_cells
    assert "row_index" in frame.columns
    assert "is_imputed" in frame.columns
    assert set(frame["row_index"].tolist()) == set(range(n_rows))
    grouped = frame.groupby(
        ["row_index", "sample_id", "feature"],
        sort=False,
    ).size()
    assert int(grouped.max()) == 1
    assert int(grouped.min()) == 1


def test_samples_long_freezes_native_observed_values() -> None:
    """Long export overwrites ``m_native == 1`` draws with native values."""
    from app.export_frames import sample_request_from_windows

    n_rows = 3
    values = np.arange(
        1,
        n_rows * _EXPORT_N_FEAT + 1,
        dtype=np.float32,
    ).reshape(n_rows, _EXPORT_N_FEAT)
    mask = np.ones((n_rows, _EXPORT_N_FEAT), dtype=np.float32)
    frame = ValidatedFrame(
        values=values,
        native_mask=mask,
        feature_names=_EXPORT_FEAT_NAMES,
        timestamps=None,
        entity_id=_ENTITY,
    )
    adapted = adapt_validated_frame(
        frame,
        WindowAdaptRequest(seq_len=_SEQ_LEN_SHORT, max_windows=None),
    )
    fake_draw = 99.0
    samples = np.full(
        (adapted.n_windows(), _LONG_K, _SEQ_LEN_SHORT, _EXPORT_N_FEAT),
        fake_draw,
        dtype=np.float32,
    )
    long_frame = samples_long_dataframe(
        sample_request_from_windows(samples, adapted)
    )
    observed = long_frame.loc[~long_frame["is_imputed"]]
    assert len(observed) == n_rows * _EXPORT_N_FEAT * _LONG_K
    assert not bool((observed["value"] == fake_draw).any())


def test_impute_mean_fill_preserves_native_under_demo_mask(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B4b: download merge keeps every ``m_native == 1`` cell under demo gaps."""
    seq_len = _SEQ_LEN_SHORT
    values = np.arange(
        1,
        seq_len * _EXPORT_N_FEAT + 1,
        dtype=np.float32,
    ).reshape(1, seq_len, _EXPORT_N_FEAT)
    m_native = np.ones((1, seq_len, _EXPORT_N_FEAT), dtype=np.float32)
    windows = AdaptedWindows(
        values=values,
        native_mask=m_native,
        feature_names=_EXPORT_FEAT_NAMES,
        entity_ids=np.array([_ENTITY], dtype=object),
    )
    fake_mean = values + _FAKE_MEAN_OFFSET

    def _fake_prepare(
            _self: SessionImputer,
            request: SessionImputeRequest,
    ) -> _PreparedObservations:
        vals = np.asarray(request.windows.values, dtype=np.float32)
        native = np.asarray(request.windows.native_mask, dtype=np.float32)
        _x_obs, m_eval, _m_art = request.demo_mask.compose(
            jnp.asarray(vals),
            jnp.asarray(native),
        )
        return _PreparedObservations(
            observations=vals,
            m_native=native,
            m_eval=np.asarray(m_eval, dtype=np.float32),
            x_obs_n=jnp.asarray(vals),
        )

    def _fake_restore(
            _self: SessionImputer,
            prepared: _PreparedObservations,
            _plan: object,
    ) -> ImputeTensors:
        quantiles = np.stack([fake_mean, fake_mean], axis=-1)
        samples = fake_mean[:, None, :, :]
        return ImputeTensors(
            mean=fake_mean,
            quantiles=quantiles,
            samples=samples,
        )

    monkeypatch.setattr(
        SessionImputer,
        "_prepare_observations",
        _fake_prepare,
    )
    monkeypatch.setattr(
        SessionImputer,
        "_impute_and_restore",
        _fake_restore,
    )
    imputer = object.__new__(SessionImputer)
    imputer.sample = AppM4SampleKnobs(
        k_samples=1,
        solver_steps=1,
        seed_mask=_DEMO_SEED,
        seed_sample=1,
    )
    result = imputer.impute(
        SessionImputeRequest(
            windows=windows,
            demo_mask=DemoMaskRequest(
                mode="random",
                rate=_DEMO_RATE,
                seed=_DEMO_SEED,
            ),
        )
    )
    native_obs = np.abs(result.context.m_native - MASK_OBSERVED) < _MASK_TOL
    demo_hidden = native_obs & (
        np.abs(result.context.m_eval - MASK_OBSERVED) >= _MASK_TOL
    )
    assert bool(np.any(demo_hidden))
    assert np.allclose(
        result.context.mean_fill[native_obs],
        result.context.observations[native_obs],
        rtol=0.0,
        atol=0.0,
    )


def test_partial_download_filename_and_warning() -> None:
    """Partial window cap names the file and warns with filled vs total hours."""
    # Imported inside the test so collection still runs when helpers are absent.
    from app.export_frames import (
        mean_fill_download_filename,
        truncation_warning_message,
    )

    n_rows = _SEQ_LEN_SHORT * _N_FULL_WINDOWS
    adapted = adapt_validated_frame(
        _ones_frame(n_rows),
        WindowAdaptRequest(
            seq_len=_SEQ_LEN_SHORT,
            max_windows=_MAX_WINDOWS_CAP,
        ),
    )
    assert adapted.is_partial() is True
    filled_hours = int(adapted.valid_len.sum())
    total_hours = int(adapted.source_n_rows)
    filename = mean_fill_download_filename(
        is_partial=True,
        row_start=0,
        row_end=filled_hours - 1,
    )
    assert "partial" in filename
    assert f"0-{filled_hours - 1}" in filename
    warning = truncation_warning_message(
        filled_hours,
        total_hours,
        "window.max_windows",
    )
    assert str(filled_hours) in warning
    assert str(total_hours) in warning
    assert "window.max_windows" in warning
    assert "rows" in warning


def test_cli_partial_mean_fill_out_name() -> None:
    """CLI mean-fill filename marks partial coverage under ``max_windows``."""
    from app.adapt_upload_m7 import _mean_fill_out_name

    n_rows = _SEQ_LEN_SHORT * _N_FULL_WINDOWS
    adapted = adapt_validated_frame(
        _ones_frame(n_rows),
        WindowAdaptRequest(
            seq_len=_SEQ_LEN_SHORT,
            max_windows=_MAX_WINDOWS_CAP,
        ),
    )
    name = _mean_fill_out_name(adapted)
    assert "partial" in name
    assert name.startswith("mean_fill_partial_rows_")


def test_write_mean_fill_partial_file_and_stderr(
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
) -> None:
    """Capped CLI write uses a partial filename and warns on stderr."""
    from app.adapt_upload_m7 import _write_mean_fill

    n_rows = _SEQ_LEN_SHORT * _N_FULL_WINDOWS
    adapted = adapt_validated_frame(
        _ones_frame(n_rows),
        WindowAdaptRequest(
            seq_len=_SEQ_LEN_SHORT,
            max_windows=_MAX_WINDOWS_CAP,
        ),
    )

    class _Session:
        """Minimal stand-in with ``run_dir`` for ``_write_mean_fill``."""

        def __init__(self, run_dir: Path) -> None:
            self.run_dir = run_dir

    out_path = _write_mean_fill(
        _Session(tmp_path),  # type: ignore[arg-type]
        adapted,
        adapted.values,
    )
    assert out_path.is_file()
    assert "partial" in out_path.name
    err = capsys.readouterr().err
    assert "window.max_windows" in err
    assert "rows" in err
