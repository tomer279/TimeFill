"""WP1/WP2 hermetic tests: M7 adapt loop + CLI upload pipeline."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from app.adapt_upload_m7 import (
    AdaptUploadFlags,
    AdaptUploadPaths,
    adapt_upload_main,
    run_adapt_upload_pipeline,
)
from src.data.adapt_config import (
    AdaptCliOverrides,
    AdaptLoopCliOverrides,
    AdaptSampleSection,
    AdaptWindowCliOverrides,
)
from src.data.config import SdeConfig, TrainConfig
from src.data.normalize import FeatureScaler
from src.data.real.specs import default_beijing_features
from src.data.schema import MASK_MISSING
from src.train.adapt_m7 import (
    AdaptCaps,
    AdaptGeometry,
    AdaptInitPolicy,
    AdaptM7Config,
    AdaptModeSection,
    AdaptSessionRequest,
    AdaptSessionRunner,
    AdaptWindowTensors,
    load_adapt_m7_config,
    run_adapt_session,
    split_adapt_windows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPT_CONFIG_PATH = REPO_ROOT / "configs" / "adapt_m7.yaml"
TOY_NON_BEIJING = (
    REPO_ROOT / "tests" / "fixtures" / "toy_non_beijing.csv"
)

_SEQ = 8
_FEAT = 3
_N_WINDOWS = 5
_STATS_ATOL = 1e-5
_SEED = 0
_TRAIN_FRACTION = 0.80
_SMOKE_SHIFT = 1000.0
_EXIT_INGEST = 2
_CLI_SEQ_LEN = 4
_CLI_N_ROWS = 8
_BASE_TEMP_C = 20.0
_BASE_HUMIDITY = 40.0
_BASE_PRESSURE = 1013.0


def _tiny_config(tmp_path: Path) -> AdaptM7Config:
    """Build a CPU-tiny AdaptM7Config rooted at ``tmp_path``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    AdaptM7Config
        Validated smoke-cap config with 1 epoch / 1 step.
    """
    train_map = {
        "lr": 0.0003,
        "epochs": 1,
        "steps_per_epoch": 1,
        "weight_decay": 0.0,
        "seed": _SEED,
        "t_eps": 1.0e-5,
        "obs_loss_weight": 0.0,
        "grad_clip_norm": 1.0,
        "mask_mode": "random",
        "artifact_dir": str(tmp_path / "m7"),
        "val_every": 1,
        "log_every": 1,
    }
    config = AdaptM7Config(
        caps=AdaptCaps(
            artifact_dir=str(tmp_path / "m7"),
            timeout_seconds=120.0,
            min_windows=2,
            train_fraction=_TRAIN_FRACTION,
            smoke_mcar_rate=0.30,
        ),
        mode=AdaptModeSection(
            default_mode="scratch",
            allow_fine_tune_when_f_matches=True,
            require_feature_name_match_for_fine_tune=True,
            base_checkpoint="artifacts/m2/checkpoints/best_score_net.eqx",
        ),
        geometry=AdaptGeometry(
            seq_len=_SEQ,
            batch_size=2,
            block_len=2,
            max_windows=4,
        ),
        train=TrainConfig(train_map),
        sde=SdeConfig({"beta_min": 0.1, "beta_max": 20.0, "T_diff": 1.0}),
        sample=AdaptSampleSection(
            k_samples=2,
            solver_steps=4,
            seed_mask=0,
            seed_sample=1,
        ),
    )
    config.validate()
    return config


def _synthetic_windows(
        n_windows: int = _N_WINDOWS,
        n_features: int = _FEAT,
        smoke_shift: float = 0.0,
        with_native_gap: bool = False,
) -> AdaptWindowTensors:
    """Build synthetic windows with optional smoke-only value shift.

    Parameters
    ----------
    n_windows : int, optional
        Batch size ``B``.
    n_features : int, optional
        Feature count ``F``.
    smoke_shift : float, optional
        If nonzero, overwrite smoke windows with this constant.
    with_native_gap : bool, optional
        If true, mark one cell natively missing.

    Returns
    -------
    AdaptWindowTensors
        Synthetic upload windows.
    """
    values = np.full(
        (n_windows, _SEQ, n_features),
        2.0,
        dtype=np.float32,
    )
    masks = np.ones_like(values, dtype=np.float32)
    if n_windows >= 2 and smoke_shift != 0.0:
        split = split_adapt_windows(n_windows, _TRAIN_FRACTION)
        values[split.smoke_idx] = smoke_shift
    if with_native_gap:
        values[:, 0, 0] = 0.0
        masks[:, 0, 0] = MASK_MISSING
    names = tuple(f"f{i}" for i in range(n_features))
    return AdaptWindowTensors(
        values=values,
        native_masks=masks,
        feature_names=names,
    )


def _write_toy_csv(path: Path, n_rows: int = _CLI_N_ROWS) -> Path:
    """Write a tiny non-Beijing CSV with hourly rows.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV path.
    n_rows : int, optional
        Number of hourly data rows.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    lines = ["time,sensor_id,temp_c,humidity_pct,pressure_hpa"]
    for hour in range(n_rows):
        temp_c = _BASE_TEMP_C + float(hour)
        humidity = _BASE_HUMIDITY + float(hour)
        lines.append(
            f"2020-01-01T{hour:02d}:00:00,lab-A,"
            f"{temp_c},{humidity},{_BASE_PRESSURE}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _aggressive_cli_overrides() -> AdaptCliOverrides:
    """Return CPU-tiny CLI overrides for hermetic pipeline tests.

    Returns
    -------
    AdaptCliOverrides
        Aggressive smoke caps.
    """
    return AdaptCliOverrides(
        window=AdaptWindowCliOverrides(
            seq_len=_CLI_SEQ_LEN,
            max_windows=4,
            batch_size=2,
        ),
        loop=AdaptLoopCliOverrides(
            epochs=1,
            steps_per_epoch=1,
            k_samples=2,
            solver_steps=4,
            timeout_seconds=120.0,
        ),
    )


def test_load_adapt_m7_yaml_smoke_caps() -> None:
    """Locked YAML loads with ASSUMPTIONS smoke caps."""
    cfg = load_adapt_m7_config(ADAPT_CONFIG_PATH)
    assert cfg.train.schedule.epochs == 5
    assert cfg.train.schedule.steps_per_epoch == 20
    assert cfg.geometry.batch_size == 4
    assert cfg.geometry.max_windows == 4
    assert cfg.caps.train_fraction == pytest.approx(0.80)
    assert cfg.caps.min_windows == 2
    assert cfg.mode.default_mode == "scratch"
    assert cfg.sample.k_samples == 8
    assert cfg.sample.solver_steps == 50


def test_split_deterministic_keeps_smoke_out() -> None:
    """80/20 index split is contiguous and reserves smoke when B>=2."""
    split = split_adapt_windows(5, 0.80)
    assert split.n_train() == 4
    assert split.n_smoke() == 1
    assert np.array_equal(split.train_idx, np.arange(4))
    assert np.array_equal(split.smoke_idx, np.array([4]))


def test_scaler_ignores_smoke_held_out_cells() -> None:
    """Scaler fit on train must ignore extreme smoke-only values."""
    windows = _synthetic_windows(smoke_shift=_SMOKE_SHIFT)
    split = split_adapt_windows(windows.n_windows(), _TRAIN_FRACTION)
    runner = AdaptSessionRunner()
    scaler, _pool = runner.build_scaled_pool(windows, split)
    assert bool(jnp.allclose(scaler.mean, 2.0, atol=_STATS_ATOL))
    global_scaler = FeatureScaler().fit(windows.values, windows.native_masks)
    assert float(jnp.max(jnp.abs(global_scaler.mean - 2.0))) > 1.0


def test_fine_tune_refuses_f_mismatch(tmp_path: Path) -> None:
    """Fine-tune with F != base F raises a clear refusal."""
    windows = _synthetic_windows(n_features=3)
    request = AdaptSessionRequest(
        windows=windows,
        config=_tiny_config(tmp_path),
        policy=AdaptInitPolicy(
            mode="fine_tune",
            base_checkpoint=tmp_path / "missing.eqx",
            base_feature_names=default_beijing_features(),
        ),
        artifact_root=tmp_path / "m7",
        run_id="f_mismatch",
    )
    with pytest.raises(ValueError, match="Fine-tune refused: upload F="):
        run_adapt_session(request)
    assert not (tmp_path / "m7" / "f_mismatch" / "best_score_net.eqx").exists()


def test_adapt_scratch_writes_session_artifacts(tmp_path: Path) -> None:
    """Tiny synthetic windows complete >=1 step and promote m7 artifacts."""
    windows = _synthetic_windows(with_native_gap=True)
    request = AdaptSessionRequest(
        windows=windows,
        config=_tiny_config(tmp_path),
        policy=AdaptInitPolicy(mode="scratch"),
        artifact_root=tmp_path / "m7",
        run_id="scratch_smoke",
    )
    result = run_adapt_session(request)
    assert result.checkpoint_path().is_file()
    assert result.scaler_path().is_file()
    meta_path = result.run_dir / "adapt_meta.json"
    assert meta_path.is_file()
    text = meta_path.read_text(encoding="utf-8")
    assert "scratch" in text
    assert "success" in text
    assert int(result.meta["F"]) == _FEAT
    assert result.model is not None
    assert "artifacts/m2" not in str(result.run_dir).replace("\\", "/")


def test_min_windows_rejects_without_artifacts(tmp_path: Path) -> None:
    """B < min_windows does not create a promoted checkpoint."""
    request = AdaptSessionRequest(
        windows=AdaptWindowTensors(
            values=np.ones((1, _SEQ, _FEAT), dtype=np.float32),
            native_masks=np.ones((1, _SEQ, _FEAT), dtype=np.float32),
            feature_names=tuple(f"f{i}" for i in range(_FEAT)),
        ),
        config=_tiny_config(tmp_path),
        policy=AdaptInitPolicy(mode="scratch"),
        artifact_root=tmp_path / "m7",
        run_id="too_few",
    )
    with pytest.raises(ValueError, match="too few windows"):
        run_adapt_session(request)
    assert not list(tmp_path.glob("**/best_score_net.eqx"))


def test_cli_pipeline_writes_mean_fill(tmp_path: Path) -> None:
    """Aggressive-cap CLI path: ingest → adapt → mean_fill.csv."""
    csv_path = _write_toy_csv(tmp_path / "toy.csv", n_rows=_CLI_N_ROWS)
    paths = AdaptUploadPaths(
        csv_path=csv_path,
        config_path=ADAPT_CONFIG_PATH,
        map_path=None,
        artifact_root=tmp_path / "m7",
    )
    flags = AdaptUploadFlags(
        smoke_eval=True,
        export_samples=True,
        mode="scratch",
    )
    result = run_adapt_upload_pipeline(
        paths,
        flags,
        _aggressive_cli_overrides(),
    )
    assert result.mean_fill_path().is_file()
    assert result.samples_path is not None and result.samples_path.is_file()
    assert result.session.checkpoint_path().is_file()
    assert "artifacts/m2" not in str(result.run_dir()).replace("\\", "/")
    text = result.mean_fill_path().read_text(encoding="utf-8")
    assert "temp_c" in text
    assert "window_id" in text


def test_cli_exit_too_few_windows(tmp_path: Path) -> None:
    """Default seq_len on short series → ingest reject exit code 2."""
    csv_path = _write_toy_csv(tmp_path / "short.csv", n_rows=4)
    code = adapt_upload_main(
        [
            "--csv",
            str(csv_path),
            "--config",
            str(ADAPT_CONFIG_PATH),
            "--artifact-dir",
            str(tmp_path / "m7"),
            "--seq-len",
            "64",
            "--epochs",
            "1",
            "--steps-per-epoch",
            "1",
        ]
    )
    assert code == _EXIT_INGEST
    assert not list(tmp_path.glob("**/best_score_net.eqx"))


def test_cli_toy_fixture_with_small_seq_len(tmp_path: Path) -> None:
    """Bundled toy_non_beijing.csv completes with --seq-len 2."""
    assert TOY_NON_BEIJING.is_file()
    paths = AdaptUploadPaths(
        csv_path=TOY_NON_BEIJING,
        config_path=ADAPT_CONFIG_PATH,
        map_path=None,
        artifact_root=tmp_path / "m7",
    )
    flags = AdaptUploadFlags(
        smoke_eval=False,
        export_samples=False,
        mode="scratch",
    )
    overrides = AdaptCliOverrides(
        window=AdaptWindowCliOverrides(
            seq_len=2,
            max_windows=4,
            batch_size=2,
        ),
        loop=AdaptLoopCliOverrides(
            epochs=1,
            steps_per_epoch=1,
            k_samples=2,
            solver_steps=4,
            timeout_seconds=120.0,
        ),
    )
    result = run_adapt_upload_pipeline(paths, flags, overrides)
    assert result.mean_fill_path().is_file()
    assert result.session.checkpoint_path().is_file()
