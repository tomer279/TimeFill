"""WP8 Done-when: adapt smoke_only honesty + M7.5 flag opt-in.

Tests
-----
test_budget_status_smoke_only
    ``adapt_budget_status`` maps step budget to success / smoke_only.
test_adapt_meta_smoke_only_still_promotes
    Under ``min_adapt_steps``, meta is smoke_only but artifacts promote.
test_adapt_meta_success_when_budget_met
    Meeting the budget keeps ``status: success``.
test_product_yaml_sets_min_adapt_steps
    ``configs/adapt_m7.yaml`` opts into ``min_adapt_steps: 200``.
test_adapt_m7_5_flags_on_toy
    Opt-in YAML on the toy fixture records flags, smoke_only, Fill wiring.
test_smoke_only_warning_text
    UI helper warns only for ``smoke_only`` status.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from app.adapt_session_ui import imputer_from_adapt_result
from app.adapt_upload_m7 import (
    AdaptUploadFlags,
    AdaptUploadPaths,
    run_adapt_upload_pipeline,
)
from src.data.adapt_config import (
    AdaptCaps,
    AdaptCliOverrides,
    AdaptGeometry,
    AdaptLoopCliOverrides,
    AdaptM7Config,
    AdaptModeSection,
    AdaptSampleSection,
    AdaptWindowCliOverrides,
    apply_adapt_cli_overrides,
    load_adapt_m7_config,
)
from src.data.config import SdeConfig, TrainConfig
from src.train.adapt_m7 import (
    AdaptInitPolicy,
    AdaptSessionRequest,
    AdaptWindowTensors,
    run_adapt_session,
)

_REPO = Path(__file__).resolve().parents[1]
_ADAPT_YAML = _REPO / "configs" / "adapt_m7.yaml"
_ADAPT_M75_YAML = _REPO / "configs" / "adapt_m7_5.yaml"
_TOY_CSV = _REPO / "tests" / "fixtures" / "toy_non_beijing.csv"

_SEQ = 8
_FEAT = 3
_N_WINDOWS = 5
_SEED = 0
_TRAIN_FRACTION = 0.80
_MIN_ADAPT_STEPS = 200
_STATUS_SMOKE = "smoke_only"
_STATUS_SUCCESS = "success"
_TOY_SEQ_LEN = 2


def _synthetic_windows() -> AdaptWindowTensors:
    """Build dense synthetic upload windows for short adapt runs.

    Returns
    -------
    AdaptWindowTensors
        ``(B, T, F)`` batch with all cells observed.
    """
    values = np.full(
        (_N_WINDOWS, _SEQ, _FEAT),
        2.0,
        dtype=np.float32,
    )
    masks = np.ones_like(values, dtype=np.float32)
    names = tuple(f"f{i}" for i in range(_FEAT))
    return AdaptWindowTensors(
        values=values,
        native_masks=masks,
        feature_names=names,
    )


def _tiny_config(
        tmp_path: Path,
        *,
        epochs: int,
        steps_per_epoch: int,
        min_adapt_steps: int | None,
) -> AdaptM7Config:
    """Build a CPU-tiny adapt config with an explicit adapt-step budget.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    epochs : int
        ``train.epochs``.
    steps_per_epoch : int
        ``train.steps_per_epoch``.
    min_adapt_steps : int or None
        Honesty threshold (``None`` disables the check).

    Returns
    -------
    AdaptM7Config
        Validated config rooted under ``tmp_path``.
    """
    train_map = {
        "lr": 0.0003,
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
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
            min_adapt_steps=min_adapt_steps,
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


def test_budget_status_smoke_only() -> None:
    """Step budget helper returns smoke_only / success / disabled."""
    from src.data.adapt_config import adapt_budget_status

    assert adapt_budget_status(1, 1, _MIN_ADAPT_STEPS) == _STATUS_SMOKE
    assert adapt_budget_status(20, 20, _MIN_ADAPT_STEPS) == _STATUS_SUCCESS
    assert adapt_budget_status(1, 1, None) == _STATUS_SUCCESS


def test_adapt_meta_smoke_only_still_promotes(tmp_path: Path) -> None:
    """Under-budget adapt promotes artifacts and marks smoke_only."""
    config = _tiny_config(
        tmp_path,
        epochs=1,
        steps_per_epoch=1,
        min_adapt_steps=_MIN_ADAPT_STEPS,
    )
    result = run_adapt_session(
        AdaptSessionRequest(
            windows=_synthetic_windows(),
            config=config,
            policy=AdaptInitPolicy(mode="scratch"),
            artifact_root=tmp_path / "m7",
            run_id="smoke_only_budget",
        )
    )
    assert result.checkpoint_path().is_file()
    assert result.scaler_path().is_file()
    assert result.meta["status"] == _STATUS_SMOKE
    assert result.meta["total_adapt_steps"] == 1
    assert result.meta["min_adapt_steps"] == _MIN_ADAPT_STEPS


def test_adapt_meta_success_when_budget_met(tmp_path: Path) -> None:
    """Meeting min_adapt_steps keeps status success."""
    config = _tiny_config(
        tmp_path,
        epochs=1,
        steps_per_epoch=1,
        min_adapt_steps=1,
    )
    result = run_adapt_session(
        AdaptSessionRequest(
            windows=_synthetic_windows(),
            config=config,
            policy=AdaptInitPolicy(mode="scratch"),
            artifact_root=tmp_path / "m7",
            run_id="budget_ok",
        )
    )
    assert result.meta["status"] == _STATUS_SUCCESS
    assert result.meta["total_adapt_steps"] == 1


def test_product_yaml_sets_min_adapt_steps() -> None:
    """Shipped adapt_m7.yaml opts into the honesty threshold."""
    config = load_adapt_m7_config(_ADAPT_YAML)
    assert config.caps.min_adapt_steps == _MIN_ADAPT_STEPS
    raw = yaml.safe_load(_ADAPT_YAML.read_text(encoding="utf-8"))
    assert int(raw["adapt"]["min_adapt_steps"]) == _MIN_ADAPT_STEPS


def test_adapt_m7_5_flags_on_toy(tmp_path: Path) -> None:
    """Toy fixture + adapt_m7_5.yaml: flags, smoke_only, Fill solve knobs."""
    assert _ADAPT_M75_YAML.is_file()
    assert _TOY_CSV.is_file()
    overrides = AdaptCliOverrides(
        window=AdaptWindowCliOverrides(
            seq_len=_TOY_SEQ_LEN,
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
    config = apply_adapt_cli_overrides(
        load_adapt_m7_config(_ADAPT_M75_YAML),
        overrides,
    )
    assert config.caps.min_adapt_steps == _MIN_ADAPT_STEPS
    assert config.train.protocol.parameterization == "eps"
    assert config.train.protocol.observed_input == "clean"
    assert config.sample.diffusion_impl == "diagonal"
    result = run_adapt_upload_pipeline(
        AdaptUploadPaths(
            csv_path=_TOY_CSV,
            config_path=_ADAPT_M75_YAML,
            map_path=None,
            artifact_root=tmp_path / "m7_5",
        ),
        AdaptUploadFlags(
            smoke_eval=False,
            export_samples=False,
            mode="scratch",
        ),
        overrides,
    )
    meta = result.session.meta
    assert result.session.checkpoint_path().is_file()
    assert meta["status"] == _STATUS_SMOKE
    assert meta["min_adapt_steps"] == _MIN_ADAPT_STEPS
    assert meta["parameterization"] == "eps"
    assert meta["observed_input"] == "clean"
    assert meta["diffusion_impl"] == "diagonal"
    assert meta["sample_t_eps"] == pytest.approx(1.0e-3)
    imputer = imputer_from_adapt_result(result.session, config)
    assert imputer.solve.parameterization == "eps"
    assert imputer.solve.diffusion_impl == "diagonal"
    assert imputer.solve.observed_mode == "frozen_clean"


def test_smoke_only_warning_text() -> None:
    """UI copy warns for smoke_only and is silent for success."""
    from app.adapt_session_ui import smoke_only_warning_text

    warning = smoke_only_warning_text(_STATUS_SMOKE)
    assert warning
    assert "quality" in warning.lower() or "claim" in warning.lower()
    assert smoke_only_warning_text(_STATUS_SUCCESS) == ""
