"""WP-B4 Done-when: abort-without-promote + shared imputer flag parity.

Tests
-----
test_nan_loss_aborts_without_promote
    Stub non-finite loss → no session checkpoint / scaler.
test_timeout_aborts_without_promote
    Injected clock past deadline → no session checkpoint / scaler.
test_adapt_sample_section_exposes_m75_solve_flags
    ``AdaptSampleSection`` carries ``t_eps`` / ``observed_mode`` /
    ``diffusion_impl``.
test_adapt_config_rejects_sample_t_eps_at_or_above_t_diff
    Config validate refuses resolved ``t_eps >= T_diff``.
test_fine_tune_refuses_eps_parameterization
    Fine-tune + ``eps`` is refused (base M2 nets are score).
test_cli_and_ui_imputers_match_config_solve_flags
    CLI and UI builders yield identical solver settings that match the
    adapt config M7.5 flags (including ``ScoreFromEps`` wrap).
test_legacy_adapt_yaml_keeps_legacy_solve_defaults
    ``adapt_m7.yaml`` stays dense / frozen_clean / train ``t_eps``.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.adapt_session_ui import imputer_from_adapt_result
from app.adapt_upload_m7 import imputer_from_adapt_result as cli_imputer_from_adapt
from app.session_impute import DemoMaskRequest, SessionImputeRequest
from app.window_adapter import AdaptedWindows
from src.data.adapt_config import (
    AdaptCaps,
    AdaptGeometry,
    AdaptM7Config,
    AdaptModeSection,
    AdaptSampleSection,
    load_adapt_m7_config,
)
from src.data.config import SdeConfig, TrainConfig
from src.data.normalize import FeatureScaler
from src.models.score_from_eps import ScoreFromEps
from src.models.score_net import ConditionalScoreNet, default_score_net_config
from src.sample.reverse_sde import ImputeContext, ImputeSolverSettings
from src.train.adapt_m7 import (
    AdaptInitPolicy,
    AdaptSessionRequest,
    AdaptSessionResult,
    AdaptWindowSplit,
    AdaptWindowTensors,
    run_adapt_session,
)
from src.train.real_loss import RealConditionalDSMLoss

_REPO = Path(__file__).resolve().parents[1]
_ADAPT_YAML = _REPO / "configs" / "adapt_m7.yaml"

_SEQ = 8
_FEAT = 3
_N_WINDOWS = 5
_SEED = 0
_TRAIN_FRACTION = 0.80
_SAMPLE_T_EPS = 1.0e-3
_TRAIN_T_EPS = 1.0e-5
_OBSERVED_MODE = "replace_noised"
_DIFFUSION_IMPL = "diagonal"
_PARAM_EPS = "eps"
_IMPUTE_K = 1
_IMPUTE_STEPS = 2


def _tiny_config(
        tmp_path: Path,
        *,
        timeout_seconds: float = 120.0,
) -> AdaptM7Config:
    """Build a CPU-tiny adapt config rooted at ``tmp_path``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    timeout_seconds : float, optional
        Wall-clock abort budget.

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
        "t_eps": _TRAIN_T_EPS,
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
            timeout_seconds=float(timeout_seconds),
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
        sample=_m75_sample_section(),
    )
    config.validate()
    return config


def _m75_sample_section() -> AdaptSampleSection:
    """Return sample knobs including M7.5 solve flags (WP-B4 contract).

    Returns
    -------
    AdaptSampleSection
        Section with non-legacy ``t_eps`` / mode / diffusion impl.
    """
    return AdaptSampleSection(
        k_samples=2,
        solver_steps=4,
        seed_mask=0,
        seed_sample=1,
        t_eps=_SAMPLE_T_EPS,
        observed_mode=_OBSERVED_MODE,
        diffusion_impl=_DIFFUSION_IMPL,
    )


def _synthetic_windows() -> AdaptWindowTensors:
    """Build synthetic upload windows for abort tests.

    Returns
    -------
    AdaptWindowTensors
        Dense synthetic batch.
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


def _assert_no_promoted_artifacts(run_dir: Path) -> None:
    """Assert abort left no checkpoint, scaler, or adapt_meta.

    Parameters
    ----------
    run_dir : pathlib.Path
        Expected session run directory.
    """
    assert not (run_dir / "best_score_net.eqx").exists()
    assert not (run_dir / "feature_scaler.npz").exists()
    assert not (run_dir / "adapt_meta.json").exists()


def _fake_session(tmp_path: Path, config: AdaptM7Config) -> AdaptSessionResult:
    """Build an in-memory session result without running adapt.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Artifact root parent.
    config : AdaptM7Config
        Supplies ``F`` via geometry / feature count from a tiny model.

    Returns
    -------
    AdaptSessionResult
        Model + scaler suitable for imputer builders.
    """
    del config
    key = jax.random.PRNGKey(_SEED)
    model = ConditionalScoreNet(key, default_score_net_config(_FEAT))
    values = np.ones((2, _SEQ, _FEAT), dtype=np.float32)
    masks = np.ones_like(values, dtype=np.float32)
    scaler = FeatureScaler().fit(values, masks)
    run_dir = tmp_path / "m7" / "fake_session"
    run_dir.mkdir(parents=True, exist_ok=True)
    return AdaptSessionResult(
        run_id="fake_session",
        run_dir=run_dir,
        model=model,
        scaler=scaler,
        split=AdaptWindowSplit(
            train_idx=np.array([0], dtype=np.int64),
            smoke_idx=np.array([1], dtype=np.int64),
        ),
        meta={"status": "success"},
    )


def _adapted_windows() -> AdaptedWindows:
    """Tiny windows for a one-shot impute spy.

    Returns
    -------
    AdaptedWindows
        Dense synthetic batch.
    """
    names = tuple(f"f{i}" for i in range(_FEAT))
    return AdaptedWindows(
        values=np.ones((1, _SEQ, _FEAT), dtype=np.float32),
        native_mask=np.ones((1, _SEQ, _FEAT), dtype=np.float32),
        feature_names=names,
        entity_ids=np.array(["e0"], dtype=object),
    )


def _capture_impute_context(
        imputer: Any,
        monkeypatch: pytest.MonkeyPatch,
) -> ImputeContext:
    """Run a tiny impute and return the ``ImputeContext`` that was built.

    Parameters
    ----------
    imputer : Any
        ``SessionImputer`` under test.
    monkeypatch : pytest.MonkeyPatch
        Patch fixture.

    Returns
    -------
    ImputeContext
        Context passed to ``ReverseSDEImputer``.
    """
    captured: dict[str, ImputeContext] = {}
    real_init = ImputeContext.__init__

    def _spy(
            self: ImputeContext,
            model: Any,
            sde: Any,
            settings: ImputeSolverSettings,
    ) -> None:
        real_init(self, model=model, sde=sde, settings=settings)
        captured["context"] = self

    monkeypatch.setattr(ImputeContext, "__init__", _spy)
    imputer.impute(
        SessionImputeRequest(
            windows=_adapted_windows(),
            demo_mask=DemoMaskRequest(mode="none", rate=0.0, block_len=2, seed=0),
            k_samples=_IMPUTE_K,
            solver_steps=_IMPUTE_STEPS,
        )
    )
    assert "context" in captured
    return captured["context"]


def test_nan_loss_aborts_without_promote(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite DSM loss must not write checkpoint or scaler."""
    monkeypatch.setattr(
        RealConditionalDSMLoss,
        "__call__",
        lambda self, model, batch, key: jnp.asarray(
            float("nan"),
            dtype=jnp.float32,
        ),
    )
    run_id = "nan_abort"
    artifact_root = tmp_path / "m7"
    # Legacy sample section for abort path (flags not required here).
    config = _tiny_config_legacy(tmp_path)
    with pytest.raises(RuntimeError, match="non-finite"):
        run_adapt_session(
            AdaptSessionRequest(
                windows=_synthetic_windows(),
                config=config,
                policy=AdaptInitPolicy(mode="scratch"),
                artifact_root=artifact_root,
                run_id=run_id,
            )
        )
    _assert_no_promoted_artifacts(artifact_root / run_id)


def test_timeout_aborts_without_promote(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wall-clock timeout must not write checkpoint or scaler."""
    ticks = itertools.count(start=0.0, step=1.0e6)
    monkeypatch.setattr(
        "src.train.adapt_loop.time.monotonic",
        lambda: float(next(ticks)),
    )
    run_id = "timeout_abort"
    artifact_root = tmp_path / "m7"
    config = _tiny_config_legacy(tmp_path, timeout_seconds=1.0)
    with pytest.raises(RuntimeError, match="timeout"):
        run_adapt_session(
            AdaptSessionRequest(
                windows=_synthetic_windows(),
                config=config,
                policy=AdaptInitPolicy(mode="scratch"),
                artifact_root=artifact_root,
                run_id=run_id,
            )
        )
    _assert_no_promoted_artifacts(artifact_root / run_id)


def _tiny_config_legacy(
        tmp_path: Path,
        *,
        timeout_seconds: float = 120.0,
) -> AdaptM7Config:
    """Tiny adapt config using today's ``AdaptSampleSection`` fields only.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    timeout_seconds : float, optional
        Wall-clock abort budget.

    Returns
    -------
    AdaptM7Config
        Validated smoke-cap config.
    """
    train_map = {
        "lr": 0.0003,
        "epochs": 1,
        "steps_per_epoch": 1,
        "weight_decay": 0.0,
        "seed": _SEED,
        "t_eps": _TRAIN_T_EPS,
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
            timeout_seconds=float(timeout_seconds),
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


def test_adapt_sample_section_exposes_m75_solve_flags() -> None:
    """Adapt sample section must expose sampler M7.5 flags (WP-B4)."""
    sample = _m75_sample_section()
    assert sample.t_eps == _SAMPLE_T_EPS
    assert sample.observed_mode == _OBSERVED_MODE
    assert sample.diffusion_impl == _DIFFUSION_IMPL


def test_adapt_config_rejects_sample_t_eps_at_or_above_t_diff(
        tmp_path: Path,
) -> None:
    """Resolved sample ``t_eps`` must be strictly below ``T_diff``."""
    with pytest.raises(ValueError, match="t_eps"):
        AdaptM7Config(
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
                base_checkpoint=(
                    "artifacts/m2/checkpoints/best_score_net.eqx"
                ),
            ),
            geometry=AdaptGeometry(
                seq_len=_SEQ,
                batch_size=2,
                block_len=2,
                max_windows=4,
            ),
            train=TrainConfig(
                {
                    "lr": 0.0003,
                    "epochs": 1,
                    "steps_per_epoch": 1,
                    "weight_decay": 0.0,
                    "seed": _SEED,
                    "t_eps": _TRAIN_T_EPS,
                    "obs_loss_weight": 0.0,
                    "grad_clip_norm": 1.0,
                    "mask_mode": "random",
                    "artifact_dir": str(tmp_path / "m7"),
                }
            ),
            sde=SdeConfig(
                {"beta_min": 0.1, "beta_max": 20.0, "T_diff": 1.0}
            ),
            sample=AdaptSampleSection(
                k_samples=2,
                solver_steps=4,
                seed_mask=0,
                seed_sample=1,
                t_eps=1.5,
            ),
        ).validate()


def test_fine_tune_refuses_eps_parameterization(tmp_path: Path) -> None:
    """Fine-tune must refuse ``eps`` (base checkpoints are score nets)."""
    config = _tiny_config_legacy(tmp_path)
    config.train.protocol.extras.loss.parameterization = _PARAM_EPS
    with pytest.raises(ValueError, match="parameterization"):
        run_adapt_session(
            AdaptSessionRequest(
                windows=_synthetic_windows(),
                config=config,
                policy=AdaptInitPolicy(
                    mode="fine_tune",
                    base_checkpoint=(
                        "artifacts/m2/checkpoints/best_score_net.eqx"
                    ),
                ),
                artifact_root=tmp_path / "m7",
                run_id="fine_tune_eps",
            )
        )
    assert not list((tmp_path / "m7").glob("**/best_score_net.eqx"))


def test_legacy_adapt_yaml_keeps_legacy_solve_defaults(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``adapt_m7.yaml`` Fill stays dense / frozen_clean / train ``t_eps``."""
    config = load_adapt_m7_config(_ADAPT_YAML)
    assert config.sample.t_eps is None
    assert config.sample.observed_mode == "frozen_clean"
    assert config.sample.diffusion_impl == "dense"
    assert config.train.protocol.parameterization == "score"
    session = _fake_session(tmp_path, config)
    ctx = _capture_impute_context(
        imputer_from_adapt_result(session, config),
        monkeypatch,
    )
    assert ctx.settings.t_eps == pytest.approx(
        float(config.train.protocol.t_eps)
    )
    assert ctx.settings.observed_mode == "frozen_clean"
    assert ctx.settings.diffusion_impl == "dense"
    assert not isinstance(ctx.model, ScoreFromEps)


def test_cli_and_ui_imputers_match_config_solve_flags(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI and UI imputers must share solver settings from one config."""
    config = _tiny_config(tmp_path)
    config.train.protocol.extras.loss.parameterization = _PARAM_EPS
    session = _fake_session(tmp_path, config)
    assert cli_imputer_from_adapt is imputer_from_adapt_result
    ui_imputer = imputer_from_adapt_result(session, config)
    cli_imputer = cli_imputer_from_adapt(session, config)
    ui_ctx = _capture_impute_context(ui_imputer, monkeypatch)
    cli_ctx = _capture_impute_context(cli_imputer, monkeypatch)
    expected = ImputeSolverSettings(
        solver_steps=_IMPUTE_STEPS,
        t_eps=_SAMPLE_T_EPS,
        observed_mode=_OBSERVED_MODE,
        diffusion_impl=_DIFFUSION_IMPL,
    ).to_dict()
    assert ui_ctx.settings.to_dict() == expected
    assert cli_ctx.settings.to_dict() == expected
    assert ui_ctx.settings.to_dict() == cli_ctx.settings.to_dict()
    assert isinstance(ui_ctx.model, ScoreFromEps)
    assert isinstance(cli_ctx.model, ScoreFromEps)
    assert ui_ctx.sde.beta_min == pytest.approx(config.sde.beta_min)
    assert ui_ctx.sde.beta_max == pytest.approx(config.sde.beta_max)
    assert ui_ctx.sde.t_diff == pytest.approx(config.sde.t_diff)
    assert cli_ctx.sde.beta_min == pytest.approx(config.sde.beta_min)
