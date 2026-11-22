"""WP4 tests: EMA, fixed val-impute set, mask hygiene, train_meta.

Functions
---------
test_val_impute_set_identical_across_two_calls
    Fixed val-impute set matches across two builds (masks and sample key).
test_ema_differs_after_update_equals_when_decay_zero
    EMA diverges from raw at decay>0 and copies raw at decay=0.
test_val_impute_runner_uses_model_argument
    Jitted runner scores differ when the model argument changes.
test_per_window_mixed_yields_both_mask_types
    ``mixed_select=per_window`` puts random and block windows in one batch.
test_block_len_range_lengths_within_bounds
    Block masks with a range only produce lengths in ``[lo, hi]``.
test_assert_wp0_still_refuses_wp3_and_wp5
    Train guard still blocks ``model.*`` / transform until those WPs.
test_train_meta_writes_required_keys
    ``write_train_meta`` persists the WP4 required fields.
test_fallback_best_ckpt_when_no_impute_selection
    Short val-impute run writes final weights; ``selection_fallback`` true.
test_impute_select_skips_val_png_hook
    Val-impute selection does not call the PNG val snapshot hook.
test_val_impute_set_is_seeded_subset_not_prefix
    Window indices are a seeded subset, not always ``0..n_take-1``.
test_with_mask_rate_keeps_wp4_extras
    ``with_mask_rate`` preserves mixed_select / range / max_blocks.
test_hygiene_yaml_passes_wp0_consumable
    ``real_m7_5_hygiene.yaml`` loads and clears the WP0 refuse guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.data.masking import BlockMasker
from src.data.real_config import load_real_m2_config
from src.data.schema import MASK_MISSING
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import ImputeSolverSettings
from src.sde.vp_sde import VPSDE
from src.train.artifacts import ArtifactPaths, ValSnapshotPayload
from src.train.loop import build_optimizer
from src.train.loss import DSMLossSettings, DsmLossExtras
from src.train.real_loop import M2Trainer, M2TrainerBundle, _assert_wp0_consumable
from src.train.real_loss import (
    RealConditionalDSMLoss,
    RealDSMBatch,
    dsm_settings_from_real_config,
)
from src.train.real_pool import RealBatchPool
from src.train.real_val_select import (
    SELECT_MAE,
    ValImputeBundle,
    build_fixed_val_impute_set,
    init_ema,
    make_val_impute_runner,
    score_val_impute,
    update_ema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_M2_PATH = REPO_ROOT / "configs" / "real_m2.yaml"
REAL_M75_PATH = REPO_ROOT / "configs" / "real_m7_5.yaml"
REAL_M75_HYGIENE_PATH = REPO_ROOT / "configs" / "real_m7_5_hygiene.yaml"

_SEED = 0
_BATCH = 32
_SEQ = 16
_FEAT = 2
_BLOCK_LEN = 8
_RANGE_LO = 4
_RANGE_HI = 16
_MASK_RATE = 0.3
_T_EPS = 1.0e-3
_N_VAL_WINDOWS = 4
_N_LENGTH_DRAWS = 64
_EMA_DECAY = 0.999
_BATCH_AXIS = 0
_MISSING_TOL = 1.0e-6
_RUNNER_STEPS = 4
_RUNNER_K = 2
_LOOP_FEAT = 6
_LOOP_SEQ = 8
_LOOP_POOL = 4
_LOOP_BATCH = 2

_TRAIN_META_REQUIRED_KEYS = (
    "clipped_step_fraction",
    "median_grad_norm",
    "ema_decay",
    "parameter_count",
    "receptive_field",
    "selected_epoch",
)


def test_val_impute_set_identical_across_two_calls() -> None:
    """Fixed val-impute set is identical across two builds with the same seeds."""
    values, m_native = _synthetic_val_windows()
    spec = {
        "n_windows": _N_VAL_WINDOWS,
        "mask_seed": 0,
        "sample_seed": 1,
        "protocols": ("mcar_0.3", "block_8"),
    }
    first = build_fixed_val_impute_set(values, m_native, spec)
    second = build_fixed_val_impute_set(values, m_native, spec)
    for protocol in spec["protocols"]:
        np.testing.assert_array_equal(
            first.masks[protocol],
            second.masks[protocol],
        )
    np.testing.assert_array_equal(first.sample_key, second.sample_key)
    np.testing.assert_array_equal(first.window_indices, second.window_indices)


def test_ema_differs_after_update_equals_when_decay_zero() -> None:
    """EMA differs from raw after an update; decay=0 copies the raw tree."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(_SEED))
    cfg = default_score_net_config(_FEAT)
    model_a = ConditionalScoreNet(key_a, cfg)
    model_b = ConditionalScoreNet(key_b, cfg)
    ema = init_ema(model_a)
    ema_decayed = update_ema(ema, model_b, decay=_EMA_DECAY)
    ema_reset = update_ema(init_ema(model_a), model_b, decay=0.0)

    assert not _array_trees_close(ema_decayed, model_b)
    assert _array_trees_close(ema_reset, model_b)


def test_val_impute_runner_uses_model_argument() -> None:
    """Val-impute scoring changes when the model argument changes."""
    values, m_native = _synthetic_val_windows()
    spec = {
        "n_windows": _N_VAL_WINDOWS,
        "mask_seed": 0,
        "sample_seed": 1,
        "protocols": ("mcar_0.3",),
    }
    fixed = build_fixed_val_impute_set(values, m_native, spec)
    sde = VPSDE(load_real_m2_config(REAL_M2_PATH).sde)
    runner = make_val_impute_runner(
        sde,
        ImputeSolverSettings(
            solver_steps=_RUNNER_STEPS,
            t_eps=_T_EPS,
            diffusion_impl="diagonal",
        ),
        _SEQ,
        _RUNNER_K,
        "score",
    )
    bundle = ValImputeBundle(
        fixed=fixed,
        values=values,
        m_native=m_native,
        runner=runner,
        select_by=SELECT_MAE,
    )
    key_a, key_b = jax.random.split(jax.random.PRNGKey(_SEED))
    cfg = default_score_net_config(_FEAT)
    model_a = ConditionalScoreNet(key_a, cfg)
    model_b = ConditionalScoreNet(key_b, cfg)
    score_a = score_val_impute(model_a, bundle)
    score_b = score_val_impute(model_b, bundle)
    assert score_a != score_b


def test_per_window_mixed_yields_both_mask_types() -> None:
    """With ``per_window``, one mixed batch contains both mask families."""
    settings = DSMLossSettings(
        mask_rate=_MASK_RATE,
        block_len=_BLOCK_LEN,
        t_eps=_T_EPS,
        obs_loss_weight=0.0,
        mask_mode="mixed",
        extras=DsmLossExtras(mixed_select="per_window"),
    )
    sde = VPSDE(load_real_m2_config(REAL_M2_PATH).sde)
    loss_fn = RealConditionalDSMLoss(sde, settings)
    batch = _full_observed_batch(_BATCH, _SEQ, _FEAT)
    _x_obs, _m_train, m_art = loss_fn.sample_mask(
        jax.random.PRNGKey(_SEED),
        batch,
    )
    is_block = _window_looks_like_block(m_art, _BLOCK_LEN)
    n_block = int(np.sum(is_block))
    n_random = int(np.sum(~is_block))
    assert n_block > 0
    assert n_random > 0


def test_block_len_range_lengths_within_bounds() -> None:
    """Block lengths sampled from a range stay inside ``[lo, hi]``."""
    masker = BlockMasker(
        block_len=_BLOCK_LEN,
        block_len_range=(_RANGE_LO, _RANGE_HI),
    )
    values = jnp.zeros((_N_LENGTH_DRAWS, _SEQ, _FEAT), dtype=jnp.float32)
    _x_obs, mask = masker.apply(jax.random.PRNGKey(_SEED), values)
    lengths = _contiguous_missing_lengths(mask)
    assert lengths.size > 0
    assert int(lengths.min()) >= _RANGE_LO
    assert int(lengths.max()) <= _RANGE_HI


def test_assert_wp0_still_refuses_wp3_and_wp5() -> None:
    """``real_m7_5.yaml`` still refuses WP3 model / WP5 transform flags."""
    config = load_real_m2_config(REAL_M75_PATH)
    with pytest.raises(NotImplementedError, match="model|transform"):
        _assert_wp0_consumable(config)


def test_train_meta_writes_required_keys(tmp_path: Path) -> None:
    """``write_train_meta`` writes every WP4 required train_meta field."""
    from src.train.artifacts import write_train_meta

    payload = {
        "clipped_step_fraction": 0.1,
        "median_grad_norm": 0.5,
        "ema_decay": _EMA_DECAY,
        "parameter_count": 1000,
        "receptive_field": 17,
        "selected_epoch": 3,
        "select_by": "val_impute_crps",
        "observed_input": "clean",
        "observed_mode": "frozen_clean",
    }
    path = write_train_meta(tmp_path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    for key in _TRAIN_META_REQUIRED_KEYS:
        assert key in loaded


def test_fallback_best_ckpt_when_no_impute_selection(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """epochs < val_every writes final eval weights, not init."""
    monkeypatch.setattr(
        "src.train.real_loop.make_val_impute_runner",
        _stub_val_impute_runner_factory,
    )
    hook_calls: list[int] = []
    trainer = _tiny_impute_trainer(
        tmp_path,
        epochs=2,
        val_every=5,
        hook_calls=hook_calls,
    )
    before = eqx.filter(trainer.model, eqx.is_array)
    trainer.run(jax.random.PRNGKey(_SEED))
    assert trainer.paths.best_checkpoint().is_file()
    meta = json.loads(
        (trainer.paths.root / "train_meta.json").read_text(encoding="utf-8")
    )
    assert meta["selected_epoch"] is None
    assert meta["selection_fallback"] is True
    assert hook_calls == []
    assert not _array_trees_close(before, trainer.model)


def test_impute_select_skips_val_png_hook(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On val_every under val-impute select, the PNG hook is not called."""
    monkeypatch.setattr(
        "src.train.real_loop.make_val_impute_runner",
        _stub_val_impute_runner_factory,
    )
    hook_calls: list[int] = []
    trainer = _tiny_impute_trainer(
        tmp_path,
        epochs=2,
        val_every=1,
        hook_calls=hook_calls,
    )
    trainer.run(jax.random.PRNGKey(_SEED))
    assert hook_calls == []
    assert trainer.paths.best_checkpoint().is_file()
    meta = json.loads(
        (trainer.paths.root / "train_meta.json").read_text(encoding="utf-8")
    )
    assert meta["selected_epoch"] == 1
    assert meta["selection_fallback"] is False


def test_val_impute_set_is_seeded_subset_not_prefix() -> None:
    """With N > n_windows, indices are not the trivial prefix ``0..n-1``."""
    n_pool = 16
    n_take = 4
    values = jax.random.normal(
        jax.random.PRNGKey(_SEED),
        (n_pool, _SEQ, _FEAT),
    )
    m_native = jnp.ones_like(values, dtype=jnp.float32)
    fixed = build_fixed_val_impute_set(
        values,
        m_native,
        {
            "n_windows": n_take,
            "mask_seed": 0,
            "sample_seed": 1,
            "protocols": ("mcar_0.3",),
        },
    )
    indices = np.asarray(fixed.window_indices)
    assert indices.shape == (n_take,)
    assert not np.array_equal(indices, np.arange(n_take))


def test_with_mask_rate_keeps_wp4_extras() -> None:
    """``with_mask_rate`` copies mixed_select / range / max_blocks."""
    settings = DSMLossSettings(
        mask_rate=_MASK_RATE,
        block_len=_BLOCK_LEN,
        t_eps=_T_EPS,
        obs_loss_weight=0.0,
        mask_mode="mixed",
        extras=DsmLossExtras(
            mixed_select="per_window",
            block_len_range=(_RANGE_LO, _RANGE_HI),
            max_blocks_per_feature=2,
        ),
    )
    sde = VPSDE(load_real_m2_config(REAL_M2_PATH).sde)
    probe = RealConditionalDSMLoss(sde, settings).with_mask_rate(0.5)
    assert probe.settings.mask_rate == pytest.approx(0.5)
    assert probe.settings.mixed_select == "per_window"
    assert probe.settings.block_len_range == (_RANGE_LO, _RANGE_HI)
    assert probe.settings.max_blocks_per_feature == 2


def test_hygiene_yaml_passes_wp0_consumable() -> None:
    """Hygiene YAML is consumable (legacy model, no transform)."""
    config = load_real_m2_config(REAL_M75_HYGIENE_PATH)
    _assert_wp0_consumable(config)


def test_val_impute_rejects_bad_protocol_name() -> None:
    """Config validate rejects malformed ``val_impute.protocols`` entries."""
    from src.data.val_impute_spec import validate_val_impute_mapping

    with pytest.raises(ValueError, match="protocol"):
        validate_val_impute_mapping(
            {
                "n_windows": 4,
                "K_samples": 2,
                "solver_steps": 10,
                "mask_seed": 0,
                "sample_seed": 1,
                "protocols": ["block8"],
            }
        )


def test_warmup_steps_must_be_below_total_steps() -> None:
    """``warmup_cosine`` rejects ``warmup_steps >= epochs * steps_per_epoch``."""
    config = load_real_m2_config(REAL_M75_HYGIENE_PATH)
    config.train.schedule.epochs = 2
    config.train.schedule.steps_per_epoch = 10
    config.train.protocol.extras.loss.warmup_steps = 20
    with pytest.raises(ValueError, match="warmup_steps"):
        config.train.validate()


def _stub_val_impute_runner_factory(
        *_args: Any,
        **_kwargs: Any,
) -> Any:
    """Return a cheap runner so loop tests skip Diffrax compile."""

    def run(
            _model: ConditionalScoreNet,
            observations: jnp.ndarray,
            _mask: jnp.ndarray,
            _key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, seq_len, n_features = observations.shape
        mean = jnp.zeros(
            (batch_size, seq_len, n_features),
            dtype=jnp.float32,
        )
        samples = jnp.zeros(
            (batch_size, _RUNNER_K, seq_len, n_features),
            dtype=jnp.float32,
        )
        return samples, mean

    return run


def _tiny_impute_trainer(
        tmp_path: Path,
        epochs: int,
        val_every: int,
        hook_calls: list[int],
) -> M2Trainer:
    """Build a hermetic M2Trainer with val-impute selection."""
    config = load_real_m2_config(REAL_M2_PATH)
    config.data.batch_size = _LOOP_BATCH
    config.train.schedule.epochs = epochs
    config.train.schedule.steps_per_epoch = 1
    config.train.schedule.val_every = val_every
    config.train.schedule.log_every = 100
    config.train.protocol.extras.loss.select_by = "val_impute_crps"
    config.train.protocol.val_impute = {
        "n_windows": 2,
        "K_samples": _RUNNER_K,
        "solver_steps": 2,
        "mask_seed": 0,
        "sample_seed": 1,
        "protocols": ["mcar_0.3"],
    }
    config.train.protocol.artifact_dir = str(tmp_path / "artifacts")
    values = jnp.zeros(
        (_LOOP_POOL, _LOOP_SEQ, _LOOP_FEAT),
        dtype=jnp.float32,
    )
    masks = jnp.ones_like(values)
    pool = RealBatchPool(
        train_values=values,
        train_masks=masks,
        val_values=values,
        val_masks=masks,
    )
    paths = ArtifactPaths(tmp_path / "artifacts")
    paths.ensure()
    sde = VPSDE(config.sde)
    key = jax.random.PRNGKey(_SEED)

    def _hook(payload: ValSnapshotPayload) -> Path:
        hook_calls.append(int(payload.step))
        out = payload.out_dir / f"val_impute_step_{payload.step}.png"
        out.write_text("stub", encoding="utf-8")
        return out

    bundle = M2TrainerBundle(
        config=config,
        model=ConditionalScoreNet(
            key,
            default_score_net_config(_LOOP_FEAT),
        ),
        loss_fn=RealConditionalDSMLoss(
            sde,
            dsm_settings_from_real_config(config),
        ),
        optimizer=build_optimizer(config.train),
        paths=paths,
        pool=pool,
        val_hook=_hook,
    )
    return M2Trainer(bundle)


def _synthetic_val_windows() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return synthetic ``(values, m_native)`` for val-impute set builds."""
    values = jax.random.normal(
        jax.random.PRNGKey(_SEED),
        (_N_VAL_WINDOWS, _SEQ, _FEAT),
    )
    m_native = jnp.ones_like(values, dtype=jnp.float32)
    return values, m_native


def _full_observed_batch(
        batch_size: int,
        seq_len: int,
        n_features: int,
) -> RealDSMBatch:
    """Build a fully observed real DSM batch of zeros."""
    shape = (batch_size, seq_len, n_features)
    return RealDSMBatch(
        x0=jnp.zeros(shape, dtype=jnp.float32),
        m_native=jnp.ones(shape, dtype=jnp.float32),
    )

def _window_looks_like_block(
        m_artificial: jnp.ndarray,
        block_len: int,
) -> np.ndarray:
    """Return per-window True when every feature has one contiguous gap."""
    missing = np.asarray(
        jnp.abs(m_artificial - MASK_MISSING) < _MISSING_TOL
    )
    flags = []
    for batch_index in range(missing.shape[_BATCH_AXIS]):
        flags.append(
            _batch_item_is_block(missing[batch_index], block_len)
        )
    return np.asarray(flags, dtype=bool)


def _batch_item_is_block(missing_tf: np.ndarray, block_len: int) -> bool:
    """True if each feature has exactly one contiguous True run of block_len."""
    n_features = missing_tf.shape[1]
    for feature_index in range(n_features):
        run = missing_tf[:, feature_index]
        if int(run.sum()) != block_len:
            return False
        if not _is_single_contiguous_run(run):
            return False
    return True


def _is_single_contiguous_run(run: np.ndarray) -> bool:
    """True when ``run`` is False* + True+ + False* (one gap)."""
    if not bool(run.any()):
        return False
    first = int(np.argmax(run))
    last = int(len(run) - 1 - np.argmax(run[::-1]))
    return bool(np.all(run[first: last + 1]))


def _contiguous_missing_lengths(mask: jnp.ndarray) -> np.ndarray:
    """Collect contiguous missing-run lengths over all batch/feature pairs."""
    missing = np.asarray(jnp.abs(mask - MASK_MISSING) < _MISSING_TOL)
    lengths: list[int] = []
    n_batch, _seq_len, n_features = missing.shape
    for batch_index in range(n_batch):
        for feature_index in range(n_features):
            run = missing[batch_index, :, feature_index]
            lengths.extend(_run_lengths(run))
    return np.asarray(lengths, dtype=np.int32)


def _run_lengths(run: np.ndarray) -> list[int]:
    """Return lengths of contiguous True segments in a 1-D boolean run."""
    padded = np.concatenate(([False], run.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [int(end - start) for start, end in zip(starts, ends)]


def _array_trees_close(tree_a: object, tree_b: object) -> bool:
    """True when filtered array leaves of two Equinox trees all match."""
    leaves_a = jax.tree_util.tree_leaves(eqx.filter(tree_a, eqx.is_array))
    leaves_b = jax.tree_util.tree_leaves(eqx.filter(tree_b, eqx.is_array))
    if len(leaves_a) != len(leaves_b):
        return False
    return all(
        np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(leaves_a, leaves_b)
    )
