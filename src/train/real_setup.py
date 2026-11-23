"""Pre-train setup, WP0 guards, and the jitted real train step.

Functions
---------
prepare_artifacts_and_pool
    Ensure artifact dirs, fit/save scaler, write split pin, return pool.
assert_not_frozen_artifact_dir
    Refuse train writes into frozen claim artifact roots.
assert_wp0_consumable
    Raise if YAML asks for M7.5 flags that training does not yet consume.
build_real_train_step
    Jitted AdamW step closed over ``RealConditionalDSMLoss``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.data.normalize import resolve_scaler_path
from src.data.real import load_processed_dataset
from src.data.real_config import RealM2Config
from src.eval.split_pin import build_split_manifest, write_split_manifest
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.train.artifacts import ArtifactPaths, resolve_artifact_paths
from src.train.real_loss import (
    RealConditionalDSMLoss,
    RealDSMBatch,
    dsm_settings_from_real_config,
)
from src.train.real_pool import RealBatchPool, load_normalized_pool

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FROZEN_ARTIFACT_DIRS = ("artifacts/m2", "artifacts/m3", "artifacts/m7")

__all__ = [
    "assert_not_frozen_artifact_dir",
    "assert_wp0_consumable",
    "build_real_train_step",
    "prepare_artifacts_and_pool",
]


def prepare_artifacts_and_pool(
        config: RealM2Config,
) -> tuple[ArtifactPaths, RealBatchPool]:
    """Ensure artifact dirs, fit/save scaler, write split pin, return pool.

    Parameters
    ----------
    config : RealM2Config
        Validated real-m2 config.

    Returns
    -------
    paths : ArtifactPaths
        Resolved artifact layout.
    pool : RealBatchPool
        Normalized train/val pools.
    """
    paths = resolve_artifact_paths(config.train.protocol.artifact_dir)
    assert_not_frozen_artifact_dir(paths)
    paths.ensure()
    # shortcut: delete best before manifest so pin never outlives the ckpt.
    _clear_stale_checkpoint_files(paths)
    pool, scaler = load_normalized_pool(config)
    scaler.save(resolve_scaler_path(paths.checkpoint_dir()))
    processed = config.data.resolve_processed_dir(_REPO_ROOT)
    _dataset, split = load_processed_dataset(processed)
    write_split_manifest(
        paths.checkpoint_dir(),
        build_split_manifest(
            processed_dir=processed,
            split_mode=config.data.split_mode,
            split_seed=config.data.split_seed,
            fractions={
                "train": config.data.train_fraction,
                "val": config.data.val_fraction,
                "test": config.data.test_fraction,
            },
            split=split,
        ),
    )
    return paths, pool


def _clear_stale_checkpoint_files(paths: ArtifactPaths) -> None:
    """Remove prior best checkpoint / sidecar / meta before writing a new pin."""
    best = paths.best_checkpoint()
    if best.is_file():
        best.unlink()
    sidecar = paths.model_config()
    if sidecar.is_file():
        sidecar.unlink()
    meta = paths.root / "train_meta.json"
    if meta.is_file():
        meta.unlink()


def assert_not_frozen_artifact_dir(paths: ArtifactPaths) -> None:
    """Refuse train writes into frozen claim artifact roots.

    Parameters
    ----------
    paths : ArtifactPaths
        Resolved artifact layout for this run.

    Raises
    ------
    RuntimeError
        When ``paths.root`` is ``artifacts/m2``, ``m3``, or ``m7``.
    """
    root = paths.root.resolve()
    for rel in _FROZEN_ARTIFACT_DIRS:
        frozen = (_REPO_ROOT / rel).resolve()
        if root == frozen or root.is_relative_to(frozen):
            raise RuntimeError(
                f"refusing to train into frozen claim dir {paths.root}"
            )


def assert_wp0_consumable(config: RealM2Config) -> None:
    """Raise if YAML asks for M7.5 flags that training does not yet consume.

    Also builds DSM settings so invalid loss combos fail before artifact wipe.

    Parameters
    ----------
    config : RealM2Config
        Config about to train.

    Raises
    ------
    NotImplementedError
        Naming each non-legacy unused setting.
    ValueError
        From ``DSMLossSettings.validate`` (e.g. clean + nonzero obs weight).
    """
    sample = config.sample
    transform = config.data.transform
    legacy_model = default_score_net_config(config.data.n_features)
    # WP4 consumes EMA / schedule / select / masks / eps + sample.t_eps.
    # shortcut: refuse model.* until pre-WP7 wires build_score_net(config.model)
    # + sidecar write; refuse WP5 transform until WP5.
    checks: tuple[tuple[bool, str], ...] = (
        (config.model != legacy_model, "model.*"),
        (sample.observed_mode != "frozen_clean", "sample.observed_mode"),
        (bool(transform.log1p_features), "data.transform.log1p_features"),
        (transform.clip_nonnegative, "data.transform.clip_nonnegative"),
        (
            not config.data.allow_empty_feature_fallback,
            "data.allow_empty_feature_fallback",
        ),
    )
    bad = [name for is_bad, name in checks if is_bad]
    if bad:
        raise NotImplementedError(
            "M7.5 flags not wired yet: " + ", ".join(bad)
        )
    dsm_settings_from_real_config(config)


def build_real_train_step(
        optimizer: optax.GradientTransformation,
        loss_fn: RealConditionalDSMLoss,
) -> Callable:
    """Create a jitted train step for ``RealDSMBatch`` inputs.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        Optax optimizer.
    loss_fn : RealConditionalDSMLoss
        DSM loss closed over by the step.

    Returns
    -------
    callable
        ``(model, opt_state, batch, key) -> (model, opt_state, loss, grad_norm)``.
    """

    @eqx.filter_jit
    def _step(
            model: ConditionalScoreNet,
            opt_state: optax.OptState,
            batch: RealDSMBatch,
            key: jax.Array,
    ) -> tuple[
        ConditionalScoreNet,
        optax.OptState,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        def _loss_fn(score_model: ConditionalScoreNet) -> jnp.ndarray:
            return loss_fn(score_model, batch, key)

        loss, grads = eqx.filter_value_and_grad(_loss_fn)(model)
        grad_norm = _tree_grad_norm(grads)
        updates, new_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss, grad_norm

    return _step


def _tree_grad_norm(grads: Any) -> jnp.ndarray:
    """Compute the global L2 norm of a pytree of gradients.

    Parameters
    ----------
    grads : Any
        Gradient pytree (Equinox-filtered).

    Returns
    -------
    jax.Array
        Scalar grad norm.
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        total = total + jnp.sum(leaf.astype(jnp.float32) ** 2)
    return jnp.sqrt(total)
