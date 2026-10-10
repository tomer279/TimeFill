"""WP3 real-window DSM trainer for Milestone 2.

Loads processed Beijing windows, fits a train-only FeatureScaler, trains
the conditional score net with native+artificial mask composition, and
writes checkpoints / loss curves / lightweight val imputation snapshots
under ``artifacts/m2/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

from src.data.normalize import resolve_scaler_path
from src.data.real_config import load_real_m2_config
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sde.vp_sde import VPSDE
from src.train.artifacts import resolve_artifact_paths
from src.train.loop import build_optimizer
from src.train.real_loop import (
    M2Trainer,
    M2TrainerBundle,
    run_m2_training,
    run_mask_rate_sensitivity,
)
from src.train.real_pool import load_normalized_pool
from src.train.real_val import M2ValHookBundle
from src.train.real_loss import (
    RealConditionalDSMLoss,
    dsm_settings_from_real_config,
)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "real_m2.yaml"
_SENSITIVITY_RATES = (0.1, 0.3, 0.5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the M2 real trainer.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed flags.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to real_m2 YAML config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs for a shorter CPU smoke run",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Override train.steps_per_epoch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override data.batch_size",
    )
    parser.add_argument(
        "--snapshot-k",
        type=int,
        default=None,
        help="Override train.snapshot_k_samples",
    )
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        default=None,
        help="Override train.snapshot_solver_steps",
    )
    parser.add_argument(
        "--sensitivity-smoke",
        action="store_true",
        help="Force sensitivity_mask_rates to 0.1 / 0.3 / 0.5 after train",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Run M2 real training and return the loss-curve CSV path.

    Parameters
    ----------
    argv : list of str or None, optional
        Optional CLI args for tests.

    Returns
    -------
    pathlib.Path
        Path to ``loss_curves.csv`` under the configured artifact dir.
    """
    args = parse_args(argv)
    if _needs_override(args):
        _train_with_overrides(args)
    else:
        run_m2_training(args.config)

    config = load_real_m2_config(args.config)
    paths = resolve_artifact_paths(config.train.protocol.artifact_dir)
    print(f"Wrote {paths.loss_csv()}")
    print(f"Checkpoint {paths.best_checkpoint()}")
    print(f"Scaler {resolve_scaler_path(paths.checkpoint_dir())}")
    return paths.loss_csv()


def _needs_override(args: argparse.Namespace) -> bool:
    """Return whether any CLI override / sensitivity flag is set.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI flags.

    Returns
    -------
    bool
        ``True`` if in-memory config mutation is required.
    """
    return any(
        (
            args.epochs is not None,
            args.steps_per_epoch is not None,
            args.batch_size is not None,
            args.snapshot_k is not None,
            args.snapshot_steps is not None,
            args.sensitivity_smoke,
        )
    )


def _train_with_overrides(args: argparse.Namespace) -> None:
    """Load config, apply CLI overrides, and train.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI flags including ``config``.
    """
    config = load_real_m2_config(args.config)
    if args.epochs is not None:
        config.train.schedule.epochs = int(args.epochs)
    if args.steps_per_epoch is not None:
        config.train.schedule.steps_per_epoch = int(args.steps_per_epoch)
    if args.batch_size is not None:
        config.data.batch_size = int(args.batch_size)
    if args.snapshot_k is not None:
        config.extras.snapshot_k_samples = int(args.snapshot_k)
    if args.snapshot_steps is not None:
        config.extras.snapshot_solver_steps = int(args.snapshot_steps)
    if args.sensitivity_smoke:
        config.extras.sensitivity_mask_rates = _SENSITIVITY_RATES
    config.validate()

    paths = resolve_artifact_paths(config.train.protocol.artifact_dir)
    paths.ensure()
    pool, scaler = load_normalized_pool(config)
    scaler.save(resolve_scaler_path(paths.checkpoint_dir()))
    key = jax.random.PRNGKey(config.train.protocol.seed)
    key_model, key_train = jax.random.split(key)
    model = ConditionalScoreNet(
        key_model,
        default_score_net_config(config.data.n_features),
    )
    sde = VPSDE(config.sde)
    loss_fn = RealConditionalDSMLoss(
        sde,
        dsm_settings_from_real_config(config),
    )
    val_hook = M2ValHookBundle(
        sde=sde,
        t_eps=config.train.protocol.t_eps,
        k_samples=config.extras.snapshot_k_samples,
        solver_steps=config.extras.snapshot_solver_steps,
    ).as_hook()
    bundle = M2TrainerBundle(
        config=config,
        model=model,
        loss_fn=loss_fn,
        optimizer=build_optimizer(config.train),
        paths=paths,
        pool=pool,
        val_hook=val_hook,
    )
    trainer = M2Trainer(bundle.with_paths())
    trainer.run(key_train)
    if config.extras.sensitivity_mask_rates:
        run_mask_rate_sensitivity(
            trainer,
            pool.fixed_val_batch(config.data.batch_size),
            config.extras.sensitivity_mask_rates,
            jax.random.PRNGKey(config.train.protocol.seed + 1),
        )


if __name__ == "__main__":
    main()
