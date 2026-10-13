"""Short CPU-friendly WP4 training smoke on synthetic sine windows.

Writes under ``artifacts/m1/`` (checkpoints, loss CSV/PNG, val stubs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax

from src.data.config import load_config
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sde.vp_sde import VPSDE
from src.train.loss import ConditionalDSMLoss, dsm_settings_from_config
from src.train.artifacts import (
    resolve_artifact_paths,
    stub_val_imputation_hook,
)
from src.train.loop import (
    M1Trainer,
    TrainerBundle,
    build_optimizer,
    run_m1_training,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "toy_m1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the WP4 smoke trainer.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed flags with ``config`` and optional ``epochs`` override.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to toy_m1 YAML config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs for a shorter CPU smoke run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Run DSM training and return the loss-curve CSV path.

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
    if args.epochs is None:
        run_m1_training(args.config)
    else:
        _train_with_epoch_override(args.config, args.epochs)

    config = load_config(args.config)
    paths = resolve_artifact_paths(config.train.protocol.artifact_dir)
    print(f"Wrote {paths.loss_csv()}")
    print(f"Checkpoint {paths.best_checkpoint()}")
    return paths.loss_csv()


def _train_with_epoch_override(config_path: Path, epochs: int) -> None:
    """Load config, override epochs in-memory, and train.

    Parameters
    ----------
    config_path : pathlib.Path
        Base YAML path.
    epochs : int
        Epoch override for CPU smoke.
    """
    config = load_config(config_path)
    config.train.schedule.epochs = int(epochs)
    paths = resolve_artifact_paths(config.train.protocol.artifact_dir)
    key = jax.random.PRNGKey(config.train.protocol.seed)
    key_model, key_train = jax.random.split(key)
    model = ConditionalScoreNet(
        key_model,
        default_score_net_config(config.data.n_features),
    )
    loss_fn = ConditionalDSMLoss(
        VPSDE(config.sde),
        dsm_settings_from_config(config),
    )
    bundle = TrainerBundle(
        config=config,
        model=model,
        loss_fn=loss_fn,
        optimizer=build_optimizer(config.train),
        paths=paths,
        val_hook=stub_val_imputation_hook,
    )
    M1Trainer(bundle.with_paths()).run(key_train)


if __name__ == "__main__":
    main()
