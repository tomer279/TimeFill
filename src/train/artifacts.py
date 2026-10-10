"""Training artifact paths, history export, and checkpoints for M1.

Classes
-------
ArtifactPaths
    Resolve checkpoint / curve / val-snapshot paths under artifacts.
TrainHistory
    Collect epoch metrics and export CSV / Matplotlib curves.
ValSnapshotPayload
    Packed inputs for the periodic val snapshot hook.

Functions
---------
save_checkpoint
    Serialize an Equinox model with ``eqx.tree_serialise_leaves``.
load_checkpoint
    Deserialize into a like-shaped Equinox template.
resolve_artifact_paths
    Build ``ArtifactPaths`` from a config ``artifact_dir`` string.
stub_val_imputation_hook
    WP5 stub: write a val snapshot marker (no reverse SDE).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.models.score_net import ConditionalScoreNet
from src.plot_io import save_figure

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BEST_CKPT_NAME = "best_score_net.eqx"
_LOSS_CSV_NAME = "loss_curves.csv"
_LOSS_PNG_NAME = "loss_curves.png"
_VAL_SUBDIR = "val_snapshots"
_CKPT_SUBDIR = "checkpoints"


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem layout for M1 training artifacts.

    Public methods
    --------------
    ensure
        Create checkpoint and val-snapshot directories.
    checkpoint_dir
        Return ``<root>/checkpoints``.
    best_checkpoint
        Return path to the best-val score-net file.
    loss_csv
        Return path to the loss-history CSV.
    loss_png
        Return path to the loss-curve PNG.
    val_dir
        Return ``<root>/val_snapshots``.

    Instance variables
    ------------------
    root : pathlib.Path
        Artifact root (typically ``artifacts/m1``).
    """

    root: Path

    def ensure(self) -> None:
        """Create checkpoint and validation snapshot directories."""
        self.checkpoint_dir().mkdir(parents=True, exist_ok=True)
        self.val_dir().mkdir(parents=True, exist_ok=True)

    def checkpoint_dir(self) -> Path:
        """Return the checkpoint directory path.

        Returns
        -------
        pathlib.Path
            ``root / checkpoints``.
        """
        return self.root / _CKPT_SUBDIR

    def best_checkpoint(self) -> Path:
        """Return the best-validation checkpoint path.

        Returns
        -------
        pathlib.Path
            ``.../checkpoints/best_score_net.eqx``.
        """
        return self.checkpoint_dir() / _BEST_CKPT_NAME

    def loss_csv(self) -> Path:
        """Return the loss-history CSV path.

        Returns
        -------
        pathlib.Path
            ``root / loss_curves.csv``.
        """
        return self.root / _LOSS_CSV_NAME

    def loss_png(self) -> Path:
        """Return the loss-curve plot path.

        Returns
        -------
        pathlib.Path
            ``root / loss_curves.png``.
        """
        return self.root / _LOSS_PNG_NAME

    def val_dir(self) -> Path:
        """Return the validation snapshot directory.

        Returns
        -------
        pathlib.Path
            ``root / val_snapshots``.
        """
        return self.root / _VAL_SUBDIR


class TrainHistory:
    """In-memory table of per-epoch training metrics.

    Public methods
    --------------
    append_row
        Append one metric mapping.
    save_csv
        Write all rows to a CSV file.
    save_plot
        Save train/val loss curves with Matplotlib.

    Instance variables
    ------------------
    rows : list of dict
        Metric rows with keys such as ``epoch``, ``train_loss``.
    """

    def __init__(self) -> None:
        """Initialize an empty metric table."""
        self.rows: list[dict[str, Any]] = []

    def append_row(self, row: Mapping[str, Any]) -> None:
        """Append a metric row.

        Parameters
        ----------
        row : Mapping[str, Any]
            Metric name to scalar / string value.
        """
        self.rows.append(dict(row))

    def save_csv(self, path: Path) -> None:
        """Write the history to CSV.

        Parameters
        ----------
        path : pathlib.Path
            Destination CSV path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(self.rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def save_plot(
            self,
            path: Path,
            title: str = "M1 conditional DSM",
    ) -> None:
        """Plot train/val loss versus epoch.

        Parameters
        ----------
        path : pathlib.Path
            Destination PNG path.
        title : str, optional
            Axes title (M2 passes a real-data label).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        epochs = [int(row["epoch"]) for row in self.rows]
        train_y = [float(row["train_loss"]) for row in self.rows]
        val_y = [float(row["val_loss"]) for row in self.rows]
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(epochs, train_y, label="train")
        axis.plot(epochs, val_y, label="val")
        axis.set_xlabel("epoch")
        axis.set_ylabel("DSM loss")
        axis.set_title(title)
        axis.legend()
        fig.tight_layout()
        save_figure(fig, path)
        plt.close(fig)


@dataclass(frozen=True)
class ValSnapshotPayload:
    """Inputs for the periodic validation snapshot hook.

    Public methods
    --------------
    marker_path
        Path for the WP5 stub marker at this step.
    summary_line
        One-line human-readable summary for the stub file.

    Instance variables
    ------------------
    model : ConditionalScoreNet
        Current score network (unused by the WP5 stub).
    x0 : jax.Array
        Clean validation windows.
    mask : jax.Array
        Validation observation mask.
    x_obs : jax.Array
        Sentinel-filled observations.
    step : int
        Epoch index used as the snapshot label.
    out_dir : pathlib.Path
        Directory for snapshot artifacts.
    """

    model: ConditionalScoreNet
    x0: jnp.ndarray
    mask: jnp.ndarray
    x_obs: jnp.ndarray
    step: int
    out_dir: Path

    def marker_path(self) -> Path:
        """Return the stub marker path for this step.

        Returns
        -------
        pathlib.Path
            ``out_dir / val_stub_step_{step}.txt``.
        """
        return self.out_dir / f"val_stub_step_{self.step}.txt"

    def summary_line(self) -> str:
        """Build a short stub summary string.

        Returns
        -------
        str
            Shape summary noting that WP5 impute is not wired yet.
        """
        return (
            f"step={self.step} x0_shape={tuple(self.x0.shape)} "
            f"mask_shape={tuple(self.mask.shape)} "
            "WP5 reverse impute not implemented; stub only"
        )


def stub_val_imputation_hook(payload: ValSnapshotPayload) -> Path:
    """Write a WP5-pending validation snapshot marker.

    Does **not** run Diffrax reverse sampling. WP5 should replace this
    hook with a real imputation mean ± band figure.

    Parameters
    ----------
    payload : ValSnapshotPayload
        Model, val tensors, step, and output directory.

    Returns
    -------
    pathlib.Path
        Path to the written stub marker file.
    """
    payload.out_dir.mkdir(parents=True, exist_ok=True)
    path = payload.marker_path()
    path.write_text(payload.summary_line() + "\n", encoding="utf-8")
    return path


def save_checkpoint(path: Path, model: ConditionalScoreNet) -> None:
    """Serialize an Equinox score network to ``path``.

    Parameters
    ----------
    path : pathlib.Path
        Destination file (``.eqx``).
    model : ConditionalScoreNet
        Model to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_checkpoint(
        path: Path,
        like: ConditionalScoreNet,
) -> ConditionalScoreNet:
    """Load a score network checkpoint into a template module.

    Parameters
    ----------
    path : pathlib.Path
        Checkpoint file written by ``save_checkpoint``.
    like : ConditionalScoreNet
        Template with the same architecture / static fields.

    Returns
    -------
    ConditionalScoreNet
        Restored module.
    """
    return eqx.tree_deserialise_leaves(path, like)


def resolve_artifact_paths(artifact_dir: str) -> ArtifactPaths:
    """Build ``ArtifactPaths`` from a config ``artifact_dir`` string.

    Parameters
    ----------
    artifact_dir : str
        Absolute or repo-relative artifact root.

    Returns
    -------
    ArtifactPaths
        Resolved paths helper.
    """
    return ArtifactPaths(_resolve_artifact_root(artifact_dir))


def _resolve_artifact_root(artifact_dir: str) -> Path:
    """Resolve artifact_dir relative to the repo root when needed.

    Parameters
    ----------
    artifact_dir : str
        Absolute path or repo-relative path string.

    Returns
    -------
    pathlib.Path
        Absolute artifact root.
    """
    path = Path(artifact_dir)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path
