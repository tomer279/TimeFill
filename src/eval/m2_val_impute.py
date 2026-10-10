"""Val-batch imputation helpers for Milestone 2 (WP4).

Classes
-------
M2ValImputePaths
    Figure / metrics paths under ``artifacts/m2/``.
OriginalScaleRestorer
    Inverse-transform windows / samples / quantiles with a fitted scaler.
MetricsArtifactWriter
    Dump a ``MetricsTable`` to JSON and CSV.

Functions
---------
metric_observation_mask
    Build an EvalTensors mask (``0`` = artificial target with GT).
load_val_pool_with_scaler
    Normalize processed windows with a **saved** (not re-fit) scaler.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.data.normalize import FeatureScaler
from src.data.real import load_processed_dataset
from src.data.real.masks import artificial_target_mask
from src.data.real_config import RealM2Config
from src.data.schema import BATCH_AXIS, N_AXES
from src.eval.metrics import MetricsTable
from src.train.real_pool import RealBatchPool

_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMPUTE_PNG = "imputation_mean_bands.png"
_METRICS_JSON = "val_metrics.json"
_METRICS_CSV = "val_metrics.csv"
_SAMPLE_AXIS = 1
_SAMPLE_TIME_AXIS = 2
_SAMPLE_FEATURE_AXIS = 3
_Q_LOW_INDEX = 0
_Q_HIGH_INDEX = 1
_Q_STACK_AXIS = -1
_ONES = 1.0


class M2ValImputePaths:
    """Resolve WP4 figure and metrics paths under an artifact root.

    Public methods
    --------------
    ensure
        Create the artifact root if needed.
    imputation_bands
        Path for the mean ± bands PNG.
    metrics_json
        Path for the metrics JSON dump.
    metrics_csv
        Path for the metrics CSV dump.

    Instance variables
    ------------------
    root : pathlib.Path
        Artifact root (typically ``artifacts/m2``).
    """

    def __init__(self, root: Path | str) -> None:
        """Store the artifact root path.

        Parameters
        ----------
        root : path-like
            Directory that will hold WP4 outputs.
        """
        self.root = Path(root)

    def ensure(self) -> None:
        """Create the artifact root directory."""
        self.root.mkdir(parents=True, exist_ok=True)

    def imputation_bands(self) -> Path:
        """Return the imputation bands figure path.

        Returns
        -------
        pathlib.Path
            ``root / imputation_mean_bands.png``.
        """
        return self.root / _IMPUTE_PNG

    def metrics_json(self) -> Path:
        """Return the metrics JSON path.

        Returns
        -------
        pathlib.Path
            ``root / val_metrics.json``.
        """
        return self.root / _METRICS_JSON

    def metrics_csv(self) -> Path:
        """Return the metrics CSV path.

        Returns
        -------
        pathlib.Path
            ``root / val_metrics.csv``.
        """
        return self.root / _METRICS_CSV


class OriginalScaleRestorer:
    """Inverse-transform imputation tensors with a fitted ``FeatureScaler``.

    Missing cells under the provided restore mask stay the sentinel ``0.0``.
    Use ``m_native`` so artificially held-out cells (with GT) are restored.

    Public methods
    --------------
    restore_windows
        Inverse-transform a ``(B, T, F)`` tensor.
    restore_samples
        Inverse-transform a ``(B, K, T, F)`` sample stack.
    restore_quantiles
        Inverse-transform low/high quantiles ``(B, T, F, 2)``.

    Instance variables
    ------------------
    scaler : FeatureScaler
        Train-fitted scaler loaded from disk (not re-fit at eval).
    """

    def __init__(self, scaler: FeatureScaler) -> None:
        """Attach a fitted scaler.

        Parameters
        ----------
        scaler : FeatureScaler
            Must already be ``fit`` or ``load``-ed.
        """
        if not scaler.is_fitted():
            raise RuntimeError("scaler must be fitted or loaded before restore")
        self.scaler = scaler

    def restore_windows(
            self,
            values: jnp.ndarray,
            restore_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Map normalized windows back to original units.

        Parameters
        ----------
        values : jax.Array
            Normalized ``(B, T, F)``.
        restore_mask : jax.Array
            Cells with ``1`` are inverse-transformed.

        Returns
        -------
        jax.Array
            Original-scale windows.
        """
        return self.scaler.inverse_transform(values, restore_mask)

    def restore_samples(
            self,
            samples: jnp.ndarray,
            restore_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Map a posterior sample stack back to original units.

        Parameters
        ----------
        samples : jax.Array
            Normalized samples ``(B, K, T, F)``.
        restore_mask : jax.Array
            Window mask ``(B, T, F)`` broadcast over ``K``.

        Returns
        -------
        jax.Array
            Original-scale samples ``(B, K, T, F)``.
        """
        if samples.ndim != N_AXES + 1:
            raise ValueError(
                "samples must have shape (B, K, T, F); "
                f"got {samples.shape}"
            )
        if restore_mask.ndim != N_AXES:
            raise ValueError(
                "restore_mask must have shape (B, T, F); "
                f"got {restore_mask.shape}"
            )
        batch_size = int(samples.shape[BATCH_AXIS])
        n_samples = int(samples.shape[_SAMPLE_AXIS])
        seq_len = int(samples.shape[_SAMPLE_TIME_AXIS])
        n_features = int(samples.shape[_SAMPLE_FEATURE_AXIS])
        flat = samples.reshape((batch_size * n_samples, seq_len, n_features))
        mask_flat = jnp.repeat(restore_mask, n_samples, axis=BATCH_AXIS)
        restored = self.scaler.inverse_transform(flat, mask_flat)
        return restored.reshape((batch_size, n_samples, seq_len, n_features))

    def restore_quantiles(
            self,
            quantiles: jnp.ndarray,
            restore_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Map low/high quantile bands back to original units.

        Parameters
        ----------
        quantiles : jax.Array
            Normalized quantiles ``(B, T, F, 2)``.
        restore_mask : jax.Array
            Window mask ``(B, T, F)``.

        Returns
        -------
        jax.Array
            Original-scale quantiles ``(B, T, F, 2)``.
        """
        low = self.restore_windows(quantiles[..., _Q_LOW_INDEX], restore_mask)
        high = self.restore_windows(quantiles[..., _Q_HIGH_INDEX], restore_mask)
        return jnp.stack([low, high], axis=_Q_STACK_AXIS)


class MetricsArtifactWriter:
    """Write a tiny metrics dump next to WP4 figures.

    Public methods
    --------------
    write_json
        Serialize metrics as JSON.
    write_csv
        Serialize metrics as a two-column CSV.

    Instance variables
    ------------------
    (none)
    """

    def write_json(self, path: Path, table: MetricsTable) -> Path:
        """Write ``table.as_dict()`` to JSON.

        Parameters
        ----------
        path : pathlib.Path
            Destination JSON path.
        table : MetricsTable
            Metrics to dump.

        Returns
        -------
        pathlib.Path
            Path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(table.as_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return out

    def write_csv(self, path: Path, table: MetricsTable) -> Path:
        """Write metric name/value rows to CSV.

        Parameters
        ----------
        path : pathlib.Path
            Destination CSV path.
        table : MetricsTable
            Metrics to dump.

        Returns
        -------
        pathlib.Path
            Path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for name, value in table.rows():
                writer.writerow([name, float(value)])
        return out


def metric_observation_mask(
        m_native: jnp.ndarray,
        m_artificial: jnp.ndarray,
) -> jnp.ndarray:
    """Build an observation-style mask for ``EvalTensors`` / plots.

    ``ImputationEvaluator`` scores cells where ``mask == 0``. This helper
    returns ``0`` only on artificial targets with known ground truth
    (natively observed ∧ artificially missing), and ``1`` elsewhere so
    natively missing cells are **not** scored.

    Parameters
    ----------
    m_native : jax.Array
        Native masks ``(B, T, F)``.
    m_artificial : jax.Array
        Restricted artificial masks ``(B, T, F)``.

    Returns
    -------
    jax.Array
        Float32 mask; ``0`` = score this cell, ``1`` = ignore.
    """
    targets = artificial_target_mask(m_native, m_artificial)
    return (_ONES - targets).astype(jnp.float32)


def load_val_pool_with_scaler(
        config: RealM2Config,
        scaler: FeatureScaler,
) -> RealBatchPool:
    """Load processed windows and normalize with a saved scaler.

    Does **not** re-fit; use the WP3 ``feature_scaler.npz`` so eval matches
    the training normalization.

    Parameters
    ----------
    config : RealM2Config
        Validated real-m2 config (processed_dir, etc.).
    scaler : FeatureScaler
        Scaler loaded from the checkpoint directory.

    Returns
    -------
    RealBatchPool
        Normalized train/val pools (val used for the fixed batch).
    """
    if not scaler.is_fitted():
        raise RuntimeError("scaler must be loaded before normalizing val")
    processed = config.data.resolve_processed_dir(_REPO_ROOT)
    dataset, split = load_processed_dataset(processed)
    values_n = np.asarray(
        scaler.transform(dataset.values, dataset.native_masks),
        dtype=np.float32,
    )
    masks = np.asarray(dataset.native_masks, dtype=np.float32)
    train_idx = np.asarray(split.train_idx, dtype=np.int64)
    val_idx = np.asarray(split.val_idx, dtype=np.int64)
    if val_idx.size == 0:
        raise ValueError("val split is empty; prepare data before WP4 impute")
    return RealBatchPool(
        train_values=jnp.asarray(values_n[train_idx], dtype=jnp.float32),
        train_masks=jnp.asarray(masks[train_idx], dtype=jnp.float32),
        val_values=jnp.asarray(values_n[val_idx], dtype=jnp.float32),
        val_masks=jnp.asarray(masks[val_idx], dtype=jnp.float32),
    )
