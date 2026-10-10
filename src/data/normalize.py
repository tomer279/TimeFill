"""Per-feature scalers for M2 real windows (train-observed cells only).

Classes
-------
FeatureScaler
    Fit / transform / inverse-transform with mask-aware missing sentinels.

Functions
---------
resolve_scaler_path
    ``<checkpoint_dir>/feature_scaler.npz`` next to the score-net checkpoint.
fit_scaler_on_train_split
    Fit a scaler on train indices of a ``RealWindowDataset``.
default_scaler_filename
    Locked NPZ basename used under ``artifacts/m2/checkpoints/``.

Notes
-----
DSM / SDE training uses normalized tensors. Reported MAE/RMSE/CRPS should be
computed in **original units** after ``inverse_transform`` (see ``docs/EVAL.md``).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.data.real.windows import RealWindowDataset
from src.data.schema import (
    FEATURE_AXIS,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    N_AXES,
)

SCALER_FILENAME = "feature_scaler.npz"
_MASK_TOL = 1e-6
_DEFAULT_EPS = 1e-6


def default_scaler_filename() -> str:
    """Return the locked scaler NPZ basename.

    Returns
    -------
    str
        ``feature_scaler.npz``.
    """
    return SCALER_FILENAME


def resolve_scaler_path(checkpoint_dir: Path) -> Path:
    """Build the scaler path beside score-net checkpoints.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Typically ``artifacts/m2/checkpoints`` from ``ArtifactPaths``.

    Returns
    -------
    pathlib.Path
        ``checkpoint_dir / feature_scaler.npz``.
    """
    return Path(checkpoint_dir) / SCALER_FILENAME


class FeatureScaler:
    """Per-feature mean/std scaler fit on observed cells only.

    Missing cells (``m == 0``) never enter fit statistics and stay the
    schema sentinel ``0.0`` under ``transform`` / ``inverse_transform``.
    Fit only on the train split; apply the same stats to val/test.

    Public methods
    --------------
    fit
        Estimate per-feature mean/std from observed cells.
    transform
        Standardize observed cells; leave missing at the sentinel.
    inverse_transform
        Map normalized observed cells back to original units.
    save
        Persist ``mean`` / ``std`` / ``eps`` to an NPZ path.
    load
        Classmethod: restore a scaler from NPZ.
    is_fitted
        Whether ``fit`` or ``load`` has populated stats.
    n_features
        Feature count ``F`` after fitting.

    Instance variables
    ------------------
    eps : float
        Floor for near-zero standard deviations.
    mean : jax.Array or None
        Per-feature means shaped ``(F,)`` when fitted.
    std : jax.Array or None
        Per-feature stds shaped ``(F,)`` when fitted.
    """

    def __init__(self, eps: float = _DEFAULT_EPS) -> None:
        """Create an unfitted scaler.

        Parameters
        ----------
        eps : float, optional
            Positive floor applied when an observed std is too small.
        """
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)
        self.mean: jnp.ndarray | None = None
        self.std: jnp.ndarray | None = None

    def is_fitted(self) -> bool:
        """Return whether mean/std are available.

        Returns
        -------
        bool
            ``True`` after a successful ``fit`` or ``load``.
        """
        return self.mean is not None and self.std is not None

    def n_features(self) -> int:
        """Return ``F`` for a fitted scaler.

        Returns
        -------
        int
            Length of the mean vector.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        """
        self._require_fitted()
        return int(self.mean.shape[0])

    def fit(
            self,
            values: jnp.ndarray | np.ndarray,
            mask: jnp.ndarray | np.ndarray,
    ) -> FeatureScaler:
        """Fit per-feature mean/std on observed cells only.

        Parameters
        ----------
        values : array-like
            Windows shaped ``(N, T, F)`` (or batch ``B`` on axis 0).
        mask : array-like
            Same shape; ``1`` = observed, ``0`` = missing.

        Returns
        -------
        FeatureScaler
            ``self`` (fitted in place).
        """
        arr, observed = _as_masked_windows(values, mask)
        self.mean, self.std = _feature_moments(arr, observed, self.eps)
        return self

    def transform(
            self,
            values: jnp.ndarray | np.ndarray,
            mask: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray:
        """Standardize observed cells; keep missing at the sentinel.

        Parameters
        ----------
        values : array-like
            Windows shaped ``(N, T, F)``.
        mask : array-like
            Observation mask with the same shape.

        Returns
        -------
        jax.Array
            Normalized windows; missing cells equal ``0.0``.
        """
        self._require_fitted()
        arr, observed = _as_masked_windows(values, mask)
        scaled = (arr - self.mean) / self.std
        return jnp.where(observed, scaled, MISSING_VALUE_SENTINEL)

    def inverse_transform(
            self,
            values: jnp.ndarray | np.ndarray,
            mask: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray:
        """Map normalized observed cells back to original units.

        Parameters
        ----------
        values : array-like
            Normalized windows shaped ``(N, T, F)``.
        mask : array-like
            Observation mask with the same shape.

        Returns
        -------
        jax.Array
            Original-scale windows; missing cells equal ``0.0``.
        """
        self._require_fitted()
        arr, observed = _as_masked_windows(values, mask)
        restored = arr * self.std + self.mean
        return jnp.where(observed, restored, MISSING_VALUE_SENTINEL)

    def save(self, path: Path) -> Path:
        """Write scaler stats to ``path`` (NPZ).

        Parameters
        ----------
        path : pathlib.Path
            Destination NPZ (parents created as needed).

        Returns
        -------
        pathlib.Path
            The written path.
        """
        self._require_fitted()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            mean=np.asarray(self.mean, dtype=np.float32),
            std=np.asarray(self.std, dtype=np.float32),
            eps=np.asarray(self.eps, dtype=np.float32),
        )
        return out

    @classmethod
    def load(cls, path: Path) -> FeatureScaler:
        """Load scaler stats from an NPZ written by ``save``.

        Parameters
        ----------
        path : pathlib.Path
            NPZ path (typically from ``resolve_scaler_path``).

        Returns
        -------
        FeatureScaler
            Fitted scaler instance.
        """
        with np.load(Path(path), allow_pickle=False) as handle:
            eps = float(handle["eps"]) if "eps" in handle.files else _DEFAULT_EPS
            scaler = cls(eps=eps)
            scaler.mean = jnp.asarray(handle["mean"], dtype=jnp.float32)
            scaler.std = jnp.asarray(handle["std"], dtype=jnp.float32)
        return scaler

    def _require_fitted(self) -> None:
        """Raise if mean/std are missing.

        Raises
        ------
        RuntimeError
            When the scaler has not been fitted or loaded.
        """
        if not self.is_fitted():
            raise RuntimeError("FeatureScaler must be fit() or load() first")


def fit_scaler_on_train_split(
        dataset: RealWindowDataset,
        train_idx: np.ndarray,
) -> FeatureScaler:
    """Fit a scaler on train windows only (native observed cells).

    Parameters
    ----------
    dataset : RealWindowDataset
        Full window stack.
    train_idx : numpy.ndarray
        Train split indices (1-D).

    Returns
    -------
    FeatureScaler
        Scaler fitted on ``dataset.subset(train_idx)``.
    """
    train = dataset.subset(np.asarray(train_idx, dtype=np.int64))
    scaler = FeatureScaler()
    return scaler.fit(train.values, train.native_masks)


def _as_masked_windows(
        values: jnp.ndarray | np.ndarray,
        mask: jnp.ndarray | np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cast inputs and build a boolean observed mask.

    Parameters
    ----------
    values : array-like
        Rank-3 value tensor.
    mask : array-like
        Rank-3 mask tensor.

    Returns
    -------
    values : jax.Array
        ``float32`` values.
    observed : jax.Array
        Boolean mask of observed cells.
    """
    arr = jnp.asarray(values, dtype=jnp.float32)
    m_arr = jnp.asarray(mask, dtype=jnp.float32)
    if arr.ndim != N_AXES or m_arr.ndim != N_AXES:
        raise ValueError(
            "values and mask must have rank 3 (N, T, F); "
            f"got values.ndim={arr.ndim}, mask.ndim={m_arr.ndim}"
        )
    if arr.shape != m_arr.shape:
        raise ValueError(
            "values and mask shapes must match; "
            f"got values={arr.shape}, mask={m_arr.shape}"
        )
    observed = jnp.abs(m_arr - MASK_OBSERVED) < _MASK_TOL
    return arr, observed


def _feature_moments(
        values: jnp.ndarray,
        observed: jnp.ndarray,
        eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute per-feature mean and std on observed cells.

    Parameters
    ----------
    values : jax.Array
        Windows ``(N, T, F)``.
    observed : jax.Array
        Boolean observed mask, same shape.
    eps : float
        Floor for near-zero std; empty features get mean ``0``, std ``1``.

    Returns
    -------
    mean : jax.Array
        Shape ``(F,)``.
    std : jax.Array
        Shape ``(F,)``.
    """
    weights = observed.astype(jnp.float32)
    counts = jnp.sum(weights, axis=(0, 1))
    safe_counts = jnp.maximum(counts, 1.0)
    sums = jnp.sum(values * weights, axis=(0, 1))
    means = sums / safe_counts
    centered = (values - means) * weights
    vars_ = jnp.sum(centered * centered, axis=(0, 1)) / safe_counts
    stds = jnp.sqrt(vars_)
    stds = jnp.where(stds < eps, 1.0, stds)
    empty = counts < 0.5
    means = jnp.where(empty, 0.0, means)
    stds = jnp.where(empty, 1.0, stds)
    n_features = int(values.shape[FEATURE_AXIS])
    if int(means.shape[0]) != n_features:
        raise ValueError("internal feature-axis mismatch in scaler moments")
    return means.astype(jnp.float32), stds.astype(jnp.float32)
