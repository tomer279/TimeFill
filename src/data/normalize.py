"""Per-feature scalers for M2 real windows (train-observed cells only).

Classes
-------
FeatureScalerOptions
    Log1p / clip / empty-feature knobs for ``FeatureScaler``.
FeatureScaler
    Fit / transform / inverse-transform with optional log1p and clip.

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

from dataclasses import dataclass, replace
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
_DEFAULT_FIT_SLICE = "fit slice"
_EMPTY_COUNT = 0.5


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


@dataclass(frozen=True)
class FeatureScalerOptions:
    """WP5 transform / empty-feature policy for ``FeatureScaler``.

    Public methods
    --------------
    validate
        Check name lists are well-formed.
    to_dict
        Serialize fields for logs / metadata.

    Instance variables
    ------------------
    log1p_features : tuple of str
        Feature names to ``log1p`` before z-score.
    clip_nonnegative : bool
        Clip restored values at 0 when True.
    allow_empty_feature_fallback : bool
        Legacy identity mean 0 / std 1 when a feature has no observations.
    feature_names : tuple of str
        Names aligned with axis ``F``.
    fit_slice_label : str
        Label embedded in empty-feature errors.
    """

    log1p_features: tuple[str, ...] = ()
    clip_nonnegative: bool = False
    allow_empty_feature_fallback: bool = True
    feature_names: tuple[str, ...] = ()
    fit_slice_label: str = _DEFAULT_FIT_SLICE

    def validate(self) -> None:
        """Raise ``ValueError`` if option fields are invalid."""
        if any(not name for name in self.log1p_features):
            raise ValueError("log1p_features entries must be non-empty strings")
        if any(not name for name in self.feature_names):
            raise ValueError("feature_names entries must be non-empty strings")
        if not self.fit_slice_label:
            raise ValueError("fit_slice_label must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict of option fields."""
        return {
            "log1p_features": list(self.log1p_features),
            "clip_nonnegative": self.clip_nonnegative,
            "allow_empty_feature_fallback": (
                self.allow_empty_feature_fallback
            ),
            "feature_names": list(self.feature_names),
            "fit_slice_label": self.fit_slice_label,
        }


class FeatureScaler:
    """Per-feature mean/std scaler with optional log1p and restore clip.

    Missing cells (``m == 0``) never enter fit statistics and stay the
    schema sentinel ``0.0`` under ``transform`` / ``inverse_transform``.
    Fit only on the train split; apply the same stats to val/test.

    Public methods
    --------------
    fit
        Estimate per-feature mean/std from observed cells.
    transform
        Optional log1p then standardize observed cells.
    inverse_transform
        Z-inverse, optional expm1, optional nonnegative clip.
    save
        Persist mean/std/eps plus transform flags to NPZ.
    load
        Classmethod: restore a scaler from NPZ (legacy keys = no transform).
    is_fitted
        Whether ``fit`` or ``load`` has populated stats.
    n_features
        Feature count ``F`` after fitting.

    Instance variables
    ------------------
    eps : float
        Floor for near-zero standard deviations.
    options : FeatureScalerOptions
        Transform / empty-feature policy.
    mean : jax.Array or None
        Per-feature means shaped ``(F,)`` when fitted.
    std : jax.Array or None
        Per-feature stds shaped ``(F,)`` when fitted.
    log1p_mask : jax.Array or None
        Boolean ``(F,)`` mask of log1p features when fitted/loaded.
    """

    eps: float
    options: FeatureScalerOptions
    mean: jnp.ndarray | None
    std: jnp.ndarray | None
    log1p_mask: jnp.ndarray | None

    def __init__(
            self,
            eps: float = _DEFAULT_EPS,
            options: FeatureScalerOptions | None = None,
    ) -> None:
        """Create an unfitted scaler.

        Parameters
        ----------
        eps : float, optional
            Positive floor applied when an observed std is too small.
        options : FeatureScalerOptions or None, optional
            Transform / empty-feature policy (legacy defaults when omitted).
        """
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        resolved = options if options is not None else FeatureScalerOptions()
        resolved.validate()
        self.eps = float(eps)
        self.options = resolved
        self.mean = None
        self.std = None
        self.log1p_mask = None

    @property
    def log1p_features(self) -> tuple[str, ...]:
        """Feature names that use ``log1p`` before z-score."""
        return self.options.log1p_features

    @property
    def clip_nonnegative(self) -> bool:
        """Whether restore clips at 0."""
        return self.options.clip_nonnegative

    @property
    def allow_empty_feature_fallback(self) -> bool:
        """Whether empty features keep identity mean/std."""
        return self.options.allow_empty_feature_fallback

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names aligned with axis ``F``."""
        return self.options.feature_names

    @property
    def fit_slice_label(self) -> str:
        """Label used in empty-feature errors."""
        return self.options.fit_slice_label

    @fit_slice_label.setter
    def fit_slice_label(self, value: str) -> None:
        """Update the fit-slice label used in empty-feature errors."""
        self.options = replace(self.options, fit_slice_label=str(value))

    def is_fitted(self) -> bool:
        """Return whether mean/std are available.

        Returns
        -------
        bool
            ``True`` after a successful ``fit`` or ``load``.
        """
        return self.mean is not None and self.std is not None

    def original_unit_means(self) -> jnp.ndarray:
        """Return fitted means mapped back to original units.

        Returns
        -------
        jax.Array
            Shape ``(F,)``; ``expm1`` applied on log1p features.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        """
        mean, _std = self._require_fitted()
        log1p_mask = self._require_log1p_mask()
        return jnp.where(log1p_mask, jnp.expm1(mean), mean)

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
        mean, _std = self._require_fitted()
        return int(mean.shape[0])

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
        n_features = int(arr.shape[FEATURE_AXIS])
        log1p_mask = self._resolve_log1p_mask(n_features)
        self._assert_log1p_nonnegative(arr, observed, log1p_mask)
        working = _apply_log1p(arr, observed, log1p_mask)
        mean, std, counts = _feature_moments(working, observed, self.eps)
        self._assert_or_fallback_empty(counts, n_features)
        self.mean = mean
        self.std = std
        self.log1p_mask = log1p_mask
        return self

    def transform(
            self,
            values: jnp.ndarray | np.ndarray,
            mask: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray:
        """Optional log1p then standardize; keep missing at the sentinel.

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
        mean, std = self._require_fitted()
        log1p_mask = self._require_log1p_mask()
        arr, observed = _as_masked_windows(values, mask)
        self._assert_log1p_nonnegative(arr, observed, log1p_mask)
        working = _apply_log1p(arr, observed, log1p_mask)
        scaled = (working - mean) / std
        return jnp.where(observed, scaled, MISSING_VALUE_SENTINEL)

    def inverse_transform(
            self,
            values: jnp.ndarray | np.ndarray,
            mask: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray:
        """Z-inverse, optional expm1, optional nonnegative clip.

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
        mean, std = self._require_fitted()
        log1p_mask = self._require_log1p_mask()
        arr, observed = _as_masked_windows(values, mask)
        restored = arr * std + mean
        restored = jnp.where(
            log1p_mask,
            jnp.expm1(restored),
            restored,
        )
        if self.clip_nonnegative:
            restored = jnp.maximum(restored, 0.0)
        return jnp.where(observed, restored, MISSING_VALUE_SENTINEL)

    def save(self, path: Path) -> Path:
        """Write scaler stats and transform flags to ``path`` (NPZ).

        Parameters
        ----------
        path : pathlib.Path
            Destination NPZ (parents created as needed).

        Returns
        -------
        pathlib.Path
            The written path.
        """
        mean, std = self._require_fitted()
        log1p_mask = self._require_log1p_mask()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        names = np.asarray(list(self.feature_names), dtype=str)
        np.savez_compressed(
            out,
            mean=np.asarray(mean, dtype=np.float32),
            std=np.asarray(std, dtype=np.float32),
            eps=np.asarray(self.eps, dtype=np.float32),
            log1p_mask=np.asarray(log1p_mask, dtype=np.bool_),
            clip_nonnegative=np.asarray(
                self.clip_nonnegative, dtype=np.bool_
            ),
            feature_names=names,
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
            mean = jnp.asarray(handle["mean"], dtype=jnp.float32)
            n_features = int(mean.shape[0])
            if "log1p_mask" in handle.files:
                log1p_mask = np.asarray(handle["log1p_mask"], dtype=bool)
            else:
                log1p_mask = np.zeros(n_features, dtype=bool)
            clip = False
            if "clip_nonnegative" in handle.files:
                clip = bool(np.asarray(handle["clip_nonnegative"]).item())
            if "feature_names" in handle.files:
                names = tuple(
                    str(name) for name in np.asarray(handle["feature_names"])
                )
            else:
                names = ()
            log1p_names = tuple(
                names[idx]
                for idx, flag in enumerate(log1p_mask)
                if flag and idx < len(names)
            )
            scaler = cls(
                eps=eps,
                options=FeatureScalerOptions(
                    log1p_features=log1p_names,
                    clip_nonnegative=clip,
                    feature_names=names,
                ),
            )
            scaler.mean = mean
            scaler.std = jnp.asarray(handle["std"], dtype=jnp.float32)
            scaler.log1p_mask = jnp.asarray(log1p_mask, dtype=bool)
        return scaler

    def _require_fitted(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return fitted mean/std, or raise if missing."""
        mean = self.mean
        std = self.std
        if mean is None or std is None:
            raise RuntimeError("FeatureScaler must be fit() or load() first")
        return mean, std

    def _require_log1p_mask(self) -> jnp.ndarray:
        """Return the fitted log1p mask, or raise if missing."""
        mask = self.log1p_mask
        if mask is None:
            raise RuntimeError("FeatureScaler must be fit() or load() first")
        return mask

    def _resolve_log1p_mask(self, n_features: int) -> jnp.ndarray:
        """Build a boolean ``(F,)`` log1p mask from configured names."""
        if not self.log1p_features:
            return jnp.zeros((n_features,), dtype=bool)
        if not self.feature_names:
            raise ValueError(
                "feature_names are required when log1p_features is set"
            )
        if len(self.feature_names) != n_features:
            raise ValueError(
                "len(feature_names) must equal F; "
                f"got {len(self.feature_names)} vs F={n_features}"
            )
        name_to_idx = {
            name: idx for idx, name in enumerate(self.feature_names)
        }
        flags = np.zeros(n_features, dtype=bool)
        for name in self.log1p_features:
            if name not in name_to_idx:
                raise ValueError(
                    f"log1p_features name {name!r} not in feature_names"
                )
            flags[name_to_idx[name]] = True
        return jnp.asarray(flags, dtype=bool)

    def _assert_log1p_nonnegative(
            self,
            values: jnp.ndarray,
            observed: jnp.ndarray,
            log1p_mask: jnp.ndarray,
    ) -> None:
        """Raise if any observed log1p cell is negative."""
        if not bool(jnp.any(log1p_mask)):
            return
        bad = observed & log1p_mask & (values < 0.0)
        if not bool(jnp.any(bad)):
            return
        names = self.feature_names or tuple(
            f"f{idx}" for idx in range(int(values.shape[FEATURE_AXIS]))
        )
        hit = [
            names[idx]
            for idx in range(len(names))
            if bool(jnp.any(bad[:, :, idx]))
        ]
        raise ValueError(
            "log1p features require non-negative observed values; "
            f"got negatives in {hit}"
        )

    def _assert_or_fallback_empty(
            self,
            counts: jnp.ndarray,
            n_features: int,
    ) -> None:
        """Raise or keep identity fallback for empty features."""
        empty = counts < _EMPTY_COUNT
        if not bool(jnp.any(empty)):
            return
        names = self.feature_names
        if len(names) != n_features:
            names = tuple(f"f{idx}" for idx in range(n_features))
        empty_names = [
            names[idx]
            for idx in range(n_features)
            if bool(empty[idx])
        ]
        if self.allow_empty_feature_fallback:
            return
        raise ValueError(
            "no observed cells for feature(s) "
            f"{empty_names} in {self.fit_slice_label}"
        )


def fit_scaler_on_train_split(
        dataset: RealWindowDataset,
        train_idx: np.ndarray,
        scaler: FeatureScaler | None = None,
) -> FeatureScaler:
    """Fit a scaler on train windows only (native observed cells).

    Parameters
    ----------
    dataset : RealWindowDataset
        Full window stack.
    train_idx : numpy.ndarray
        Train split indices (1-D).
    scaler : FeatureScaler or None, optional
        Preconfigured scaler (transform / empty-feature policy). When
        omitted, a legacy empty-transform scaler is used.

    Returns
    -------
    FeatureScaler
        Scaler fitted on ``dataset.subset(train_idx)``.
    """
    train = dataset.subset(np.asarray(train_idx, dtype=np.int64))
    fitted = scaler if scaler is not None else FeatureScaler()
    fitted.fit_slice_label = "train split"
    return fitted.fit(train.values, train.native_masks)


def _as_masked_windows(
        values: jnp.ndarray | np.ndarray,
        mask: jnp.ndarray | np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cast inputs and build a boolean observed mask."""
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


def _apply_log1p(
        values: jnp.ndarray,
        observed: jnp.ndarray,
        log1p_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Return values with log1p on observed log1p features."""
    logged = jnp.log1p(values)
    mixed = jnp.where(log1p_mask, logged, values)
    return jnp.where(observed, mixed, values)


def _feature_moments(
        values: jnp.ndarray,
        observed: jnp.ndarray,
        eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute per-feature mean and std on observed cells.

    Non-observed cells are zeroed before reduction so NaNs under ``m==0``
    do not pollute moments.
    """
    weights = observed.astype(jnp.float32)
    counts = jnp.sum(weights, axis=(0, 1))
    safe_counts = jnp.maximum(counts, 1.0)
    safe_values = jnp.where(observed, values, 0.0)
    sums = jnp.sum(safe_values, axis=(0, 1))
    means = sums / safe_counts
    centered = jnp.where(observed, values - means, 0.0)
    vars_ = jnp.sum(centered * centered, axis=(0, 1)) / safe_counts
    stds = jnp.sqrt(vars_)
    stds = jnp.where(stds < eps, 1.0, stds)
    empty = counts < _EMPTY_COUNT
    means = jnp.where(empty, 0.0, means)
    stds = jnp.where(empty, 1.0, stds)
    n_features = int(values.shape[FEATURE_AXIS])
    if int(means.shape[0]) != n_features:
        raise ValueError("internal feature-axis mismatch in scaler moments")
    return (
        means.astype(jnp.float32),
        stds.astype(jnp.float32),
        counts.astype(jnp.float32),
    )
