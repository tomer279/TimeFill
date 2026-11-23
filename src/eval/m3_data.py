"""M3 report-split windows, mask composition, and protocol grid helpers.

Classes
-------
SplitWindowBundle
    Normalized windows + native masks for one report split.
ProtocolMaskBundle
    Composed ``m_eval`` / artificial masks and metric helpers.
SplitLoadRequest
    Arguments for loading a normalized split.
ProtocolComposeRequest
    Arguments for drawing artificial masks.
DataBundle
    Windows + fitted scaler for the report split.

Functions
---------
count_target_cells
    Count cells with ``m_native == 1`` and ``m_artificial == 0``.
build_protocol_grid
    Expand ``bench_m3`` protocol config into ``ProtocolKey`` list.
load_split_window_bundle
    Load processed windows, normalize with a saved scaler, take a split.
compose_protocol_masks
    Draw artificial masks and compose ``m_eval``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.data.normalize import FeatureScaler
from src.data.real import load_processed_dataset
from src.data.real.masks import MaskComposer, artificial_target_mask
from src.data.schema import BATCH_AXIS
from src.eval.m2_val_impute import metric_observation_mask
from src.eval.m3_table import ProtocolKey
from src.eval.split_pin import read_split_manifest, verify_split_manifest

_MCAR_BLOCK_STUB = 1
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitWindowBundle:
    """Normalized report-split windows and native masks.

    Public methods
    --------------
    n_windows
        Number of windows in the bundle.
    truncate
        Keep at most ``max_windows`` leading windows.

    Instance variables
    ------------------
    values_n : jax.Array
        Normalized values ``(B, T, F)``.
    m_native : jax.Array
        Native masks ``(B, T, F)``.
    split_name : str
        Split label (``test`` / ``val`` / ``train``).
    split_pinned : bool
        ``True`` when a matching ``split_manifest.json`` was verified.
    """

    values_n: jnp.ndarray
    m_native: jnp.ndarray
    split_name: str
    split_pinned: bool = False

    def n_windows(self) -> int:
        """Return batch size ``B``."""
        return int(self.values_n.shape[BATCH_AXIS])

    def truncate(self, max_windows: int | None) -> "SplitWindowBundle":
        """Return a leading subset when ``max_windows`` is set.

        Parameters
        ----------
        max_windows : int or None
            Cap on windows; ``None`` keeps all.

        Returns
        -------
        SplitWindowBundle
            Possibly truncated bundle.
        """
        if max_windows is None:
            return self
        if max_windows <= 0:
            raise ValueError(
                f"max_windows must be positive when set, got {max_windows}"
            )
        limit = min(int(max_windows), self.n_windows())
        return SplitWindowBundle(
            values_n=self.values_n[:limit],
            m_native=self.m_native[:limit],
            split_name=self.split_name,
            split_pinned=self.split_pinned,
        )


@dataclass(frozen=True)
class ProtocolMaskBundle:
    """Composed evaluation masks for one protocol draw.

    Public methods
    --------------
    metric_mask
        Observation-style mask for ``EvalTensors`` (0 = score).
    n_target_cells
        Count of artificial targets with ground truth.

    Instance variables
    ------------------
    x_obs_n : jax.Array
        Sentinel-filled normalized observations under ``m_eval``.
    m_eval : jax.Array
        ``m_native * m_artificial``.
    m_artificial : jax.Array
        Restricted artificial mask.
    m_native : jax.Array
        Native observation mask.
    """

    x_obs_n: jnp.ndarray
    m_eval: jnp.ndarray
    m_artificial: jnp.ndarray
    m_native: jnp.ndarray

    def metric_mask(self) -> jnp.ndarray:
        """Return mask with 0 on artificial targets that have GT."""
        return metric_observation_mask(self.m_native, self.m_artificial)

    def n_target_cells(self) -> int:
        """Return the number of scored target cells."""
        return count_target_cells(self.m_native, self.m_artificial)


@dataclass(frozen=True)
class SplitLoadRequest:
    """Arguments for ``load_split_window_bundle``.

    Public methods
    --------------
    resolve_processed
        Absolute processed directory.
    describe
        Short string for logs.

    Instance variables
    ------------------
    processed_dir : pathlib.Path
        ``data/processed/m2``.
    scaler : FeatureScaler
        Loaded train scaler.
    split_name : str
        ``test`` / ``val`` / ``train``.
    max_windows : int or None
        Optional leading-window cap for smoke.
    checkpoint_dir : pathlib.Path or None
        Optional dir with ``split_manifest.json`` for A5 pin checks.
    """

    processed_dir: Path
    scaler: FeatureScaler
    split_name: str
    max_windows: int | None
    checkpoint_dir: Path | None = None

    def resolve_processed(self) -> Path:
        """Return the processed directory path."""
        return Path(self.processed_dir)

    def describe(self) -> str:
        """Return a short log label."""
        cap = "all" if self.max_windows is None else str(self.max_windows)
        return f"split={self.split_name}, max_windows={cap}"


@dataclass(frozen=True)
class ProtocolComposeRequest:
    """Inputs for ``compose_protocol_masks``.

    Public methods
    --------------
    composer
        Build a ``MaskComposer`` for this protocol.
    block_len_or_default
        Block length used by the composer (MCAR uses a positive stub).

    Instance variables
    ------------------
    protocol : ProtocolKey
        Protocol identity.
    windows : SplitWindowBundle
        Normalized windows.
    key : jax.Array
        PRNG key for the artificial mask draw.
    """

    protocol: ProtocolKey
    windows: SplitWindowBundle
    key: jnp.ndarray

    def composer(self) -> MaskComposer:
        """Return a ``MaskComposer`` for this protocol."""
        rate = (
            0.0
            if self.protocol.mask_rate is None
            else float(self.protocol.mask_rate)
        )
        return MaskComposer(rate, self.block_len_or_default())

    def block_len_or_default(self) -> int:
        """Return block length (stub when unused for MCAR)."""
        if self.protocol.block_len is None:
            return _MCAR_BLOCK_STUB
        return int(self.protocol.block_len)


@dataclass(frozen=True)
class DataBundle:
    """Normalized report-split windows and the fitted train scaler.

    Public methods
    --------------
    feature_means
        Train-feature means in original units.
    require_feature_count
        Fail if scaler ``F`` mismatches the bench config.

    Instance variables
    ------------------
    windows : SplitWindowBundle
        Normalized values and native masks.
    scaler : FeatureScaler
        Loaded M2 train scaler.
    config_feature_names : tuple of str
        Bench-config names used when the scaler NPZ has none.
    """

    windows: SplitWindowBundle
    scaler: FeatureScaler
    config_feature_names: tuple[str, ...] = ()

    def feature_means(self) -> jnp.ndarray:
        """Return per-feature means in original units for baselines.

        Applies ``expm1`` on log1p features so LOCF / linear fill gaps in
        the same units as restored observations.

        Returns
        -------
        jax.Array
            Means shaped ``(F,)``.
        """
        return self.scaler.original_unit_means()

    def feature_names(self) -> tuple[str, ...]:
        """Return scaler names, else bench-config names, else ``f{i}``."""
        names = tuple(self.scaler.feature_names)
        n_features = int(self.scaler.n_features())
        if len(names) == n_features:
            return names
        if len(self.config_feature_names) == n_features:
            return tuple(self.config_feature_names)
        return tuple(f"f{idx}" for idx in range(n_features))

    def require_feature_count(self, n_features: int) -> None:
        """Raise if the scaler feature count mismatches.

        Parameters
        ----------
        n_features : int
            Expected ``F`` from ``bench_m3.yaml``.
        """
        fitted = int(self.scaler.n_features())
        if fitted != int(n_features):
            raise ValueError(
                "scaler feature count mismatch: "
                f"scaler={fitted}, config={int(n_features)}"
            )


def count_target_cells(
        m_native: jnp.ndarray,
        m_artificial: jnp.ndarray,
) -> int:
    """Count cells with native GT held out by the artificial mask.

    Parameters
    ----------
    m_native : jax.Array
        Native masks ``(B, T, F)``.
    m_artificial : jax.Array
        Restricted artificial masks ``(B, T, F)``.

    Returns
    -------
    int
        Number of metric target cells.
    """
    targets = artificial_target_mask(m_native, m_artificial)
    return int(jnp.sum(targets))


def build_protocol_grid(protocols_cfg: dict) -> tuple[ProtocolKey, ...]:
    """Expand YAML ``protocols`` into a frozen protocol tuple.

    Parameters
    ----------
    protocols_cfg : dict
        Mapping with ``mcar_rates`` and optional ``block``.

    Returns
    -------
    tuple of ProtocolKey
        MCAR entries then optional block.
    """
    rates = tuple(float(rate) for rate in protocols_cfg["mcar_rates"])
    items = [
        ProtocolKey(name="mcar", mask_rate=rate, block_len=None)
        for rate in rates
    ]
    block_cfg = protocols_cfg.get("block") or {}
    if bool(block_cfg.get("enabled", False)):
        items.append(
            ProtocolKey(
                name="block",
                mask_rate=None,
                block_len=int(block_cfg["block_len"]),
            )
        )
    return tuple(items)


def load_split_window_bundle(request: SplitLoadRequest) -> SplitWindowBundle:
    """Load processed windows and normalize with a saved scaler.

    Parameters
    ----------
    request : SplitLoadRequest
        Processed dir, scaler, split name, optional window cap.

    Returns
    -------
    SplitWindowBundle
        Normalized values and native masks.
    """
    if not request.scaler.is_fitted():
        raise RuntimeError("scaler must be loaded before normalizing split")
    split_pinned = _apply_split_pin(
        request.checkpoint_dir,
        request.processed_dir,
    )
    dataset, split = load_processed_dataset(request.processed_dir)
    index = _split_indices(split, request.split_name)
    if index.size == 0:
        raise ValueError(
            f"{request.split_name} split is empty; prepare M2 data first"
        )
    values_n = np.asarray(
        request.scaler.transform(dataset.values, dataset.native_masks),
        dtype=np.float32,
    )
    masks = np.asarray(dataset.native_masks, dtype=np.float32)
    bundle = SplitWindowBundle(
        values_n=jnp.asarray(values_n[index], dtype=jnp.float32),
        m_native=jnp.asarray(masks[index], dtype=jnp.float32),
        split_name=request.split_name,
        split_pinned=split_pinned,
    )
    return bundle.truncate(request.max_windows)


def _apply_split_pin(
        checkpoint_dir: Path | None,
        processed_dir: Path,
) -> bool:
    """Verify a split manifest when ``checkpoint_dir`` is set.

    Returns
    -------
    bool
        ``True`` when a matching manifest was verified; ``False`` when
        absent (warns) or when no checkpoint dir was provided.
    """
    if checkpoint_dir is None:
        return False
    manifest = read_split_manifest(checkpoint_dir)
    if manifest is None:
        _LOG.warning(
            "no split manifest beside checkpoint; split_pinned=false"
        )
        return False
    verify_split_manifest(manifest, processed_dir)
    return True


def compose_protocol_masks(
        request: ProtocolComposeRequest,
) -> ProtocolMaskBundle:
    """Draw artificial masks and compose ``m_eval``.

    Parameters
    ----------
    request : ProtocolComposeRequest
        Protocol, windows, and PRNG key.

    Returns
    -------
    ProtocolMaskBundle
        ``x_obs_n``, ``m_eval``, ``m_artificial``.
    """
    composer = request.composer()
    values = request.windows.values_n
    m_native = request.windows.m_native
    if request.protocol.is_block():
        x_obs, m_eval, m_art = composer.compose_block(
            request.key, values, m_native,
        )
    else:
        x_obs, m_eval, m_art = composer.compose_random(
            request.key, values, m_native,
        )
    return ProtocolMaskBundle(
        x_obs_n=x_obs,
        m_eval=m_eval,
        m_artificial=m_art,
        m_native=m_native,
    )


def _split_indices(split, split_name: str) -> np.ndarray:
    """Select train/val/test indices from a ``SplitIndices`` object."""
    mapping = {
        "train": split.train_idx,
        "val": split.val_idx,
        "test": split.test_idx,
    }
    if split_name not in mapping:
        raise ValueError(
            f"split must be one of {sorted(mapping)}, got {split_name!r}"
        )
    return np.asarray(mapping[split_name], dtype=np.int64)


__all__ = [
    "DataBundle",
    "ProtocolComposeRequest",
    "ProtocolMaskBundle",
    "SplitLoadRequest",
    "SplitWindowBundle",
    "build_protocol_grid",
    "compose_protocol_masks",
    "count_target_cells",
    "load_split_window_bundle",
]
