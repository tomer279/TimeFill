"""Conditional temporal score network for M1 imputation (WP3).

Classes
-------
ScoreNetConfig
    Bundled hyperparameters for `ConditionalScoreNet` (keeps ctor arity small).
TimeConditioner
    Sinusoidal diffusion-time embedding projected to TCN hidden width.
ConditionalScoreNet
    Legacy ``tcn_v1`` dilated-TCN score model (checkpoint-compatible).
ResidualTCNBlock
    One ``tcn_v2`` residual block with GroupNorm and FiLM.
ConditionalScoreNetV2
    ``tcn_v2`` residual TCN with per-block FiLM time conditioning.

Functions
---------
default_score_net_config
    Sensible M1 defaults for a given feature count ``F``.
sinusoidal_time_embedding
    Sinusoidal embedding of diffusion time (optional ``scale``).
receptive_field
    Temporal receptive field including the stem.
build_score_net
    Build ``tcn_v1`` or ``tcn_v2`` from ``ScoreNetConfig.arch``.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp

from src.data.schema import BATCH_AXIS

_N_CONTEXT_STREAMS = 3  # y_t, m, x_obs
# Unbatched window layout (T, F) before / after the TCN.
_TF_TIME_AXIS = 0
_TF_FEATURE_AXIS = 1
# Channels-first layout (C, T) expected by eqx.nn.Conv1d.
_CT_CHANNEL_AXIS = 0
_CT_TIME_AXIS = 1
_DILATION_BASE = 2
_DEFAULT_HIDDEN = 32
_DEFAULT_N_LAYERS = 3
_DEFAULT_KERNEL = 3
_DEFAULT_TIME_EMBED = 32
_DEFAULT_INDEX_FREQS = 4
_DEFAULT_ARCH = "tcn_v1"
_DEFAULT_TIME_EMBED_SCALE = 1.0
_DEFAULT_TIME_COND = "stem"
_DEFAULT_NORM = "none"
_DEFAULT_NORM_GROUPS = 8
_ALLOWED_ARCH = frozenset({"tcn_v1", "tcn_v2"})
_ALLOWED_TIME_COND = frozenset({"stem", "per_block"})
_ALLOWED_NORM = frozenset({"none", "group"})
_TIME_MLP_WIDTH_MULT = 2
_TIME_MLP_DEPTH = 1
_OUT_KERNEL = 1
_HALF = 2
_LOG_TIME_PERIOD = 10000.0
_INDEX_PI = jnp.pi
_RESIDUAL_SCALE = 2.0 ** 0.5


@dataclass(frozen=True)
class ScoreNetConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters for the conditional temporal score network.

    Public methods
    --------------
    validate
        Check M1-compatible ranges (features, odd kernel, even time embed).
    to_dict
        Serialize fields to a plain dict.
    from_dict
        Build a config from a sidecar / YAML mapping.
    in_channels
        Channel count after fusing ``[y_t, m, x_obs, index_enc]``.

    Instance variables
    ------------------
    n_features : int
        Data feature count ``F`` (score output channels).
    hidden_channels : int
        TCN hidden width.
    n_layers : int
        Number of dilated temporal conv layers after the stem.
    kernel_size : int
        Odd temporal kernel size (SAME padding).
    time_embed_dim : int
        Even sinusoidal diffusion-time embedding size.
    n_index_frequencies : int
        Number of sin/cos pairs for regular-grid index encoding along ``T``.
    arch : str
        Architecture id (``tcn_v1`` legacy; ``tcn_v2`` for WP3).
    dilations : tuple of int or None
        Explicit dilations; ``None`` means derive from ``n_layers``.
    time_embed_scale : float
        Multiplier on raw ``t`` before sinusoids (legacy ``1.0``).
    time_cond : str
        ``stem`` (legacy) or ``per_block`` (WP3).
    norm : str
        ``none`` (legacy) or ``group`` (WP3).
    norm_groups : int
        GroupNorm groups when ``norm == group``.
    """

    n_features: int
    hidden_channels: int = _DEFAULT_HIDDEN
    n_layers: int = _DEFAULT_N_LAYERS
    kernel_size: int = _DEFAULT_KERNEL
    time_embed_dim: int = _DEFAULT_TIME_EMBED
    n_index_frequencies: int = _DEFAULT_INDEX_FREQS
    arch: str = _DEFAULT_ARCH
    dilations: tuple[int, ...] | None = None
    time_embed_scale: float = _DEFAULT_TIME_EMBED_SCALE
    time_cond: str = _DEFAULT_TIME_COND
    norm: str = _DEFAULT_NORM
    norm_groups: int = _DEFAULT_NORM_GROUPS

    def validate(self) -> None:
        """Raise ``ValueError`` if hyperparameters are invalid.

        M1 toy path uses ``F in {1, 2, 3}``; M2 real path uses ``F = 6``.
        """
        self._validate_numeric()
        self._validate_arch_options()

    def _validate_numeric(self) -> None:
        """Check sizes / parities used by the TCN body."""
        if self.n_features <= 0:
            raise ValueError(
                "n_features must be positive, "
                f"got {self.n_features}"
            )
        if self.hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive, "
                f"got {self.hidden_channels}"
            )
        if self.n_layers <= 0:
            raise ValueError(
                f"n_layers must be positive, got {self.n_layers}"
            )
        if self.kernel_size < 1 or self.kernel_size % _HALF == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer, "
                f"got {self.kernel_size}"
            )
        if self.time_embed_dim < _HALF or self.time_embed_dim % _HALF != 0:
            raise ValueError(
                "time_embed_dim must be a positive even integer, "
                f"got {self.time_embed_dim}"
            )
        if self.n_index_frequencies <= 0:
            raise ValueError(
                "n_index_frequencies must be positive, "
                f"got {self.n_index_frequencies}"
            )
        if self.time_embed_scale <= 0.0:
            raise ValueError(
                "time_embed_scale must be positive, "
                f"got {self.time_embed_scale}"
            )
        if self.norm_groups <= 0:
            raise ValueError(
                f"norm_groups must be positive, got {self.norm_groups}"
            )
        if (
                self.norm == "group"
                and self.hidden_channels % self.norm_groups != 0
        ):
            raise ValueError(
                "hidden_channels must be divisible by norm_groups, "
                f"got {self.hidden_channels} and {self.norm_groups}"
            )

    def _validate_arch_options(self) -> None:
        """Check arch enums, dilation schedule, and arch-specific knobs."""
        if self.arch not in _ALLOWED_ARCH:
            raise ValueError(
                f"arch must be one of {sorted(_ALLOWED_ARCH)}, "
                f"got {self.arch!r}"
            )
        if self.time_cond not in _ALLOWED_TIME_COND:
            raise ValueError(
                "time_cond must be one of "
                f"{sorted(_ALLOWED_TIME_COND)}, got {self.time_cond!r}"
            )
        if self.norm not in _ALLOWED_NORM:
            raise ValueError(
                f"norm must be one of {sorted(_ALLOWED_NORM)}, "
                f"got {self.norm!r}"
            )
        if self.dilations is not None:
            if len(self.dilations) != self.n_layers:
                raise ValueError(
                    "len(dilations) must equal n_layers, "
                    f"got {len(self.dilations)} vs {self.n_layers}"
                )
            if any(int(item) <= 0 for item in self.dilations):
                raise ValueError(
                    f"dilations must be positive, got {self.dilations!r}"
                )
        self._validate_arch_knob_honesty()

    def _validate_arch_knob_honesty(self) -> None:
        """Refuse knobs the selected arch does not implement."""
        if self.arch == "tcn_v1":
            if self.dilations is not None:
                raise ValueError(
                    "tcn_v1 ignores dilations; leave dilations=null "
                    "(derived as 2**i from n_layers)"
                )
            if self.time_embed_scale != _DEFAULT_TIME_EMBED_SCALE:
                raise ValueError(
                    "tcn_v1 ignores time_embed_scale; use "
                    f"{_DEFAULT_TIME_EMBED_SCALE}"
                )
            if self.time_cond != _DEFAULT_TIME_COND:
                raise ValueError(
                    "tcn_v1 only supports time_cond="
                    f"{_DEFAULT_TIME_COND!r}"
                )
            if self.norm != _DEFAULT_NORM:
                raise ValueError(
                    f"tcn_v1 only supports norm={_DEFAULT_NORM!r}"
                )
            return
        if self.arch == "tcn_v2" and self.time_cond != "per_block":
            raise ValueError(
                "tcn_v2 requires time_cond='per_block' "
                f"(got {self.time_cond!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "n_features": self.n_features,
            "hidden_channels": self.hidden_channels,
            "n_layers": self.n_layers,
            "kernel_size": self.kernel_size,
            "time_embed_dim": self.time_embed_dim,
            "n_index_frequencies": self.n_index_frequencies,
            "arch": self.arch,
            "dilations": (
                None if self.dilations is None else list(self.dilations)
            ),
            "time_embed_scale": self.time_embed_scale,
            "time_cond": self.time_cond,
            "norm": self.norm,
            "norm_groups": self.norm_groups,
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "ScoreNetConfig":
        """Build a config from a sidecar or YAML ``model`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must include ``n_features`` (or callers pass it in).

        Returns
        -------
        ScoreNetConfig
            Parsed config with legacy defaults for missing keys.
        """
        raw_dil = mapping.get("dilations", None)
        dilations = (
            None if raw_dil is None
            else tuple(int(item) for item in raw_dil)
        )
        if dilations is not None and "n_layers" not in mapping:
            n_layers = len(dilations)
        else:
            n_layers = int(mapping.get("n_layers", _DEFAULT_N_LAYERS))
        return cls(
            n_features=int(mapping["n_features"]),
            hidden_channels=int(
                mapping.get("hidden_channels", _DEFAULT_HIDDEN)
            ),
            n_layers=n_layers,
            kernel_size=int(mapping.get("kernel_size", _DEFAULT_KERNEL)),
            time_embed_dim=int(
                mapping.get("time_embed_dim", _DEFAULT_TIME_EMBED)
            ),
            n_index_frequencies=int(
                mapping.get("n_index_frequencies", _DEFAULT_INDEX_FREQS)
            ),
            arch=str(mapping.get("arch", _DEFAULT_ARCH)),
            dilations=dilations,
            time_embed_scale=float(
                mapping.get("time_embed_scale", _DEFAULT_TIME_EMBED_SCALE)
            ),
            time_cond=str(mapping.get("time_cond", _DEFAULT_TIME_COND)),
            norm=str(mapping.get("norm", _DEFAULT_NORM)),
            norm_groups=int(
                mapping.get("norm_groups", _DEFAULT_NORM_GROUPS)
            ),
        )

    def in_channels(self) -> int:
        """Return fused input channel count for the temporal stem.

        Returns
        -------
        int
            ``3 * F + 2 * n_index_frequencies``.
        """
        index_dim = _HALF * self.n_index_frequencies
        return _N_CONTEXT_STREAMS * self.n_features + index_dim


class TimeConditioner(eqx.Module):
    """Sinusoidal diffusion-time embedding projected to TCN hidden width.

    Public methods
    --------------
    __call__
        Map a scalar time to a per-channel bias vector.
    hidden_width
        Return the output channel count of the projection.

    Instance variables
    ------------------
    embed_dim : int
        Sinusoidal embedding size (static).
    mlp : eqx.nn.MLP
        MLP on sinusoidal features.
    to_hidden : eqx.nn.Linear
        Linear map to TCN hidden channels.
    """

    embed_dim: int = eqx.field(static=True)
    mlp: eqx.nn.MLP
    to_hidden: eqx.nn.Linear

    def __init__(
            self,
            key: jax.Array,
            embed_dim: int,
            hidden_channels: int,
    ) -> None:
        """Build MLP and linear projection for time conditioning.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        embed_dim : int
            Even sinusoidal embedding dimension.
        hidden_channels : int
            TCN hidden width (projection output size).
        """
        mlp_key, lin_key = jax.random.split(key)
        self.embed_dim = int(embed_dim)
        self.mlp = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim,
            width_size=embed_dim * _TIME_MLP_WIDTH_MULT,
            depth=_TIME_MLP_DEPTH,
            key=mlp_key,
        )
        self.to_hidden = eqx.nn.Linear(
            embed_dim,
            hidden_channels,
            key=lin_key,
        )

    def __call__(self, time: jnp.ndarray) -> jnp.ndarray:
        """Embed scalar ``time`` into a hidden-channel bias vector.

        Parameters
        ----------
        time : jax.Array
            Scalar diffusion time.

        Returns
        -------
        jax.Array
            Bias shaped ``(hidden_channels,)``.
        """
        features = sinusoidal_time_embedding(time, self.embed_dim)
        features = self.mlp(features)
        return self.to_hidden(jax.nn.silu(features))

    def hidden_width(self) -> int:
        """Return the projection output dimension.

        Returns
        -------
        int
            ``to_hidden`` output size (TCN hidden channels).
        """
        return int(self.to_hidden.out_features)


class ConditionalScoreNet(eqx.Module):
    """Conditional score network with dilated temporal convolutions.

    Inputs are fused along the feature axis as
    ``[y_t, m, x_obs * m, index_encoding(T)]``, diffusion time ``t`` is
    injected via ``TimeConditioner``, and a non-causal dilated TCN
    predicts a score tensor with the same shape as ``y_t``.

    Public methods
    --------------
    __call__
        Batched forward: ``(B, T, F)`` score estimate.
    count_parameters
        Count trainable array parameters.

    Instance variables
    ------------------
    n_features : int
        Output feature count ``F`` (static).
    n_index_frequencies : int
        Index-encoding frequency count (static).
    time_cond : TimeConditioner
        Diffusion-time embedding and hidden projection.
    in_conv : eqx.nn.Conv1d
        Stem convolution over fused channels.
    convs : tuple of eqx.nn.Conv1d
        Dilated residual temporal layers.
    out_conv : eqx.nn.Conv1d
        ``1 x 1`` projection back to ``F`` score channels.
    """

    n_features: int = eqx.field(static=True)
    n_index_frequencies: int = eqx.field(static=True)
    time_cond: TimeConditioner
    in_conv: eqx.nn.Conv1d
    convs: tuple[eqx.nn.Conv1d, ...]
    out_conv: eqx.nn.Conv1d

    def __init__(self, key: jax.Array, config: ScoreNetConfig) -> None:
        """Build time conditioner, stem, dilated TCN body, and output head.

        Parameters
        ----------
        key : jax.Array
            PRNG key for Equinox layer initialization.
        config : ScoreNetConfig
            Validated hyperparameters (including ``n_features``).
        """
        config.validate()
        self.n_features = int(config.n_features)
        self.n_index_frequencies = int(config.n_index_frequencies)

        n_keys = config.n_layers + 3
        keys = jax.random.split(key, n_keys)
        time_key, stem_key, out_key = keys[0], keys[1], keys[-1]
        body_keys = keys[2:-1]

        self.time_cond = TimeConditioner(
            time_key,
            config.time_embed_dim,
            config.hidden_channels,
        )
        self.in_conv = eqx.nn.Conv1d(
            config.in_channels(),
            config.hidden_channels,
            kernel_size=config.kernel_size,
            padding="SAME",
            key=stem_key,
        )
        self.convs = tuple(
            _make_dilated_conv(
                body_keys[layer_idx],
                config.hidden_channels,
                config.kernel_size,
                _DILATION_BASE ** layer_idx,
            )
            for layer_idx in range(config.n_layers)
        )
        self.out_conv = eqx.nn.Conv1d(
            config.hidden_channels,
            config.n_features,
            kernel_size=_OUT_KERNEL,
            padding="SAME",
            key=out_key,
        )

    def __call__(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray | float,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict the conditional score for a batch of noisy windows.

        Parameters
        ----------
        y_t : jax.Array
            Noisy state shaped ``(B, T, F)``.
        time : float or jax.Array
            Diffusion time: scalar or shape ``(B,)``.
        mask : jax.Array
            Observation mask shaped ``(B, T, F)`` (``1`` observed).
        x_obs : jax.Array
            Observed context shaped ``(B, T, F)``; missing cells should be
            the sentinel ``0.0`` (multiplied by ``mask`` internally).

        Returns
        -------
        jax.Array
            Score estimate shaped ``(B, T, F)``.
        """
        batch_size = int(y_t.shape[BATCH_AXIS])
        time_b = _batch_time(time, batch_size)
        return jax.vmap(self._forward_one)(y_t, time_b, mask, x_obs)

    def count_parameters(self) -> int:
        """Count trainable array parameters in this module.

        Returns
        -------
        int
            Total number of scalar parameters.
        """
        arrays = eqx.filter(self, eqx.is_array)
        leaves = jax.tree_util.tree_leaves(arrays)
        return int(sum(int(leaf.size) for leaf in leaves))

    def _forward_one(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Unbatched forward on shapes ``(T, F)`` with scalar ``time``.

        Parameters
        ----------
        y_t : jax.Array
            Noisy window ``(T, F)``.
        time : jax.Array
            Scalar diffusion time.
        mask : jax.Array
            Mask ``(T, F)``.
        x_obs : jax.Array
            Context ``(T, F)``.

        Returns
        -------
        jax.Array
            Score ``(T, F)``.
        """
        seq_len = int(y_t.shape[_TF_TIME_AXIS])
        context = x_obs * mask
        index = _index_encoding(seq_len, self.n_index_frequencies)
        fused = jnp.concatenate(
            [y_t, mask, context, index],
            axis=_TF_FEATURE_AXIS,
        )
        # (T, C) -> (C, T) for Equinox Conv1d.
        fused_ct = jnp.swapaxes(fused, _TF_TIME_AXIS, _TF_FEATURE_AXIS)
        hidden = self.in_conv(fused_ct)
        time_bias = self.time_cond(time)
        hidden = jax.nn.silu(hidden + time_bias[:, None])
        for conv in self.convs:
            hidden = hidden + jax.nn.silu(conv(hidden))
        score_ct = self.out_conv(hidden)
        # (C, T) -> (T, F)
        return jnp.swapaxes(score_ct, _CT_CHANNEL_AXIS, _CT_TIME_AXIS)


@dataclass(frozen=True)
class _ResidualBlockSpec:
    """Ctor bundle for ``ResidualTCNBlock`` (keeps arity small)."""

    hidden_channels: int
    kernel_size: int
    dilation: int
    time_embed_dim: int
    norm_groups: int  # 0 → skip GroupNorm


class ResidualTCNBlock(eqx.Module):
    """One v2 residual block: GN → SiLU → dilated → FiLM → GN → SiLU → 1×1.

    Public methods
    --------------
    __call__
        Apply the block to channels-first ``(C, T)`` with a time embedding.
    output_channels
        Return the hidden channel count.
    """

    norm1: eqx.nn.GroupNorm | None
    conv_dilated: eqx.nn.Conv1d
    film: eqx.nn.Linear
    norm2: eqx.nn.GroupNorm | None
    conv_1x1: eqx.nn.Conv1d

    def __init__(self, key: jax.Array, spec: _ResidualBlockSpec) -> None:
        """Build norms, dilated conv, FiLM, and 1×1 from ``spec``.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        spec : _ResidualBlockSpec
            Channel / kernel / dilation / FiLM / norm knobs.
        """
        keys = jax.random.split(key, 3)
        channels = int(spec.hidden_channels)
        if int(spec.norm_groups) > 0:
            groups = int(spec.norm_groups)
            self.norm1 = eqx.nn.GroupNorm(groups, channels)
            self.norm2 = eqx.nn.GroupNorm(groups, channels)
        else:
            self.norm1 = None
            self.norm2 = None
        self.conv_dilated = _make_dilated_conv(
            keys[0],
            channels,
            spec.kernel_size,
            spec.dilation,
        )
        self.film = eqx.nn.Linear(
            spec.time_embed_dim,
            _HALF * channels,
            key=keys[1],
        )
        self.conv_1x1 = eqx.nn.Conv1d(
            channels,
            channels,
            kernel_size=_OUT_KERNEL,
            padding="SAME",
            key=keys[2],
        )

    def __call__(
            self,
            hidden: jnp.ndarray,
            time_emb: jnp.ndarray,
    ) -> jnp.ndarray:
        """Residual block on channels-first ``hidden`` shaped ``(C, T)``.

        Parameters
        ----------
        hidden : jax.Array
            Activations ``(C, T)``.
        time_emb : jax.Array
            Time embedding vector ``(time_embed_dim,)``.

        Returns
        -------
        jax.Array
            Updated activations ``(C, T)``.
        """
        residual = hidden
        out = hidden
        if self.norm1 is not None:
            out = self.norm1(out)
        out = jax.nn.silu(out)
        out = self.conv_dilated(out)
        film_params = self.film(time_emb)
        scale, shift = jnp.split(film_params, _HALF, axis=0)
        # (1+γ) keeps init near-identity; plain γ flips/shrinks channels.
        out = out * (1.0 + scale[:, None]) + shift[:, None]
        if self.norm2 is not None:
            out = self.norm2(out)
        out = jax.nn.silu(out)
        out = self.conv_1x1(out)
        return (residual + out) / jnp.asarray(
            _RESIDUAL_SCALE,
            dtype=hidden.dtype,
        )

    def output_channels(self) -> int:
        """Return the block channel width.

        Returns
        -------
        int
            ``conv_1x1`` output channels.
        """
        return int(self.conv_1x1.out_channels)


class ConditionalScoreNetV2(eqx.Module):
    """tcn_v2 residual TCN with per-block FiLM time conditioning.

    ``ScoreNetConfig`` requires ``time_cond='per_block'`` for this arch.

    Public methods
    --------------
    __call__
        Batched forward ``(B, T, F)``.
    count_parameters
        Trainable parameter count.

    Instance variables
    ------------------
    n_features : int
        Output feature count ``F`` (static).
    n_index_frequencies : int
        Index-encoding frequency count (static).
    time_embed_scale : float
        Multiplier on raw ``t`` before sinusoids (static).
    time_mlp : eqx.nn.MLP
        MLP on sinusoidal time features (``in_size`` is embed dim).
    in_conv : eqx.nn.Conv1d
        Stem convolution over fused channels.
    blocks : tuple of ResidualTCNBlock
        Dilated residual body.
    out_conv : eqx.nn.Conv1d
        ``1 x 1`` projection back to ``F``.
    """

    n_features: int = eqx.field(static=True)
    n_index_frequencies: int = eqx.field(static=True)
    time_embed_scale: float = eqx.field(static=True)
    time_mlp: eqx.nn.MLP
    in_conv: eqx.nn.Conv1d
    blocks: tuple[ResidualTCNBlock, ...]
    out_conv: eqx.nn.Conv1d

    def __init__(self, key: jax.Array, config: ScoreNetConfig) -> None:
        """Build stem, time MLP, residual blocks, and output head.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        config : ScoreNetConfig
            Must have ``arch='tcn_v2'``.
        """
        config.validate()
        if config.arch != "tcn_v2":
            raise ValueError(
                "ConditionalScoreNetV2 requires arch='tcn_v2', "
                f"got {config.arch!r}"
            )
        self.n_features = int(config.n_features)
        self.n_index_frequencies = int(config.n_index_frequencies)
        self.time_embed_scale = float(config.time_embed_scale)

        dilations = _resolved_dilations(config)
        n_keys = 3 + len(dilations)
        keys = jax.random.split(key, n_keys)
        time_key, stem_key, out_key = keys[0], keys[1], keys[-1]
        block_keys = keys[2:-1]

        self.time_mlp = eqx.nn.MLP(
            in_size=config.time_embed_dim,
            out_size=config.time_embed_dim,
            width_size=config.time_embed_dim * _TIME_MLP_WIDTH_MULT,
            depth=_TIME_MLP_DEPTH,
            key=time_key,
        )
        self.in_conv = eqx.nn.Conv1d(
            config.in_channels(),
            config.hidden_channels,
            kernel_size=config.kernel_size,
            padding="SAME",
            key=stem_key,
        )
        norm_groups = (
            int(config.norm_groups) if config.norm == "group" else 0
        )
        self.blocks = tuple(
            ResidualTCNBlock(
                block_keys[idx],
                _ResidualBlockSpec(
                    hidden_channels=config.hidden_channels,
                    kernel_size=config.kernel_size,
                    dilation=dilations[idx],
                    time_embed_dim=config.time_embed_dim,
                    norm_groups=norm_groups,
                ),
            )
            for idx in range(len(dilations))
        )
        self.out_conv = eqx.nn.Conv1d(
            config.hidden_channels,
            config.n_features,
            kernel_size=_OUT_KERNEL,
            padding="SAME",
            key=out_key,
        )

    def __call__(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray | float,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict the conditional score for a batch of noisy windows.

        Parameters
        ----------
        y_t : jax.Array
            Noisy state shaped ``(B, T, F)``.
        time : float or jax.Array
            Diffusion time: scalar or shape ``(B,)``.
        mask : jax.Array
            Observation mask shaped ``(B, T, F)``.
        x_obs : jax.Array
            Observed context shaped ``(B, T, F)``.

        Returns
        -------
        jax.Array
            Score estimate shaped ``(B, T, F)``.
        """
        batch_size = int(y_t.shape[BATCH_AXIS])
        time_b = _batch_time(time, batch_size)
        return jax.vmap(self._forward_one)(y_t, time_b, mask, x_obs)

    def count_parameters(self) -> int:
        """Count trainable array parameters in this module.

        Returns
        -------
        int
            Total number of scalar parameters.
        """
        arrays = eqx.filter(self, eqx.is_array)
        leaves = jax.tree_util.tree_leaves(arrays)
        return int(sum(int(leaf.size) for leaf in leaves))

    def _forward_one(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Unbatched forward on shapes ``(T, F)`` with scalar ``time``."""
        seq_len = int(y_t.shape[_TF_TIME_AXIS])
        context = x_obs * mask
        index = _index_encoding(seq_len, self.n_index_frequencies)
        fused = jnp.concatenate(
            [y_t, mask, context, index],
            axis=_TF_FEATURE_AXIS,
        )
        hidden = self.in_conv(
            jnp.swapaxes(fused, _TF_TIME_AXIS, _TF_FEATURE_AXIS)
        )
        time_emb = sinusoidal_time_embedding(
            time,
            int(self.time_mlp.in_size),
            scale=self.time_embed_scale,
        )
        time_emb = jax.nn.silu(self.time_mlp(time_emb))
        for block in self.blocks:
            hidden = block(hidden, time_emb)
        score_ct = self.out_conv(hidden)
        return jnp.swapaxes(score_ct, _CT_CHANNEL_AXIS, _CT_TIME_AXIS)


def default_score_net_config(n_features: int) -> ScoreNetConfig:
    """Build default score-net hyperparameters for feature count ``F``.

    Parameters
    ----------
    n_features : int
        Number of data channels ``F`` (M1: ``{1,2,3}``; M2: ``6``).

    Returns
    -------
    ScoreNetConfig
        Validated default configuration.
    """
    config = ScoreNetConfig(n_features=int(n_features))
    config.validate()
    return config


def _resolved_dilations(config: ScoreNetConfig) -> tuple[int, ...]:
    """Return explicit dilations or ``2**i`` from ``n_layers``."""
    if config.dilations is not None:
        return tuple(int(item) for item in config.dilations)
    return tuple(
        _DILATION_BASE ** layer_idx
        for layer_idx in range(config.n_layers)
    )


def receptive_field(config: ScoreNetConfig) -> int:
    """Temporal receptive field including the stem (dilation 1).

    ``1 + (k-1) * (1 + sum(body_dilations))`` so legacy defaults → 17.

    Parameters
    ----------
    config : ScoreNetConfig
        Score-net hyperparameters.

    Returns
    -------
    int
        Receptive-field width in time steps.
    """
    config.validate()
    body = _resolved_dilations(config)
    span = 1 + sum(body)
    return 1 + (int(config.kernel_size) - 1) * span


def build_score_net(
        key: jax.Array,
        config: ScoreNetConfig,
) -> ConditionalScoreNet | ConditionalScoreNetV2:
    """Build ``tcn_v1`` or ``tcn_v2`` from ``config.arch``.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    config : ScoreNetConfig
        Validated architecture config.

    Returns
    -------
    ConditionalScoreNet or ConditionalScoreNetV2
        Constructed score network.
    """
    config.validate()
    if config.arch == "tcn_v1":
        return ConditionalScoreNet(key, config)
    if config.arch == "tcn_v2":
        return ConditionalScoreNetV2(key, config)
    raise ValueError(f"unsupported arch={config.arch!r}")


def _make_dilated_conv(
        key: jax.Array,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
) -> eqx.nn.Conv1d:
    """Construct one SAME-padded dilated temporal convolution.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    hidden_channels : int
        Input/output channel width.
    kernel_size : int
        Odd kernel size.
    dilation : int
        Dilation factor.

    Returns
    -------
    eqx.nn.Conv1d
        Configured convolution layer.
    """
    return eqx.nn.Conv1d(
        hidden_channels,
        hidden_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        padding="SAME",
        key=key,
    )


def _batch_time(
        time: jnp.ndarray | float,
        batch_size: int,
) -> jnp.ndarray:
    """Broadcast scalar diffusion time to shape ``(B,)``.

    Parameters
    ----------
    time : float or jax.Array
        Scalar or per-example times.
    batch_size : int
        Batch size ``B``.

    Returns
    -------
    jax.Array
        Times shaped ``(B,)``.

    Raises
    ------
    ValueError
        If ``time`` is not scalar or rank-1 with length ``B``.
    """
    time_arr = jnp.asarray(time, dtype=jnp.float32)
    if time_arr.ndim == 0:
        return jnp.broadcast_to(time_arr, (batch_size,))
    if time_arr.ndim == 1 and int(time_arr.shape[0]) == batch_size:
        return time_arr
    raise ValueError(
        "time must be scalar or shape (B,), "
        f"got shape={time_arr.shape} for B={batch_size}"
    )


def sinusoidal_time_embedding(
        time: jnp.ndarray,
        embed_dim: int,
        scale: float = _DEFAULT_TIME_EMBED_SCALE,
) -> jnp.ndarray:
    """Sinusoidal embedding of a scalar diffusion time.

    Parameters
    ----------
    time : jax.Array
        Scalar time.
    embed_dim : int
        Even embedding dimension.
    scale : float, optional
        Multiplier on ``time`` before sinusoids (legacy ``1.0``).

    Returns
    -------
    jax.Array
        Vector of length ``embed_dim``.
    """
    half = embed_dim // _HALF
    frequencies = jnp.exp(
        -jnp.log(jnp.asarray(_LOG_TIME_PERIOD, dtype=jnp.float32))
        * jnp.arange(half, dtype=jnp.float32)
        / float(half)
    )
    angles = (
        jnp.asarray(time, dtype=jnp.float32)
        * jnp.asarray(scale, dtype=jnp.float32)
        * frequencies
    )
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def _index_encoding(
        seq_len: int,
        n_frequencies: int,
) -> jnp.ndarray:
    """Regular-grid positional encoding along the time axis.

    Parameters
    ----------
    seq_len : int
        Window length ``T``.
    n_frequencies : int
        Number of sin/cos frequency pairs.

    Returns
    -------
    jax.Array
        Encoding shaped ``(T, 2 * n_frequencies)``.
    """
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    freq_idx = jnp.arange(n_frequencies, dtype=jnp.float32)[None, :]
    angles = (
        positions
        * (_INDEX_PI / float(seq_len))
        * (_DILATION_BASE ** freq_idx)
    )
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
