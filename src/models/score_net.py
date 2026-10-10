"""Conditional temporal score network for M1 imputation (WP3).

Classes
-------
ScoreNetConfig
    Bundled hyperparameters for `ConditionalScoreNet` (keeps ctor arity small).
TimeConditioner
    Sinusoidal diffusion-time embedding projected to TCN hidden width.
ConditionalScoreNet
    Equinox dilated-TCN score model with mask / observation conditioning.

Functions
---------
default_score_net_config
    Sensible M1 defaults for a given feature count ``F``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
_TIME_MLP_WIDTH_MULT = 2
_TIME_MLP_DEPTH = 1
_OUT_KERNEL = 1
_HALF = 2
_LOG_TIME_PERIOD = 10000.0
_INDEX_PI = jnp.pi


@dataclass(frozen=True)
class ScoreNetConfig:
    """Hyperparameters for the conditional temporal score network.

    Public methods
    --------------
    validate
        Check M1-compatible ranges (features, odd kernel, even time embed).
    to_dict
        Serialize fields to a plain dict.
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
    """

    n_features: int
    hidden_channels: int = _DEFAULT_HIDDEN
    n_layers: int = _DEFAULT_N_LAYERS
    kernel_size: int = _DEFAULT_KERNEL
    time_embed_dim: int = _DEFAULT_TIME_EMBED
    n_index_frequencies: int = _DEFAULT_INDEX_FREQS

    def validate(self) -> None:
        """Raise ``ValueError`` if hyperparameters violate M1 constraints."""
        if self.n_features not in (1, 2, 3):
            raise ValueError(
                "n_features must be in {1, 2, 3}, "
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
        }

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
        features = _sinusoidal_time_embedding(time, self.embed_dim)
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


def default_score_net_config(n_features: int) -> ScoreNetConfig:
    """Build default M1 score-net hyperparameters for feature count ``F``.

    Parameters
    ----------
    n_features : int
        Number of data channels ``F`` in ``{1, 2, 3}``.

    Returns
    -------
    ScoreNetConfig
        Validated default configuration.
    """
    config = ScoreNetConfig(n_features=int(n_features))
    config.validate()
    return config


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


def _sinusoidal_time_embedding(
        time: jnp.ndarray,
        embed_dim: int,
) -> jnp.ndarray:
    """Sinusoidal embedding of a scalar diffusion time.

    Parameters
    ----------
    time : jax.Array
        Scalar time.
    embed_dim : int
        Even embedding dimension.

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
    angles = jnp.asarray(time, dtype=jnp.float32) * frequencies
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
