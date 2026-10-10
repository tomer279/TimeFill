"""Synthetic fixed-length window generators for the M1 toy path.

Classes
-------
SineWindowGenerator
    Multi-frequency, phase-shifted sine channels plus Gaussian noise.
ArWindowGenerator
    Discrete AR(1) / diagonal VAR(1) windows on a regular grid.

Functions
---------
generate_sine_windows
    Convenience wrapper returning clean sine `x0` of shape `(B, T, F)`.
generate_ar_windows
    Convenience wrapper returning clean AR/VAR `x0` of shape `(B, T, F)`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_SHAPE_BATCH = 0
_SHAPE_TIME = 1
_SHAPE_FEATURES = 2
_N_SHAPE_DIMS = 3
_STACK_FEATURE_AXIS =1

_DEFAULT_SINE_NOISE_STD = 0.05
_DEFAULT_AR_COEFF = 0.8
_DEFAULT_AR_DRIVE_STD = 0.1

_TWO_PI = 2.0 * jnp.pi
_BATCH_MAJOR_FROM_TIME = (1, 0, 2)


class SineWindowGenerator:
    """Build multivariate sine windows with phase-shifted channels.

    Each feature ``f`` uses frequency ``f + 1`` cycles per window and phase
    ``2π f / F``, then adds i.i.d. Gaussian noise.

    Public methods
    --------------
    generate
        Draw a batch of clean windows shaped `(B, T, F)`.
    channel_specs
        Return per-feature ``(frequency_cycles, phase_radians)``.

    Instance variables
    ------------------
    seq_len : int
        Window length `T`.
    n_features : int
        Feature count `F` in `{1, 2, 3}`.
    noise_std : float
        Std of additive Gaussian noise.
    """

    def __init__(
            self,
            seq_len: int,
            n_features: int,
            noise_std: float = _DEFAULT_SINE_NOISE_STD,
    ) -> None:
        """Store window geometry and noise scale.

        Parameters
        ----------
        seq_len : int
            Positive window length `T`.
        n_features : int
            Feature count; must be in `{1, 2, 3}`.
        noise_std : float, optional
            Non-negative Gaussian noise standard deviation.
        """
        _validate_window_geometry(seq_len, n_features)
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        self.seq_len = int(seq_len)
        self.n_features = int(n_features)
        self.noise_std = float(noise_std)

    def generate(self, key: jax.Array, batch_size: int) -> jnp.ndarray:
        """Sample sine windows with additive Gaussian noise.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        batch_size : int
            Number of windows `B`; must be positive.

        Returns
        -------
        jax.Array
            Clean values `x0` with shape `(B, T, F)` and dtype float32.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        time = jnp.arange(self.seq_len, dtype=jnp.float32)
        specs = self.channel_specs()
        channels = []
        for freq_cycles, phase in specs:
            angle = (
                _TWO_PI * freq_cycles * time / float(self.seq_len) + phase
            )
            channels.append(jnp.sin(angle))
        base = jnp.stack(channels, axis=_STACK_FEATURE_AXIS)
        base = jnp.broadcast_to(
            base[None, :, :],
            (batch_size, self.seq_len, self.n_features),
        )
        noise = (
            jax.random.normal(key, base.shape, dtype=jnp.float32)
            * self.noise_std
        )
        return base + noise

    def channel_specs(self) -> tuple[tuple[float, float], ...]:
        """Return ``(cycles_per_window, phase)`` for each feature.

        Returns
        -------
        tuple of tuple of float
            Length-`F` sequence of frequency/phase pairs.
        """
        specs = []
        for feature_idx in range(self.n_features):
            freq_cycles = float(feature_idx + 1)
            phase = _TWO_PI * float(feature_idx) / float(self.n_features)
            specs.append((freq_cycles, float(phase)))
        return tuple(specs)


class ArWindowGenerator:
    """Build AR(1) / diagonal VAR(1) windows on a regular grid.

    The transition is ``A = ar_coeff * I_F`` (stable when
    ``|ar_coeff| < 1``). Innovations are i.i.d. Gaussian.

    Public methods
    --------------
    generate
        Draw a batch of clean windows shaped `(B, T, F)`.
    transition_matrix
        Return the ``(F, F)`` VAR(1) coefficient matrix.

    Instance variables
    ------------------
    seq_len : int
        Window length `T`.
    n_features : int
        Feature count `F` in `{1, 2, 3}`.
    ar_coeff : float
        Diagonal autoregressive coefficient.
    drive_std : float
        Innovation / initial-state standard deviation.
    """

    def __init__(
            self,
            seq_len: int,
            n_features: int,
            ar_coeff: float = _DEFAULT_AR_COEFF,
            drive_std: float = _DEFAULT_AR_DRIVE_STD,
    ) -> None:
        """Store AR geometry and dynamics.

        Parameters
        ----------
        seq_len : int
            Positive window length `T`.
        n_features : int
            Feature count; must be in `{1, 2, 3}`.
        ar_coeff : float, optional
            Diagonal AR coefficient with absolute value ``< 1``.
        drive_std : float, optional
            Non-negative innovation scale.
        """
        _validate_window_geometry(seq_len, n_features)
        if abs(ar_coeff) >= 1.0:
            raise ValueError(
                f"ar_coeff must satisfy |a| < 1 for stability, got {ar_coeff}"
            )
        if drive_std < 0.0:
            raise ValueError(f"drive_std must be >= 0, got {drive_std}")
        self.seq_len = int(seq_len)
        self.n_features = int(n_features)
        self.ar_coeff = float(ar_coeff)
        self.drive_std = float(drive_std)

    def generate(self, key: jax.Array, batch_size: int) -> jnp.ndarray:
        """Sample AR/VAR windows via a time scan.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        batch_size : int
            Number of windows `B`; must be positive.

        Returns
        -------
        jax.Array
            Clean values `x0` with shape `(B, T, F)` and dtype float32.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        init_key, drive_key = jax.random.split(key)
        transition = self.transition_matrix()
        x_init = (
            jax.random.normal(
                init_key,
                (batch_size, self.n_features),
                dtype=jnp.float32,
            )
            * self.drive_std
        )
        if self.seq_len == 1:
            return x_init[:, None, :]

        n_steps = self.seq_len - 1
        drive = (
            jax.random.normal(
                drive_key,
                (n_steps, batch_size, self.n_features),
                dtype=jnp.float32,
            )
            * self.drive_std
        )

        def _step(
                carry: jnp.ndarray,
                eps_t: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            x_next = carry @ transition.T + eps_t
            return x_next, x_next

        _, xs_tail = jax.lax.scan(_step, x_init, drive)
        xs_time_major = jnp.concatenate(
            [x_init[None, :, :], xs_tail],
            axis=0,
        )
        return jnp.transpose(xs_time_major, _BATCH_MAJOR_FROM_TIME)

    def transition_matrix(self) -> jnp.ndarray:
        """Return the diagonal VAR(1) transition matrix.

        Returns
        -------
        jax.Array
            Shape `(F, F)`; ``A = ar_coeff * I``.
        """
        return self.ar_coeff * jnp.eye(self.n_features, dtype=jnp.float32)


def generate_sine_windows(
        key: jax.Array,
        shape: tuple[int, int, int],
        noise_std: float = _DEFAULT_SINE_NOISE_STD,
) -> jnp.ndarray:
    """Generate clean multi-frequency sine windows.

    Parameters
    ----------
    key : jax.Array
        PRNG key for additive noise.
    shape : tuple of int
        ``(B, T, F)`` batch shape.
    noise_std : float, optional
        Gaussian noise standard deviation.

    Returns
    -------
    jax.Array
        Clean `x0` with shape `(B, T, F)`.
    """
    batch_size, seq_len, n_features = _unpack_shape(shape)
    generator = SineWindowGenerator(seq_len, n_features, noise_std)
    return generator.generate(key, batch_size)


def generate_ar_windows(
        key: jax.Array,
        shape: tuple[int, int, int],
        ar_coeff: float = _DEFAULT_AR_COEFF,
        drive_std: float = _DEFAULT_AR_DRIVE_STD,
) -> jnp.ndarray:
    """Generate clean AR(1) / diagonal VAR(1) windows.

    Parameters
    ----------
    key : jax.Array
        PRNG key for initial state and innovations.
    shape : tuple of int
        ``(B, T, F)`` batch shape.
    ar_coeff : float, optional
        Diagonal autoregressive coefficient (``|a| < 1``).
    drive_std : float, optional
        Innovation / initial-state scale.

    Returns
    -------
    jax.Array
        Clean `x0` with shape `(B, T, F)`.
    """
    batch_size, seq_len, n_features = _unpack_shape(shape)
    generator = ArWindowGenerator(
        seq_len,
        n_features,
        ar_coeff,
        drive_std,
    )
    return generator.generate(key, batch_size)


def _unpack_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate and unpack a ``(B, T, F)`` shape tuple.

    Parameters
    ----------
    shape : tuple of int
        Expected length-3 batch shape.

    Returns
    -------
    batch_size : int
        Batch size `B`.
    seq_len : int
        Window length `T`.
    n_features : int
        Feature count `F`.

    Raises
    ------
    ValueError
        If `shape` is not a length-3 tuple of positive-compatible ints.
    """
    if len(shape) != _N_SHAPE_DIMS:
        raise ValueError(
            f"shape must be (B, T, F) with {_N_SHAPE_DIMS} entries; "
            f"got {shape!r}"
        )
    batch_size = int(shape[_SHAPE_BATCH])
    seq_len = int(shape[_SHAPE_TIME])
    n_features = int(shape[_SHAPE_FEATURES])
    return batch_size, seq_len, n_features


def _validate_window_geometry(seq_len: int, n_features: int) -> None:
    """Validate `T` and `F` against M1 constraints.

    Parameters
    ----------
    seq_len : int
        Window length; must be positive.
    n_features : int
        Feature count; must be in `{1, 2, 3}`.

    Raises
    ------
    ValueError
        If geometry violates the M1 contract.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if n_features not in (1, 2, 3):
        raise ValueError(
            f"n_features must be in {{1, 2, 3}} for M1, got {n_features}"
        )
