"""Non-UI M4 / M7 impute service (Streamlit calls this later).

Classes
-------
DemoMaskRequest
    Optional MCAR / block demo corruption composed with ``m_native``.
SessionImputeRequest
    Adapted windows plus optional ``K`` / step / quantile / mask overrides.
ImputeTensors
    Original-unit posterior mean / bands / samples.
ImputeContextArrays
    Observations, mean fill, and run masks.
ImputeRunMeta
    Effective ``K`` / solver steps for the run.
SessionImputeResult
    Nested original-unit impute outputs for export / charts.
SessionAssetRequest
    In-memory session model / scaler / SDE knobs for M7 adapt impute.
SessionImputer
    Load pretrained or session assets and run reverse-SDE imputation.

Functions
---------
merge_mean_fill
    Preserve observed cells; fill only ``m == 0`` targets.
impute_bundled_sample
    Convenience: validate bundled CSV → impute with config defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from app.m4_config import AppM4Config, AppM4SampleKnobs, load_app_m4_config
from app.schema_validate import default_app_schema, validate_upload_csv
from app.window_adapter import AdaptedWindows, adapt_validated_frame
from src.data.config import SdeConfig
from src.data.normalize import FeatureScaler
from src.data.real.masks import MaskComposer
from src.data.real_config import load_real_m2_config
from src.data.schema import (
    BATCH_AXIS,
    MASK_OBSERVED,
    TIME_AXIS,
    apply_missing_sentinel,
)
from src.eval.m2_val_impute import OriginalScaleRestorer
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE
from src.train.artifacts import load_checkpoint

_MASK_COMPARE_TOL = 1e-6
_MODE_NONE = "none"
_MODE_RANDOM = "random"
_MODE_BLOCK = "block"
_ALLOWED_MASK_MODES = frozenset({_MODE_NONE, _MODE_RANDOM, _MODE_BLOCK})
_TEMPLATE_SEED = 0
_DEFAULT_QUANTILE_LOW = 0.1
_DEFAULT_QUANTILE_HIGH = 0.9


@dataclass(frozen=True)
class DemoMaskRequest:
    """Optional artificial demo corruption on natively observed cells.

    Public methods
    --------------
    validate
        Check mode / rate / block length.
    compose
        Return ``(x_obs, m_eval, m_artificial)`` in the same units as ``values``.

    Instance variables
    ------------------
    mode : str
        ``none``, ``random`` (MCAR), or ``block``.
    rate : float
        MCAR drop probability (ignored for ``none`` / ``block``).
    block_len : int
        Contiguous block length for ``block`` mode.
    seed : int
        PRNG seed (``seeds.mask``).
    """

    mode: str = _MODE_NONE
    rate: float = 0.0
    block_len: int = 8
    seed: int = 0

    def validate(self) -> None:
        """Raise ``ValueError`` if demo-mask knobs are invalid."""
        if self.mode not in _ALLOWED_MASK_MODES:
            raise ValueError(
                "demo mask mode must be one of "
                f"{sorted(_ALLOWED_MASK_MODES)}, got {self.mode!r}"
            )
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")
        if self.block_len <= 0:
            raise ValueError(
                f"block_len must be positive, got {self.block_len}"
            )

    def compose(
            self,
            values: jnp.ndarray,
            m_native: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compose demo corruption with native masks.

        Parameters
        ----------
        values : jax.Array
            Original-unit windows ``(B, T, F)``.
        m_native : jax.Array
            Native masks ``(B, T, F)``.

        Returns
        -------
        x_obs : jax.Array
            Sentinel-filled observations under ``m_eval``.
        m_eval : jax.Array
            ``m_native * m_artificial`` (or ``m_native`` when mode is none).
        m_artificial : jax.Array
            Restricted artificial mask (all-ones when mode is none).
        """
        self.validate()
        if self.mode == _MODE_NONE:
            m_eval = m_native
            m_artificial = jnp.ones_like(m_native)
            x_obs = apply_missing_sentinel(values, m_eval)
            return x_obs, m_eval, m_artificial
        composer = MaskComposer(self.rate, self.block_len)
        key = jax.random.PRNGKey(self.seed)
        if self.mode == _MODE_RANDOM:
            return composer.compose_random(key, values, m_native)
        return composer.compose_block(key, values, m_native)


@dataclass(frozen=True)
class SessionImputeRequest:
    """Inputs for one non-UI impute call.

    Public methods
    --------------
    validate
        Check that windows are non-empty.
    n_windows
        Batch size ``B``.

    Instance variables
    ------------------
    windows : AdaptedWindows
        WP1 adapter output in original units.
    demo_mask : DemoMaskRequest
        Optional artificial corruption.
    k_samples : int or None
        Override ``sample.K_samples``.
    solver_steps : int or None
        Override ``sample.solver_steps``.
    quantile_low : float or None
        Override lower predictive quantile (default solver 0.1).
    quantile_high : float or None
        Override upper predictive quantile (default solver 0.9).
    """

    windows: AdaptedWindows
    demo_mask: DemoMaskRequest
    k_samples: int | None = None
    solver_steps: int | None = None
    quantile_low: float | None = None
    quantile_high: float | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` if the request cannot be imputed."""
        if self.windows.n_windows() <= 0:
            raise ValueError("AdaptedWindows must contain at least one window")
        self.demo_mask.validate()
        if self.quantile_low is not None or self.quantile_high is not None:
            low = (
                _DEFAULT_QUANTILE_LOW
                if self.quantile_low is None
                else float(self.quantile_low)
            )
            high = (
                _DEFAULT_QUANTILE_HIGH
                if self.quantile_high is None
                else float(self.quantile_high)
            )
            if not 0.0 <= low < high <= 1.0:
                raise ValueError(
                    "require 0 <= quantile_low < quantile_high <= 1, "
                    f"got {low}, {high}"
                )

    def n_windows(self) -> int:
        """Return the number of windows in the request.

        Returns
        -------
        int
            ``B``.
        """
        return self.windows.n_windows()


@dataclass(frozen=True)
class _ImputeSolvePlan:
    """Effective reverse-SDE knobs for one impute call.

    Public methods
    --------------
    build_settings
        Build ``ImputeSolverSettings`` with optional quantile overrides.
    describe
        Short ``K`` / steps summary.

    Instance variables
    ------------------
    knobs : AppM4SampleKnobs
        Effective ``K`` / steps / seeds.
    quantile_low : float or None
        Optional lower quantile override.
    quantile_high : float or None
        Optional upper quantile override.
    """

    knobs: AppM4SampleKnobs
    quantile_low: float | None
    quantile_high: float | None

    def build_settings(self, t_eps: float) -> ImputeSolverSettings:
        """Build Diffrax solver settings for this plan.

        Parameters
        ----------
        t_eps : float
            Terminal diffusion time.

        Returns
        -------
        ImputeSolverSettings
            Validated solver settings.
        """
        low = (
            float(self.quantile_low)
            if self.quantile_low is not None
            else _DEFAULT_QUANTILE_LOW
        )
        high = (
            float(self.quantile_high)
            if self.quantile_high is not None
            else _DEFAULT_QUANTILE_HIGH
        )
        return ImputeSolverSettings(
            solver_steps=int(self.knobs.solver_steps),
            t_eps=float(t_eps),
            quantile_low=low,
            quantile_high=high,
        )

    def describe(self) -> str:
        """Return a short knob summary.

        Returns
        -------
        str
            ``K=... steps=...``.
        """
        return (
            f"K={self.knobs.k_samples} steps={self.knobs.solver_steps}"
        )

@dataclass(frozen=True)
class ImputeTensors:
    """Original-unit posterior tensors from reverse-SDE sampling.

    Public methods
    --------------
    n_windows
        Batch size ``B``.
    mean_shape
        Shape of the posterior mean.

    Instance variables
    ------------------
    mean : numpy.ndarray
        Posterior mean ``(B, T, F)``.
    quantiles : numpy.ndarray
        Low/high bands ``(B, T, F, 2)``.
    samples : numpy.ndarray
        Posterior samples ``(B, K, T, F)``.
    """

    mean: np.ndarray
    quantiles: np.ndarray
    samples: np.ndarray

    def n_windows(self) -> int:
        """Return the number of imputed windows.

        Returns
        -------
        int
            ``B``.
        """
        return int(self.mean.shape[BATCH_AXIS])

    def mean_shape(self) -> tuple[int, ...]:
        """Return the posterior mean shape.

        Returns
        -------
        tuple of int
            ``mean.shape``.
        """
        return tuple(int(dim) for dim in self.mean.shape)


@dataclass(frozen=True)
class ImputeContextArrays:
    """Context arrays and masks used for mean-fill export.

    Public methods
    --------------
    n_windows
        Batch size ``B``.
    observed_cell_count
        Count of ``m_eval == 1`` cells.

    Instance variables
    ------------------
    observations : numpy.ndarray
        Original-unit context ``(B, T, F)`` before fill merge.
    mean_fill : numpy.ndarray
        Export tensor ``(B, T, F)``; observed cells preserved.
    m_eval : numpy.ndarray
        Composed conditioning mask used for the run.
    m_native : numpy.ndarray
        Native / pad mask before demo corruption.
    """

    observations: np.ndarray
    mean_fill: np.ndarray
    m_eval: np.ndarray
    m_native: np.ndarray

    def n_windows(self) -> int:
        """Return the number of windows.

        Returns
        -------
        int
            ``B``.
        """
        return int(self.mean_fill.shape[BATCH_AXIS])

    def observed_cell_count(self) -> int:
        """Count preserved observed cells under ``m_eval``.

        Returns
        -------
        int
            Number of cells with ``m_eval == 1``.
        """
        observed = np.abs(self.m_eval - MASK_OBSERVED) < _MASK_COMPARE_TOL
        return int(np.sum(observed))


@dataclass(frozen=True)
class ImputeRunMeta:
    """Effective sampler knobs for one impute run.

    Public methods
    --------------
    describe
        Short ``K`` / steps summary.
    as_dict
        Plain dict for session / download metadata.

    Instance variables
    ------------------
    k_samples : int
        Effective ``K``.
    solver_steps : int
        Effective solver steps.
    """

    k_samples: int
    solver_steps: int

    def describe(self) -> str:
        """Return a short knob summary.

        Returns
        -------
        str
            ``K=... steps=...``.
        """
        return f"K={self.k_samples} steps={self.solver_steps}"

    def as_dict(self) -> dict[str, int]:
        """Return knobs as a plain dict.

        Returns
        -------
        dict of str to int
            ``k_samples`` and ``solver_steps``.
        """
        return {
            "k_samples": self.k_samples,
            "solver_steps": self.solver_steps,
        }


@dataclass(frozen=True)
class SessionImputeResult:
    """Nested original-unit impute outputs for export / Streamlit.

    Public methods
    --------------
    n_windows
        Batch size ``B``.
    observed_cell_count
        Count of ``m_eval == 1`` cells preserved in the mean fill.

    Instance variables
    ------------------
    tensors : ImputeTensors
        Posterior mean / bands / samples.
    context : ImputeContextArrays
        Observations, mean fill, and masks.
    meta : ImputeRunMeta
        Effective ``K`` / solver steps.
    """

    tensors: ImputeTensors
    context: ImputeContextArrays
    meta: ImputeRunMeta

    def n_windows(self) -> int:
        """Return the number of imputed windows.

        Returns
        -------
        int
            ``B``.
        """
        return self.context.n_windows()

    def observed_cell_count(self) -> int:
        """Count preserved observed cells under ``m_eval``.

        Returns
        -------
        int
            Number of cells with ``m_eval == 1``.
        """
        return self.context.observed_cell_count()


@dataclass(frozen=True)
class _PreparedObservations:
    """Normalized observations and masks ready for reverse-SDE.

    Public methods
    --------------
    n_windows
        Batch size ``B``.
    seq_len
        Window length ``T``.
    """

    observations: np.ndarray
    m_native: np.ndarray
    m_eval: np.ndarray
    x_obs_n: jnp.ndarray

    def n_windows(self) -> int:
        """Return the number of windows.

        Returns
        -------
        int
            ``B``.
        """
        return int(self.observations.shape[BATCH_AXIS])

    def seq_len(self) -> int:
        """Return the window length ``T``.

        Returns
        -------
        int
            ``T``.
        """
        return int(self.x_obs_n.shape[TIME_AXIS])


@dataclass(frozen=True)
class SessionAssetRequest:
    """In-memory session assets for reverse-SDE impute (M7).

    Public methods
    --------------
    validate
        Check ``t_eps``, sample knobs, and scaler ``F``.
    n_features
        Scaler feature count.

    Instance variables
    ------------------
    model : ConditionalScoreNet
        Session score net.
    scaler : FeatureScaler
        Train-only session scaler.
    sde_cfg : SdeConfig
        VP-SDE schedule used during adapt.
    t_eps : float
        Terminal diffusion time.
    sample : AppM4SampleKnobs
        Effective ``K`` / steps / seeds.
    """

    model: ConditionalScoreNet
    scaler: FeatureScaler
    sde_cfg: SdeConfig
    t_eps: float
    sample: AppM4SampleKnobs

    def validate(self) -> None:
        """Raise ``ValueError`` if assets are inconsistent."""
        if self.t_eps <= 0.0:
            raise ValueError(f"t_eps must be positive, got {self.t_eps}")
        self.sample.validate()
        if self.scaler.n_features() <= 0:
            raise ValueError("scaler must have positive n_features")

    def n_features(self) -> int:
        """Return scaler ``F``.

        Returns
        -------
        int
            Feature count.
        """
        return self.scaler.n_features()


@dataclass(frozen=True)
class _LoadedBundle:
    """Internal asset bundle for ``SessionImputer``.

    Public methods
    --------------
    n_features
        Scaler / schema feature count.
    describe
        Short asset summary.
    """

    model: ConditionalScoreNet
    scaler: FeatureScaler
    t_eps: float
    sde_cfg: SdeConfig
    sample: AppM4SampleKnobs
    config: AppM4Config | None = None

    def n_features(self) -> int:
        """Return ``F`` from the loaded scaler.

        Returns
        -------
        int
            Feature count.
        """
        return self.scaler.n_features()

    def describe(self) -> str:
        """Return a short asset summary.

        Returns
        -------
        str
            Path summary from the app config, or session ``F`` / ``K``.
        """
        if self.config is None:
            return (
                f"session_assets F={self.scaler.n_features()} "
                f"K={self.sample.k_samples} "
                f"steps={self.sample.solver_steps}"
            )
        return self.config.paths.describe()


class SessionImputer:
    """Reverse-SDE impute path for M4 pretrained or M7 session assets.

    Public methods
    --------------
    from_app_config
        Load checkpoint, scaler, and SDE knobs from ``app_m4.yaml``.
    from_session_assets
        Bind an in-memory adapt session model / scaler (M7).
    impute
        Run normalize → reverse SDE → inverse → mean-fill merge.

    Instance variables
    ------------------
    config : AppM4Config or None
        Product config when loaded from ``app_m4.yaml``.
    model : ConditionalScoreNet
        Score network (pretrained or session).
    scaler : FeatureScaler
        Train scaler from disk or adapt session.
    t_eps : float
        Terminal diffusion time.
    sde_cfg : SdeConfig
        ``SdeConfig`` matching the bound model.
    sample : AppM4SampleKnobs
        Default ``K`` / steps / seeds for ``impute``.
    """

    def __init__(self, bundle: _LoadedBundle) -> None:
        """Attach a validated asset bundle.

        Parameters
        ----------
        bundle : _LoadedBundle
            Internal loaded assets.
        """
        self.config = bundle.config
        self.model = bundle.model
        self.scaler = bundle.scaler
        self.t_eps = bundle.t_eps
        self.sde_cfg: SdeConfig = bundle.sde_cfg
        self.sample = bundle.sample

    @classmethod
    def from_app_config(cls, path: Path | str) -> SessionImputer:
        """Load pretrained assets from an app YAML path.

        Parameters
        ----------
        path : path-like
            ``configs/app_m4.yaml``.

        Returns
        -------
        SessionImputer
            Ready for ``impute``.

        Raises
        ------
        FileNotFoundError
            If checkpoint or scaler is missing.
        ValueError
            If scaler ``F`` does not match the app schema.
        """
        config = load_app_m4_config(path)
        config.paths.require_exist()
        scaler = FeatureScaler.load(config.paths.scaler)
        if scaler.n_features() != config.schema.n_features:
            raise ValueError(
                "scaler feature count mismatch: "
                f"scaler F={scaler.n_features()}, "
                f"app F={config.schema.n_features}"
            )
        real_cfg = load_real_m2_config(config.paths.real_config)
        template = ConditionalScoreNet(
            jax.random.PRNGKey(_TEMPLATE_SEED),
            default_score_net_config(config.schema.n_features),
        )
        model = load_checkpoint(config.paths.checkpoint, template)
        bundle = _LoadedBundle(
            model=model,
            scaler=scaler,
            t_eps=float(real_cfg.train.protocol.t_eps),
            sde_cfg=real_cfg.sde,
            sample=config.sample,
            config=config,
        )
        return cls(bundle)

    @classmethod
    def from_session_assets(cls, request: SessionAssetRequest) -> SessionImputer:
        """Build an imputer from in-memory adapt session assets.

        Parameters
        ----------
        request : SessionAssetRequest
            Model, scaler, SDE, ``t_eps``, and sample knobs.

        Returns
        -------
        SessionImputer
            Ready for ``impute`` (no M2 claim paths required).
        """
        request.validate()
        bundle = _LoadedBundle(
            model=request.model,
            scaler=request.scaler,
            t_eps=float(request.t_eps),
            sde_cfg=request.sde_cfg,
            sample=request.sample,
            config=None,
        )
        return cls(bundle)

    def impute(self, request: SessionImputeRequest) -> SessionImputeResult:
        """Impute adapted windows and return original-unit fills.

        Parameters
        ----------
        request : SessionImputeRequest
            WP1 windows plus optional demo mask / ``K`` overrides.

        Returns
        -------
        SessionImputeResult
            Mean fill (observed preserved), bands, and samples.
        """
        request.validate()
        knobs = self.sample.effective(
            request.k_samples,
            request.solver_steps,
        )
        plan = _ImputeSolvePlan(
            knobs=knobs,
            quantile_low=request.quantile_low,
            quantile_high=request.quantile_high,
        )
        prepared = self._prepare_observations(request)
        tensors = self._impute_and_restore(prepared, plan)
        mean_fill = merge_mean_fill(
            prepared.observations,
            tensors.mean,
            prepared.m_eval,
        )
        return SessionImputeResult(
            tensors=tensors,
            context=ImputeContextArrays(
                observations=prepared.observations,
                mean_fill=mean_fill,
                m_eval=prepared.m_eval,
                m_native=prepared.m_native,
            ),
            meta=ImputeRunMeta(
                k_samples=knobs.k_samples,
                solver_steps=knobs.solver_steps,
            ),
        )

    def _prepare_observations(
            self,
            request: SessionImputeRequest,
    ) -> _PreparedObservations:
        """Build normalized observations and composed masks.

        Parameters
        ----------
        request : SessionImputeRequest
            Adapted windows and demo-mask knobs.

        Returns
        -------
        _PreparedObservations
            Context arrays plus normalized ``x_obs``.
        """
        values = jnp.asarray(request.windows.values, dtype=jnp.float32)
        m_native = jnp.asarray(request.windows.native_mask, dtype=jnp.float32)
        x_obs, m_eval, _m_art = request.demo_mask.compose(values, m_native)
        x_obs_n = self.scaler.transform(x_obs, m_eval)
        return _PreparedObservations(
            observations=np.asarray(values, dtype=np.float32),
            m_native=np.asarray(m_native, dtype=np.float32),
            m_eval=np.asarray(m_eval, dtype=np.float32),
            x_obs_n=x_obs_n,
        )

    def _impute_and_restore(
            self,
            prepared: _PreparedObservations,
            plan: _ImputeSolvePlan,
    ) -> ImputeTensors:
        """Run reverse SDE and inverse-transform all cells.

        Parameters
        ----------
        prepared : _PreparedObservations
            Normalized observations and masks.
        plan : _ImputeSolvePlan
            Effective ``K`` / steps / quantiles / sample seed.

        Returns
        -------
        ImputeTensors
            Posterior tensors in original units.
        """
        samples_n, mean_n, quantiles_n = self._run_reverse_sde(
            prepared.x_obs_n,
            jnp.asarray(prepared.m_eval, dtype=jnp.float32),
            plan,
        )
        restorer = OriginalScaleRestorer(self.scaler)
        ones = jnp.ones_like(prepared.x_obs_n)
        return ImputeTensors(
            mean=np.asarray(
                restorer.restore_windows(mean_n, ones),
                np.float32,
            ),
            quantiles=np.asarray(
                restorer.restore_quantiles(quantiles_n, ones),
                np.float32,
            ),
            samples=np.asarray(
                restorer.restore_samples(samples_n, ones),
                np.float32,
            ),
        )

    def _run_reverse_sde(
            self,
            x_obs_n: jnp.ndarray,
            m_eval: jnp.ndarray,
            plan: _ImputeSolvePlan,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Call existing ``ReverseSDEImputer.impute`` in normalized space.

        Parameters
        ----------
        x_obs_n : jax.Array
            Normalized observations.
        m_eval : jax.Array
            Composed conditioning mask.
        plan : _ImputeSolvePlan
            Effective ``K`` / steps / quantiles / sample seed.

        Returns
        -------
        samples, mean, quantiles : jax.Array
            Normalized impute tensors.
        """
        settings = plan.build_settings(self.t_eps)
        context = ImputeContext(
            model=self.model,
            sde=VPSDE(self.sde_cfg),
            settings=settings,
        )
        context.validate()
        seq_len = int(x_obs_n.shape[TIME_AXIS])
        knobs = plan.knobs
        return ReverseSDEImputer(context).impute(
            ImputeQuery(
                observations=x_obs_n,
                mask=m_eval,
                timestamps=grid_timestamps(seq_len),
                key=jax.random.PRNGKey(knobs.seed_sample),
                k_samples=knobs.k_samples,
            )
        )


def merge_mean_fill(
        observed_original: np.ndarray,
        mean_original: np.ndarray,
        mask: np.ndarray,
) -> np.ndarray:
    """Preserve observed cells; fill only ``m == 0`` with the posterior mean.

    Parameters
    ----------
    observed_original : numpy.ndarray
        Original-unit context ``(B, T, F)`` (sentinels allowed on gaps).
    mean_original : numpy.ndarray
        Posterior mean in original units, same shape.
    mask : numpy.ndarray
        Composed run mask ``(B, T, F)``; ``1`` preserved, ``0`` filled.

    Returns
    -------
    numpy.ndarray
        Mean-fill export tensor in original units.
    """
    if observed_original.shape != mean_original.shape:
        raise ValueError(
            "observed and mean shapes must match; "
            f"got {observed_original.shape} vs {mean_original.shape}"
        )
    if mask.shape != observed_original.shape:
        raise ValueError(
            "mask shape must match observed; "
            f"got {mask.shape} vs {observed_original.shape}"
        )
    observed = np.abs(mask - MASK_OBSERVED) < _MASK_COMPARE_TOL
    return np.where(
        observed,
        observed_original.astype(np.float32),
        mean_original.astype(np.float32),
    ).astype(np.float32)


def impute_bundled_sample(app_config_path: Path | str) -> SessionImputeResult:
    """Validate the bundled sample CSV and impute with config defaults.

    Parameters
    ----------
    app_config_path : path-like
        ``configs/app_m4.yaml``.

    Returns
    -------
    SessionImputeResult
        Original-unit fills for the demo slice.
    """
    config = load_app_m4_config(app_config_path)
    frame = validate_upload_csv(
        config.paths.sample_csv,
        default_app_schema(),
    )
    windows = adapt_validated_frame(
        frame,
        config.schema.window_request(config.smoke.max_windows),
    )
    imputer = SessionImputer.from_app_config(app_config_path)
    request = SessionImputeRequest(
        windows=windows,
        demo_mask=DemoMaskRequest(
            mode=_MODE_NONE,
            rate=0.0,
            block_len=config.smoke.block_len,
            seed=config.sample.seed_mask,
        ),
    )
    return imputer.impute(request)
