"""Research-grade state-space model for a spot-perpetual basis series.

Model
-----
    y_t = level_t + transient_t + observation_noise_t
    level_t = level_{t-1} + level_innovation_t
    transient_t = b * transient_{t-1} + transient_innovation_t

The module deliberately separates parameter fitting from filtering.  A strategy
must fit parameters on historical data only and then call ``filter_state_space``
or ``OnlineBasisFilter`` on later observations.  No smoother is used anywhere.

This is research infrastructure, not an exchange execution client.  It contains
no order placement, authentication, or portfolio management code.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, logit
from scipy.stats import chi2, jarque_bera

try:  # Optional but strongly recommended for repeated maximum-likelihood fitting.
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    njit = None  # type: ignore[assignment]
    _NUMBA_AVAILABLE = False

ArrayLike = Sequence[float] | np.ndarray
SampleMode = Literal["head", "tail"]

_EPS = np.finfo(float).eps
_MIN_SIGMA = 1e-12
_MAX_B = 1.0 - 1e-7
_MIN_B = 1e-7


@dataclass(frozen=True)
class StateSpaceParams:
    """Structural parameters of the local-level plus AR(1) model."""

    b: float
    sigma_level: float
    sigma_transient: float
    sigma_observation: float
    dt_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not (_MIN_B <= self.b <= _MAX_B):
            raise ValueError(f"b must be in [{_MIN_B}, {_MAX_B}], got {self.b}")
        for name in ("sigma_level", "sigma_transient", "sigma_observation"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if not (math.isfinite(self.dt_seconds) and self.dt_seconds > 0.0):
            raise ValueError("dt_seconds must be finite and positive")

    @property
    def half_life_seconds(self) -> float:
        return self.dt_seconds * math.log(2.0) / -math.log(self.b)

    @property
    def transient_sd(self) -> float:
        return self.sigma_transient / math.sqrt(1.0 - self.b * self.b)

    def expected_fraction_reverted(self, horizon_seconds: float) -> float:
        if horizon_seconds < 0:
            raise ValueError("horizon_seconds cannot be negative")
        steps = horizon_seconds / self.dt_seconds
        return 1.0 - self.b**steps

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "StateSpaceParams":
        return cls(**data)


@dataclass
class OptimizerRun:
    start_id: int
    success: bool
    objective: float
    iterations: int
    message: str
    parameters: dict[str, float]


@dataclass
class StateSpaceFit:
    """Result of maximum-likelihood estimation."""

    params: StateSpaceParams
    loglik: float
    converged: bool
    message: str
    n_obs: int
    n_effective: int
    burn_in: int
    sample_mode: SampleMode
    source_n_obs: int
    optimizer_runs: list[OptimizerRun] = field(default_factory=list)
    null_loglik: float | None = None
    null_bic: float | None = None

    @property
    def b(self) -> float:
        return self.params.b

    @property
    def sigma_w(self) -> float:
        """Compatibility alias for the level innovation standard deviation."""
        return self.params.sigma_level

    @property
    def sigma_v(self) -> float:
        """Compatibility alias for the transient innovation standard deviation."""
        return self.params.sigma_transient

    @property
    def sigma_eps(self) -> float:
        """Compatibility alias for the observation-noise standard deviation."""
        return self.params.sigma_observation

    @property
    def half_life(self) -> float:
        """Compatibility alias, in seconds."""
        return self.params.half_life_seconds

    @property
    def aic(self) -> float:
        return 2.0 * 4 - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        return math.log(max(self.n_effective, 1)) * 4 - 2.0 * self.loglik

    @property
    def delta_bic_vs_local_level(self) -> float | None:
        if self.null_bic is None:
            return None
        return self.null_bic - self.bic

    def report(self, scale: float = 1e4, unit: str = "bp") -> str:
        p = self.params
        lines = [
            "State-space basis decomposition",
            f"  source observations       : {self.source_n_obs:,}",
            f"  fitted observations       : {self.n_obs:,}",
            f"  effective likelihood obs  : {self.n_effective:,}",
            f"  converged                 : {self.converged}",
            f"  optimiser message         : {self.message}",
            f"  b (AR1 persistence)       : {p.b:.8f}",
            f"  half-life                 : {p.half_life_seconds:.3f} s",
            f"  sigma_level               : {p.sigma_level * scale:.6f} {unit}/step",
            f"  sigma_transient           : {p.sigma_transient * scale:.6f} {unit}/step",
            f"  sigma_observation         : {p.sigma_observation * scale:.6f} {unit}",
            f"  stationary transient sd   : {p.transient_sd * scale:.6f} {unit}",
            f"  log-likelihood            : {self.loglik:,.3f}",
            f"  AIC / BIC                 : {self.aic:,.3f} / {self.bic:,.3f}",
        ]
        if self.delta_bic_vs_local_level is not None:
            lines.append(
                f"  BIC improvement vs null   : {self.delta_bic_vs_local_level:,.3f}"
            )
        identification = self.identifiability_summary()
        lines.extend([
            f"  near-optimal solutions    : {int(identification['n_near_optimal'])}",
            f"  near-optimal HL ratio     : {identification['half_life_ratio']:.3f}",
            f"  near-optimal signal ratio : {identification['transient_sd_ratio']:.3f}",
        ])
        return "\n".join(lines)

    def identifiability_summary(self, delta_nll: float = 2.0) -> dict[str, float]:
        """Dispersion across optimiser solutions close to the best likelihood.

        Large dispersion means that materially different decompositions explain
        the data almost equally well; convergence alone is then not sufficient.
        """

        if delta_nll < 0:
            raise ValueError("delta_nll cannot be negative")
        if not self.optimizer_runs:
            return {
                "n_near_optimal": 0.0,
                "half_life_ratio": float("inf"),
                "transient_sd_ratio": float("inf"),
            }
        best = min(run.objective for run in self.optimizer_runs)
        near = [run for run in self.optimizer_runs if run.objective <= best + delta_nll]
        half_lives: list[float] = []
        transient_sds: list[float] = []
        for run in near:
            params = StateSpaceParams.from_dict(run.parameters)
            half_lives.append(params.half_life_seconds)
            transient_sds.append(params.transient_sd)

        def ratio(values: list[float]) -> float:
            finite = [value for value in values if math.isfinite(value) and value > 0]
            return max(finite) / min(finite) if finite else float("inf")

        return {
            "n_near_optimal": float(len(near)),
            "half_life_ratio": float(ratio(half_lives)),
            "transient_sd_ratio": float(ratio(transient_sds)),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "params": self.params.to_dict(),
            "loglik": self.loglik,
            "converged": self.converged,
            "message": self.message,
            "n_obs": self.n_obs,
            "n_effective": self.n_effective,
            "burn_in": self.burn_in,
            "sample_mode": self.sample_mode,
            "source_n_obs": self.source_n_obs,
            "optimizer_runs": [asdict(run) for run in self.optimizer_runs],
            "null_loglik": self.null_loglik,
            "null_bic": self.null_bic,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "StateSpaceFit":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["params"] = StateSpaceParams.from_dict(raw["params"])
        raw["optimizer_runs"] = [OptimizerRun(**item) for item in raw["optimizer_runs"]]
        return cls(**raw)


@dataclass(frozen=True)
class ModelAdequacyReport:
    """Explicit verdict on innovation diagnostics.

    ``dynamic_passed`` covers the parts needed for a usable conditional-mean
    model: approximately centred/unit-scaled innovations without detectable
    residual autocorrelation. ``gaussian_passed`` is reported separately because
    heavy tails are common in market data and call for a robust likelihood even
    when the conditional dynamics are otherwise acceptable.
    """

    passed: bool
    dynamic_passed: bool
    gaussian_passed: bool
    diagnostics: dict[str, float]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    alpha: float

    def summary(self) -> str:
        lines = [
            "State-space model adequacy",
            f"  overall pass       : {self.passed}",
            f"  dynamic pass       : {self.dynamic_passed}",
            f"  Gaussian pass      : {self.gaussian_passed}",
        ]
        lines.extend(f"  failure            : {item}" for item in self.failures)
        lines.extend(f"  warning            : {item}" for item in self.warnings)
        return "\n".join(lines)


@dataclass
class FilterResult:
    level: np.ndarray
    transient: np.ndarray
    innovation: np.ndarray
    innovation_variance: np.ndarray
    standardized_innovation: np.ndarray
    filtered_variance_level: np.ndarray
    filtered_covariance: np.ndarray
    filtered_variance_transient: np.ndarray
    observed: np.ndarray
    loglik_contribution: np.ndarray

    @property
    def transient_filter_t(self) -> np.ndarray:
        """Posterior signal-to-filter-uncertainty ratio.

        This is useful for state-estimation confidence, but it is not the
        economic amplitude used for entry thresholds.
        """

        sd = np.sqrt(np.maximum(self.filtered_variance_transient, _EPS))
        return self.transient / sd

    def transient_stationary_z(self, params: StateSpaceParams) -> np.ndarray:
        """Transient divided by its stationary structural standard deviation."""

        return self.transient / params.transient_sd

    @property
    def transient_z(self) -> np.ndarray:
        """Deprecated ambiguous alias for :attr:`transient_filter_t`."""

        warnings.warn(
            "FilterResult.transient_z is ambiguous; use transient_filter_t for "
            "posterior confidence or transient_stationary_z(params) for the "
            "economic signal amplitude",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.transient_filter_t

    def slice(self, start: int | None = None, stop: int | None = None) -> "FilterResult":
        """Return a view-like result restricted to an evaluation interval."""

        sl = slice(start, stop)
        return FilterResult(
            level=self.level[sl],
            transient=self.transient[sl],
            innovation=self.innovation[sl],
            innovation_variance=self.innovation_variance[sl],
            standardized_innovation=self.standardized_innovation[sl],
            filtered_variance_level=self.filtered_variance_level[sl],
            filtered_covariance=self.filtered_covariance[sl],
            filtered_variance_transient=self.filtered_variance_transient[sl],
            observed=self.observed[sl],
            loglik_contribution=self.loglik_contribution[sl],
        )

    def diagnostics(self, max_lag: int = 20) -> dict[str, float]:
        z = self.standardized_innovation[np.isfinite(self.standardized_innovation)]
        if z.size < max(30, max_lag + 5):
            return {
                "n": float(z.size),
                "mean": float("nan"),
                "sd": float("nan"),
                "ljung_box_q": float("nan"),
                "ljung_box_p": float("nan"),
                "jarque_bera": float("nan"),
                "jarque_bera_p": float("nan"),
            }
        centered = z - z.mean()
        denom = float(np.dot(centered, centered))
        acf = []
        for lag in range(1, max_lag + 1):
            acf.append(float(np.dot(centered[lag:], centered[:-lag]) / denom))
        n = z.size
        q = n * (n + 2.0) * sum(
            rho * rho / max(n - lag, 1) for lag, rho in enumerate(acf, start=1)
        )
        jb = jarque_bera(z)
        return {
            "n": float(n),
            "mean": float(z.mean()),
            "sd": float(z.std(ddof=1)),
            "ljung_box_q": float(q),
            "ljung_box_p": float(chi2.sf(q, df=max_lag)),
            "jarque_bera": float(jb.statistic),
            "jarque_bera_p": float(jb.pvalue),
        }

    def adequacy_report(
        self,
        *,
        max_lag: int = 20,
        alpha: float = 0.01,
        mean_tolerance: float = 0.10,
        sd_bounds: tuple[float, float] = (0.80, 1.20),
        require_gaussian: bool = False,
    ) -> ModelAdequacyReport:
        """Turn diagnostics into an explicit research gate.

        The Gaussian verdict is always reported.  Set ``require_gaussian=True``
        when the normal likelihood itself must be accepted rather than treated
        as a quasi-likelihood.
        """

        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1")
        if mean_tolerance < 0.0:
            raise ValueError("mean_tolerance cannot be negative")
        if not (0.0 < sd_bounds[0] < sd_bounds[1]):
            raise ValueError("sd_bounds must be positive and increasing")

        diag = self.diagnostics(max_lag=max_lag)
        failures: list[str] = []
        report_warnings: list[str] = []

        if not math.isfinite(diag["mean"]):
            failures.append("insufficient finite innovations")
        else:
            if abs(diag["mean"]) > mean_tolerance:
                failures.append(
                    f"innovation mean {diag['mean']:.4g} exceeds tolerance "
                    f"{mean_tolerance:.4g}"
                )
            if not (sd_bounds[0] <= diag["sd"] <= sd_bounds[1]):
                failures.append(
                    f"innovation sd {diag['sd']:.4g} outside "
                    f"[{sd_bounds[0]:.4g}, {sd_bounds[1]:.4g}]"
                )
            if diag["ljung_box_p"] < alpha:
                failures.append(
                    f"residual autocorrelation rejected at alpha={alpha:g} "
                    f"(p={diag['ljung_box_p']:.3g})"
                )

        gaussian_passed = bool(
            math.isfinite(diag["jarque_bera_p"])
            and diag["jarque_bera_p"] >= alpha
        )
        if not gaussian_passed:
            report_warnings.append(
                f"Gaussian innovations rejected at alpha={alpha:g} "
                f"(p={diag['jarque_bera_p']:.3g}); use robust inference or a "
                "heavy-tailed observation model"
            )

        dynamic_passed = not failures
        passed = dynamic_passed and (gaussian_passed or not require_gaussian)
        return ModelAdequacyReport(
            passed=passed,
            dynamic_passed=dynamic_passed,
            gaussian_passed=gaussian_passed,
            diagnostics=diag,
            failures=tuple(failures),
            warnings=tuple(report_warnings),
            alpha=float(alpha),
        )


@dataclass
class _FilterState:
    level: float
    transient: float
    p11: float
    p12: float
    p22: float


def _clean_series(values: ArrayLike) -> np.ndarray:
    y = np.asarray(values, dtype=float).reshape(-1)
    if y.size == 0:
        raise ValueError("empty input series")
    return y


def _robust_scale(y: np.ndarray) -> float:
    finite = y[np.isfinite(y)]
    if finite.size < 3:
        return 1e-6
    differences = np.diff(finite)
    mad = np.median(np.abs(differences - np.median(differences)))
    scale = mad / 0.6744897501960817 if mad > 0 else differences.std(ddof=1)
    if not (math.isfinite(scale) and scale > 0):
        scale = max(np.std(finite, ddof=1), 1e-6)
    return max(float(scale), _MIN_SIGMA)


def _theta_to_params(theta: np.ndarray, dt_seconds: float) -> StateSpaceParams:
    b = _MIN_B + (_MAX_B - _MIN_B) * expit(float(theta[0]))
    sigmas = np.exp(np.asarray(theta[1:4], dtype=float))
    return StateSpaceParams(
        b=float(b),
        sigma_level=float(max(sigmas[0], _MIN_SIGMA)),
        sigma_transient=float(max(sigmas[1], _MIN_SIGMA)),
        sigma_observation=float(max(sigmas[2], _MIN_SIGMA)),
        dt_seconds=float(dt_seconds),
    )


def _params_to_theta(params: StateSpaceParams) -> np.ndarray:
    scaled_b = (params.b - _MIN_B) / (_MAX_B - _MIN_B)
    return np.array(
        [
            logit(np.clip(scaled_b, 1e-12, 1 - 1e-12)),
            math.log(params.sigma_level),
            math.log(params.sigma_transient),
            math.log(params.sigma_observation),
        ],
        dtype=float,
    )


def _initial_state(y: np.ndarray, params: StateSpaceParams) -> _FilterState:
    finite_idx = np.flatnonzero(np.isfinite(y))
    if finite_idx.size == 0:
        raise ValueError("series contains no finite observations")
    first = float(y[finite_idx[0]])
    # Initial uncertainty must not depend on observations that arrive later.
    # Use fitted structural scales rather than a full-series sample statistic.
    scale = max(
        params.sigma_level,
        params.sigma_transient,
        params.sigma_observation,
    )
    # Approximately diffuse initial uncertainty for the non-stationary level.
    p11 = max((100.0 * scale) ** 2, 1e-10)
    p22 = params.sigma_transient**2 / max(1.0 - params.b**2, 1e-12)
    return _FilterState(first, 0.0, p11, 0.0, p22)


def _filter_core(
    y: np.ndarray,
    params: StateSpaceParams,
    *,
    initial_state: _FilterState | None = None,
    burn_in: int = 10,
    store: bool = True,
) -> tuple[float, _FilterState, FilterResult | None, int]:
    n = y.size
    state = initial_state or _initial_state(y, params)
    b = params.b
    q_level = params.sigma_level**2
    q_transient = params.sigma_transient**2
    r = params.sigma_observation**2

    if store:
        level = np.full(n, np.nan)
        transient = np.full(n, np.nan)
        innovation = np.full(n, np.nan)
        innovation_variance = np.full(n, np.nan)
        standardized = np.full(n, np.nan)
        p11_out = np.full(n, np.nan)
        p12_out = np.full(n, np.nan)
        p22_out = np.full(n, np.nan)
        ll_out = np.full(n, np.nan)
    else:
        level = transient = innovation = innovation_variance = standardized = None
        p11_out = p12_out = p22_out = ll_out = None

    loglik = 0.0
    effective = 0
    log_two_pi = math.log(2.0 * math.pi)

    for t, observation in enumerate(y):
        # One-step prediction.
        level_pred = state.level
        transient_pred = b * state.transient
        p11_pred = state.p11 + q_level
        p12_pred = b * state.p12
        p22_pred = b * b * state.p22 + q_transient

        if np.isfinite(observation):
            residual = float(observation - level_pred - transient_pred)
            s = p11_pred + 2.0 * p12_pred + p22_pred + r
            if not (math.isfinite(s) and s > 0.0):
                return -math.inf, state, None, effective

            hp1 = p11_pred + p12_pred
            hp2 = p12_pred + p22_pred
            k1 = hp1 / s
            k2 = hp2 / s

            level_new = level_pred + k1 * residual
            transient_new = transient_pred + k2 * residual

            # Covariance update P = P^- - K H P^-; then enforce symmetry/floors.
            p11_new = p11_pred - k1 * hp1
            p12_new = p12_pred - k1 * hp2
            p21_new = p12_pred - k2 * hp1
            p22_new = p22_pred - k2 * hp2
            p12_new = 0.5 * (p12_new + p21_new)
            p11_new = max(float(p11_new), _EPS)
            p22_new = max(float(p22_new), _EPS)

            ll = -0.5 * (log_two_pi + math.log(s) + residual * residual / s)
            if t >= burn_in:
                loglik += ll
                effective += 1

            state = _FilterState(
                level=float(level_new),
                transient=float(transient_new),
                p11=p11_new,
                p12=float(p12_new),
                p22=p22_new,
            )

            if store:
                innovation[t] = residual
                innovation_variance[t] = s
                standardized[t] = residual / math.sqrt(s)
                ll_out[t] = ll
        else:
            # Missing observation: prediction becomes the filtered state.
            state = _FilterState(
                level=float(level_pred),
                transient=float(transient_pred),
                p11=float(p11_pred),
                p12=float(p12_pred),
                p22=float(p22_pred),
            )

        if store:
            level[t] = state.level
            transient[t] = state.transient
            p11_out[t] = state.p11
            p12_out[t] = state.p12
            p22_out[t] = state.p22

    if store:
        result = FilterResult(
            level=level,
            transient=transient,
            innovation=innovation,
            innovation_variance=innovation_variance,
            standardized_innovation=standardized,
            filtered_variance_level=p11_out,
            filtered_covariance=p12_out,
            filtered_variance_transient=p22_out,
            observed=y.copy(),
            loglik_contribution=ll_out,
        )
    else:
        result = None
    return float(loglik), state, result, effective


def _select_sample(y: np.ndarray, max_obs: int | None, mode: SampleMode) -> np.ndarray:
    if max_obs is None or y.size <= max_obs:
        return y.copy()
    if max_obs < 500:
        raise ValueError("max_obs must be at least 500")
    if mode == "head":
        return y[:max_obs].copy()
    if mode == "tail":
        return y[-max_obs:].copy()
    raise ValueError(f"unknown sample mode: {mode}; use 'head' or 'tail'")


if _NUMBA_AVAILABLE:

    @njit(cache=True)  # type: ignore[misc]
    def _numba_loglik(
        theta: np.ndarray,
        y: np.ndarray,
        burn_in: int,
    ) -> tuple[float, int]:
        logistic = 1.0 / (1.0 + math.exp(-theta[0]))
        b = _MIN_B + (_MAX_B - _MIN_B) * logistic
        sigma_level = math.exp(theta[1])
        sigma_transient = math.exp(theta[2])
        sigma_observation = math.exp(theta[3])
        q_level = sigma_level * sigma_level
        q_transient = sigma_transient * sigma_transient
        r = sigma_observation * sigma_observation
        structural_scale = max(sigma_level, sigma_transient, sigma_observation)
        initial_p11 = max((100.0 * structural_scale) ** 2, 1e-10)

        first = -1
        for i in range(y.size):
            if math.isfinite(y[i]):
                first = i
                break
        if first < 0:
            return -math.inf, 0

        level = y[first]
        transient = 0.0
        p11 = initial_p11
        p12 = 0.0
        p22 = q_transient / max(1.0 - b * b, 1e-12)
        loglik = 0.0
        effective = 0
        log_two_pi = math.log(2.0 * math.pi)

        for t in range(y.size):
            level_pred = level
            transient_pred = b * transient
            p11_pred = p11 + q_level
            p12_pred = b * p12
            p22_pred = b * b * p22 + q_transient

            observation = y[t]
            if math.isfinite(observation):
                residual = observation - level_pred - transient_pred
                s = p11_pred + 2.0 * p12_pred + p22_pred + r
                if not math.isfinite(s) or s <= 0.0:
                    return -math.inf, effective
                hp1 = p11_pred + p12_pred
                hp2 = p12_pred + p22_pred
                k1 = hp1 / s
                k2 = hp2 / s
                level = level_pred + k1 * residual
                transient = transient_pred + k2 * residual
                p11_new = p11_pred - k1 * hp1
                p12_a = p12_pred - k1 * hp2
                p12_b = p12_pred - k2 * hp1
                p22_new = p22_pred - k2 * hp2
                p11 = max(p11_new, _EPS)
                p12 = 0.5 * (p12_a + p12_b)
                p22 = max(p22_new, _EPS)
                if t >= burn_in:
                    loglik += -0.5 * (
                        log_two_pi + math.log(s) + residual * residual / s
                    )
                    effective += 1
            else:
                level = level_pred
                transient = transient_pred
                p11 = p11_pred
                p12 = p12_pred
                p22 = p22_pred
        return loglik, effective


def _objective(
    theta: np.ndarray,
    y: np.ndarray,
    dt_seconds: float,
    burn_in: int,
) -> float:
    try:
        if _NUMBA_AVAILABLE:
            ll, effective = _numba_loglik(theta, y, burn_in)
        else:
            params = _theta_to_params(theta, dt_seconds)
            ll, _, _, effective = _filter_core(y, params, burn_in=burn_in, store=False)
    except (ValueError, OverflowError, FloatingPointError):
        return 1e100
    if effective <= 0 or not math.isfinite(ll):
        return 1e100
    return -ll


def _make_starts(scale: float, dt_seconds: float) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    for b in (0.50, 0.90, 0.98):
        for level_share, transient_share, noise_share in (
            (0.55, 0.35, 0.10),
            (0.30, 0.60, 0.10),
        ):
            params = StateSpaceParams(
                b=b,
                sigma_level=max(scale * level_share, _MIN_SIGMA),
                sigma_transient=max(scale * transient_share, _MIN_SIGMA),
                sigma_observation=max(scale * noise_share, _MIN_SIGMA),
                dt_seconds=dt_seconds,
            )
            starts.append(_params_to_theta(params))
    return starts


def _fit_local_level_null(
    y: np.ndarray,
    *,
    dt_seconds: float,
    burn_in: int,
    scale: float,
) -> tuple[float, float]:
    """Fit y_t = level_t + eps_t for a BIC comparison."""

    def objective(theta: np.ndarray) -> float:
        sigma_level, sigma_observation = np.exp(theta)
        # A negligible transient innovation and nearly-zero persistence reproduce
        # the local-level model while reusing the validated filter implementation.
        params = StateSpaceParams(
            b=_MIN_B,
            sigma_level=max(float(sigma_level), _MIN_SIGMA),
            sigma_transient=_MIN_SIGMA,
            sigma_observation=max(float(sigma_observation), _MIN_SIGMA),
            dt_seconds=dt_seconds,
        )
        if _NUMBA_AVAILABLE:
            full_theta = _params_to_theta(params)
            ll, effective = _numba_loglik(full_theta, y, burn_in)
        else:
            ll, _, _, effective = _filter_core(y, params, burn_in=burn_in, store=False)
        return -ll if effective > 0 and math.isfinite(ll) else 1e100

    bounds = [
        (math.log(scale) - 12.0, math.log(scale) + 6.0),
        (math.log(scale) - 12.0, math.log(scale) + 6.0),
    ]
    starts = [
        np.log([scale * 0.8, scale * 0.2]),
        np.log([scale * 0.4, scale * 0.6]),
    ]
    best: OptimizeResult | None = None
    for start in starts:
        candidate = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if best is None or candidate.fun < best.fun:
            best = candidate
    assert best is not None
    loglik = -float(best.fun)
    effective = max(int(np.isfinite(y).sum()) - burn_in, 1)
    bic = math.log(effective) * 2 - 2.0 * loglik
    return loglik, bic


def fit_state_space(
    values: ArrayLike,
    dt: float = 1.0,
    *,
    max_obs: int | None = None,
    sample_mode: SampleMode = "tail",
    burn_in: int = 10,
    min_obs: int = 500,
    starts: Iterable[np.ndarray] | None = None,
    compare_null: bool = True,
    require_convergence: bool = True,
) -> StateSpaceFit:
    """Estimate parameters by innovations maximum likelihood.

    Parameters are fitted only to the supplied array.  For a backtest, the caller
    is responsible for passing a training slice that ends before the evaluation
    period.  ``max_obs`` defaults to ``None``: silent truncation is intentionally
    forbidden.  If a cap is requested, only a contiguous head or tail may be
    selected; uniform thinning is rejected because it changes the effective time
    step and would corrupt persistence and half-life interpretation.
    """

    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive")
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative")
    if sample_mode not in {"head", "tail"}:
        raise ValueError(
            "sample_mode must be 'head' or 'tail'; uniform thinning changes "
            "the effective dt and is intentionally unsupported"
        )

    source = _clean_series(values)
    y = _select_sample(source, max_obs, sample_mode)
    finite_count = int(np.isfinite(y).sum())
    if finite_count < min_obs:
        raise ValueError(f"need at least {min_obs} finite observations, got {finite_count}")

    scale = _robust_scale(y)
    start_list = list(starts) if starts is not None else _make_starts(scale, dt)
    if not start_list:
        raise ValueError("at least one optimiser start is required")

    # Data-scaled bounds prevent variance components from collapsing to machine
    # zero or exploding into numerically equivalent likelihood plateaus.
    sigma_low = math.log(scale) - 12.0
    sigma_high = math.log(scale) + 6.0
    bounds = [(-12.0, 16.0)] + [(sigma_low, sigma_high)] * 3

    best: OptimizeResult | None = None
    run_records: list[OptimizerRun] = []
    for start_id, start in enumerate(start_list):
        start = np.asarray(start, dtype=float)
        if start.shape != (4,):
            raise ValueError("each optimiser start must have shape (4,)")
        candidate = minimize(
            _objective,
            start,
            args=(y, dt, burn_in),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7, "maxls": 40},
        )
        # Powell is slower but useful when the numerical gradient stalls.
        if not candidate.success:
            fallback = minimize(
                _objective,
                candidate.x,
                args=(y, dt, burn_in),
                method="Powell",
                bounds=bounds,
                options={"maxiter": 700, "xtol": 1e-7, "ftol": 1e-8},
            )
            if fallback.fun < candidate.fun:
                candidate = fallback

        params = _theta_to_params(candidate.x, dt)
        run_records.append(
            OptimizerRun(
                start_id=start_id,
                success=bool(candidate.success),
                objective=float(candidate.fun),
                iterations=int(getattr(candidate, "nit", -1)),
                message=str(candidate.message),
                parameters=params.to_dict(),
            )
        )
        if best is None or candidate.fun < best.fun:
            best = candidate

    assert best is not None
    params = _theta_to_params(best.x, dt)
    loglik, _, _, effective = _filter_core(y, params, burn_in=burn_in, store=False)
    converged = bool(best.success and math.isfinite(loglik))
    if require_convergence and not converged:
        raise RuntimeError(f"state-space optimisation did not converge: {best.message}")

    null_loglik: float | None = None
    null_bic: float | None = None
    if compare_null:
        null_loglik, null_bic = _fit_local_level_null(
            y, dt_seconds=dt, burn_in=burn_in, scale=scale
        )

    return StateSpaceFit(
        params=params,
        loglik=float(loglik),
        converged=converged,
        message=str(best.message),
        n_obs=int(y.size),
        n_effective=int(effective),
        burn_in=int(burn_in),
        sample_mode=sample_mode,
        source_n_obs=int(source.size),
        optimizer_runs=run_records,
        null_loglik=null_loglik,
        null_bic=null_bic,
    )


def filter_state_space(
    values: ArrayLike,
    fit_or_params: StateSpaceFit | StateSpaceParams,
    *,
    burn_in: int = 0,
) -> FilterResult:
    """One-sided Kalman filter; it never reads observations after time t."""

    y = _clean_series(values)
    params = fit_or_params.params if isinstance(fit_or_params, StateSpaceFit) else fit_or_params
    _, _, result, _ = _filter_core(y, params, burn_in=burn_in, store=True)
    if result is None:  # pragma: no cover - defensive
        raise RuntimeError("filter failed")
    return result


def smooth_transient(values: ArrayLike, fit: StateSpaceFit) -> np.ndarray:
    """Backward-compatible alias; despite the old name, this is not a smoother."""

    warnings.warn(
        "smooth_transient is a one-sided filter and is deprecated; use "
        "filter_state_space(...).transient",
        DeprecationWarning,
        stacklevel=2,
    )
    return filter_state_space(values, fit).transient


class OnlineBasisFilter:
    """Stateful one-observation-at-a-time Kalman filter for paper/live trading."""

    def __init__(
        self,
        params: StateSpaceParams,
        initial_observation: float,
        *,
        initial_state: _FilterState | None = None,
    ) -> None:
        if not math.isfinite(initial_observation):
            raise ValueError("initial_observation must be finite")
        self.params = params
        seed = np.array([initial_observation], dtype=float)
        self._state = initial_state or _initial_state(seed, params)
        self.n_updates = 0

    @classmethod
    def from_history(
        cls,
        params: StateSpaceParams,
        history: ArrayLike,
    ) -> "OnlineBasisFilter":
        y = _clean_series(history)
        _, state, _, _ = _filter_core(y, params, burn_in=0, store=False)
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            raise ValueError("history has no finite observation")
        obj = cls(params, float(finite[0]), initial_state=state)
        obj.n_updates = int(y.size)
        return obj

    def update(self, observation: float | None) -> dict[str, float]:
        value = np.nan if observation is None else float(observation)
        y = np.array([value], dtype=float)
        _, state, result, _ = _filter_core(
            y,
            self.params,
            initial_state=self._state,
            burn_in=0,
            store=True,
        )
        self._state = state
        self.n_updates += 1
        assert result is not None
        posterior_sd = math.sqrt(max(state.p22, _EPS))
        stationary_z = state.transient / self.params.transient_sd
        filter_t = state.transient / posterior_sd
        return {
            "level": state.level,
            "transient": state.transient,
            "transient_posterior_sd": posterior_sd,
            "transient_stationary_z": stationary_z,
            "transient_filter_t": filter_t,
            # Backward-compatible key, now aligned with the notebook's economic
            # threshold definition rather than posterior uncertainty.
            "transient_z": stationary_z,
            "innovation": float(result.innovation[-1]),
            "innovation_sd": math.sqrt(float(result.innovation_variance[-1])),
        }

    def snapshot(self) -> dict[str, float]:
        return asdict(self._state)


def simulate_state_space(
    n: int,
    params: StateSpaceParams,
    *,
    initial_level: float = 0.0,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = np.random.default_rng(seed)
    level = np.zeros(n, dtype=float)
    transient = np.zeros(n, dtype=float)
    level[0] = initial_level
    transient[0] = rng.normal(scale=params.transient_sd)
    for t in range(1, n):
        level[t] = level[t - 1] + rng.normal(scale=params.sigma_level)
        transient[t] = (
            params.b * transient[t - 1]
            + rng.normal(scale=params.sigma_transient)
        )
    observation_noise = rng.normal(scale=params.sigma_observation, size=n)
    observed = level + transient + observation_noise
    return {
        "observed": observed,
        "level": level,
        "transient": transient,
        "observation_noise": observation_noise,
    }


def no_lookahead_check(
    values: ArrayLike,
    fit_or_params: StateSpaceFit | StateSpaceParams,
    split: int,
    *,
    perturbation: float = 1000.0,
    atol: float = 1e-12,
) -> bool:
    """Verify that changing future observations cannot alter earlier states."""

    y = _clean_series(values)
    if not (1 <= split < y.size):
        raise ValueError("split must be inside the series")
    baseline = filter_state_space(y, fit_or_params).transient
    altered = y.copy()
    altered[split:] = altered[split:] + perturbation
    counterfactual = filter_state_space(altered, fit_or_params).transient
    return bool(np.allclose(baseline[:split], counterfactual[:split], atol=atol, rtol=0.0))


__all__ = [
    "FilterResult",
    "ModelAdequacyReport",
    "OnlineBasisFilter",
    "OptimizerRun",
    "StateSpaceFit",
    "StateSpaceParams",
    "filter_state_space",
    "fit_state_space",
    "no_lookahead_check",
    "simulate_state_space",
    "smooth_transient",
]
