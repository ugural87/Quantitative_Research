"""Strictly causal baseline signals for the one-day basis workbench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

ArrayLike = Sequence[float] | np.ndarray | pd.Series


@dataclass(frozen=True)
class AR1BaselineParams:
    """Training-only stationary AR(1), the discrete-time OU baseline."""

    intercept: float
    phi: float
    residual_sd: float
    unconditional_mean: float
    stationary_sd: float
    dt_seconds: float
    n_pairs: int

    def __post_init__(self) -> None:
        if not (0.0 < self.phi < 1.0 - 1e-8):
            raise ValueError("AR(1) phi must be in (0, 1 - 1e-8) for an OU baseline")
        if not (math.isfinite(self.residual_sd) and self.residual_sd > 0.0):
            raise ValueError("residual_sd must be finite and positive")
        if not (math.isfinite(self.stationary_sd) and self.stationary_sd > 0.0):
            raise ValueError("stationary_sd must be finite and positive")
        if not (math.isfinite(self.dt_seconds) and self.dt_seconds > 0.0):
            raise ValueError("dt_seconds must be finite and positive")
        if self.n_pairs < 3:
            raise ValueError("n_pairs must be at least three")

    @property
    def half_life_seconds(self) -> float:
        return self.dt_seconds * math.log(2.0) / -math.log(self.phi)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_array(values: ArrayLike) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("values cannot be empty")
    return x


def causal_rolling_z(
    values: ArrayLike,
    *,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    """Current value relative to a mean and scale estimated through ``t-1``.

    The current observation is deliberately excluded from its own reference
    window.  This makes the baseline a clean causal comparator rather than a
    rolling statistic that partially normalises away the event being traded.
    """

    if window < 3:
        raise ValueError("window must be at least three")
    required = window if min_periods is None else min_periods
    if not (2 <= required <= window):
        raise ValueError("min_periods must be between two and window")

    series = pd.Series(_as_array(values), dtype=float)
    history = series.shift(1)
    mean = history.rolling(window=window, min_periods=required).mean()
    sd = history.rolling(window=window, min_periods=required).std(ddof=1)
    z = (series - mean) / sd
    z = z.where(np.isfinite(z) & (sd > 0.0))
    return z.to_numpy(dtype=float)


def fit_ar1_ou_baseline(
    values: ArrayLike,
    *,
    dt_seconds: float = 1.0,
    min_pairs: int = 100,
) -> AR1BaselineParams:
    """Fit ``y_t = intercept + phi*y_(t-1) + error`` on training data only."""

    if not (math.isfinite(dt_seconds) and dt_seconds > 0.0):
        raise ValueError("dt_seconds must be finite and positive")
    if min_pairs < 3:
        raise ValueError("min_pairs must be at least three")

    y = _as_array(values)
    lag = y[:-1]
    current = y[1:]
    valid = np.isfinite(lag) & np.isfinite(current)
    x = lag[valid]
    target = current[valid]
    if x.size < min_pairs:
        raise ValueError(f"need at least {min_pairs} consecutive finite pairs")

    design = np.column_stack([np.ones(x.size), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    intercept, phi = map(float, coefficients)
    if not (0.0 < phi < 1.0 - 1e-8):
        raise ValueError(
            f"fitted phi={phi:.6g} is not a stationary positive-persistence OU baseline"
        )

    residual = target - (intercept + phi * x)
    residual_sd = float(np.std(residual, ddof=2))
    if not (math.isfinite(residual_sd) and residual_sd > 0.0):
        raise ValueError("AR(1) residual scale is not positive")

    unconditional_mean = intercept / (1.0 - phi)
    stationary_sd = residual_sd / math.sqrt(1.0 - phi * phi)
    return AR1BaselineParams(
        intercept=intercept,
        phi=phi,
        residual_sd=residual_sd,
        unconditional_mean=float(unconditional_mean),
        stationary_sd=float(stationary_sd),
        dt_seconds=float(dt_seconds),
        n_pairs=int(x.size),
    )


def ar1_ou_signal_z(values: ArrayLike, params: AR1BaselineParams) -> np.ndarray:
    """Standardised deviation from the training-fitted AR(1) long-run mean."""

    y = _as_array(values)
    signal = (y - params.unconditional_mean) / params.stationary_sd
    signal[~np.isfinite(y)] = np.nan
    return signal


__all__ = [
    "AR1BaselineParams",
    "ar1_ou_signal_z",
    "causal_rolling_z",
    "fit_ar1_ou_baseline",
]
