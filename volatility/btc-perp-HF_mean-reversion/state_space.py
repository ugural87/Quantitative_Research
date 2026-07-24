"""
State-space decomposition of a high-frequency price series.

    y_t = m_t + x_t + eps_t          observation (log price)
    m_t = m_{t-1} + w_t              efficient price, random walk (NOT tradeable)
    x_t = b * x_{t-1} + v_t          transient deviation, AR(1)  (tradeable)
    eps_t ~ N(0, s_eps^2)            bid-ask bounce / microstructure noise

This is the fix for the moving-average detrending problem: the mean is estimated
jointly with theta instead of being imposed by a window choice, and the bounce is
absorbed by eps rather than contaminating the AR(1) slope.

Parameters are estimated by exact Kalman-filter MLE. The 2x2 linear algebra is
written out by hand because np.linalg calls inside the per-observation loop
dominate runtime otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class StateSpaceFit:
    b: float                 # AR(1) slope of the transient component
    sigma_w: float           # efficient-price innovation sd (per step)
    sigma_v: float           # transient innovation sd (per step)
    sigma_eps: float         # microstructure noise sd
    half_life: float         # seconds
    loglik: float
    converged: bool
    n_obs: int

    def report(self, dt: float = 1.0) -> str:
        signal_sd = (self.sigma_v / math.sqrt(1 - self.b**2)
                     if abs(self.b) < 1 else float("nan"))
        return "\n".join([
            "State-space decomposition",
            f"  observations        : {self.n_obs:,}",
            f"  converged           : {self.converged}",
            f"  b (AR1 slope)       : {self.b:.6f}",
            f"  half-life           : {self.half_life:.2f} s",
            f"  sigma_w  (efficient): {self.sigma_w * 1e4:.4f} bp / step",
            f"  sigma_v  (transient): {self.sigma_v * 1e4:.4f} bp / step",
            f"  sigma_eps (bounce)  : {self.sigma_eps * 1e4:.4f} bp",
            f"  transient sd        : {signal_sd * 1e4:.4f} bp  <- tradeable amplitude",
            f"  loglik              : {self.loglik:,.1f}",
        ])


def _kalman_loglik(params: np.ndarray, y: np.ndarray) -> float:
    """
    Negative log-likelihood. Parameters arrive unconstrained:
        params = [logit(b), log(sigma_w), log(sigma_v), log(sigma_eps)]
    """
    logit_b, log_w, log_v, log_e = params
    b = 1.0 / (1.0 + math.exp(-logit_b))          # constrained to (0, 1)
    q_w = math.exp(2.0 * log_w)
    q_v = math.exp(2.0 * log_v)
    r = math.exp(2.0 * log_e)

    # State a = [m, x], covariance P written out as p11, p12, p22.
    m, x = y[0], 0.0
    p11, p12, p22 = 1e-4, 0.0, q_v / max(1.0 - b * b, 1e-8)

    loglik = 0.0
    two_pi = math.log(2.0 * math.pi)

    for t in range(1, y.size):
        # Predict: m stays, x decays by b.
        m_pred = m
        x_pred = b * x
        p11_pred = p11 + q_w
        p12_pred = p12 * b
        p22_pred = p22 * b * b + q_v

        # Observe y = m + x + eps.
        innovation = y[t] - (m_pred + x_pred)
        s = p11_pred + 2.0 * p12_pred + p22_pred + r
        if s <= 0 or not math.isfinite(s):
            return 1e12

        loglik += -0.5 * (two_pi + math.log(s) + innovation * innovation / s)

        # Gain K = P H' / S, with H = [1, 1].
        k1 = (p11_pred + p12_pred) / s
        k2 = (p12_pred + p22_pred) / s

        m = m_pred + k1 * innovation
        x = x_pred + k2 * innovation

        p11 = p11_pred - k1 * (p11_pred + p12_pred)
        p12 = p12_pred - k1 * (p12_pred + p22_pred)
        p22 = p22_pred - k2 * (p12_pred + p22_pred)

    if not math.isfinite(loglik):
        return 1e12
    return -loglik


def fit_state_space(log_prices: np.ndarray, dt: float = 1.0,
                    max_obs: int = 60_000) -> StateSpaceFit:
    """
    Estimate the decomposition by MLE.

    `max_obs` caps the sample used for optimisation; the filter is a Python loop
    so a full trading day costs a few minutes otherwise. Estimates are stable
    well before 60k observations.
    """
    y = np.asarray(log_prices, dtype=float)
    y = y[np.isfinite(y)]
    if y.size > max_obs:
        y = y[:max_obs]
    if y.size < 500:
        raise ValueError("need at least 500 observations")

    scale = np.diff(y).std(ddof=1)

    # Multi-start: the likelihood surface has local optima, and a single start
    # silently returns the wrong basin in slow-reversion regimes.
    starts = []
    for logit_b in (1.5, 3.5, 5.5):
        for signal_share in (0.35, 0.7):
            starts.append(np.array([
                logit_b,
                math.log(scale * (1.0 - signal_share) + 1e-12),
                math.log(scale * signal_share + 1e-12),
                math.log(scale * 0.2 + 1e-12),
            ]))

    best = None
    for start in starts:
        candidate = minimize(_kalman_loglik, start, args=(y,),
                             method="Nelder-Mead",
                             options={"maxiter": 1200, "xatol": 1e-6, "fatol": 1e-4})
        if best is None or candidate.fun < best.fun:
            best = candidate
    result = best

    logit_b, log_w, log_v, log_e = result.x
    b = 1.0 / (1.0 + math.exp(-logit_b))
    half_life = math.log(2.0) / (-math.log(b) / dt) if 0 < b < 1 else float("nan")

    return StateSpaceFit(
        b=b,
        sigma_w=math.exp(log_w),
        sigma_v=math.exp(log_v),
        sigma_eps=math.exp(log_e),
        half_life=half_life,
        loglik=-result.fun,
        converged=bool(result.success),
        n_obs=y.size,
    )


def smooth_transient(log_prices: np.ndarray, fit: StateSpaceFit) -> np.ndarray:
    """
    Run the filter at the fitted parameters and return the filtered transient
    component x_t. This is the series a strategy would actually trade.
    """
    y = np.asarray(log_prices, dtype=float)
    b, q_w = fit.b, fit.sigma_w**2
    q_v, r = fit.sigma_v**2, fit.sigma_eps**2

    m, x = y[0], 0.0
    p11, p12, p22 = 1e-4, 0.0, q_v / max(1.0 - b * b, 1e-8)
    out = np.zeros_like(y)

    for t in range(1, y.size):
        m_pred, x_pred = m, b * x
        p11_pred = p11 + q_w
        p12_pred = p12 * b
        p22_pred = p22 * b * b + q_v

        innovation = y[t] - (m_pred + x_pred)
        s = p11_pred + 2.0 * p12_pred + p22_pred + r
        k1 = (p11_pred + p12_pred) / s
        k2 = (p12_pred + p22_pred) / s

        m = m_pred + k1 * innovation
        x = x_pred + k2 * innovation
        p11 = p11_pred - k1 * (p11_pred + p12_pred)
        p12 = p12_pred - k1 * (p12_pred + p22_pred)
        p22 = p22_pred - k2 * (p12_pred + p22_pred)
        out[t] = x

    return out
