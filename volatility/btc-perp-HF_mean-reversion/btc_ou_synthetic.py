"""
BTC perpetual high-frequency mean-reversion research toolkit.

Pipeline:
    1. Download aggTrades from data.binance.vision (free, no API key, back to 2019)
    2. Resample to fixed-interval bars (default 1 second)
    3. Estimate the Ornstein-Uhlenbeck mean-reversion speed theta -> half-life
    4. Lo-MacKinlay variance ratio test (heteroskedasticity-robust)
    5. Roll (1984) implied effective spread -> how much of the reversion is bid-ask bounce
    6. Break-even economics under a given fee tier

Model:
    dX = theta * (mu - X) dt + sigma dW
    Discretised at step dt, this is an AR(1):
        X_{t+1} = a + b * X_t + eps,   b = exp(-theta * dt)
    so theta = -ln(b) / dt  and  half_life = ln(2) / theta.

Usage:
    python btc_ou.py --symbol BTCUSDT --date 2026-07-01 --bar-seconds 1
    python btc_ou.py --synthetic          # offline self-test, no network needed
"""

from __future__ import annotations

import argparse
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
CACHE_DIR = Path("./data")

AGGTRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


# --------------------------------------------------------------------------
# 1. Data acquisition
# --------------------------------------------------------------------------
def download_agg_trades(symbol: str, date: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """
    Fetch one day of aggregated trades for a USD-M perpetual contract.

    `date` must be formatted as YYYY-MM-DD. Files are cached locally as parquet
    so repeated runs do not re-download.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}_{date}_aggtrades.parquet"

    if cache_path.exists():
        print(f"[cache] loading {cache_path}")
        return pd.read_parquet(cache_path)

    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{date}.zip"
    print(f"[download] {url}")
    with urlopen(url) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        inner_name = archive.namelist()[0]
        with archive.open(inner_name) as handle:
            head = handle.read(200).decode("utf-8", errors="ignore")
            handle.seek(0)
            # Binance added a header row to these files at some point; detect it.
            has_header = "agg_trade_id" in head or "transact_time" in head
            frame = pd.read_csv(
                handle,
                header=0 if has_header else None,
                names=None if has_header else AGGTRADE_COLUMNS,
            )

    frame.columns = [c.strip() for c in frame.columns]
    rename_map = {
        "a": "agg_trade_id", "p": "price", "q": "quantity",
        "f": "first_trade_id", "l": "last_trade_id",
        "T": "transact_time", "m": "is_buyer_maker",
    }
    frame = frame.rename(columns=rename_map)

    frame = frame[["transact_time", "price", "quantity", "is_buyer_maker"]].copy()
    frame["price"] = frame["price"].astype(float)
    frame["quantity"] = frame["quantity"].astype(float)

    # Timestamps switched from milliseconds to microseconds in 2025 files.
    unit = "us" if frame["transact_time"].iloc[0] > 1e15 else "ms"
    frame["timestamp"] = pd.to_datetime(frame["transact_time"], unit=unit, utc=True)
    frame = frame.drop(columns=["transact_time"]).set_index("timestamp").sort_index()

    frame.to_parquet(cache_path)
    print(f"[cache] wrote {cache_path} ({len(frame):,} trades)")
    return frame


def build_bars(trades: pd.DataFrame, bar_seconds: int = 1) -> pd.DataFrame:
    """
    Collapse the trade tape into fixed-interval bars.

    Returns last price, VWAP, traded quantity and trade count per bar.
    Empty intervals are forward-filled on price and zero-filled on activity,
    which matters: dropping them would silently distort the time step and
    therefore every theta estimate downstream.
    """
    rule = f"{bar_seconds}s"
    notional = trades["price"] * trades["quantity"]

    bars = pd.DataFrame({
        "last": trades["price"].resample(rule).last(),
        "quantity": trades["quantity"].resample(rule).sum(),
        "trade_count": trades["price"].resample(rule).count(),
        "notional": notional.resample(rule).sum(),
    })
    bars["vwap"] = np.where(bars["quantity"] > 0, bars["notional"] / bars["quantity"], np.nan)
    bars["vwap"] = bars["vwap"].ffill()
    bars["last"] = bars["last"].ffill()
    bars["quantity"] = bars["quantity"].fillna(0.0)
    bars["trade_count"] = bars["trade_count"].fillna(0).astype(int)

    return bars.dropna(subset=["last"])


# --------------------------------------------------------------------------
# 2. Ornstein-Uhlenbeck estimation
# --------------------------------------------------------------------------
@dataclass
class OUFit:
    theta: float          # mean-reversion speed, per second
    mu: float             # long-run level (in the units of the fitted series)
    sigma: float          # diffusion coefficient, per sqrt(second)
    half_life: float      # seconds
    ar1_beta: float       # AR(1) slope
    r_squared: float
    n_obs: int

    def report(self) -> str:
        lines = [
            "Ornstein-Uhlenbeck fit",
            f"  observations   : {self.n_obs:,}",
            f"  AR(1) beta     : {self.ar1_beta:.6f}",
            f"  theta (1/s)    : {self.theta:.6f}",
            f"  half-life      : {self.half_life:.2f} s",
            f"  sigma (1/sqrt s): {self.sigma:.6e}",
            f"  R^2            : {self.r_squared:.5f}",
        ]
        return "\n".join(lines)


def fit_ou(series: np.ndarray, dt: float = 1.0) -> OUFit:
    """
    Estimate OU parameters by exact-discretisation OLS on the AR(1) form.

    This is the conditional MLE for a Gaussian OU process, so there is nothing
    to gain from a numerical optimiser here.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        raise ValueError("need at least 100 observations to fit")

    x_lag, x_now = x[:-1], x[1:]
    design = np.column_stack([np.ones_like(x_lag), x_lag])
    coefficients, *_ = np.linalg.lstsq(design, x_now, rcond=None)
    intercept, beta = coefficients

    residuals = x_now - design @ coefficients
    residual_variance = residuals.var(ddof=2)
    total_variance = x_now.var(ddof=1)
    r_squared = 1.0 - residual_variance / total_variance

    # beta must be inside (0, 1) for a stationary mean-reverting process.
    if not 0.0 < beta < 1.0:
        theta = float("inf") if beta <= 0 else 0.0
        half_life = 0.0 if beta <= 0 else float("inf")
        sigma = math.sqrt(residual_variance / dt)
        return OUFit(theta, float(x.mean()), sigma, half_life, beta, r_squared, x.size)

    theta = -math.log(beta) / dt
    mu = intercept / (1.0 - beta)
    sigma = math.sqrt(residual_variance * 2.0 * theta / (1.0 - beta**2))
    half_life = math.log(2.0) / theta

    return OUFit(theta, mu, sigma, half_life, beta, r_squared, x.size)


# --------------------------------------------------------------------------
# 3. Lo-MacKinlay variance ratio
# --------------------------------------------------------------------------
def variance_ratio(log_prices: np.ndarray, q: int) -> tuple[float, float]:
    """
    Lo & MacKinlay (1988) variance ratio with the heteroskedasticity-robust
    test statistic.

    VR < 1  -> negative serial correlation (mean reversion)
    VR > 1  -> positive serial correlation (trending)
    Returns (VR, robust z-statistic).
    """
    p = np.asarray(log_prices, dtype=float)
    p = p[np.isfinite(p)]
    n = p.size - 1
    if n < 10 * q:
        raise ValueError(f"series too short for q={q}")

    diffs = np.diff(p)
    mu_hat = diffs.mean()

    var_1 = np.sum((diffs - mu_hat) ** 2) / (n - 1)

    q_diffs = p[q:] - p[:-q]
    m = q * (n - q + 1) * (1.0 - q / n)
    var_q = np.sum((q_diffs - q * mu_hat) ** 2) / m

    vr = var_q / var_1

    # Heteroskedasticity-robust variance of the VR statistic.
    centered_sq = (diffs - mu_hat) ** 2
    denominator = centered_sq.sum() ** 2
    theta_star = 0.0
    for j in range(1, q):
        numerator = np.sum(centered_sq[j:] * centered_sq[:-j])
        delta_j = numerator / denominator
        weight = 2.0 * (q - j) / q
        theta_star += (weight**2) * delta_j

    z_stat = (vr - 1.0) / math.sqrt(theta_star) if theta_star > 0 else float("nan")
    return vr, z_stat


def roll_effective_spread(prices: np.ndarray) -> float:
    """
    Roll (1984) implied effective spread, in basis points of the mean price.

    Cov(dP_t, dP_{t-1}) = -s^2 / 4 under the bid-ask bounce model.
    A non-negative covariance means the bounce model does not bind and the
    estimator is undefined; returns NaN in that case.
    """
    p = np.asarray(prices, dtype=float)
    changes = np.diff(p)
    covariance = np.cov(changes[1:], changes[:-1])[0, 1]
    if covariance >= 0:
        return float("nan")
    spread = 2.0 * math.sqrt(-covariance)
    return spread / p.mean() * 1e4


# --------------------------------------------------------------------------
# 4. Fee-tier economics
# --------------------------------------------------------------------------
@dataclass
class TradeEconomics:
    maker_fee_bp: float
    spread_bp: float
    breakeven_bp: float
    sigma_per_second_bp: float
    seconds_to_breakeven: float

    def report(self) -> str:
        return "\n".join([
            "Break-even economics",
            f"  maker fee per side  : {self.maker_fee_bp:.4f} bp",
            f"  effective spread    : {self.spread_bp:.4f} bp",
            f"  break-even deviation: {self.breakeven_bp:.4f} bp",
            f"  realised vol        : {self.sigma_per_second_bp:.4f} bp / sqrt(s)",
            f"  diffusion time to it: {self.seconds_to_breakeven:.1f} s",
        ])


def trade_economics(log_prices: np.ndarray, maker_fee_bp: float,
                    spread_bp: float, dt: float = 1.0) -> TradeEconomics:
    """
    A deviation d is only worth trading if the expected capture (roughly d/2
    when entering at d and exiting near the mean) clears the round trip:

        d / 2 > 2 * maker_fee + spread   ->   d > 2 * (2f + s)

    The diffusion time to reach that deviation follows from t = (d / sigma)^2.
    """
    breakeven_bp = 2.0 * (2.0 * maker_fee_bp + spread_bp)

    diffs = np.diff(np.asarray(log_prices, dtype=float))
    sigma_per_step_bp = diffs.std(ddof=1) * 1e4
    sigma_per_second_bp = sigma_per_step_bp / math.sqrt(dt)

    seconds = (breakeven_bp / sigma_per_second_bp) ** 2 if sigma_per_second_bp > 0 else float("inf")
    return TradeEconomics(maker_fee_bp, spread_bp, breakeven_bp,
                          sigma_per_second_bp, seconds)


def sharpe_from_trade_stats(edge_bp: float, trade_vol_bp: float,
                            trades_per_year: float) -> float:
    """
    Sharpe_annual = (edge / vol) per trade * sqrt(number of trades per year).
    """
    if trade_vol_bp <= 0:
        return float("nan")
    return (edge_bp / trade_vol_bp) * math.sqrt(trades_per_year)


# --------------------------------------------------------------------------
# 5. Analysis driver
# --------------------------------------------------------------------------
def analyse(bars: pd.DataFrame, bar_seconds: int, maker_fee_bp: float) -> None:
    log_price = np.log(bars["last"].to_numpy())

    print(f"\nbars: {len(bars):,} at {bar_seconds}s "
          f"({len(bars) * bar_seconds / 3600:.1f} hours of tape)")
    print(f"empty bars: {(bars['trade_count'] == 0).mean() * 100:.2f}%")

    spread_bp = roll_effective_spread(bars["last"].to_numpy())
    print(f"\nRoll implied effective spread: {spread_bp:.4f} bp")

    # OU is fitted on the deviation from a slow moving average, because the
    # raw log price is not stationary. The window sets what counts as "the mean".
    for window in (30, 120, 600):
        if window * 5 > len(bars):
            continue
        centre = pd.Series(log_price).rolling(window, min_periods=window).mean()
        deviation = (pd.Series(log_price) - centre).dropna().to_numpy()
        fit = fit_ou(deviation, dt=float(bar_seconds))
        print(f"\n--- centred on {window * bar_seconds}s moving average ---")
        print(fit.report())

    print()
    for q in (2, 5, 15, 60, 300):
        try:
            vr, z = variance_ratio(log_price, q)
        except ValueError:
            continue
        verdict = "mean reverting" if vr < 1 else "trending"
        print(f"VR(q={q:>3}) = {vr:.4f}   z = {z:+7.2f}   {verdict}")

    spread_for_economics = 0.0 if math.isnan(spread_bp) else spread_bp
    econ = trade_economics(log_price, maker_fee_bp, spread_for_economics,
                           dt=float(bar_seconds))
    print()
    print(econ.report())

    # Sweep the entry threshold. At exactly break-even the net edge is zero by
    # construction, so a single point tells us nothing -- the trade-off between
    # edge per trade and number of trades only shows up across a range.
    cost_bp = 2.0 * maker_fee_bp + spread_for_economics
    print("\nEntry threshold sweep (empirical crossing frequency, no adverse selection)")
    print("  mult   entry(bp)   edge(bp)   trades/day   Sharpe_ceiling")

    log_series = pd.Series(log_price)
    centre = log_series.rolling(120, min_periods=120).mean()
    deviation_bp = ((log_series - centre).dropna() * 1e4).to_numpy()

    for multiple in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        entry_bp = econ.breakeven_bp * multiple
        edge_bp = entry_bp / 2.0 - cost_bp

        # Count actual threshold crossings rather than assuming a diffusion time.
        outside = np.abs(deviation_bp) > entry_bp
        crossings = int(np.sum(outside[1:] & ~outside[:-1]))
        seconds_observed = len(deviation_bp) * bar_seconds
        trades_per_day = crossings / seconds_observed * 86400.0 if seconds_observed else 0.0

        holding_seconds = (entry_bp / econ.sigma_per_second_bp) ** 2
        trade_vol_bp = econ.sigma_per_second_bp * math.sqrt(max(holding_seconds, 1.0))
        sharpe = (sharpe_from_trade_stats(edge_bp, trade_vol_bp, trades_per_day * 365)
                  if trades_per_day > 0 and edge_bp > 0 else 0.0)

        print(f"  {multiple:4.1f}   {entry_bp:8.3f}   {edge_bp:8.3f}   "
              f"{trades_per_day:10,.0f}   {sharpe:14.2f}")


def synthetic_bars(n: int = 40_000, half_life: float = 45.0, seed: int = 7) -> pd.DataFrame:
    """
    Generate a known OU-plus-bounce series so the estimators can be validated
    against a ground truth before touching real data.
    """
    rng = np.random.default_rng(seed)
    theta = math.log(2.0) / half_life
    beta = math.exp(-theta)

    sigma_innovation = 0.8e-4  # ~0.8 bp per second
    deviation = np.zeros(n)
    for t in range(1, n):
        deviation[t] = beta * deviation[t - 1] + sigma_innovation * rng.standard_normal()

    drift = np.cumsum(rng.standard_normal(n)) * 0.3e-4
    log_price = math.log(100_000.0) + drift + deviation

    # Superimpose a bid-ask bounce of one tick on a 100k price (~0.01 bp).
    tick = 0.10 / 100_000.0
    bounce = rng.choice([-0.5, 0.5], size=n) * tick
    prices = np.exp(log_price + bounce)

    index = pd.date_range("2026-07-01", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({
        "last": prices,
        "quantity": rng.gamma(2.0, 0.5, size=n),
        "trade_count": rng.poisson(12, size=n),
        "notional": 0.0,
        "vwap": prices,
    }, index=index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", default="2026-07-01", help="YYYY-MM-DD")
    parser.add_argument("--bar-seconds", type=int, default=1)
    parser.add_argument("--maker-fee-bp", type=float, default=2.0,
                        help="maker fee per side in bp (Binance base tier = 2.0)")
    parser.add_argument("--synthetic", action="store_true",
                        help="run on generated data, no network access needed")
    args = parser.parse_args()

    if args.synthetic:
        print("=== SYNTHETIC SELF-TEST: true half-life = 45.0 s ===")
        bars = synthetic_bars()
    else:
        trades = download_agg_trades(args.symbol, args.date)
        bars = build_bars(trades, args.bar_seconds)

    analyse(bars, args.bar_seconds, args.maker_fee_bp)


if __name__ == "__main__":
    main()
