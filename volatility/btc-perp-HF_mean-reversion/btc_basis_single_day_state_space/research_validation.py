"""Validation utilities for a single-day dependent financial time series."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
import pandas as pd

Alternative = Literal["greater", "less", "two-sided"]


@dataclass(frozen=True)
class IntradayResearchSplit:
    """Chronological train/development/purge/final-holdout partition."""

    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    development_start: pd.Timestamp
    development_end_exclusive: pd.Timestamp
    purge_start: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end_exclusive: pd.Timestamp
    train_positions: np.ndarray
    development_positions: np.ndarray
    purge_positions: np.ndarray
    holdout_positions: np.ndarray

    def summary(self) -> pd.DataFrame:
        rows = {
            "training": {
                "start": self.train_start,
                "end_exclusive": self.train_end_exclusive,
                "clock_observations": self.train_positions.size,
            },
            "threshold_development": {
                "start": self.development_start,
                "end_exclusive": self.development_end_exclusive,
                "clock_observations": self.development_positions.size,
            },
            "purge": {
                "start": self.purge_start,
                "end_exclusive": self.holdout_start,
                "clock_observations": self.purge_positions.size,
            },
            "final_holdout": {
                "start": self.holdout_start,
                "end_exclusive": self.holdout_end_exclusive,
                "clock_observations": self.holdout_positions.size,
            },
        }
        return pd.DataFrame.from_dict(rows, orient="index")


def _validate_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if index.tz is None:
        raise ValueError("index must be timezone-aware")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("index must be sorted and unique")


def make_intraday_research_split(
    index: pd.DatetimeIndex,
    *,
    train_duration: str | pd.Timedelta,
    development_duration: str | pd.Timedelta,
    purge_duration: str | pd.Timedelta,
    minimum_holdout_duration: str | pd.Timedelta = "1h",
) -> IntradayResearchSplit:
    """Create the one chronological split used by this one-day workbench.

    The development interval is reserved for exploratory threshold sensitivity.
    A wall-clock purge follows it so a position initiated in development cannot
    complete inside the final holdout.  The final holdout is never used to pick a
    threshold or model.
    """

    _validate_index(index)
    if index.empty:
        raise ValueError("index cannot be empty")

    train_td = pd.Timedelta(train_duration)
    development_td = pd.Timedelta(development_duration)
    purge_td = pd.Timedelta(purge_duration)
    minimum_holdout_td = pd.Timedelta(minimum_holdout_duration)
    if min(train_td, development_td, minimum_holdout_td) <= pd.Timedelta(0):
        raise ValueError("train, development and minimum holdout must be positive")
    if purge_td < pd.Timedelta(0):
        raise ValueError("purge_duration cannot be negative")

    if len(index) > 1:
        differences = np.diff(index.asi8)
        if not np.all(differences == differences[0]):
            raise ValueError("index must use a regular wall-clock interval")
        step = pd.Timedelta(int(differences[0]), unit="ns")
    else:
        step = pd.Timedelta("1ns")
    start = index[0]
    train_end = start + train_td
    development_start = train_end
    development_end = development_start + development_td
    purge_start = development_end
    holdout_start = purge_start + purge_td
    holdout_end = index[-1] + step

    if holdout_end - holdout_start < minimum_holdout_td:
        raise ValueError(
            "configured partitions leave less than the minimum final holdout"
        )

    train_pos = np.flatnonzero((index >= start) & (index < train_end))
    development_pos = np.flatnonzero(
        (index >= development_start) & (index < development_end)
    )
    purge_pos = np.flatnonzero((index >= purge_start) & (index < holdout_start))
    holdout_pos = np.flatnonzero((index >= holdout_start) & (index < holdout_end))

    if not train_pos.size or not development_pos.size or not holdout_pos.size:
        raise ValueError("each substantive partition must contain observations")

    return IntradayResearchSplit(
        train_start=index[train_pos[0]],
        train_end_exclusive=train_end,
        development_start=index[development_pos[0]],
        development_end_exclusive=development_end,
        purge_start=purge_start,
        holdout_start=index[holdout_pos[0]],
        holdout_end_exclusive=holdout_end,
        train_positions=train_pos,
        development_positions=development_pos,
        purge_positions=purge_pos,
        holdout_positions=holdout_pos,
    )


def _finite_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def _validate_bootstrap_inputs(
    x: np.ndarray,
    *,
    block_length: int,
    n_boot: int,
) -> None:
    if x.size < 2:
        raise ValueError("need at least two finite observations")
    if not (1 <= block_length <= x.size):
        raise ValueError("block_length must be between 1 and sample size")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")


def _circular_block_means(
    x: np.ndarray,
    *,
    block_length: int,
    n_boot: int,
    seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.size
    n_blocks = math.ceil(n / block_length)
    offsets = np.arange(block_length)
    means = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        indices = (starts[:, None] + offsets[None, :]) % n
        sample = x[indices.reshape(-1)[:n]]
        means[b] = sample.mean()
    return means


def circular_block_bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    block_length: int,
    n_boot: int = 2_000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float]:
    """Dependent-data percentile confidence interval for a sample mean."""

    x = _finite_values(values)
    _validate_bootstrap_inputs(x, block_length=block_length, n_boot=n_boot)
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")

    means = _circular_block_means(
        x,
        block_length=block_length,
        n_boot=n_boot,
        seed=seed,
    )
    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n": float(x.size),
        "mean": float(x.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "probability_mean_positive": float(np.mean(means > 0.0)),
        "block_length": float(block_length),
        "n_boot": float(n_boot),
    }


def circular_block_bootstrap_mean_test(
    values: Sequence[float] | np.ndarray,
    *,
    block_length: int,
    n_boot: int = 2_000,
    alternative: Alternative = "greater",
    seed: int | None = 0,
) -> dict[str, float | str]:
    """Centred-null block-bootstrap test for a mean different from zero.

    The observed dependent sequence is centred to impose the null mean.  Circular
    blocks are then resampled from that centred sequence.  The plus-one correction
    prevents a zero Monte Carlo p-value.
    """

    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")
    x = _finite_values(values)
    _validate_bootstrap_inputs(x, block_length=block_length, n_boot=n_boot)

    observed = float(x.mean())
    null_means = _circular_block_means(
        x - observed,
        block_length=block_length,
        n_boot=n_boot,
        seed=seed,
    )
    if alternative == "greater":
        exceedances = int(np.count_nonzero(null_means >= observed))
    elif alternative == "less":
        exceedances = int(np.count_nonzero(null_means <= observed))
    else:
        exceedances = int(np.count_nonzero(np.abs(null_means) >= abs(observed)))

    p_value = (exceedances + 1.0) / (n_boot + 1.0)
    return {
        "n": float(x.size),
        "mean": observed,
        "p_value": float(p_value),
        "alternative": alternative,
        "block_length": float(block_length),
        "n_boot": float(n_boot),
    }


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values for an exploratory sweep."""

    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must be finite and in [0, 1]")
    if p.size == 0:
        return p.copy()

    order = np.argsort(p)
    ranked = p[order]
    m = p.size
    adjusted_ranked = ranked * m / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


__all__ = [
    "IntradayResearchSplit",
    "benjamini_hochberg",
    "circular_block_bootstrap_mean_ci",
    "circular_block_bootstrap_mean_test",
    "make_intraday_research_split",
]
