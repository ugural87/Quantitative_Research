from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_validation import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_ci,
    circular_block_bootstrap_mean_test,
    make_intraday_research_split,
)


def test_intraday_split_is_chronological_and_purged():
    index = pd.date_range("2026-01-01", periods=24 * 60, freq="1min", tz="UTC")
    split = make_intraday_research_split(
        index,
        train_duration="12h",
        development_duration="4h",
        purge_duration="30min",
        minimum_holdout_duration="6h",
    )

    assert split.train_positions.size == 12 * 60
    assert split.development_positions.size == 4 * 60
    assert split.purge_positions.size == 30
    assert split.holdout_positions.size == 7 * 60 + 30
    assert split.development_end_exclusive <= split.holdout_start
    assert not np.intersect1d(
        split.development_positions, split.holdout_positions
    ).size
    assert set(split.summary().index) == {
        "training",
        "threshold_development",
        "purge",
        "final_holdout",
    }


def test_intraday_split_rejects_insufficient_holdout():
    index = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
    with pytest.raises(ValueError):
        make_intraday_research_split(
            index,
            train_duration="20h",
            development_duration="3h",
            purge_duration="1h",
            minimum_holdout_duration="2h",
        )


def test_block_bootstrap_ci_is_deterministic_and_reports_positive_probability():
    rng = np.random.default_rng(1)
    x = 0.2 + rng.normal(size=400)
    a = circular_block_bootstrap_mean_ci(
        x, block_length=10, n_boot=500, seed=42
    )
    b = circular_block_bootstrap_mean_ci(
        x, block_length=10, n_boot=500, seed=42
    )
    assert a == b
    assert 0.0 <= a["probability_mean_positive"] <= 1.0
    assert a["lower"] <= a["mean"] <= a["upper"]


def test_block_bootstrap_mean_test_detects_large_positive_mean():
    rng = np.random.default_rng(2)
    x = 2.0 + rng.normal(scale=0.5, size=300)
    result = circular_block_bootstrap_mean_test(
        x,
        block_length=12,
        n_boot=500,
        alternative="greater",
        seed=5,
    )
    assert result["mean"] > 1.5
    assert result["p_value"] < 0.01


def test_block_bootstrap_mean_test_validates_alternative():
    with pytest.raises(ValueError):
        circular_block_bootstrap_mean_test(
            [1.0, 2.0],
            block_length=1,
            n_boot=100,
            alternative="invalid",  # type: ignore[arg-type]
        )


def test_benjamini_hochberg_is_monotone_in_ranked_space():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adjusted = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all(adjusted >= p)


def test_intraday_split_rejects_irregular_clock():
    index = pd.DatetimeIndex([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
        "2026-01-01T00:00:03Z",
    ])
    with pytest.raises(ValueError, match="regular wall-clock"):
        make_intraday_research_split(
            index,
            train_duration="1s",
            development_duration="1s",
            purge_duration="0s",
            minimum_holdout_duration="1s",
        )
