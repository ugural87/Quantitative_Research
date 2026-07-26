from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest import ProxyBacktestConfig, run_proxy_backtest_detailed
from research_baselines import causal_rolling_z
from research_data import write_research_manifest
from research_validation import make_intraday_research_split
from state_space import StateSpaceParams, filter_state_space, fit_state_space, simulate_state_space


def test_single_day_pipeline_smoke(tmp_path: Path):
    params = StateSpaceParams(
        b=0.90,
        sigma_level=2e-5,
        sigma_transient=8e-5,
        sigma_observation=1e-5,
    )
    simulated = simulate_state_space(4_000, params, seed=101)
    index = pd.date_range("2026-01-01", periods=4_000, freq="1s", tz="UTC")
    split = make_intraday_research_split(
        index,
        train_duration="2000s",
        development_duration="600s",
        purge_duration="66s",
        minimum_holdout_duration="1000s",
    )

    fitted = fit_state_space(
        simulated["observed"][split.train_positions],
        burn_in=30,
        compare_null=True,
    )
    filtered = filter_state_space(simulated["observed"], fitted)
    signal = filtered.transient_stationary_z(fitted.params)
    rolling = causal_rolling_z(simulated["observed"], window=120, min_periods=60)

    frame = pd.DataFrame(
        {
            "basis_bp": simulated["observed"] * 1e4,
            "signal_state_space_z": signal,
            "signal_rolling_z": rolling,
            "valid": True,
        },
        index=index,
    )
    config = ProxyBacktestConfig(
        entry_z=2.0,
        exit_z=0.25,
        stop_z=4.0,
        max_hold_seconds=60,
        cooldown_seconds=5,
        round_trip_cost_bp=1.0,
    )
    state_result = run_proxy_backtest_detailed(
        frame,
        start_time=split.holdout_start,
        end_time=split.holdout_end_exclusive,
        config=config,
        signal_col="signal_state_space_z",
    )
    rolling_result = run_proxy_backtest_detailed(
        frame,
        start_time=split.holdout_start,
        end_time=split.holdout_end_exclusive,
        config=config,
        signal_col="signal_rolling_z",
    )

    assert state_result.final_state.completed_trades == len(state_result.trades)
    assert rolling_result.final_state.completed_trades == len(rolling_result.trades)
    assert state_result.final_state.last_clock_time < split.holdout_end_exclusive

    manifest = write_research_manifest(
        tmp_path / "manifest.json",
        study={"scope": "single-day"},
        archives=[],
        configuration={"entry_z": config.entry_z},
        outputs={"trades": len(state_result.trades)},
    )
    payload = json.loads(manifest.read_text())
    assert payload["study"]["scope"] == "single-day"
