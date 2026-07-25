from __future__ import annotations

import pandas as pd
import pytest

from research_backtest import ProxyBacktestConfig, run_proxy_backtest


def _config(**overrides):
    base = dict(
        entry_z=2.0,
        exit_z=0.25,
        stop_z=4.0,
        max_hold_seconds=60.0,
        cooldown_seconds=5.0,
        round_trip_cost_bp=1.0,
        execution_delay_seconds=1.0,
        max_fill_delay_seconds=2.0,
        allow_short_spot=False,
    )
    base.update(overrides)
    return ProxyBacktestConfig(**base)


def _frame(index, basis, signal, valid=None):
    if valid is None:
        valid = [True] * len(index)
    return pd.DataFrame(
        {"basis_bp": basis, "signal_stationary_z": signal, "valid": valid},
        index=index,
    )


def test_signal_fills_one_second_later_and_pnl_sign_is_correct():
    index = pd.date_range("2026-01-01", periods=6, freq="1s", tz="UTC")
    frame = _frame(
        index,
        basis=[0.0, 10.0, 9.0, 8.0, 7.0, 7.0],
        signal=[2.5, 2.2, 0.1, 0.1, 0.1, 0.1],
    )
    trades = run_proxy_backtest(frame, start_time=index[0], config=_config())

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_decision_time"] == index[0]
    assert trade["entry_time"] == index[1]
    assert trade["exit_decision_time"] == index[2]
    assert trade["exit_time"] == index[3]
    assert trade["entry_fill_delay_s"] == pytest.approx(1.0)
    assert trade["exit_fill_delay_s"] == pytest.approx(1.0)
    assert trade["holding_s"] == pytest.approx(2.0)
    assert trade["gross_pnl_bp"] == pytest.approx(2.0)
    assert trade["net_pnl_bp"] == pytest.approx(1.0)


def test_fill_waits_for_next_tradable_timestamp_within_lateness_budget():
    index = pd.date_range("2026-01-01", periods=7, freq="1s", tz="UTC")
    frame = _frame(
        index,
        basis=[10, 10, 10, 9, 8, 8, 8],
        signal=[2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0],
        valid=[True, False, False, True, True, True, True],
    )
    trades = run_proxy_backtest(
        frame,
        start_time=index[0],
        config=_config(max_fill_delay_seconds=2.0),
    )

    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == index[3]
    assert trades.iloc[0]["entry_fill_delay_s"] == pytest.approx(3.0)


def test_pending_entry_is_cancelled_across_large_gap():
    index = pd.date_range("2026-01-01", periods=8, freq="1s", tz="UTC")
    frame = _frame(
        index,
        basis=[10] * 8,
        signal=[2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        valid=[True, False, False, False, True, True, True, True],
    )
    trades = run_proxy_backtest(
        frame,
        start_time=index[0],
        config=_config(max_fill_delay_seconds=1.0),
    )
    assert trades.empty


def test_holding_period_uses_wall_clock_not_row_count():
    index = pd.DatetimeIndex(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:10Z",
            "2026-01-01T00:00:11Z",
        ]
    )
    frame = _frame(index, basis=[10, 10, 9, 8], signal=[2.5, 2.5, 0.0, 0.0])
    trades = run_proxy_backtest(
        frame,
        start_time=index[0],
        config=_config(max_fill_delay_seconds=20.0),
    )
    assert len(trades) == 1
    assert trades.iloc[0]["holding_s"] == pytest.approx(10.0)


def test_mean_reversion_exit_accepts_zero_crossing_overshoot():
    from research_backtest import run_proxy_backtest_detailed

    index = pd.date_range("2026-01-01", periods=6, freq="1s", tz="UTC")
    frame = _frame(
        index,
        basis=[10.0, 10.0, 9.0, 8.0, 7.0, 7.0],
        signal=[2.5, 2.2, -1.0, -1.0, -1.0, -1.0],
    )
    result = run_proxy_backtest_detailed(
        frame,
        start_time=index[0],
        config=_config(),
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "mean_reversion"
    assert result.trades.iloc[0]["exit_decision_time"] == index[2]
    assert result.trades.iloc[0]["exit_time"] == index[3]


def test_open_position_is_reported_in_final_state_not_silently_dropped():
    from research_backtest import run_proxy_backtest_detailed

    index = pd.date_range("2026-01-01", periods=5, freq="1s", tz="UTC")
    frame = _frame(
        index,
        basis=[10.0, 10.0, 9.0, 8.0, 7.0],
        signal=[2.5, 2.5, 2.0, 1.5, 1.0],
    )
    result = run_proxy_backtest_detailed(
        frame,
        start_time=index[0],
        config=_config(),
    )

    assert result.trades.empty
    state = result.final_state
    assert state.has_open_position
    assert state.position == -1
    assert state.side == "short_basis"
    assert state.entry_time == index[1]
    assert state.last_tradable_basis_bp == pytest.approx(7.0)
    assert state.mark_to_market_gross_bp == pytest.approx(3.0)
    assert state.mark_to_market_net_bp_if_liquidated == pytest.approx(2.0)


def test_pending_exit_is_visible_at_end_of_interval():
    from research_backtest import run_proxy_backtest_detailed

    index = pd.date_range("2026-01-01", periods=3, freq="1s", tz="UTC")
    frame = _frame(
        index,
        basis=[10.0, 10.0, 9.0],
        signal=[2.5, 2.5, 0.0],
    )
    result = run_proxy_backtest_detailed(
        frame,
        start_time=index[0],
        config=_config(),
    )

    assert result.trades.empty
    assert result.final_state.has_open_position
    assert result.final_state.pending_action == "exit"
    assert result.final_state.pending_reason == "mean_reversion"
    assert result.final_state.pending_due_time == index[2] + pd.Timedelta(seconds=1)


def test_end_time_is_exclusive_and_validated():
    from research_backtest import run_proxy_backtest_detailed

    index = pd.date_range("2026-01-01", periods=6, freq="1s", tz="UTC")
    frame = _frame(
        index,
        basis=[10.0, 10.0, 9.0, 8.0, 7.0, 6.0],
        signal=[2.5, 2.5, 0.0, 0.0, 0.0, 0.0],
    )
    result = run_proxy_backtest_detailed(
        frame,
        start_time=index[0],
        end_time=index[3],
        config=_config(),
    )

    assert result.trades.empty
    assert result.final_state.last_clock_time == index[2]
    assert result.final_state.pending_action == "exit"

    with pytest.raises(ValueError):
        run_proxy_backtest_detailed(
            frame,
            start_time=index[1],
            end_time=index[1],
            config=_config(),
        )
