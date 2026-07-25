"""Timestamp-aware proxy backtest for basis-signal diagnostics.

The engine models only claims that one-second trade-price bars can support:
causal decisions, delayed fills, tradability checks, wall-clock holding periods,
explicit stale-instruction cancellation, deterministic cost accounting and an
honest end-of-sample state.  It is not an exchange execution simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd

Position = Literal[-1, 0, 1]

_TRADE_COLUMNS = [
    "entry_decision_time",
    "entry_time",
    "exit_decision_time",
    "exit_time",
    "side",
    "entry_basis_bp",
    "exit_basis_bp",
    "entry_signal_z",
    "exit_signal_z",
    "entry_fill_delay_s",
    "exit_fill_delay_s",
    "holding_s",
    "gross_pnl_bp",
    "cost_bp",
    "net_pnl_bp",
    "entry_reason",
    "exit_reason",
]


@dataclass(frozen=True)
class ProxyBacktestConfig:
    entry_z: float
    exit_z: float
    stop_z: float
    max_hold_seconds: float
    cooldown_seconds: float
    round_trip_cost_bp: float
    execution_delay_seconds: float = 1.0
    max_fill_delay_seconds: float = 2.0
    allow_short_spot: bool = False

    def __post_init__(self) -> None:
        if not (self.entry_z > self.exit_z >= 0.0):
            raise ValueError("require entry_z > exit_z >= 0")
        if self.stop_z <= self.entry_z:
            raise ValueError("stop_z must exceed entry_z")
        for name in (
            "max_hold_seconds",
            "cooldown_seconds",
            "round_trip_cost_bp",
            "execution_delay_seconds",
            "max_fill_delay_seconds",
        ):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.execution_delay_seconds <= 0.0:
            raise ValueError("execution_delay_seconds must be strictly positive")


@dataclass(frozen=True)
class ProxyBacktestFinalState:
    """Observable state at the exclusive end of the evaluation interval."""

    interval_start: pd.Timestamp
    interval_end_exclusive: pd.Timestamp | None
    last_clock_time: pd.Timestamp | None
    last_tradable_time: pd.Timestamp | None
    position: Position
    side: str | None
    entry_decision_time: pd.Timestamp | None
    entry_time: pd.Timestamp | None
    entry_basis_bp: float | None
    entry_signal_z: float | None
    pending_action: str | None
    pending_reason: str | None
    pending_decision_time: pd.Timestamp | None
    pending_due_time: pd.Timestamp | None
    last_tradable_basis_bp: float | None
    mark_to_market_gross_bp: float | None
    mark_to_market_net_bp_if_liquidated: float | None
    completed_trades: int
    decisions: int
    fills: int
    cancelled_stale_actions: int

    @property
    def has_open_position(self) -> bool:
        return self.position != 0

    @property
    def has_pending_action(self) -> bool:
        return self.pending_action is not None

    def to_frame(self) -> pd.DataFrame:
        values = asdict(self)
        values["has_open_position"] = self.has_open_position
        values["has_pending_action"] = self.has_pending_action
        return pd.Series(values, name="value").to_frame()


@dataclass(frozen=True)
class ProxyBacktestResult:
    trades: pd.DataFrame
    final_state: ProxyBacktestFinalState


@dataclass(frozen=True)
class _PendingAction:
    action: Position
    reason: str
    signal_z: float
    decision_time: pd.Timestamp
    due_time: pd.Timestamp


def _validate_frame(
    frame: pd.DataFrame,
    *,
    basis_col: str,
    signal_col: str,
    tradable_col: str,
) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame index must be a pandas DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("frame index must be timezone-aware")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted")
    if frame.index.has_duplicates:
        raise ValueError("frame index must be unique")
    missing = {basis_col, signal_col, tradable_col} - set(frame.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")


def _action_name(action: Position) -> str:
    if action == -1:
        return "enter_short_basis"
    if action == 1:
        return "enter_long_basis"
    return "exit"


def _side_name(position: Position) -> str | None:
    if position == -1:
        return "short_basis"
    if position == 1:
        return "long_basis"
    return None


def run_proxy_backtest_detailed(
    frame: pd.DataFrame,
    *,
    start_time: pd.Timestamp,
    config: ProxyBacktestConfig,
    end_time: pd.Timestamp | None = None,
    basis_col: str = "basis_bp",
    signal_col: str = "signal_stationary_z",
    tradable_col: str = "valid",
) -> ProxyBacktestResult:
    """Run a causal proxy backtest and retain the end-of-sample state.

    ``start_time`` is inclusive and ``end_time`` is exclusive.  A decision at
    time ``t`` may fill no earlier than ``t + execution_delay_seconds``.  The
    first tradable observation at or after that due time is accepted only while
    its lateness remains within ``max_fill_delay_seconds``.

    Completed trades never silently include an unfinished final position.  Any
    open inventory or pending instruction is reported in ``final_state`` with a
    hypothetical liquidation mark that is separate from realised trade P&L.
    """

    _validate_frame(
        frame,
        basis_col=basis_col,
        signal_col=signal_col,
        tradable_col=tradable_col,
    )
    if start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware")
    if end_time is not None:
        if end_time.tzinfo is None:
            raise ValueError("end_time must be timezone-aware")
        if end_time <= start_time:
            raise ValueError("end_time must be later than start_time")

    basis = pd.to_numeric(frame[basis_col], errors="coerce").to_numpy(float)
    signal = pd.to_numeric(frame[signal_col], errors="coerce").to_numpy(float)
    tradable = frame[tradable_col].fillna(False).astype(bool).to_numpy()
    timestamps = frame.index

    trades: list[dict[str, object]] = []
    position: Position = 0
    pending: _PendingAction | None = None
    entry_time: pd.Timestamp | None = None
    entry_decision_time: pd.Timestamp | None = None
    entry_basis: float | None = None
    entry_signal: float | None = None
    entry_reason: str | None = None
    next_allowed_time = start_time

    last_clock_time: pd.Timestamp | None = None
    last_tradable_time: pd.Timestamp | None = None
    last_tradable_basis: float | None = None
    decisions = 0
    fills = 0
    cancelled_stale_actions = 0

    delay = pd.Timedelta(seconds=config.execution_delay_seconds)
    max_lateness = pd.Timedelta(seconds=config.max_fill_delay_seconds)
    cooldown = pd.Timedelta(seconds=config.cooldown_seconds)
    max_hold = pd.Timedelta(seconds=config.max_hold_seconds)

    for i, now in enumerate(timestamps):
        if now < start_time:
            continue
        if end_time is not None and now >= end_time:
            break

        last_clock_time = now
        if tradable[i] and np.isfinite(basis[i]):
            last_tradable_time = now
            last_tradable_basis = float(basis[i])

        filled_this_row = False

        if pending is not None and now >= pending.due_time:
            lateness = now - pending.due_time
            can_fill = tradable[i] and np.isfinite(basis[i])

            if can_fill and lateness <= max_lateness:
                filled_this_row = True
                fills += 1
                action = pending.action

                if action in (-1, 1) and position == 0:
                    position = action
                    entry_time = now
                    entry_decision_time = pending.decision_time
                    entry_basis = float(basis[i])
                    entry_signal = float(pending.signal_z)
                    entry_reason = pending.reason

                elif action == 0 and position != 0:
                    assert entry_time is not None
                    assert entry_decision_time is not None
                    assert entry_basis is not None
                    assert entry_signal is not None
                    assert entry_reason is not None

                    gross_bp = float(position * (basis[i] - entry_basis))
                    net_bp = gross_bp - config.round_trip_cost_bp
                    trades.append(
                        {
                            "entry_decision_time": entry_decision_time,
                            "entry_time": entry_time,
                            "exit_decision_time": pending.decision_time,
                            "exit_time": now,
                            "side": _side_name(position),
                            "entry_basis_bp": entry_basis,
                            "exit_basis_bp": float(basis[i]),
                            "entry_signal_z": entry_signal,
                            "exit_signal_z": float(pending.signal_z),
                            "entry_fill_delay_s": (
                                entry_time - entry_decision_time
                            ).total_seconds(),
                            "exit_fill_delay_s": (
                                now - pending.decision_time
                            ).total_seconds(),
                            "holding_s": (now - entry_time).total_seconds(),
                            "gross_pnl_bp": gross_bp,
                            "cost_bp": float(config.round_trip_cost_bp),
                            "net_pnl_bp": net_bp,
                            "entry_reason": entry_reason,
                            "exit_reason": pending.reason,
                        }
                    )
                    position = 0
                    entry_time = None
                    entry_decision_time = None
                    entry_basis = None
                    entry_signal = None
                    entry_reason = None
                    next_allowed_time = now + cooldown

                pending = None

            elif lateness > max_lateness:
                cancelled_stale_actions += 1
                pending = None

        if filled_this_row or pending is not None:
            continue

        z = signal[i]
        if not np.isfinite(z) or not tradable[i]:
            continue

        if position == 0:
            if now < next_allowed_time:
                continue
            if z >= config.entry_z:
                pending = _PendingAction(
                    action=-1,
                    reason="positive_transient",
                    signal_z=float(z),
                    decision_time=now,
                    due_time=now + delay,
                )
                decisions += 1
            elif z <= -config.entry_z and config.allow_short_spot:
                pending = _PendingAction(
                    action=1,
                    reason="negative_transient",
                    signal_z=float(z),
                    decision_time=now,
                    due_time=now + delay,
                )
                decisions += 1

        else:
            assert entry_time is not None
            held = now - entry_time

            # The exit is directional: crossing through zero and overshooting the
            # opposite side still constitutes completed mean reversion.
            reverted = (
                (position == -1 and z <= config.exit_z)
                or (position == 1 and z >= -config.exit_z)
            )
            stopped = (
                (position == -1 and z >= config.stop_z)
                or (position == 1 and z <= -config.stop_z)
            )
            timed_out = held >= max_hold

            reason: str | None = None
            if stopped:
                reason = "stop"
            elif reverted:
                reason = "mean_reversion"
            elif timed_out:
                reason = "max_hold"

            if reason is not None:
                pending = _PendingAction(
                    action=0,
                    reason=reason,
                    signal_z=float(z),
                    decision_time=now,
                    due_time=now + delay,
                )
                decisions += 1

    mtm_gross: float | None = None
    mtm_net: float | None = None
    if position != 0 and entry_basis is not None and last_tradable_basis is not None:
        mtm_gross = float(position * (last_tradable_basis - entry_basis))
        mtm_net = mtm_gross - float(config.round_trip_cost_bp)

    final_state = ProxyBacktestFinalState(
        interval_start=start_time,
        interval_end_exclusive=end_time,
        last_clock_time=last_clock_time,
        last_tradable_time=last_tradable_time,
        position=position,
        side=_side_name(position),
        entry_decision_time=entry_decision_time,
        entry_time=entry_time,
        entry_basis_bp=entry_basis,
        entry_signal_z=entry_signal,
        pending_action=None if pending is None else _action_name(pending.action),
        pending_reason=None if pending is None else pending.reason,
        pending_decision_time=None if pending is None else pending.decision_time,
        pending_due_time=None if pending is None else pending.due_time,
        last_tradable_basis_bp=last_tradable_basis,
        mark_to_market_gross_bp=mtm_gross,
        mark_to_market_net_bp_if_liquidated=mtm_net,
        completed_trades=len(trades),
        decisions=decisions,
        fills=fills,
        cancelled_stale_actions=cancelled_stale_actions,
    )
    return ProxyBacktestResult(
        trades=pd.DataFrame(trades, columns=_TRADE_COLUMNS),
        final_state=final_state,
    )


def run_proxy_backtest(
    frame: pd.DataFrame,
    *,
    start_time: pd.Timestamp,
    config: ProxyBacktestConfig,
    end_time: pd.Timestamp | None = None,
    basis_col: str = "basis_bp",
    signal_col: str = "signal_stationary_z",
    tradable_col: str = "valid",
) -> pd.DataFrame:
    """Backward-compatible wrapper returning completed trades only."""

    return run_proxy_backtest_detailed(
        frame,
        start_time=start_time,
        end_time=end_time,
        config=config,
        basis_col=basis_col,
        signal_col=signal_col,
        tradable_col=tradable_col,
    ).trades


__all__ = [
    "ProxyBacktestConfig",
    "ProxyBacktestFinalState",
    "ProxyBacktestResult",
    "run_proxy_backtest",
    "run_proxy_backtest_detailed",
]
