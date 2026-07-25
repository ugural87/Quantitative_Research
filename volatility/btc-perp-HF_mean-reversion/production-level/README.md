# BTC Spot–Perpetual Basis: A State-Space Study

[![tests](https://github.com/ugural87/Quantitative_Research/actions/workflows/ci.yml/badge.svg)](https://github.0.com/ugural87/Quantitative_Research/actions/workflows/ci.yml)

<!-- Replace OWNER/REPO above with your GitHub username and repository name. -->

Research infrastructure for decomposing the Bitcoin spot–perpetual basis into a
slow *equilibrium* component and a fast *mean-reverting* component, and for
testing — honestly — whether the fast component is tradable at second scale.

**This is research infrastructure, not a live strategy.** The headline result
is negative and is stated on purpose: at any fee level a retail or prosumer
account can realistically reach, a one-second basis-reversion strategy does not
clear its own transaction costs. The value here is the estimation and
causality machinery, and the discipline of letting the economics falsify the
idea rather than curve-fitting an equity curve.

## The question

At one-second resolution, the traded basis between `BTCUSDT` spot and the USD-M
perpetual wanders around a slowly moving level. Some of that wander is a
persistent, mean-reverting deviation. Two questions follow:

1. Is the mean-reverting component *statistically* real, or an artifact of a
   drifting level and observation noise?
2. If it is real, is it *economically* real — large enough to pay for the four
   fills a round trip costs?

The repo answers (1) with a state-space model and a formal null comparison, and
(2) with a cost-stressed out-of-sample backtest and an explicit economic gate.

## The model

A local-level (random-walk) equilibrium plus a stationary AR(1) transient,
observed with noise, at interval `dt`:

```
y_t = μ_t + τ_t + ε_t          observed log-basis
μ_t = μ_{t-1} + η_t            η_t ~ N(0, σ_level²)        (permanent level)
τ_t = b · τ_{t-1} + ν_t        ν_t ~ N(0, σ_transient²)    (mean-reverting)
ε_t ~ N(0, σ_obs²)             observation noise
```

with `0 < b < 1`. The AR(1) transient is the discrete-time analogue of an
Ornstein–Uhlenbeck process, so two derived quantities carry the economics:

```
half-life          =  dt · ln 2 / (−ln b)
stationary sd(τ)   =  σ_transient / √(1 − b²)
```

Parameters are estimated by innovations maximum likelihood (a one-sided Kalman
filter) with multi-start optimisation and a Powell fallback. The fit reports a
BIC comparison against a pure local-level null (is there an AR(1) transient at
all?) and an identifiability summary (do materially different decompositions
explain the data almost equally well?), because convergence alone is not proof
of a well-identified model.

## Causality — the property a backtest depends on

Everything downstream relies on the filter being strictly one-sided: the
estimate of `τ_t` may never use an observation after time `t`.

- Parameters are fit on an earlier training slice only; all reported trades
  occur after the cutoff.
- `filter_state_space` is a forward Kalman recursion with no smoother anywhere.
- `no_lookahead_check` perturbs every observation after a split point and
  verifies, to floating-point tolerance, that no filtered value before the
  split moves.
- In the backtest a signal read at second `t` is filled at `t+1`; there is no
  same-bar execution.

These are enforced by the test suite, not just asserted in prose.

## The economic result

Each completed round trip is four fills — two legs at entry, two at exit. With
the notebook's (deliberately generous) `2.0 bp` fee and `0.10 bp` slippage per
fill:

```
round-trip cost = 4 × (2.0 + 0.10) = 8.4 bp
```

The most a single reversion can return, entering at `z_in` and exiting near
`z_out` standard deviations, is about `(z_in − z_out) × stationary sd(τ)`. So
the strategy can only break even if

```
stationary sd(τ)  >  cost / (z_in − z_out)
```

At `z_in = 2`, `z_out = 0.25` that is `8.4 / 1.75 ≈ 4.8 bp`. The one-second
transient standard deviation on `BTCUSDT` is well below that — typically a few
bp at most, often near 1 bp — so the strategy is structurally below break-even.
The notebook's economic gate (`cost / sd(τ) > 3`) trips for exactly this
reason.

And `8.4 bp` is optimistic. At standard Binance rates (spot 0.10% per side;
USD-M perp 0.02% maker / 0.05% taker), a realistic both-legs-maker round trip is
closer to `2 × (10 + 2) = 24 bp` (≈ 19 bp paying fees in BNB). The `2 bp`
per-fill blended figure only holds for the spot leg at deep VIP tiers that
require volumes a retail account will not reach. The honest cost is therefore
two to three times the modelled one, and the conclusion only hardens.

## Why publish a strategy that loses money

Because the negative result is the contribution. The repo shows a causal,
tested estimation pipeline and then uses it to *kill* a plausible-sounding
idea on cost grounds instead of shipping an overfit backtest. The gross signal
and the fee arithmetic are both laid out so the reader can see exactly where
the edge dies — which is the reasoning a costed strategy actually requires.

## Limitations (stated, not hidden)

- **`aggTrades` is a last-trade proxy.** It carries no historical bid/ask or
  queue position, so the backtest is a *signal diagnostic*, not executable
  P&L. It does not replace an order-book replay. Part of the apparent
  one-second reversion is non-synchronous last-trade prints reverting, which is
  not tradable — so even the gross figure is optimistic.
- **Single day of data.** Enough to demonstrate the pipeline; not enough to
  make any production claim. No regime or funding-event coverage.
- **No maker-fill model.** The only order-of-magnitude cost lever is post-only
  execution, and this data cannot validate it.
- The module estimates and filters a signal. Execution, risk limits, inventory,
  funding, borrow and reconciliation are explicitly out of scope.

## What would make a variant worth testing

Not a fee trick — you cannot extract an 8.4 bp cost from a sub-3 bp signal.
Each real lever changes the strategy so that edge-per-trip grows or fills-per-
edge shrink:

- **Push the horizon out.** Re-run the same pipeline at 5s / 30s / 1m / 5m
  bars and plot `cost / stationary sd(τ)` against bar size; find where it falls
  below one. Both the transient and the persistent funding basis grow relative
  to the fixed cost.
- **Maker-only with an order-book replay**, to learn whether the gross edge
  survives realistic fills and adverse selection.
- **Cut fills.** Hold a standing spot inventory and trade only the perp leg
  around it — two fills, not four — accepting that it becomes inventory and
  funding management rather than a neutral basis trade.

## Repository layout

```
state_space.py                     estimation + one-sided Kalman filter + online filter
btc_basis_realdata_workbench.ipynb real-data fit, OOS proxy backtest, threshold sweep, gate
test_state_space.py                causality, recovery, online-vs-batch, validation tests
requirements.txt
```

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the tests
pytest -v

# open the notebook (downloads one day of Binance aggTrades into ./data on first run)
jupyter lab btc_basis_realdata_workbench.ipynb
```

The notebook downloads public Binance archive data from `data.binance.vision`.
The `data/` cache is git-ignored and regenerated on first run.

## Library usage

```python
from state_space import fit_state_space, filter_state_space, OnlineBasisFilter

fit = fit_state_space(training_basis_log, dt=1.0)   # fit on history only
result = filter_state_space(all_basis_log, fit)     # one-sided, no look-ahead
signal_z = result.transient / fit.params.transient_sd

# streaming, one observation at a time, for paper/live use
online = OnlineBasisFilter.from_history(fit.params, training_basis_log)
for x in live_basis_stream:
    state = online.update(x)
    # state["transient_z"], state["level"], ...
```

## Disclaimer

For research and educational purposes only. Nothing here is investment advice
or an offer to trade. The backtest is a cost-stressed proxy and is not proof of
executable profit and loss.

## License

MIT — see [LICENSE](LICENSE).
