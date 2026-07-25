[![BTC basis single-day tests](https://github.com/ugural87/Quantitative_Research/actions/workflows/btc_basis_single_day_state_space.yml/badge.svg)](https://github.com/ugural87/Quantitative_Research/actions/workflows/btc_basis_single_day_state_space.yml)

# BTC Spot–Perpetual Basis: Single-Day Causal State-Space Research

A research-grade, deliberately **single-day** workbench for testing whether the
BTC spot–perpetual basis contains a causally observable mean-reverting component
and whether its scale is economically meaningful relative to explicit costs.

The project separates three claims:

1. **Statistical existence:** does a local-level plus AR(1) transient improve on
   a drifting local-level null?
2. **Model adequacy:** do final-holdout innovations support the fitted
   conditional dynamics?
3. **Economic relevance:** is the transient scale large enough to justify a
   separate execution-research project?

It does **not** claim executable P&L.

## Scope contract: one day by design

This repository intentionally studies one settled UTC day. That is a design
boundary, not an unfinished item.

- The day is partitioned into training, threshold development, purge and final
  holdout intervals.
- Multi-day regime stability, cross-date refits and prospective multi-session
  validation will be implemented in a **separate project**.
- Extending this repository to multiple days is therefore neither required nor
  recommended as a change to this codebase.
- Review this repository as a one-day methodological and software-engineering
  study, not as the multi-day project that follows it.

The execution layer is also a separate project. Public `aggTrades` data cannot
reveal historical queue position, counterfactual maker fills, partial fills,
own-order latency or adverse selection.

## What is implemented

### Causal state-space model

At one-second intervals:

```text
y_t = level_t + transient_t + observation_noise_t
level_t = level_(t-1) + level_innovation_t
transient_t = b * transient_(t-1) + transient_innovation_t
```

Parameters are estimated by innovations maximum likelihood with:

- a strictly one-sided Kalman filter;
- multi-start optimisation;
- bounded structural parameters;
- a local-level null comparison;
- near-optimal-solution identifiability diagnostics;
- explicit half-life and stationary transient scale.

No smoother is used. Parameters are fitted on the training interval only.

### Explicit signal semantics

```python
result = filter_state_space(values, fit)

stationary_z = result.transient_stationary_z(fit.params)
filter_t = result.transient_filter_t
```

`stationary_z` is the economic signal amplitude relative to the structural
stationary transient standard deviation. `filter_t` is posterior state-estimate
confidence. They are not interchangeable.

### One-day chronological research protocol

The notebook creates exactly one chronological partition:

1. **Training:** state-space and AR(1)/OU parameters are estimated.
2. **Threshold development:** an exploratory threshold sweep is run.
3. **Purge:** longer than the maximum episode and fill-delay assumptions.
4. **Final holdout:** the predeclared `ENTRY_Z=2.0` configuration and causal
   baselines are evaluated once.

The exploratory sweep never reads final-holdout outcomes. Benjamini–Hochberg
adjustment is applied to available block-bootstrap p-values across the sweep.

### Causal baselines

The state-space signal is compared under identical event and cost rules with:

- a rolling z-score whose reference window ends at `t-1`;
- a stationary AR(1), interpreted as a discrete-time OU baseline, fitted on
  training data only;
- an explicit no-trade benchmark.

If the training data cannot support the AR(1)/OU stationarity contract, that
baseline is reported as unavailable rather than coerced.

### Timestamp-correct proxy event engine

The proxy backtest enforces what one-second trade-price bars can establish:

- decisions and fills are separate events;
- fills occur only after an explicit delay;
- stale pending actions are cancelled across excessive data gaps;
- holding periods and cooldowns use wall-clock timestamps;
- entry, exit and fill require explicitly tradable observations;
- mean-reversion exits are directional, so a zero crossing that overshoots the
  opposite side still closes the position;
- gross and net P&L signs and costs are unit-tested;
- an unfinished terminal position or pending action is reported explicitly.

Completed-trade P&L never silently absorbs or discards terminal inventory. The
reported terminal mark is hypothetical and remains separate from realised P&L.

### Dependent-data inference

`research_validation.py` provides:

- the single-day train/development/purge/final-holdout partition;
- circular block-bootstrap confidence intervals;
- centred-null circular block-bootstrap mean tests;
- Benjamini–Hochberg false-discovery adjustment.

### Reproducibility

The notebook:

- retains raw Binance ZIP archives;
- calculates SHA-256 checksums;
- keys parsed caches by archive checksum;
- writes the fitted model to JSON;
- writes a run manifest containing archive metadata, partition boundaries,
  configuration, package versions and principal outputs.

A pinned reference environment is supplied in `requirements-lock.txt`.

## Research gates

A result remains diagnostic when any required gate fails:

1. The transient model is not materially preferred to the local-level null.
2. Final-holdout innovations fail centring, scale or residual-autocorrelation
   checks.
3. Near-optimal likelihood solutions imply unstable half-life or transient
   scale.
4. Dependent-data uncertainty includes economically irrelevant convergence.
5. Conservative round-trip costs dominate the plausible signal capture.
6. The state-space signal does not add value relative to causal simple
   baselines under the same proxy event rules.

Gaussian rejection is reported separately. It requires robust inference or a
heavy-tailed model, but it does not automatically reject the conditional mean.

## Data and execution boundary

The notebook uses Binance spot and USD-M perpetual `aggTrades`. These are market
trade prints, not quote or depth history.

The project makes no hidden assumption about:

- bid/ask spread;
- queue priority;
- maker-fill probability;
- two-leg execution latency;
- partial fills;
- adverse selection;
- spot borrow availability;
- order acknowledgements or inventory reconciliation.

The required handoff to a separate execution study is documented in
`EXECUTION_RESEARCH_HANDOFF.md`.

## Repository layout

```text
state_space.py                       causal estimation, filtering and adequacy
research_backtest.py                 timestamp-aware proxy event engine
research_baselines.py                causal rolling-z and training-only AR(1)/OU
research_validation.py               intraday split and dependent-data inference
research_data.py                     checksummed archives and run manifests
btc_basis_realdata_workbench.ipynb   narrated real-data single-day workbench
test_state_space.py                  estimation, causality and signal tests
test_research_backtest.py            event-time, exit and terminal-state tests
test_research_baselines.py           baseline causality and recovery tests
test_research_validation.py          split, bootstrap and FDR tests
test_research_data.py                checksum, atomic download and manifest tests
RESEARCH_PROTOCOL.md                 promotion and falsification contract
EXECUTION_RESEARCH_HANDOFF.md        boundary for the separate execution project
```

## Install

For normal development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest
```

For the exact supplied reference environment:

```bash
python -m pip install -r requirements-lock.txt
pytest
```

The supplied suite contains **47 deterministic tests**. The configured CI gate
requires at least **80% aggregate statement coverage**. In the supplied
environment the suite passes with approximately **80.91%** coverage.

GitHub Actions runs compilation and the full test/coverage gate on Python 3.11
and 3.12.

## Minimal usage

```python
from state_space import fit_state_space, filter_state_space, OnlineBasisFilter

fit = fit_state_space(training_basis_log, dt=1.0)
filtered = filter_state_space(all_basis_log, fit)

economic_signal = filtered.transient_stationary_z(fit.params)
adequacy = filtered.slice(holdout_start, holdout_stop).adequacy_report(max_lag=30)

online = OnlineBasisFilter.from_history(fit.params, training_basis_log)
state = online.update(new_basis_log)
print(state["transient_stationary_z"], state["transient_filter_t"])
```

## Status

This is a portfolio-grade quantitative research prototype with tested causal and
event-time semantics. It is not a live strategy, exchange execution client,
portfolio-management system or multi-day validation study.

Negative findings are first-class results.

## License

MIT — see `LICENSE`.
