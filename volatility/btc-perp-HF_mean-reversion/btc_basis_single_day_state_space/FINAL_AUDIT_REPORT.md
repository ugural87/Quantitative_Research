# Final Technical Audit Report

## Audit conclusion

The final audited implementation is internally consistent within its declared
scope: a **single-day, trade-price, causal research prototype**. The audit found
several material defects in the previous revision; all are corrected in version
`0.3.0` and locked by regression tests.

No remaining known high-severity defect was found in the state-space algebra,
chronological partition, proxy event engine, completed-trade accounting,
baseline construction, notebook graphs or reproducibility contract after the
final fixes and stress tests.

This conclusion does **not** convert the project into an executable strategy.
It validates research and proxy semantics, not bid/ask fills, queue position,
two-leg execution or multi-day generalisation.

## Material defects found and corrected

| Area | Previous defect | Final correction |
|---|---|---|
| Bar time | A one-second bar was labelled at its left edge although its last price used trades arriving later in that second. | Bars are labelled at the right-edge availability boundary; label `t` contains `[t-1s, t)`. |
| Trade age | Active bars appeared to have age zero regardless of the sub-second last-trade time. | Age is measured from the actual last event timestamp. |
| Leading gaps | The first future finite observation was copied into states before it existed. | Pre-observation states remain `NaN`; future perturbations cannot change them. |
| Burn-in | Leading missing clock rows could consume the burn-in budget. | Burn-in starts at the first finite observation. |
| Kalman covariance | The subtraction covariance update was vulnerable to cancellation at extreme scale ratios. | Joseph-form covariance update is used in Python and Numba paths. |
| Gap diagnostics | Deleting missing innovations compressed time; the first gap-aware implementation could dilute autocorrelation as missingness increased. | Exact clock-lag pairs and pairwise correlations are used, with a generalised Ljung–Box denominator. |
| Initial scale | Differences across missing clock gaps were treated as one-step differences. | Robust scale uses only adjacent finite pairs. |
| Null BIC | Null-model effective sample count could differ from the fitted-model likelihood count. | Both use the actual filter likelihood count. |
| Purge | Only one fill-delay budget was included. | Purge is `max_hold + 2*(execution_delay + max_fill_delay)`; current value is 66 seconds. |
| Exit semantics | A zero-crossing overshoot could fail to close a mean-reversion position. | Directional crossing exits are implemented and tested. |
| Terminal state | Open inventory or a pending action could disappear from completed-trade output. | Typed final state reports inventory, pending actions and hypothetical MTM separately. |
| Drawdown | Equity started at the first trade result, so an initial loss could produce zero drawdown. | Drawdown includes the zero-equity origin. |
| Decision validity | Non-finite thresholds or a non-finite basis could reach decision logic. | Configuration and decision inputs are validated. |
| Threshold sweep | High entry thresholds also changed the stop threshold, confounding interpretation. | `STOP_Z` remains fixed and the grid stops below it. |
| Exit graph | A symmetric shaded band contradicted directional overshoot exits and one-sided trading. | The graph shows the active short-basis entry and directional exit boundary. |
| Economic gate | Cost was compared with an arbitrary multiple of one transient standard deviation. | Cost is compared with nominal entry-to-exit transient movement. |
| Promotion status | Gates were discussed but not combined into an explicit verdict. | The notebook emits `diagnostic_only` or `execution_research_candidate`. |
| Cache provenance | Parser changes could reuse a checksum-matching old pickle. | Cache keys contain both archive checksum and parser version. |
| Manifest | Git/Python context and promotion gates were incomplete. | Commit, dirty state, Python/packages, gates and status are recorded. |
| Notebook schema | Cells lacked stable IDs and triggered an nbformat future-compatibility warning. | Every cell has a unique deterministic ID. |

## Final automated verification

- **58/58 tests passed**.
- Aggregate statement coverage: **81.08%**; enforced minimum: **80%**.
- Every notebook code cell parses.
- Every notebook output is cleared and every execution count is `null`.
- Every notebook cell has a unique ID.
- Source compilation passes.
- Version `0.3.0` wheel builds and imports successfully.

## Independent numerical stress tests

These checks were implemented independently of the production test assertions.

| Stress test | Result |
|---|---:|
| Matrix-form Kalman oracle, random parameters/gaps | 200 cases; max absolute state/covariance error `5.79e-15` |
| Numba versus Python likelihood | 100 cases; max log-likelihood difference `1.82e-12`; effective counts identical |
| Covariance PSD, extreme parameter ratios | 1,000 cases; worst negative eigenvalue was numerical round-off; separate 5,000-case worst relative value `-1.43e-16` |
| Level-offset equivariance | Max error `1.78e-15` |
| Scale equivariance | Max error `1.13e-17` |
| Independent proxy-event state machine | 1,000 random irregular-clock/configuration cases; every trade record and final state matched exactly |
| Pure local-level null | 10/10 simulations rejected the unnecessary transient by BIC; delta-BIC range approximately `-17.39` to `-14.86` |

The tiny negative covariance eigenvalues only occurred under extreme scale ratios
and were approximately machine precision relative to the covariance magnitude;
they do not represent a material PSD failure.

## Synthetic model-recovery benchmark

A 30,000-second controlled series was generated from the same model class. The
first 20,000 seconds were used for fitting and the final 10,000 for comparison.
Costs were set to zero solely to compare signal recovery under identical event
rules.

- True half-life: **11.2023 s**.
- Fitted half-life: **11.3067 s**.
- BIC improvement over the local-level null: **383.48**.

| Signal | Correlation with latent transient | Completed trades | Gross total (bp) | Gross/trade (bp) | Gross win rate |
|---|---:|---:|---:|---:|---:|
| Oracle latent state | 1.0000 | 78 | 332.79 | 4.2665 | 100.00% |
| State-space | 0.8846 | 40 | 149.12 | 3.7281 | 97.50% |
| Causal rolling z | 0.8717 | 95 | 325.44 | 3.4257 | 94.74% |
| Training-only AR(1)/OU | 0.1850 | 295 | 101.26 | 0.3433 | 53.56% |

Interpretation: the rolling baseline traded more often and therefore accumulated
more total convergence, while the state-space signal recovered the latent
component more accurately and produced higher convergence per completed event.
This is exactly why the notebook reports multiple estimands rather than declaring
one model the winner from a single headline metric.

## Full notebook execution on controlled trade archives

The complete notebook was executed against synthetic spot and perpetual ZIP
archives that reproduce the expected archive schema and contain one trade per
second at `+0.800 s`. This validated acquisition, parsing, availability bars,
model fitting, splits, graphs, backtest, baselines, bootstrap, cost stress,
manifest and terminal state as one integrated path.

Key consistency checks:

- 86,400 availability bars were produced.
- First label: `00:00:01`; final label: next-day `00:00:00`.
- Measured event age: **0.200 s**, as required by a trade at `+0.800 s`.
- Split sizes: 60,000 training; 8,000 development; 66 purge; 18,334 holdout.
- Fitted half-life: **10.882 s**; transient SD: **0.5786 bp**.
- Final-holdout adequacy passed; lag-1 innovation correlation was **0.0085**.
- Completed holdout episodes: **39**.
- Gross convergence: **43.2086 bp**.
- At the declared 8.4-bp round-trip scenario, net proxy total equals
  `43.2086 - 39*8.4 = -284.3914 bp` exactly.
- Nominal threshold capture: **1.0126 bp**; cost/capture: **8.2958**.
- Therefore the explicit final status is correctly **`diagnostic_only`** because
  the economic-scale gate fails despite statistical and dynamic gates passing.

## Graph audit

1. **Prices, basis and staleness:** spot/perpetual paths align; basis is the log
   price difference in bp; age is 0.2 s in the controlled data; event-time skew
   is zero.
2. **State-space decomposition:** observed basis, filtered level and transient
   use consistent units. The active one-sided entry and directional exit lines
   match the event engine.
3. **Equity:** gross and net curves begin at zero. The net curve differs from
   gross by the cumulative declared cost per completed episode.
4. **Histograms:** every net observation is the corresponding gross observation
   shifted by exactly the declared round-trip cost.
5. **Signal versus convergence:** x-values are decision-time stationary z-scores;
   y-values are completed gross basis convergence.
6. **Threshold sensitivity:** only entry threshold changes; `STOP_Z` is fixed;
   the final holdout is not read.

## What could not be verified in this environment

The repository intentionally clears notebook outputs and ignores raw data and
artifacts. Network access in the audit runtime could not resolve the Binance
archive host, so the actual 2026-07-20 Binance numerical outputs were not
re-downloaded here. Therefore:

- code, mathematics, controlled outputs and chart construction were verified;
- the user's actual real-data fit/trade numbers require one clean local notebook
  run from the checksummed archives;
- the resulting manifest and executed notebook should be retained privately or
  supplied for a final number-for-number comparison.

This is the only unresolved verification item. It is an input-access limitation,
not a known source-code defect.

## Remaining intentional research limitations

- `aggTrades` are trade prints, not synchronized executable quotes or depth.
- Last-trade event-time skew is measured and displayed, but cannot remove all
  lead/lag microstructure effects.
- Fee and slippage values are scenarios, not verified account costs.
- Basis-change bp is a convergence proxy, not exchange-replayed P&L.
- Short-spot entries are disabled by default.
- Multi-day stability is intentionally delegated to a separate project.
- A Gaussian likelihood may still require a heavy-tailed successor when real
  holdout innovations reject Gaussianity.

## Final classification

**Portfolio-grade quantitative research prototype with research-engineering and
falsification discipline.** The final audited version is suitable for detailed
study and recruiter review, provided it is described as signal research and not
as a production or proven-profit trading system.
