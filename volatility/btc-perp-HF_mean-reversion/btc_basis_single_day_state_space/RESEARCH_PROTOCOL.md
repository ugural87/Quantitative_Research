# Single-Day Research Promotion Protocol

## Fixed repository scope

The estimand and validation contract in this repository apply to one settled UTC
day only. This is intentional.

The day is divided chronologically into training, threshold development, purge
and final holdout. Multi-day regime stability and cross-date refitting are
reserved for a separate project and must not be presented as a missing step in
this repository.

## Estimands

The primary statistical estimand is the stationary scale and persistence of a
causally filtered transient component in the log spot–perpetual basis.

The primary economic estimand is gross basis convergence per completed signal
episode in the final holdout, before any claim about executable fills.

Terminal open inventory is a separate state variable. It is never silently
included in or removed from completed-trade P&L.

## Confirmatory and exploratory separation

### Training

- Fit state-space and AR(1)/OU parameters.
- Freeze all structural estimates before later intervals.

### Threshold development

- Run threshold sensitivity only here.
- Treat the full sweep as exploratory.
- Apply dependent-data inference and false-discovery adjustment where p-values
  are reported.
- Do not use this interval to rewrite final-holdout outcomes.

### Purge

The purge must be at least
`max_hold + 2 * (execution_delay + max_fill_delay)`. This covers a delayed entry
fill, the maximum hold measured from that fill, and a delayed exit fill. A
decision initiated during development must not produce a realised exit in the
final holdout.

### Final holdout

- Evaluate the predeclared primary configuration once.
- Compare the state-space signal with causal simple baselines under identical
  event and cost assumptions.
- Evaluate state-space innovation adequacy only with observations unavailable at
  parameter-estimation time.
- Report causal simple baselines as comparative context. Do not automatically
  select a winner from final-holdout trade count, total convergence or average
  convergence without predeclaring which estimand is primary.

## Predeclared rejection gates

A candidate is rejected or retained at diagnostic status when any of the
following applies:

1. The transient model is not materially preferred to the local-level null.
2. Final-holdout innovations retain material centring, scaling or serial-
   dependence failures.
3. Near-optimal likelihood solutions imply materially unstable half-lives or
   transient scales.
4. The dependent-data confidence interval for mean gross convergence includes
   economically irrelevant values.
5. The declared round-trip cost scenario exceeds the nominal
   entry-to-exit transient movement.
6. The final holdout has too few completed episodes for the declared inference
   gate, or the dependent-data gross-mean interval includes zero.
7. Terminal state, cancelled instructions or data gaps make a headline result
   incomplete or misleading.

Gaussian rejection alone does not necessarily reject the conditional mean. It
invalidates naive Gaussian inference and requires robust inference or a
heavy-tailed likelihood before stronger statistical claims.

## Causality contract

- Parameter fitting receives training observations only.
- Filtering is forward-only; no smoother is used.
- Future perturbations cannot change prior filtered states, including before
  the first finite observation.
- Burn-in is counted from the first observation, not from leading missing clock
  rows.
- Online and batch filtering agree observation by observation.
- Rolling baseline statistics end at `t-1`.
- All intervals use timezone-aware wall-clock boundaries.
- Bars are labelled at their availability boundary: label `t` contains trades
  from `[t-dt, t)`.
- Innovation autocorrelation uses exact clock-lag pairs; gaps are not deleted.

## Economic proxy contract

- A decision and a fill are distinct events.
- A fill cannot occur on the decision bar.
- Pending instructions expire when the permitted lateness budget is exceeded.
- Mean-reversion exit is directional and recognises a crossing overshoot.
- Completed trades, terminal inventory and pending actions are reported
  separately.
- Four-fill round-trip cost scenarios are explicit and are not represented as
  verified account fees.

## Separation of evidence

### Signal validity

May be studied with causal trade-price observations, strict time handling,
final-holdout diagnostics and dependent-data inference.

### Execution feasibility

Requires historical or prospectively captured quotes/depth, explicit crossing
or maker assumptions, two-leg latency, partial-fill and adverse-selection
analysis.

### Live calibration

Requires the account's own order acknowledgements, execution reports,
cancellations, partial fills and inventory reconciliation. Public market data
cannot supply this counterfactual information.

## Reproducibility

Every recorded run must include:

- exact UTC date, symbol and venue definitions;
- archive URLs, raw file paths and SHA-256 checksums;
- parsed row counts and timestamp units;
- code commit when run from version control;
- exact train, development, purge and final-holdout boundaries;
- model starts and near-optimal solutions;
- final-holdout adequacy diagnostics;
- gross economics and explicit costs;
- terminal backtest state;
- Python and package versions, Git commit/dirty state and fit artifact paths;
- explicit promotion-gate values and final research status;
- rejection reasons as prominently as continuation reasons.
