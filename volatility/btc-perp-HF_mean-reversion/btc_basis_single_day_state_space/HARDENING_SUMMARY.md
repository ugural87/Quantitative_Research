# Hardening Summary

## Final audited implementation

1. Preserved the full regular one-second availability clock; invalid
   observations remain `NaN` rather than being deleted.
2. Corrected bar timing so each bar is labelled at the right edge, when all
   included trades are observable; sub-second last-event age is retained.
3. Retained a strictly one-sided state-space filter with explicit economic and
   posterior signal scalings.
4. Removed leading-gap future leakage: states before the first observation stay
   undefined, and burn-in is counted from the first observation.
5. Replaced the covariance subtraction update with Joseph form for numerical
   positive-semidefinite stability under extreme scale ratios.
6. Made innovation diagnostics exact-clock and gap-aware rather than deleting
   missing seconds and compressing time.
7. Made the robust initial scale use only truly adjacent finite observations.
8. Matched local-level-null and transient-model effective likelihood counts.
9. Rejected uniform sample thinning because it changes effective `dt` and
   corrupts half-life interpretation.
10. Added a single-day chronological training, threshold-development, purge and
    final-holdout contract.
11. Corrected purge to cover maximum hold plus both successful fill-delay
    budgets: `max_hold + 2 * (execution_delay + max_fill_delay)`.
12. Added causal rolling-z and training-only AR(1)/OU baselines.
13. Isolated the threshold sweep from the final holdout and held `STOP_Z` fixed,
    so it is an entry-threshold sweep rather than a confounded joint sweep.
14. Added centred-null circular block-bootstrap mean tests and
    Benjamini–Hochberg adjustment for the exploratory sweep.
15. Fixed directional mean-reversion exits so zero-crossing overshoots close the
    position correctly.
16. Added a detailed backtest result that reports terminal open inventory,
    pending actions, hypothetical liquidation mark and stale-action counts.
17. Corrected maximum drawdown to include the initial zero-equity origin.
18. Required finite basis and signal values before a decision and finite
    threshold configuration values.
19. Added explicit inclusive-start/exclusive-end event intervals.
20. Added raw archive retention, SHA-256 checksums, parser-versioned caches, fit
    JSON and a reproducibility manifest with Git and Python metadata.
21. Added explicit cost scenarios and an economic gate based on nominal
    entry-to-exit transient capture rather than an arbitrary multiple of one
    stationary standard deviation.
22. Added numeric promotion gates and an explicit `diagnostic_only` versus
    `execution_research_candidate` status.
23. Corrected plots to show one-sided entry and directional exit boundaries when
    short spot is disabled.
24. Added notebook cell IDs, package metadata, a pinned reference environment and
    GitHub Actions.
25. Expanded the deterministic suite to 58 tests with an enforced 80% aggregate
    statement-coverage threshold; the final audit run passes at 81.08%.
26. Re-executed the entire notebook on controlled synthetic trade archives and
    independently stress-tested Kalman algebra, likelihood parity, covariance
    stability, parameter recovery, null rejection and simple baselines.

## Intentional boundaries

1. This repository remains a **single-day project by design**. Multi-day regime
   validation is a separate follow-on project, not unfinished work here.
2. The current data is `aggTrades`, not order-book replay.
3. Fee and slippage inputs are declared scenarios, not claims about an account's
   actual tier or achievable execution.
4. No queue, spread, maker-fill, partial-fill, latency, adverse-selection,
   funding, borrow or live inventory claim is embedded in the model.
5. Execution research requirements are documented as a handoff rather than
   falsely simulated from unavailable data.
