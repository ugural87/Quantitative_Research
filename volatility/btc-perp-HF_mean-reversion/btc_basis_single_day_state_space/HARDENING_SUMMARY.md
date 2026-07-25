# Hardening Summary

## Implemented

1. Preserved the full regular one-second clock; invalid observations remain
   `NaN` rather than being deleted.
2. Retained a strictly one-sided state-space filter with explicit economic and
   posterior signal scalings.
3. Rejected uniform sample thinning because it changes the effective `dt` and
   corrupts half-life interpretation.
4. Added a single-day chronological training, threshold-development, purge and
   final-holdout contract.
5. Added causal rolling-z and training-only AR(1)/OU baselines.
6. Isolated the threshold sweep from the final holdout.
7. Added centred-null circular block-bootstrap mean tests and Benjamini–Hochberg
   adjustment for the exploratory sweep.
8. Fixed directional mean-reversion exits so zero-crossing overshoots close the
   position correctly.
9. Added a detailed backtest result that reports terminal open inventory,
   pending actions, hypothetical liquidation mark and stale-action counts.
10. Added explicit inclusive-start/exclusive-end event intervals.
11. Added raw archive retention, SHA-256 checksums, checksum-keyed parsed caches,
    fit JSON and a reproducibility manifest.
12. Added package metadata, a pinned reference environment and GitHub Actions.
13. Expanded the deterministic suite from 23 to 47 tests with an enforced 80%
    aggregate statement-coverage threshold.
14. Rewrote the notebook as a narrated, outputs-cleared research workbench.

## Intentional boundaries

1. This repository remains a **single-day project by design**. Multi-day regime
   validation is a separate follow-on project, not unfinished work here.
2. The current data is `aggTrades`, not order-book replay.
3. No queue, spread, maker-fill, partial-fill, latency, adverse-selection,
   funding, borrow or live inventory claim is embedded in the model.
4. Execution research requirements are documented as a handoff rather than
   falsely simulated from unavailable data.
