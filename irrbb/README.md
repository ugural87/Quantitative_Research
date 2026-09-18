# Treasury Quant and IRRBB

This directory contains quantitative research for bank treasury and balance-sheet problems. The current work begins with term-structure model risk because the choice of curve representation flows directly into valuation, sensitivity and interest-rate risk measures.

## Current project

### [IRRBB Term-Structure Model Risk and Banking-Book Measurement Engine](./4F-Svensson_vs_3F-Nelson-Siegel_Opt/irrbb_dynamic_report_project/)

The project compares three-factor Nelson-Siegel and four-factor Svensson within the same data, calibration, diagnostic and banking-book framework.

```mermaid
flowchart LR
    A["Curve observations"] --> B["NS3 and NSS4 models"]
    B --> C["Temporal fit diagnostics"]
    C --> D["Banking-book cash flows"]
    D --> E["NMD and repricing assumptions"]
    E --> F["EVE, NII and stress results"]
```

The research separates two questions that are often collapsed into one:

1. Which curve specification reconstructs the observed term structure more closely?
2. Which specification is sufficiently stable and identifiable for the downstream risk measure?

The full project includes live and offline data modes, parameter and factor diagnostics, curve-shape controls, repricing gaps, EVE and NII scenario outputs, NMD sensitivity, generated artifacts and a reproducible report pipeline.

## Research direction

This category will expand toward ALM, liquidity risk, funds transfer pricing, deposit behaviour, hedging and bond-portfolio risk. Those topics are a roadmap, not implemented projects in the current directory.

