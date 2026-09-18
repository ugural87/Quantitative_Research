# Market Quant and Systematic Research

This directory contains research on market behaviour, volatility, relative value and systematic signals. The emphasis is on causal information timing, defensible backtests and an explicit boundary between a statistical finding and an executable strategy.

## Current project

### [BTC Spot-Perpetual Basis: Single-Day Causal State-Space Research](./btc-perp-HF_mean-reversion/btc_basis_single_day_state_space/)

The project studies the BTC spot-perpetual basis during one trading day. The one-day scope is deliberate: it provides a controlled research and engineering test, not evidence of cross-regime persistence.

```mermaid
flowchart LR
    A["Raw spot and perpetual events"] --> B["Availability-time bars"]
    B --> C["Causal state-space model"]
    C --> D["Signal semantics"]
    D --> E["Chronological evaluation"]
    E --> F["Statistical research gates"]
```

The implementation includes causal filtering, explicit signal timing, baseline comparisons, a timestamp-correct proxy event engine, dependent-data inference, deterministic checks and automated tests. Exchange-grade order books, fees, slippage, funding and live execution are outside the current scope.

## Research direction

This category will expand toward volatility modelling, realised-volatility forecasting, systematic factors, momentum and mean reversion, market regimes, portfolio construction and execution-aware backtesting. These are planned research lines, not current performance claims.

