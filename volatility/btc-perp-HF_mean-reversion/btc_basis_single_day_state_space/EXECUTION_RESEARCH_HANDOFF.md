# Execution Research Handoff

This repository stops at causal signal research and a last-trade proxy. A
separate execution project must not infer the following from public
`aggTrades`.

## Required market data

At minimum, capture synchronized spot and perpetual:

- exchange event timestamp and local receive timestamp;
- best bid and ask prices and displayed sizes;
- preferably depth updates with sequence numbers;
- trade prints with aggressor side where available;
- funding timestamps and realised funding rates;
- venue status, symbol filters and trading interruptions.

## Required own-order telemetry

For every order and leg:

- client and exchange order identifiers;
- local decision and submission timestamps;
- acknowledgement, rejection and cancellation timestamps;
- requested and acknowledged price/quantity;
- partial-fill sequence, price, quantity and fee;
- remaining quantity and final order state;
- inventory before and after each event.

## Questions the execution project must answer

1. Is the signal still present at executable bid/ask prices?
2. What fraction of apparent convergence survives two-leg latency?
3. How often does one leg fill without the other?
4. What is the adverse-selection profile after maker fills?
5. What crossing or maker policy is justified by queue and fill evidence?
6. How do fees, funding, borrow and inventory constraints alter net economics?
7. Can live acknowledgements and fills be reconciled exactly to inventory?

## Promotion boundary

No result from this single-day `aggTrades` workbench should be described as an
executable strategy until the separate project answers these questions with
quote/depth data and own-order telemetry.
