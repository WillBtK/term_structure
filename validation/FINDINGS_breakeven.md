# Ten-year breakevens: what moves them (2026-10-13)

Code: `model/breakeven.py`. Framework: BE = expected inflation (the scenario's own
forecasts, averaged over the ten years) + inflation risk premium (today's breakeven less
a constant per market: US 0.38pp, UK 0.34pp on a CPI footing, the premium at which the
Baseline's implied real yield converges on Baseline real GDP growth by 2035).
Phased in from today's breakeven by 1 - lam^h with the nominal projections' lam (0.8
Baseline; 0.685 for a scenario averaging 4%+ inflation, else 0.40).

Why today's breakeven is not reverted toward a fitted level (and only the gap to
E + IRP is phased out). Out of sample on real-time refits (quarterly
origins; US 2004-2026, UK 2004-2026 on a CPI footing), RMSE relative to no change:

| model | US 1y | US 3y | US 5y | UK 1y | UK 3y | UK 5y |
|---|---|---|---|---|---|---|
| level fit (expected inflation, headline, volatility), residual held | 1.74 | 2.32 | 2.74 | 1.51 | 1.54 | 1.34 |
| same, residual faded fully | 1.89 | 2.45 | 2.77 | 1.92 | 1.92 | 1.51 |
| change in headline inflation only | 0.92 | 0.82 | 0.84 | 1.08 | 0.74 | 0.75 |
| + inflation volatility | 1.53 | 1.67 | 1.91 | 1.37 | 2.80 | 3.29 |

Reverting the level toward a fitted value never helps; inflation volatility makes every
forecast worse. Twelve-month changes: US breakeven +0.11pp per 1pp of headline inflation
(t 5.1), UK +0.05 (t 2.2, ex the 2022-23 LDI crisis). Multi-year inflation averages enter
negatively (an inflation run is read as one the central bank reverses).

Today's premium (CPI footing): US -0.35pp (TIPS liquidity), UK +0.75pp (pension demand
for linkers). Euro area pending its breakeven history (`macro/data/ea_breakeven_10y.csv`).

Check against real GDP growth (implied real yield = nominal projection less breakeven,
2035, vs the scenario's ten-year forward growth): Baseline US 2.29 vs 2.10, UK 1.24 vs
1.20; Goldilocks US 2.63 vs 3.00, UK 1.91 vs 2.25; High Growth High Inflation US 3.49 vs
3.10, UK 2.37 vs 2.25. Stagflation (US 4.32 vs 1.50, UK 3.79 vs 0.80) and Hard Landing
(US 0.19 vs 1.25, UK -1.04 vs 0.50) fail it, and no constant premium fixes that: given
real yields at growth, the nominal projections would need an inflation risk premium of
+3pp in Stagflation and -0.7 to -1.2pp in Hard Landing. The gap is in the nominal
projections for those two scenarios.
