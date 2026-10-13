# Ten-year breakevens: what moves them (2026-10-13)

Code: `model/breakeven.py`. Framework: BE = expected inflation (the scenario's own
forecasts, averaged over the ten years) + inflation risk premium (today's breakeven less
today's Baseline expected inflation, held constant).
Phased in from today's breakeven by 1 - lam^h with the nominal projections' lam (0.8
Baseline; 0.685 for a scenario averaging 4%+ inflation, else 0.40).

Why the premium is held, not reverted. Out of sample on real-time refits (quarterly
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
