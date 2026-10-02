# Forecasting the ten-year yield 1-5 quarters ahead: backtest (2026-10-02)

Script: `validation/backtest_yield_forecasts.py` (`--other` for UK / euro area / Japan).
Origins: SPF survey dates, quarterly, 1992Q1-2026Q2 (US). Targets: the quarterly average
ten-year yield 1, 4 and 5 quarters on (the SPF's own horizon definition; 4-5 quarters is
the end-2027 analogue, 1 quarter end-2026). Scored on the change from the origin; each
method's error is compared with no-change on the SAME forecast dates.

## Methods

| Label | Forecast |
|---|---|
| Spot | no change from the origin date |
| Spot, 20-day avg | no change from a 20-business-day average |
| Forwards | the market forward ten-year yield |
| SPF | the SPF's mean ten-year Treasury forecast (scored against the CMT yield) |
| Regression | the real-policy-rate method: (y10 - i) on (i - pi - r*), r* = SPF ten-year bill less ten-year CPI, expanding window, SPF bill and CPI paths |
| Regression, anchored | spot plus the change that regression predicts |
| Engine | expected-rate leg blended market -> view at 1 - 0.8^H, premium reverting to its trailing ten-year mean at rho^H (the sheet's current construction, incl. the market's priced drift) |
| Spot + view tilt | spot + (1 - lam^H) x (view's ten-year average short rate over [H, H+10] - the market's); premium held |

Views: SPF (the SPF T-bill path, on the curve's basis), perfect foresight (the short rate
that actually followed), oracle (the expected-rate leg the market actually priced at the
target date: the ceiling on what any rate view can add), none (the market's own path).

## US results: RMSE, pp (relative to spot on the same dates)

| | 1 quarter | 4 quarters | 5 quarters |
|---|---|---|---|
| **Spot** | **0.40** | **0.82** | **0.90** |
| Spot, 20-day avg | 0.45 (1.12) | 0.85 (1.04) | 0.93 (1.03) |
| Forwards | 0.41 (1.03) | 0.89 (1.09) | 1.00 (1.11) |
| SPF ten-year forecast | 0.51 (1.29) | 1.01 (1.23) | - |
| Regression | 1.22 (3.23) | 1.30 (1.70) | - |
| Regression, anchored | 0.56 (1.49) | 1.08 (1.42) | - |
| Engine, SPF view (current settings) | 0.41 (1.03) | 0.88 (1.07) | 0.98 (1.08) |
| Engine, no view, premium held | 0.41 (1.02) | 0.86 (1.05) | 0.95 (1.06) |
| Spot + view tilt, SPF view, lam 0.8 | 0.40 (1.02) | 0.87 (1.06) | 0.97 (1.08) |
| Spot + view tilt, perfect foresight, lam 0.8 | 0.39 (0.98) | 0.77 (0.98) | 0.83 (0.98) |
| Spot + view tilt, oracle, lam 0.8 | 0.39 (0.99) | 0.77 (0.95) | 0.84 (0.93) |
| Spot + view tilt, oracle, full weight | 0.36 (0.91) | 0.68 (0.84) | 0.74 (0.83) |

Biases: forwards +0.30pp at 4q, SPF +0.47pp, regression +0.98pp: all three expected
yields to rise more than they did over 1992-2026.

## UK, euro area (Bund), Japan: relative to spot, 4 quarters

| | UK | Euro area | Japan |
|---|---|---|---|
| Forwards | 1.06 | 1.12 | 1.10 |
| Engine, no view, premium reverting (rho 0.9) | 1.01 | 1.09 | 1.05 |
| Engine, no view, premium held | 0.99 | 1.05 | 0.98 |
| Spot + view tilt, oracle, lam 0.8 | 0.94 | 0.92 | 0.97 |
| Spot + view tilt, oracle, full weight | 0.92 | 0.89 | 0.91 |

## What it says

1. **Spot is the benchmark at these horizons, in every market.** Forwards, the SPF's own
   ten-year forecasts and the regression method all do worse, mostly through bias.
2. **The regression method should be retired**: 70% more error than spot at four
   quarters, 40% more even anchored to today. Its level relationship drifts.
3. **Do not assume the market's priced drift materialises**, and **hold the premium at
   today's level**: both the forward drift in expected rates and reverting the premium to
   its recent average made forecasts worse (engine vs spot + tilt; rho 1 vs 0.9).
4. **A rate view adds value only if it beats the market's path.** A perfect view of where
   the priced path goes is worth 16-21% less error at 4-5 quarters (US), 8-11% elsewhere;
   phased in at the current speed (lam 0.8) it is worth 5-7%. The SPF's path, used the
   same way, cost 6-8%.
5. **The phase-in speed is right for the rate leg.** Regressing the change in the market's
   priced path on the SPF-minus-market gap gives 0.10 (1q) to 0.16 (4-5q): the market
   moves 10-16% of the way toward the surveyed path a year. lam = 0.8 phases in 20% in the
   first year. But that convergence did not show up in yields (the premium moved against
   it: slope of the yield change on the gap -0.18 at 4q), which is why the SPF view does
   not pay.
6. **Do not smooth the starting point**: a 20-day average costs 3-12% accuracy. Stability
   comes from the construction instead: the tilt is fixed by the view, so when the
   market reprices only the market leg moves, and a forecast moves by roughly
   1 - w x (share of the move in expected rates) of a spot move.

## Recommended year-end construction

    y(end-year) = spot  +  (1 - 0.8^H) x [view's 10y average policy rate over [H, H+10]
                                           - the market's, on the curve's basis]
                        (+ premium held at today's level)

Today (Baseline, 2026-09-30): US 5.28 spot, 5.30 end-2026, 5.40 end-2027; UK 5.38,
5.35, 5.28; euro area 3.63, 3.62, 3.58; Japan 3.06, 3.05, 3.04. The sheet's current
construction gives 5.24/5.13 (US), 5.31/5.06, 3.60/3.50, 3.02/2.91: lower, through the
market-drift and premium-reversion terms the backtest does not support.

Open item: the policy-rate path is compared with the market's path on the curve's own
short-rate basis without conversion (US curve short rate = fed funds target midpoint
+0.15pp, validation/survey_basis.py; not yet measured against the ECB, BoE and BoJ policy
rates). For the US this understates the view by 0.15pp, about +0.04pp on end-2027.

Look-ahead: the expected-rate curve and premium history come from the Atlas's current
estimation (full-sample parameters). The fair premium, the regression and all views are
real-time.
