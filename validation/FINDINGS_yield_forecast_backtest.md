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

## Addendum: fair value + fading deviation (validation/backtest_fair_value_reversion.py)

    y(end-year) = spot - (1 - rho^H) x resid + (1 - 0.8^H) x (view - market)
    resid = today's premium - the driver model's fair premium (refitted each quarter, real time)

Equivalently: fair value (house rate view, part-weighted, + the drivers' fair premium) plus
rho^H of today's deviation from it.

| Relative RMSE vs spot, 4q (5q) | rho 1 | rho 0.7 | rho 0.5 | rho 0.3 |
|---|---|---|---|---|
| US, no view | 1.00 | 0.95 (0.94) | 0.93 (0.91) | 0.93 (0.90) |
| US, SPF view | 1.07 | 1.02 | 1.00 | 1.00 |
| US, oracle view | 0.95 | 0.90 | 0.88 (0.84) | 0.87 |
| UK / euro area / Japan, no view | 1.00 | 1.08 / 1.05 / 1.07 | 1.15 / 1.14 / 1.22 | |

US: holds in both halves (1994-2009 0.90, 2010-2026 0.95 at rho 0.5, 4q); the sign of the
residual calls the direction of the yield change 59-70% of the time. Forecasts move by
0.75 of a spot move (sd of revisions 0.82 of spot's). UK, euro area, Japan: the residual has
no predictive value; the driver models there are too weak to anchor a forecast.

2026 (US, end-2027 target, house view held at today's Baseline):

| | 27 Feb | 31 Mar | 30 Apr | 29 May | 30 Jun | 31 Jul | 31 Aug | 30 Sep |
|---|---|---|---|---|---|---|---|---|
| 10y (zero) | 4.04 | 4.37 | 4.46 | 4.53 | 4.49 | 4.83 | 4.83 | 5.28 |
| Fair value, end-2027 | 4.43 | 4.42 | 4.43 | 4.55 | 4.54 | 4.53 | 4.49 | 4.87 |
| Forecast, rho 0.5 | 4.15 | 4.24 | 4.28 | 4.38 | 4.39 | 4.53 | 4.51 | 4.95 |
| Spot + tilt (rho 1) | 4.29 | 4.59 | 4.68 | 4.72 | 4.68 | 4.99 | 4.97 | 5.40 |

Of the 1.24pp rise, expected rates 0.21, premium 1.02: fair premium +0.44, unexplained
+0.58. The September jump in fair value is the US expected-inflation driver moving
2.43 -> 2.81, and that comes from the Cleveland Fed 10y series bridged on after the
August SPF; that series is partly market-derived, so a fair value built on it chases
yields. On the SPF alone, September fair value is 4.49 and the forecast 4.72.

## Addendum: what a correct house call is worth, by channel (validation/backtest_inflation_channel.py)

Production models (US driver model; UK / euro area / Japan as US beta + relative drivers),
refitted each quarter in real time. "Correct call" = perfect foresight on the policy path
(oracle) or on the expected-inflation and uncertainty drivers. Relative RMSE vs spot, 4q (5q).

| | US | UK | Euro area | Japan |
|---|---|---|---|---|
| Lean against residual, rho 0.5 | 0.93 (0.91) | 1.12 (1.12) | 1.16 (1.15) | 1.26 (1.27) |
| Correct rate call | 0.84 (0.83) | 0.77 (0.74) | 0.89 (0.86) | 0.93 (0.89) |
| + correct inflation call via the premium | 0.75 (0.74) | 0.86 (0.82) | 0.87 (0.84) | 1.12 (1.11) |
| Realised / modelled premium response to inflation | 1.06 (1.01) | 0.10 (0.18) | 0.76 (0.72) | -0.08 (-0.09) |

The rate call pays in every market. The inflation call pays through the premium in the US
(at the model's coefficient) and the euro area (three-quarters of it); in the UK and Japan
the modelled premium response is far too large and a correct inflation call made the
forecast WORSE -- there it moves yields through the policy path, not the premium.

US mispricing spills over: leaning against the US residual at the spread market's beta cuts
4-5q error by 4-5% for gilts and 7% for Bunds (slope of the yield change on the US residual
-0.47 to -0.71, as in the US itself); Japan neutral (slope -0.14 against beta 0.5).
Baseline path on this construction: scenarios/year_end_10y.py.
