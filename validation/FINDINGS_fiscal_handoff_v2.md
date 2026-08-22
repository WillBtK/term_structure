# Assessing the reworked fiscal-sensitivity analysis

`WillBtK/imf_fiscal_sensitivity` has been re-run on **our** term premium in place of ACM.
Five things follow for this repo — one of them a defect in our own model.

---

## 1. Their 65.6bp against our 14.2bp — reconciled, and we are the ones who are wrong

Same premium series, same driver (CBO 5y-ahead total balance), same post-2008 sample,
and a 4.6× gap in the 10y coefficient. Decomposed:

| US 10y premium on the CBO deficit, post-break | bp / 1pp GDP |
|---|---|
| deficit alone | **+31.0** (t +2.5) |
| + expected inflation | +32.8 (t +3.3) |
| + inflation uncertainty | **+14.6** (t +1.6) ← the absorber |
| + price-insensitive ownership | +14.2 (t +1.9) |

**It is not the ownership driver.** Its correlation with the deficit post-break is
−0.057 and removing it moves the coefficient by 0.4bp — so the QE-collinearity worry
flagged in `FINDINGS_fiscal_flow.md` was the wrong suspect.

It is `inflation_uncertainty`, and for a bad reason. Its full-window correlation with the
deficit is +0.569, but **within** every episode the correlation is *negative*:

| window | deficit | uncertainty | corr |
|---|---|---|---|
| 2009–2019 | 2.98% | 0.46 | **−0.76** |
| 2020–2022 | 3.71% | 0.84 | +0.34 |
| 2023–2026 | 5.38% | 1.57 | **−0.76** |

Both series merely drift up across 2008–2026. The +0.57 is Simpson's paradox — a common
trend, not a mechanism — and our uncertainty driver is absorbing it. Dropping COVID does
not fix it (32.2 → 14.1), confirming it is the whole-window drift rather than 2020–21.

Their specification carries **trend + trend²**. Ours does not. Adding one:

| spec | coefficient |
|---|---|
| ours, no trend | +14.2 (t +1.9) |
| ours + trend | +35.3 (t +3.8) |
| **ours + trend + trend²** | **+40.4 (t +5.0)** |
| trend + trend² only | +41.8 (t +6.3) |

**The trend control restores most of the gap and roughly triples the t-statistic.** The
residual 40 → 66 is their remaining controls (excess bond premium, recession dummy,
expected real growth, foreign share) and vintage frequency.

### What this means for us

Our scenario tool **understates the fiscal channel by about a factor of three**, and the
cause is a spurious trend in the inflation-uncertainty driver rather than anything
fiscal. This is a defect to fix, not a difference of opinion: the coefficient we would
have shipped was identified off a common drift that has no mechanism behind it.

Fixing it means either adding a trend control to the driver model, or de-trending the
uncertainty driver. Both need care — a trend in a *forecasting* model extrapolates, which
is exactly the failure mode `RESIDUAL_PERSISTENCE` was introduced to avoid. Not done
here pending that decision.

---

## 2. Our premium corroborated the 2008-09 break; ACM contradicted it

The reason ACM was replaced. Through the identical Quandt-Andrews test:

* **Ours**: break 2008-09 at 5y, 7y, 10y and 20y (SupF 20.3–23.5, 1%), pre-break
  coefficients ≈ 0, post-break 58–66bp.
* **ACM**: break **1995-01, with the regime shift reversed** — 60.7bp pre-break
  collapsing to 17.4bp after, inconsistent with the yield evidence and every rolling
  window in that project.

This is the strongest external validation our term-premium estimates have had. It is a
different model on a different sample reproducing a break found independently in yields,
where the incumbent series did not. Worth stating in the Atlas documentation.

---

## 3. The 0.6 yield-to-premium scalar is superseded

Measured across the curve rather than anchored on ACM's two maturities:

| maturity | 2y | 5y | 10y | 20y | 30y |
|---|---|---|---|---|---|
| premium share of the yield response | 0.48 | 0.69 | **0.80** | 0.89 | 0.94 |

At the short end a wider deficit mostly moves the expected policy path; at the long end
it is almost entirely risk compensation. Anywhere this repo converts between a yield
sensitivity and a premium sensitivity, use the maturity-specific share — 0.80 at our 10y
instrument, not 0.6.

---

## 4. Fiscal shocks are a real-rate effect, and that constrains the scenario tool

Post-break, per 1pp of GDP: the **real** premium moves 57.5bp at 5y and 61.1bp at 10y
(t = −6.8, −7.3), while the **inflation-risk** premium is insignificant at every tenor
(largest |t| = 1.5).

Our scenario tool takes inflation and the fiscal path as separate user inputs, which is
the right structure — but a user who widens the deficit **should not also raise the
inflation path on that account**. The evidence says the fiscal channel does not run
through inflation compensation. Worth saying on the sheet.

---

## 5. Our debt calibration rests on the weakest available footing

`DEBT_COEF = 0.025` is applied to Germany and Japan as a *debt-level* coefficient. That
work finds debt **unidentified** at this frequency — n=19 post-break against 11
parameters, because CBO reports a debt projection on only 45 of 100 vintages — and
recommends the total balance instead. It also warns the fiscal drivers are mutually
collinear and belong in one at a time, which independently confirms the withdrawal of
the "debt + Δdebt together" suggestion for the UK.

Our calibration is not wrong so much as unsupported by the source it points at. If the
US moves to the projected balance, DE and JP should follow rather than keep a debt-level
constant the underlying study declines to estimate.

---

## 6. Fixed on our side

`exports/README.md` told them to fetch over `raw.githubusercontent.com`. That is wrong —
this repo is **private**, and raw.githubusercontent 404s for private repos regardless of
caller. Corrected to an authenticated clone.

## 7. Outstanding: node density is their binding interpolation constraint

Their premium-curve interpolation is accurate to 0.49bp on average but **2.9bp at the 20y
node**, because our export has no node between 10y and 20y. They name denser nodes in
our export as the only real fix, and note the model can emit any maturity — only the
export grid is restricted.

They also tried re-running our model at a denser grid and were off by 46bp at 10y rising
to 90bp at 30y, because `viz/export_daily.py` runs the US nominal curve in **two passes**
with a survey-anchored shifting endpoint solved in between. That is our complexity to
absorb, not theirs. Two options, neither yet done:

* add maturities to the export — cheapest, and keeps the endpoint logic on our side;
* export the daily factor states alongside the endpoint series, after which
  `model/term_premium.py` gives exact values at any τ.

---

# Implemented: the US fiscal channel, measured with the trend and shipped without it

`model/tp_drivers.py::measure_us_fiscal` + `_fit_us_production`. The architecture keeps
estimation and projection separate, which is the design decision the whole exercise
turned on:

* **Measured**: post-break only (2008-09), at **vintage frequency** (the ~43 months CBO
  actually published, recovered from the series itself), with trend and trend² purging
  the shared drift that let uncertainty steal the coefficient. Result **+51.9bp per pp
  of GDP, t = +6.7** — inside the 40–66bp window this document establishes. Robustness:
  monthly NW(24) +40.4 (t +5.0), linear trend only +35.3, excluding COVID +42.4.
* **Shipped**: as a fixed sensitivity applied to the *change* from today's CBO reading
  (5.87%), the mechanism the calibrated debt term established. No trend regressor exists
  in the production model — the purged drift lands in the residual, where ρ decays it.
  Imposing the coefficient on the pre-break history instead would assert an effect the
  pre-break data rejects.
* **The US debt driver is gone.** Debt and the projected balance are collinear
  alternatives, not a joint model; the balance is the identified one. DE/JP keep the
  calibrated debt term and the UK its fitted debt coefficient *pending
  projected-balance sources for them* (AMECO/OBR — the concept exists there too).
* Real-time honesty in backtests: a training window ending before ~2012 has too few
  post-break vintages to measure the channel and gets a zero coefficient, which is what
  a real-time modeller would have had.

## Gates

| gate | before | after |
|---|---|---|
| frozen-2015 decade, US RMSE | 0.72pp | **0.52pp** |
| frozen-2015 end-point error | −0.25pp | −0.86pp |
| rolling 36m vs random walk, US | 0.95 (ratio 0.98) | **0.93 (ratio 0.96)** |
| spreads (UK/DE/JP) | — | unchanged, by construction |
| fiscal coefficient vs the 40–66bp window | absent | **51.9** ✓ |

The frozen-decade RMSE falls 28% — the channel does real out-of-sample work across the
hardest window available. The end-point error worsens (the 2015-frozen measurement has
only ~20 vintages and under-sizes the coefficient, leaving the 2026 projection low);
the path as a whole is much closer.

## Consequences a user sees

* The sheet's US tab gains a **Deficit (% GDP)** row, defaulting to CBO's own baseline —
  entering nothing runs the official projection, visibly, rather than a frozen constant.
  A ten-year holder prices the year-5 (long-run) deficit at ~52bp per point of GDP.
* The default is no longer premium-flat: CBO's own baseline has the deficit widening to
  ~6.6%, worth ~+0.3pp of premium at the ten-year horizon. That is CBO's view, entered
  in a visible input row — not a hidden model view.
* Stated on the page: the fiscal effect is a real-rate effect; widening the deficit is
  not by itself a reason to raise the inflation path.

## One attribution shift to keep an eye on

With debt out of the long-sample fit, the ownership coefficient eased from −4.3 to
**−3.0bp per % of GDP** — now just *below* the 4–8bp event-study band it used to sit
inside. The three slow-moving level variables (debt, ownership, uncertainty) share
their secular identification, and removing one reshuffles the others. The scenario
consequences are modest (QE scenarios move ~30% less than before), but if a future
re-estimation pushes it lower still, the ownership channel needs its own
event-study-anchored treatment rather than a free coefficient.
