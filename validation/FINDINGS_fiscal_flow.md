# Fiscal flow or fiscal stock? — assessing deficits against debt/GDP

**Question.** The production models use debt/GDP. Prior work (and Laubach 2009, who gets
~25bp per point of projected deficit against 3–4bp per point of debt) says the *deficit*
carries the stronger relationship — the fiscal analogue of the stock-versus-flow result
in the QE analysis. Tested here.

`validation/fiscal_flow_vs_stock.py`. Three candidates as a percent of GDP: the stock
(`debt_gdp`), the accrual deficit (`deficit`), and the twelve-month change in the stock
(`d_debt`). Scored on real-time refits at the tool's three-year horizon, each market in
its **production** specification — US standalone, others as a US-relative spread — with
the residual decaying at ρ = 0.9.

| market | debt | deficit | Δdebt | both | n | best |
|---|---|---|---|---|---|---|
| **US** | 0.86 | 0.76 | **0.75** | 0.85 | 33 | Δdebt |
| **UK** | 0.71 | 0.87 | 0.73 | **0.64** | 10 | both |
| **DE** | **0.70** | 0.77 | 0.86 | 0.83 | 24 | debt |
| **JP** | 1.47 | 1.47 | **1.45** | 1.45 | 21 | Δdebt (marginal) |

## Three conclusions

**1. The hypothesis replicates in the US, and only there — but decisively.** The flow
cuts out-of-sample RMSE from 0.86 to 0.75, a 13% improvement, in the market with the
longest and cleanest sample and the only one whose fiscal coefficient is fitted rather
than calibrated. That is a real result, not noise.

**2. The deficit is not the right flow measure — the change in the debt stock is.**
Δdebt beats the accrual deficit in every market (US 0.75 vs 0.76, UK 0.73 vs 0.87,
DE 0.86 vs 0.77 is the only reversal and both lose there, JP 1.45 vs 1.47). This
refines the hypothesis rather than rejecting it: what preferred habitat says the
price-sensitive market must absorb is the change in *outstanding marketable debt*, and
that differs from the accrual balance through valuation, off-budget financing and
cash-versus-accrual timing. Δdebt also needs no new data source and inherits the
committed WEO/national debt projections for free — a deficit path would need its own
forecast, which for a scenario tool matters.

**3. It does not generalise, so a blanket switch is wrong.** Germany prefers the stock
decisively (0.70 against 0.86) and the UK wants both (0.64, with debt at t = +3.9 and
Δdebt at t = +6.7 side by side). Switching everything to a flow would buy the US 0.11pp
and cost Germany 0.16pp.

## Why "just include both" does not settle it

For the UK it does — 0.64 is the best cell in the table. For the US it does not: with
both in, the sign-constrained in-sample fit picks the stock and zeros the flow, so the
US lands at 0.85 rather than 0.75. The in-sample and out-of-sample preferences genuinely
disagree there, which no automatic selection rule resolves.

## Recommendation

Add Δdebt as an available driver and select per market — Δdebt for the US, both for the
UK, stock for Germany, Δdebt for Japan — **with the caveat stated plainly**: that is
selection on 10–33 out-of-sample windows, and the UK cell in particular rests on ten.
The safer alternative is to change nothing and record that the US would fit ~13% better
with a flow measure. The evidence supports the first; sample size counsels the second.
Not a call to make silently, so it is not made here.

## CORRECTION — the test above was not a fair test

Handed the prior fiscal work (`WillBtK/imf_fiscal_sensitivity`, replicating and
extending Furceri, Goncalves and Li 2025), two things above turn out to be wrong, and
both for the same reason: **the wrong fiscal object, on the wrong sample.**

**Wrong object.** Everything above uses the *realised* annual deficit from FRED. The
driver in that literature is the **CBO five-year-ahead projected total balance,
observed in real time at each baseline vintage** — forward-looking by construction,
which is the point: what should price a ten-year bond is the fiscal path expected over
its life, not the deficit already run. It is also cyclically cleaner than the realised
balance without any adjustment.

**Wrong sample.** A Quandt-Andrews SupF search in that work puts a break at **2008-09**
(SupF 19.35, 1%): pre-break α = −0.065 (se 0.099, indistinguishable from zero),
post-break α = −0.595 (se 0.071), about nine times larger. Fitting across the break
averages two regimes.

Re-run with the right object and the break, on the US ten-year premium, Newey-West(24),
bp per 1pp of GDP:

| sample | CBO 5y-ahead deficit | realised deficit | debt | Δdebt | R² (deficit / others) |
|---|---|---|---|---|---|
| full 1984+ | +6.0bp (t +1.3) | 0 | +0.8bp (t +1.3) | 0 | 0.914 / 0.914 |
| pre-break | 0 | 0 | 0 | 0 | 0.913 / 0.914 |
| **post-break** | **+14.2bp (t +1.9)** | 0 | 0 | 0 | **0.726 / 0.699** |

Post-break, **the projected deficit is the only fiscal driver that is identified at
all** — every alternative pins at zero under the sign constraint. Pre-break nothing is,
reproducing the near-zero pre-GFC result. And the full sample gives +6.0bp against
+14.2bp post-break: a **2.4× attenuation**, close to the ~3× that work predicts.

Out of sample, training post-break only, three-year horizon: **0.82 for the projected
deficit** against 0.85 debt, 0.86 Δdebt, 0.86 realised deficit — the ranking holds,
though on 18 overlapping windows drawn from ~43 independent vintages.

### So the original hypothesis was right, and this document's first half was wrong

The deficit does beat debt/GDP — provided it is the projected balance and the sample is
the post-2008 regime. The Δdebt result above survives only as a statement about
*realised* fiscal data on a regime-averaged sample, which is not the relevant question.

Magnitude: +14.2bp per pp of deficit on the term premium here, against 39.1bp at 10y in
the source work. Ours is lower because it is a different premium (AFNS + shadow rate on
a regime-clean sample, not ACM) and because it controls for expected inflation,
inflation uncertainty and price-insensitive ownership — QE and the deficit move together
post-2008, so the ownership driver absorbs part of what a fiscal-only regression
attributes to the deficit. Scaled by that work's ~60% yield-to-premium ratio, 14.2bp of
premium implies ~24bp of yield, against Laubach's 25bp. The three numbers are consistent.

**Today this matters.** CBO's five-year-ahead deficit is 5.87% of GDP against a
post-break mean of 3.55% — 2.3pp wide, worth **+33bp of term premium** at the post-break
coefficient.

### Caveats before this is acted on

* t = +1.9. Marginal, and the 216 monthly observations rest on ~43 independent vintages.
* The post-break sample starts in 2008-09, so the fiscal driver's whole identifying
  range coincides with QE. Its coefficient and the ownership driver's are not cleanly
  separable, and the honest reading is that they share the variation.
* Fiscal drivers are mutually collinear. One at a time — never debt and deficit
  together, which invalidates the "both" recommendation made for the UK above.

## What was ruled out

The five-year average deficit — a proxy for the structural balance, closer to Laubach's
projected measure — was tested and is worse than both alternatives everywhere except a
truncated German sample (UK 2.75 out of sample against 0.71 for the stock). Raw
cyclicality is not the thing that needs stripping out.
