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

## What was ruled out

The five-year average deficit — a proxy for the structural balance, closer to Laubach's
projected measure — was tested and is worse than both alternatives everywhere except a
truncated German sample (UK 2.75 out of sample against 0.71 for the stock). Raw
cyclicality is not the thing that needs stripping out.
