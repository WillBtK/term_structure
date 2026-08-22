# Can the driver models be trusted forward? — out-of-sample evidence

`validation/tp_scenario_oos.py`. Two tests: fit frozen at end-2015 and driven with
realized drivers for a decade; and expanding-window annual refits projecting 36 months
ahead, scored against the random walk. Reproduce with
`python -m validation.tp_scenario_oos`.

Current specification (US fiscal block + projected-balance fiscal channel in every
market, relative debt stock removed from the spreads — see the dated section at the
end):

|        | frozen-2015 RMSE | end-point error | 3y model RMSE | 3y RW RMSE | ratio |
|--------|-----------------|-----------------|---------------|------------|-------|
| **US** | 0.52pp          | −0.86pp         | 0.93pp        | 0.97pp     | 0.96  |
| **UK** | 0.67pp          | +0.50pp         | 1.17pp        | 1.00pp     | 1.17  |
| **DE** | 1.02pp          | −0.57pp         | 1.31pp        | 1.00pp     | 1.30  |
| **JP** | 1.48pp          | −2.00pp         | 0.96pp        | 0.50pp     | 1.91  |

For reference, the same gates before the fiscal work (debt-stock channels in the
spreads, no US fiscal block): US 0.72 / −0.25 / 0.98; UK 0.58 / −0.33 / 0.95;
DE 1.44 / −1.17 / 1.24; JP 1.69 / −2.56 / 1.88.

## Verdict

**Credible as a conditional scenario mapping; not an unconditional forecaster — and it
must never be presented as one.** The user supplies the macro scenario; the model
prices it. Matching the random walk at three years (US 0.96) is the expected
result: nothing in this literature beats no-change unconditionally at that horizon
(Thornton–Valente 2012). Anchoring is what makes that safe — an unchanged scenario
returns today's premium exactly, so the model adds no drift of its own, only the
translation of deliberate scenario deviations. The UK's ratio moved above one when
its debt channel was retired (see the dated section at the end) — the cost of
consistency, stated there.

## Why DE and JP fail the frozen test

Identification, not specification. Frozen at 2015, Germany's fit had seen **zero
months of PSPP** (it began March 2015) and Japan's had never seen BoJ holdings above
~30% of GDP; both extrapolated coefficients into driver ranges outside their samples.
Japan's random-walk RMSE of 0.50pp is yield-curve control itself — no-change was
enforced policy, unbeatable and uninformative as a benchmark. Today's fits contain a
complete QE expansion **and** contraction, exactly the variation those coefficients
needed. This defence is fair for the ownership channel and should be retired the day
either market's coefficient moves materially on new data.

## External validity (the checks an IC can verify independently)

- QE sensitivity **−6.5bp per % of GDP** vs the event-study range of ~4–8bp
  (Ihrig et al. 2018; Williams 2014 survey).
- Fed and foreign-official coefficients land within a whisker of each other
  (−0.069 / −0.064) with no constraint forcing it — consistent with one mechanism:
  price-insensitive duration absorption.
- Fiscal at +51.9bp per point of the five-year-ahead projected deficit/GDP vs the
  Laubach (2009) 20–29bp and the deficit-based literature's 18–28bp per point —
  above the classic range, measured post-2008 only, and consistent with the
  reworked handoff analysis (FINDINGS_fiscal_handoff_v2.md).
- US-level-plus-spread structure per the spillover literature (Rey;
  Miranda-Agrippino–Rey; Diebold–Li–Yue; Bauer–Neely — US shocks move foreign yields
  mostly through term premia).

## Standing caveats for use

1. Present sensitivities, never forecasts. Scenarios carry the uncertainty (agreed:
   no bands to the IC).
2. Japan is directional only: β on the US premium is 0.74 full-sample vs 0.12 on
   post-2010 changes. With YCC ended the forward case is migration back toward the
   higher number — a judgement, not an estimate.
3. Overlapping 36m windows mean the rolling RMSEs carry wide standard errors
   (n = 10–19 per market).

---

# How persistent is the unexplained premium?

Full anchoring — carrying today's residual whole to every horizon — asserts that every
shock to the term premium is permanent. For a fair-value model that is an uncomfortable
claim: a temporary spike would be projected forward for a decade. The opposite, reverting
to the model's own fair value, asserts a mispricing: for the US today, that the ten-year
yield is **71bp rich**. Neither is obviously right, so it was measured.

`validation/tp_scenario_oos.py::residual_persistence_grid`. Real-time annual refits, so
the residual is the one a user would actually have seen. RMSE in pp:

| horizon | market | n | ρ=1.0 | ρ=0.9 | ρ=0.8 | ρ=0.7 | ρ=0.5 | ρ=0.0 |
|---|---|---|---|---|---|---|---|---|
| 3y | US | 19 | 0.93 | 0.84 | 0.77 | 0.73 | 0.69 | **0.67** |
| | UK | 10 | 1.04 | 0.98 | 0.93 | 0.90 | 0.87 | **0.86** |
| | DE | 14 | 1.12 | **1.11** | 1.11 | 1.13 | 1.17 | 1.20 |
| | JP | 14 | 0.75 | 0.71 | 0.69 | **0.68** | 0.69 | 0.71 |
| 10y | US | 12 | 0.94 | 0.76 | **0.73** | 0.73 | 0.73 | 0.73 |
| | UK | 3 | **1.02** | 1.09 | 1.12 | 1.13 | 1.14 | 1.14 |
| | DE | 7 | **1.10** | 1.43 | 1.57 | 1.62 | 1.63 | 1.64 |
| | JP | 7 | **1.11** | 1.17 | 1.22 | 1.24 | 1.25 | 1.25 |
| | *mean* | | **1.001** | 1.009 | 1.019 | 1.021 | 1.021 | 1.023 |
| | *worst cell* | | **1.12** | 1.43 | 1.57 | 1.62 | 1.64 | 1.64 |

(Numbers refreshed after the fiscal rework; the pattern is the one first measured,
with the UK's ten-year cell — n = 3 — flipping to prefer full anchoring.)

## What it says

**The market whose fair value is best identified wants reversion; the ones leaning on
borrowed coefficients mostly want none.** The US improves at both horizons, materially
(3y 0.93 → 0.67). The UK now wants reversion at three years but full anchoring at ten
(on three windows — too few to lean on). Germany and Japan are indifferent at three
years and clearly prefer full anchoring at ten — reverting *to* an unreliable level is
worse than staying at the observed one. Averaged over all eight cells the choice is
within one percent between full anchoring and ρ = 0.9, while the worst cell degrades
from 1.12 to 1.64 as reversion increases.

So the honest verdict is that the evidence does not license a strong default in either
direction. **ρ = 0.9 a year** stays the shipped default: a half-life near six and a
half years, keeping 73% of the residual at three years and 35% at ten. It concedes the
conceptual point — no shock is treated as permanent — without asserting a large
mispricing, costs at most a percent of aggregate RMSE against full anchoring, and it
is exposed as a control on the scenario sheet rather than buried. A user who judges a
particular shock temporary should turn it down.

## A trap worth recording

Estimated **in sample**, the residual looks far more mean-reverting than it is: a
full-sample AR(1) puts the US half-life at four months and a positive residual predicts
the premium falling over the next three years with t = −15.5. That is largely an
artefact — the regression forces the residual orthogonal to the drivers over the whole
span, including the future. The real-time numbers above are three to ten times less
enthusiastic about reversion, and they are the ones to quote.

---

# 2026-08: the projected-balance fiscal channel reaches every market

The US fiscal driver — the five-year-ahead OFFICIAL projected total balance, measured
post-2008 at vintage frequency with the common trend purged (+51.9bp per point of GDP,
t +6.7; see FINDINGS_fiscal_handoff_v2.md) — was extended to the spread markets. Each
market now carries its own official projection as the driver, entering the spread as
the deficit RELATIVE to the US path:

| market | source | horizon shipped |
|---|---|---|
| US | CBO baseline (GitHub mirrors of cbo.gov) | 11 years |
| UK | OBR public finances databank, PSNB % GDP | to 2030-31 (year 4), then flat |
| DE | European Commission AMECO, net lending % GDP (EDP) | 2 years, then flat |
| JP | Cabinet Office medium/long-term projection, baseline case, central+local balance | to FY2040 |

`macro/fiscal_projections.py`, committed CSV caches under `macro/data/`. No spread
market has the vintage archive that identified the US coefficient (OBR's historical
forecasts database and AMECO's twice-yearly vintages are the recorded upgrade path),
so all three borrow the US +0.519 as an adjustment measured from today's relative
reading: worth nothing at the base case, alive when a scenario moves one side's path
against the other's. **Relative debt stock was removed from the spread models
entirely** — UK's fitted +6.2bp (t 6.6), DE's fitted +1.8bp (t 18.2), JP's calibrated
+2.5bp — on the flow-vs-stock evidence: the projected balance is the priced object,
and running both channels would double-count one story.

## What it cost and bought, honestly

In-sample fit gives up what the debt trends were absorbing: UK R² 0.809 → 0.777,
DE 0.845 → 0.795 with the 2010+ sub-window going negative (−0.25), JP unchanged.
Out of sample the trade is asymmetric. **DE and JP improve substantially on the
frozen-2015 decade** (DE 1.44 → 1.02pp, JP 1.69 → 1.48pp; endpoint errors shrink by
roughly half and a fifth)
— their debt channels were exactly the sort of trend-fitted coefficient that
extrapolates badly. **The UK pays**: frozen RMSE 0.58 → 0.67pp and the rolling 3y
ratio 0.95 → 1.17, now behind the random walk. Its relative-debt term evidently
carried real out-of-sample signal, most plausibly as a proxy for the fiscal-trajectory
news the projected-balance channel now carries only from today forward (a backtest
cannot feed the channel, since no historical projection panel exists for the UK yet —
building the OBR vintage panel is the fix that would let the UK re-earn that signal
honestly, and is the strongest single argument for the upgrade path above).

Net: two markets clearly better, one clearly worse, one unchanged, and the model now
says one consistent thing about fiscal policy everywhere instead of three different
things. The residual-persistence grid moved only at the margin (aggregate optimum now
statistically indistinguishable between ρ = 0.9 and 1.0; DE at ten years now prefers
full anchoring more strongly); ρ = 0.9 remains the shipped default on the same
reasoning as before.

## Correction: the fiscal driver reads the trajectory, not one calendar year

Building the exhibits for the CIO briefing exposed a defect in how the fiscal channel was
projected. The driver was taken as `step(path, h+5)` — the single projected year sitting five
years beyond the horizon. Official projections are lumpy at annual frequency for reasons that
have nothing to do with fiscal trajectory (expiring provisions, one-off timing), and reading
one year propagated that idiosyncrasy straight into fair value: CBO's year-8 deficit of 6.63%
against year-10's 6.15% swung the US modelled premium **43bp** depending purely on which
calendar year the horizon happened to land on. The horizon profile was visibly jagged
(0.73 → 1.16 → 0.91 → 1.16 across consecutive horizons) with no economic content.

A ten-year bond bought at horizon `h` prices the fiscal path over its **whole life**, so the
driver now moves with the ten-year forward average of the projected deficit — the same
treatment `expected_inflation` already had. Two properties are preserved deliberately:

- The **level** stays the estimation object (today's five-year-ahead reading, the statistic the
  +0.519 coefficient was measured on). Only the *change* in the trajectory average is applied.
- The reference is the **official baseline's** average, not the scenario's own first year. A
  first attempt referenced the scenario against itself, which silently ignored the user's input:
  entering a flat deficit two points wider than the baseline produced no premium response at all,
  because a flat path has the same forward average at every horizon. Referencing the official
  path restores it — a genuinely wider path registers its full gap at every horizon, while
  running the baseline unchanged still contributes exactly nothing.

The profile is now smooth and monotone (0.68 → 0.77 → … → 1.06 over ten years, tracking the
CBO trajectory's deterioration). Browser-vs-engine agreement tightened from 0.28bp to 0.09bp,
since the single-year read was amplifying payload rounding. Regression tests cover both
properties: a one-year 2pp spike now moves the driver by less than 0.25, and a genuinely wider
path still moves it by the full 2.0.
