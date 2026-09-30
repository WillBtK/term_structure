# Pricing the structural deficit (2026-09)

**Decision.** The term premium prices the **cyclically adjusted** deficit,
`deficit + ε × output gap`, not the headline one (`macro/fiscal_rules.structural_deficit`;
the sheet's `structuralDeficit`). The deficit row itself, and the rule that derives it
from the growth row, are unchanged.

## Why

Every fiscal coefficient in the scenario engine — the US +0.519 and the spread
markets' pooled +0.235 — was estimated on **five-year-ahead official projections**, which
assume the output gap has closed: structural objects. The sheet's endogenous deficit
moves with the output gap, and that cyclical movement was being priced at the
structural rate. The clearest symptom was Hard Landing, US 10y premium at ten years:

| driver | contribution |
|---|---|
| inflation (falls toward target) | −0.30 |
| inflation uncertainty | +0.17 |
| **fiscal (recession deficit, +0.8pp of GDP over the decade from the cycle alone)** | **+0.65** |

A recession RAISED the premium, the opposite of 2008 and 2020, and spilled into
Bunds (+0.36, the largest single driver there). In reverse, growth above the official
baseline shrank the deficit and compressed premia: most of Goldilocks' compression
came from this channel.

## How — automatic, no new data

The deficit row is derived as `official − ε × gap`, the gap cumulating the growth row's
deviations from the deficit source's own growth baseline and closing at 30%/yr. The ε
are each institution's own semi-elasticities (CBO calibration 0.36, OBR 0.70, EC 0.504
recovered from AMECO's decomposition, OECD 0.45) — the arithmetic they use for their
own structural balances. Taking the cycle back out is the same arithmetic reversed,
on one shared gap recursion (`output_gap`), so derivation and adjustment cannot drift:

- **row untouched** → exactly the official path (verified to 1e-15, every market and
  scenario, years 1–45). A growth view alone never moves the priced deficit.
- **row typed** → the statement relative to what the cycle implies. A recession deficit
  copied alongside a recession growth row prices nothing extra; a cut the growth row
  cannot explain prices in full (verified: one point narrower moves the US premium
  −0.519 × view weight, Germany −0.235 × view weight).

The limit every cyclical adjustment shares: growth above the official baseline is read
as a positive output gap, never as higher potential growth. A structural-gain view is
entered on the deficit row.

## Two bugs fixed on the way

Checking the typed-row case exposed a pre-existing fault: **taking a deficit row manual
without changing a number moved the US 10y premium −30bp** (Baseline 1.355 → 1.05 on
the sheet as it stood before today). Two causes, both fixed in the sheet and the engine:

1. **Keying.** Typed rows were keyed by year-end horizon (2027 at 1.35), the official
   and derived paths by calendar-year index (2027 at 1), so a typed row was read a year
   late. Typed deficits are now keyed by calendar year.
2. **Truncation.** A typed row covers the six grid years and was then held flat,
   throwing away the official out-years (CBO rises to 6.6% by year 11). Past its last
   year a typed row now follows the derived path, holding the typed DEVIATION
   (`extend_stated_deficit`). A copy of the displayed row now prices identically.

The rule's derived path is also now produced to year 45 rather than frozen at 15, so
the identity holds wherever a bond window reaches (the OBR's growth baseline runs fifty
years and kept moving after 15).

## What it does to the scenarios

Sheet as committed, 10y premium at 3 and 10 years and the 10y yield at 10 years, before
→ after this change (the spread-market fiscal coefficient already at +0.235 in both).

| scenario | mkt | TP 3y | TP 10y | 10y yield at 10y |
|---|---|---|---|---|
| Baseline | US | 1.46 → 1.55 | 1.35 → 1.54 | 4.97 → 5.15 |
| Baseline | UK | 1.28 → 1.24 | 1.14 → 1.02 | 4.18 → 4.07 |
| Baseline | DE | 0.72 → 0.86 | 0.46 → 0.77 | 2.26 → 2.56 |
| Baseline | JP | 1.03 → 1.11 | 0.70 → 0.87 | 2.44 → 2.61 |
| Goldilocks | US | 0.70 → 1.35 | 0.37 → 1.12 | 4.37 → 5.12 |
| Goldilocks | UK | 0.71 → 1.26 | 0.33 → 0.91 | 4.08 → 4.66 |
| Goldilocks | DE | 0.14 → 0.85 | -0.04 → 0.83 | 2.46 → 3.33 |
| Goldilocks | JP | 1.00 → 1.30 | 0.75 → 1.12 | 2.35 → 2.72 |
| High Growth High Inflation | US | 1.70 → 2.40 | 1.51 → 2.33 | 6.51 → 7.33 |
| High Growth High Inflation | UK | 1.22 → 1.79 | 0.97 → 1.58 | 5.22 → 5.83 |
| High Growth High Inflation | DE | 1.02 → 1.60 | 0.73 → 1.45 | 3.73 → 4.45 |
| High Growth High Inflation | JP | 1.13 → 1.50 | 0.76 → 1.22 | 2.76 → 3.21 |
| Stagflation | US | 4.23 → 4.08 | 4.91 → 4.75 | 10.35 → 10.18 |
| Stagflation | UK | 2.86 → 2.48 | 3.20 → 2.60 | 7.92 → 7.32 |
| Stagflation | DE | 2.86 → 2.79 | 3.31 → 3.29 | 7.27 → 7.25 |
| Stagflation | JP | 0.62 → 0.69 | 0.84 → 0.93 | 2.84 → 2.93 |
| Hard Landing | US | 1.85 → 1.52 | 1.53 → 1.20 | 2.53 → 2.20 |
| Hard Landing | UK | 2.23 → 1.50 | 1.87 → 1.02 | 1.87 → 1.02 |
| Hard Landing | DE | 1.33 → 1.06 | 1.09 → 0.87 | 1.09 → 0.87 |
| Hard Landing | JP | 1.21 → 1.22 | 1.02 → 1.08 | 1.03 → 1.08 |

Read across:

- **Hard Landing** premia now sit at or near Baseline — US 1.20 against 1.54, UK 1.02 against 1.02, Germany 0.87 against 0.77. The residual German gap is inflation
  uncertainty and the US channel, not fiscal.
- **Goldilocks** loses most of its compression (US 0.37 → 1.12): it came almost entirely
  from cyclical deficit shrinkage. It still has the inflation and uncertainty channels.
- **Baseline** rises for the US (+0.19) and Germany (+0.31) because their growth rows
  (US 2.1% vs CBO ~1.8%; Germany 1.25% vs the official 0.65%) had been booked as lasting
  cyclical improvement. The UK falls a little: its growth row runs below the OBR's.
- **Stagflation** barely moves.
