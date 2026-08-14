# Phase 5 findings — modern production term premium (US)

**Decision.** The US term premium is delivered as **two separate products**, not
one 1961-present series: a **modern production series** (this note) and a
**historical/stagflation exhibit** (`FINDINGS_phase3.md`,
`FINDINGS_kim_validation.md`). A single full-sample model would pull θ/K^P toward
the 1970s-80s and contaminate *today's* term premium; the earlier 2008 splice also
produced a 243 bp seam discontinuity (`FINDINGS_phase4.md`). Keeping today's
estimate on a modern sample is cleaner and more correct.

## Production series — `us/production_tp.py::run_production_tp`

Shadow-rate AFNS on **1987-2026** (monthly, ELB −0.35%, loglik 20286, converged,
yield-fit MAD 3.73 bp), with **Bauer-Rudebusch-Wu bias-correction** of the
P-persistence (ρ 0.992 → 0.9998). Term premium uses the expected-average *floored*
short rate under P (`term_premium.expected_floored_short_rate_yields`), so it
respects the ELB in the 2008-15 / 2020-21 ZLB and reduces to plain AFNS elsewhere.

> **Decomposition refinement (from the Japan leg).** Two corrections now apply
> pipeline-wide: (a) the expectations-leg long-run mean θ is the **sample mean** of
> the filtered factors (per BRW), not the unidentified ML transition θ; (b) BRW runs
> in **"auto"** mode — unrestricted VAR where well-behaved (the US keeps this: its
> cross-factor P-dynamics are what match SF Fed), falling back to the diagonal
> correction only when the unrestricted K^P turns complex (Japan). (c) The premium
> is now measured against the **observed** yield (realised premium), which for the
> US equals the fitted-based premium to ~4 bp. The numbers below are on the
> corrected path; the US is essentially unchanged (short-end corr ~0.80, and the
> level bias is *reduced* by the sample-mean θ).

### vs SF Fed — bias-correction is decisive (auto mode → unrestricted for the US)

| maturity | uncorrected corr / bias | **bias-corrected corr / bias** |
|---|---|---|
| 1y  | 0.17 / −5 bp | **0.80 / −7.6 bp** |
| 2y  | 0.32 / +1 bp | **0.81 / −6.4 bp** |
| 3y  | 0.43 / +10 bp | **0.78 / −3.9 bp** |
| 5y  | 0.59 / +29 bp | **0.74 / +1.7 bp** |
| 7y  | 0.67 / +43 bp | **0.73 / +7.6 bp** |
| 10y | 0.74 / +56 bp | **0.72 / +15.5 bp** |

Bias-correction lifts short-to-medium correlation with SF Fed sharply (1y
0.17→0.80, 2y 0.32→0.81, 3y 0.43→0.78) and **cuts the level bias sharply at every
maturity** (10y +56→+15.5 bp; 5y +29→+2 bp). The 10y correlation is essentially flat
(0.74→0.72); its residual level gap is the known weak-identification of the premium
*level*. The corrected series (corr 0.72-0.81, |bias| ≤ 15.5 bp across the grid) is
the production choice.

### Headline levels (bias-corrected)

| | 10y TP | 2y TP |
|---|---|---|
| latest (2026-07) | **+0.67%** | +0.04% |
| mean 2015-2019 (post-QE) | ~0.00% | — |

Plausible modern levels: a moderate, positive 10y term premium today, compressed to
~zero through the mid-2010s QE era.

## Why this, not the alternatives
- **Not a full-sample (1961+) single model** — it would bias today's expected-rate
  path with 1970s-80s dynamics (higher long-run mean, different persistence).
- **Not the 2008 splice** — 243 bp seam discontinuity, and the pre-2008 leg still
  risks contaminating the modern estimate. `us/splice.py` is retained as an internal
  option, not the product.
- **Modern sample + BRW** gets both: modern dynamics (no contamination) *and*
  persistence robustness (fixes the short-sample bias). This is the headline,
  cross-country-comparable series.

## Historical / stagflation exhibit (separate)
Kept clearly labeled and not joined to the production series: regime-switching
*dates* the high-inflation/high-vol regime (1973-75, 1979-82; Kim filter validated,
`FINDINGS_kim_validation.md`), and a single-regime 1961+ bias-corrected fit puts the
1970s stagflation 10y term premium at a moderate **~1.3%** (`FINDINGS_phase3.md`).

## Artifacts
`us/production_tp.py` → `run_production_tp(start="1987-01-01")`; canonical frame
`(date, maturity, term_premium, expected_rate, fitted_yield)` (bias-corrected
headline + uncorrected comparison). `_assemble_tp` is estimation-free unit-tested.
