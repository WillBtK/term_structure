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
yield-fit MAD 3.73 bp, λ=0.512), with **Bauer-Rudebusch-Wu bias-correction** of the
P-persistence (ρ 0.986 → 0.9998, δ=0.89; smallest eig(K^P) 0.033 → 0.0018 — pushed
to near-unit-root, capped at the stationarity boundary). Term premium uses the
expected-average *floored* short rate under P
(`term_premium.expected_floored_short_rate_yields`), so it respects the ELB in the
2008-15 / 2020-21 ZLB and reduces to plain AFNS elsewhere.

### vs SF Fed — bias-correction is decisive

| maturity | uncorrected corr / bias | **bias-corrected corr / bias** |
|---|---|---|
| 1y  | 0.17 / −5.0 bp | **0.84 / −5.2 bp** |
| 2y  | 0.32 / +1.2 bp | **0.82 / −2.0 bp** |
| 3y  | 0.43 / +10.4 bp | **0.80 / +3.5 bp** |
| 5y  | 0.59 / +28.5 bp | **0.75 / +14.5 bp** |
| 7y  | 0.67 / +42.5 bp | **0.73 / +22.0 bp** |
| 10y | 0.74 / +55.9 bp | **0.72 / +27.1 bp** |

Bias-correction lifts short-to-medium correlation with SF Fed sharply (1y
0.17→0.84, 2y 0.32→0.82, 3y 0.43→0.80) and **roughly halves the level bias at every
maturity** (10y +55.9→+27.1 bp). The 10y correlation is essentially flat
(0.74→0.72); its residual +27 bp level gap is the known weak-identification of the
premium *level* — the corrected series is the production choice (corr 0.72-0.84,
bias 2-27 bp across the grid).

### Headline levels (bias-corrected)

| | 10y TP | 2y TP |
|---|---|---|
| latest (2026-07) | **+0.75%** | +0.10% |
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
