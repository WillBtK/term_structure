# UK (gilt) term premium — findings

Fourth country. Same engine as US/Germany/Japan (`/model` unforked); only the BoE
data layer (`uk/`) and a mild fixed ELB differ. The UK leg also carries the real
(index-linked) curve and the inflation-risk premium — see
`validation/FINDINGS_inflation.md`.

## Data
Bank of England Government Liability Curve (GLC), published as **spot rates
directly** (VRP spline; no reconstruction). Nominal month-end **1970→2026** (679
obs), grid 0.5-20y — 20y ragged early, extending to 40y from 2016. Yield-fit MAD
**4.8 bp**.

## Effective lower bound
UK is the **mildest ZLB** of the four: Bank Rate floored at 0.10% (2020-21) and
short gilt yields barely went negative, so the ELB is weakly identified. We **fix**
it at −0.35% (like the US) rather than ML-estimate; the shadow-rate machinery
barely binds.

## Nominal production TP — inflation-targeting era (1992-10+)

`uk/production_tp.py::run_uk_nominal`, 406 monthly obs, converged (loglik 16840,
BRW auto→full). The modern sample starts at UK **inflation targeting** (post-ERM,
Oct 1992) — the regime-clean analog of US-1987 / Germany-1999; the pre-1992
stagflation era is a separate historical exhibit.

**Headline levels (bias-corrected, realised):**

| | 10y TP | 2y TP | 20y TP |
|---|---|---|---|
| latest (2026-07) | **+1.49%** | +0.43% | +2.04% |

The UK 10y term premium (~1.5%) is the **highest of the four markets** (US ~0.67%
[1987+] / ~0.98% [2004+], Germany ~0.13%, Japan ~0.90%) — consistent with elevated
gilt supply/fiscal risk premia and the 2022 LDI episode.

**History (10y):** +1.71% (2008-12 crisis) → +0.29% (2016 Brexit-era low) →
**−0.68% (Sep 2022 mini-budget / LDI gilt rout**, when long yields spiked above the
expected-rate path in the forced-selling dislocation) → +1.49% now. The brief
negative reading in Sep-2022 is the realised-premium signature of the LDI crisis,
recovering as the Bank intervened.

## Cross-country nominal 10y term premium (production, latest)

| | 10y TP | sample | ELB |
|---|---|---|---|
| Germany | +0.13% | 1999+ | −0.80% (ML) |
| US | +0.67% | 1987+ | −0.35% |
| Japan | +0.90% | 1986+ | −0.17% (ML) |
| UK | +1.49% | 1992+ | −0.35% |

Ordering DE < US < JP < UK is economically coherent (Bund safe-haven lowest;
gilt/fiscal premia highest).

## Validation
No UK SF-Fed-equivalent TP benchmark; validation rests on BoE data coverage, the
4.8 bp yield fit, sensible cross-country levels, and the breakeven curve tracking
known inflation episodes (`FINDINGS_inflation.md`). External directional check vs
the Krippner UK shadow rate where reachable (`economy="UK"`).
