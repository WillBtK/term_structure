# Mean-reversion floor — fixing the over-tracking of the policy cycle

**This supersedes the persistence treatment (Bauer-Rudebusch-Wu) in the earlier
findings.** The production term-premium numbers in `FINDINGS_phase5.md`,
`FINDINGS_germany.md`, `FINDINGS_japan.md`, `FINDINGS_uk.md` and
`FINDINGS_inflation.md` predate this fix; the corrected numbers are here.

## The problem (found by inspection vs the SF Fed Christensen-Rudebusch series)

Our term premium was, to first order, **just the term spread**. For the US:
- `corr(expected 10y-avg short rate, actual 3m short rate) = 0.99` — the expected
  path moved essentially one-for-one with the current policy rate.
- `corr(our 10y term premium, the 10y-3m spread) = 0.95`.

So in the 2022-23 hiking cycle, with the policy rate at ~5%, the model projected
that forward and the 10y term premium **collapsed to −1.2%** (US) — while the SF
Fed C&R series stayed ~+1% and reads **1.19% now**.

**Root cause.** Yields-only ML lands near a unit root (the level factor's
half-life is effectively centuries), so `E[avg short over 10y] ≈ today's short
rate`. And the **BRW bias-correction made it worse** — it deliberately *raises*
persistence, pushing ρ to the stationarity boundary (0.9999). The expectations
leg then can't mean-revert to θ, so the premium degenerates into the spread.

## The fix

`model.term_premium.mean_reversion_floor(K, kappa_min)` floors the ML K^P so every
factor reverts at least as fast as a calibrated half-life, restoring anchored
expectations (E[avg short] reverts to the long-run mean θ instead of tracking the
policy rate). It **replaces BRW** for the expectations leg (`run_production_tp
(persistence="mrfloor")`, the default; BRW retained as an option). It is a
post-estimation *decomposition* change — the shadow-rate AFNS fit, the ELB, and
the shadow short rate are untouched, and no re-estimation was needed.

**Calibration (US, vs the current SF Fed C&R series to 2026): half-life = 3.5y**
(`kappa_min = ln2/3.5 ≈ 0.198`), applied uniformly to every market.

| US nominal TP vs SF Fed C&R | corr (BRW → mrfloor) | bias | latest ours / SF Fed |
|---|---|---|---|
| 2y | 0.36 → **0.43** | −0.16 | 0.24 / 0.20 |
| 5y | 0.58 → **0.73** | −0.11 | 0.63 / 0.70 |
| 10y | 0.66 → **0.80** | +0.27 → **+0.04** | **1.19 / 1.19** |

The over-tracking correlation drops and the 2022-23 collapse disappears
(US 2023: −1.2% → +0.4%). The near-unit-root version had "passed" validation only
because the cached SF-Fed window then predated the 2022-23 cycle.

## Corrected production term premia (mean-reversion floor, realised, %)

**Latest, nominal:**

| | 2y | 5y | 10y | 20y | 30y | sample | ELB | yield-fit |
|---|---|---|---|---|---|---|---|---|
| US | +0.24 | +0.63 | **+1.19** | — | — | 1987+ | −0.35 | 3.7 bp |
| Germany | +0.49 | +0.77 | **+1.29** | — | — | 1999+ | −0.80 (ML) | 3.5 bp |
| Japan | +0.33 | +0.88 | **+1.65** | +2.57 | +2.88 | 1986+ | −0.17 (ML) | 4.8 bp |
| UK | +0.30 | +0.76 | **+1.50** | +2.38 | — | 1992+ | −0.35 | 4.8 bp |
| Switzerland | −0.31 | −0.41 | **−0.37** | −0.37 | −0.50 | 1988+ | −0.75 | 5.1 bp |

**Real (no ELB, plain AFNS):** US TIPS 10y +1.26% (20y +2.23%); UK index-linked
10y +1.24% (20y +2.00%).

**10y history (nominal):**

| | 2000 | 2008 | 2012 | 2020-03 | 2023-10 | latest |
|---|---|---|---|---|---|---|
| US | +1.94 | +1.15 | −0.49 | −1.43 | +0.38 | +0.84¹ |
| UK | +0.68 | +1.63 | −0.33 | −1.79 | +0.28 | +1.27¹ |
| Germany | +2.61 | +1.84 | +0.42 | −1.38 | +0.39 | +1.03¹ |
| Japan | +0.99 | +0.47 | −0.08 | −0.44 | +0.12 | +1.58¹ |
| Switzerland | +1.79 | +1.01 | −0.43 | −0.44 | −0.31 | −0.37 |

¹ month-end 2026-07; the "latest" column in the table above is the final datum.
Remaining negatives (2012 ZLB, 2020-03 COVID) are genuine flight-to-safety /
normalisation episodes — the same sign as the SF Fed benchmark — not the
hiking-cycle over-tracking, which is fixed.

## Cross-country 10y term premium today

| Switzerland | US | Germany | UK | Japan |
|---|---|---|---|---|
| −0.4% | +1.2% | +1.3% | +1.5% | +1.7% |

Switzerland lowest (deepest safe-haven). **Japan now reads highest (1.65%)** — its
expected short rate anchors to a very low historical sample-mean θ (~0.7%) while
JGB yields have risen with BoJ normalisation, so the premium is large. This flags
a **θ-anchoring refinement** worth revisiting for Japan: the sample-mean θ may be
too low now that the regime has shifted (the market anchors to a higher neutral
than the ZLB-dominated history implies).

## What this did NOT change
- The shadow-rate AFNS estimation, the ELB, and the shadow short rate — all intact.
- The regime **detection** (Kim-filter dating) — persistence-agnostic, unaffected.
- The regime-**conditional** term premium *does* now use the same floor
  (`run_regimes(mr_halflife=3.5)`), so the stagflation-regime premium is anchored
  rather than the spurious negative a near-unit-root per-regime K^P produced;
  numbers appended when the (dating-unchanged) re-run completes.
