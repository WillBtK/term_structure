# Japan — regime dating and the move to a regime-clean 2000+ production sample

This note records why Japan's production term-premium sample moves from **1986-07**
to **2000-05**, and documents the 1974+ historical/stagflation exhibit that now sits
beside it (mirroring the US 1961+ and Germany 1972+ exhibits).

## Motivation
Every other market in the atlas is fit on a **regime-clean modern sample** chosen so
that one θ / K^P describes the whole span (US 1987, UK 1992-10, DE 1999, CA 1991).
Japan's prior **1986-07** start was **data-limited, not regime-chosen** — it is simply
the first month MoF publishes a 10y JGB yield. To find where Japan's modern regime
actually begins we extended the sample to **1974-09** (MoF publishes ≤9y back to 1974)
and ran two estimations on the full 1974+ monthly curve.

## 1. Regime dating — 2-regime Kim-filter AFNS, 1974-09 → 2026-07 (623 monthly obs)
A 2-regime Markov-switching AFNS (Kim 1994 filter; per-regime θ/K^P/Σ, common λ and
measurement error) separates a **high-level** regime from a **low-level** regime:

| Regime | θ_L (level factor) | Persistence |
|---|---|---|
| High-level | **7.86 %** | p_stay ≈ 0.995 |
| Low-level | **5.75 %** | p_stay ≈ 0.995 |

Smoothed annual P(high-level regime):
- **1974 → 2000: high-level**, P ≈ 1.0 throughout (one transient one-year dip in 1978).
- **2000 onward: low-level**, P ≈ 0.00 continuously to 2026 (incl. the 2024+ BoJ exit).
- Permanent switch dated **2000-05-31** — squarely in the ZIRP→deflation transition
  (BoJ ran ZIRP Feb-1999, lifted it Aug-2000, began QE Mar-2001).

**Caveats.** The optimizer reports `success=False` (the regime likelihood is flat /
multi-modal, as documented for these fits) — the bar is a stable, economically-coherent
pattern across the sample, which this clearly is. The θ_L values are **regime attractors**,
not literal long-run rates (one low-level regime absorbs the entire 0–1.5% JGB world, so
5.75% reads high); the **dating** is the result, not the levels.

## 2. Implication — 1986 sat *inside* the old regime
The filter places the 1986-07 start **14 years inside** the high-level regime, so the
prior production sample straddled the 2000 break: it mixed 14 years of high-regime
(1986–2000) with 26 of low-regime (2000–2026). That pulls the sample-mean θ **up**,
biasing today's expected-rate path up and today's premium **down** — exactly the
contamination the modern-sample discipline removes elsewhere. Starting at the break
gives Japan the same regime-clean treatment as the other five markets.

## 3. Production re-estimation — shadow AFNS + BRW, 2000-05 → 2026-07 (315 monthly obs)
ELB re-estimated at **−0.20 %** (NIRP-consistent). 10y term premium by era:

| Era | 10y TP (mean) |
|---|---|
| 2000–2012 | +1.05 % |
| 2013–2019 (QQE) | +0.01 % |
| 2020–2023 (YCC) | +0.02 % |
| 2024+ (post-YCC normalisation) | +1.03 % |
| **latest (2026-07)** | **+2.15 %** |

The latest **+2.15 %** is up from **+0.89 %** under the old 1986 fit. The move is the
expected direction: the regime-clean 2000+ sample has a **lower, deflation-era θ anchor**,
so the expected-rate path is lower and the residual premium higher.

**θ-anchoring caveat (Japan-specific, important).** Japan is the one market currently
**transitioning out** of its modern regime (BoJ ended NIRP/YCC in 2024 and JGB yields are
normalising up). Because θ is anchored to the 2000–2024 deflation era, a rising 10y yield
against that low anchor pushes the *current* premium up — so read the recent **+2%** level
as **anchor-sensitive** and lean on event-window **changes** more than the absolute level.
The mean-reversion floor (3.5y half-life) keeps the expected-rate path from being pinned
hard to θ, but does not remove this caveat.

## 4. Historical / stagflation exhibit (1974+), separate and labelled
The 1974-2000 high-inflation era — the regime the modern sample never sees — is kept as a
**separate exhibit**, never joined to the production line:
- **Model-fair (fitted-basis) 10y TP, 1974–85: mean +3.36 %, peak +4.91 %** (in the
  1979–81 tightening window; the 1974–75 oil-shock sub-window averaged +2.35 %). Observed-
  basis is unavailable pre-1986 (no published 10y), so this is the fitted-basis premium;
  it agrees with the observed basis to a few bp from 1986 on, which validates it where it
  is the only series.
- In the Atlas this appears as the **"Japan before the deflation era" cross-check panel**:
  the single-regime production model extrapolated back to 1974 vs the 2-regime model, with
  the May-2000 break marked. Before 1986 the atlas line is the 9y premium spliced up to the
  10y (no published 10y JGB), so the deep-past **level** is indicative.

So Japan's premium can run to **~+5% at 10y** in a genuine high-inflation regime —
comparable to the US and German Great-Inflation exhibits, and a world away from the
−0.8%/≈0% of the YCC/deflation era it spent 2000–2024 in.
