# Regime dating across markets — Germany, the UK and Canada

Every market in the atlas is fit on a **regime-clean modern sample**, chosen so no
structural break sits inside the estimation window. Until now only two of those
choices had been tested with a fitted model: the US (a 2-regime Kim filter on 1961+
confirms the high-volatility regime is confined to 1973–75 and 1979–82) and Japan
(where the same filter *moved* the start, from 1986 to 2000-05). The other four were
dated on monetary-policy grounds — monetary union, the adoption of inflation
targeting — which is the same logic applied by judgement rather than by likelihood.

This note is that test for Germany, the UK and Canada. Two of the three moved.

Method, identical to the Japan run: a 2-regime Markov-switching AFNS (Kim 1994
filter; per-regime `theta`/`K^P`/`Sigma`, common `lambda` and measurement error),
estimated by QML on the **full published monthly history** — not the production
window — then smoothed to say which regime each month was in. Warm-started from each
market's converged production fit, but with the two regimes' `theta` started at
*different* levels so the search can find a level break rather than settling into a
pure volatility split. Dating is on the **level** regime (higher long-run anchor
`theta_L`), because `theta` is what anchors the expected-rate path and is therefore
what a mis-cut sample corrupts. Runner: `scratchpad/run_regime_dating.py`.

## Summary

| | Sample | Break dated | Regime switches | Verdict |
|---|---|---|---|---|
| **Germany** | 1972-09 → 2026-07 (647 m) | **1983-11** | **1** | clean, but the old 1999 start was **15 years conservative** |
| **UK** | 1970-01 → 2026-07 (679 m) | **1995-11** | 18 (3 sustained spells) | old 1992-10 start **straddled the break by 37 months** |
| **Canada** | 1986-01 → 2026-07 (487 m) | — | 12 | **not identified** — untestable with the available history |

## Germany — the break is 1983, not 1999

| | 1972-09 … 1983-10 | 1983-11 … 2026-07 |
|---|---|---|
| long-run anchor `theta_L` | **11.94 %** | **2.52 %** |
| `sigma` (level / slope / curvature) | 67 / 978 / 1567 bp | 31 / 85 / 195 bp |
| `p_stay` | 0.9984 | 0.9990 |

Smoothed P(high-level regime) is 1.000 flat through 1982, 0.794 in 1983, and **0.000
in every year from 1984 to 2026** — one switch in 647 months, with no reversion
through reunification, the 1992 ERM crisis, the euro changeover, the 2012 sovereign
crisis or the negative-rate era. It is a level *and* volatility break: the slope and
curvature factors were an order of magnitude more volatile before it.

So the euro changeover is **invisible to the rate regime**. The old 1999 start was
regime-clean — zero months of the high regime fell inside it, unlike Japan — but it
discarded **180 monthly observations** (1984-01 … 1998-12) belonging to the same
regime as today. That is the one variance reduction a yields-only model can buy
without incurring bias, and we were declining it: more same-regime data is exactly
what Li, Meldrum & Rodriguez (FEDS Notes, 2017) show a survey-free model needs to pin
down persistence.

Term premium by era (regime-mixed, 10y): **+0.66 %** across 1972–83, **+2.27 %**
across the discarded 1984–98 window, **+1.01 %** on the old 1999+ sample. The
discarded window is the *high*-premium one, so extending the sample lifts the sample
mean anchor and is expected to **lower** the current estimated premium. This is a
substantive change, not a cosmetic one.

**Production sample moved to 1984-01** — re-estimated on 511 monthly obs. Realised
effect, against the directional prediction above:

| | old (1999-01) | new (1984-01) |
|---|---|---|
| long-run **short-rate** anchor (`theta_L + theta_S`) | 1.39 % | **2.84 %** |
| ML lower bound | −0.776 % | **−0.742 %** |
| yield-fit MAD | 8.97 bp | **10.65 bp** |
| **latest 10y term premium** | **+1.39 %** | **+0.58 %** |

The premium falls **81 bp**, in the predicted direction and for the predicted reason.
The ML lower bound barely moves, which is a stability check rather than a null: the
bound is identified off the negative-rate era, present in both windows, so adding
fifteen years at the *high*-rate end should not disturb it — and does not. The cost is
a worse in-sample fit (8.97 → 10.65 bp): one parameter set now spans 8 % Bund yields in
1984 and −0.8 % in 2020. That is the trade — lower variance on the weakly-identified
persistence, paid for in measurement error — and the published premium is the
*realised* one (observed yield minus expected path), so the fitted yield does not enter
it directly.

**A caveat this exposes, which the regime filter cannot address.** The 1984+ sample
implies a long-run short rate of **2.84 %**, above most estimates of euro-area neutral
(nearer 2 %). Regime-cleanliness and a correct anchor are different properties: r\* has
drifted down *secularly within* the post-1984 regime (Del Negro et al.), and a 42-year
sample mean can be regime-clean and still too high for today's structural conditions.
The principled fix already exists in this codebase — the **shifting-endpoint neutral
anchor** built for Japan (Kozicki-Tinsley / Kim-Orphanides), which re-anchors the
expected-rate path at decomposition time without re-estimating. Germany is now the
market that most needs it. Not applied here; it needs its own survey / r\* input series
and is the top follow-up.

## UK — the curve re-anchored three years after the policy announcement

| | high-level regime | current regime |
|---|---|---|
| long-run anchor `theta_L` | **14.21 %** | **6.85 %** |
| `sigma` (level / slope / curvature) | 117 / 289 / 387 bp | 54 / 72 / 223 bp |
| `p_stay` | 0.9732 | 0.9728 |

Sustained spells in the high-level regime: 1972-06…1973-12 (19 m), 1977-05…1982-03
(59 m), and **1982-07…1995-10 (160 m)**. Two brief excursions follow — 1999-07…2000-01
(7 m) and 2008-10…2009-06 (9 m) — both stress episodes, not regime changes.

The current regime therefore begins **1995-11**, three years after the old 1992-10
start, so **37 of 406 months (9 % of the sample)** sat inside the old high-anchor
regime. That is the Japan problem in miniature and in the same direction: those months
pull the sample-mean `theta` up, which lowers the expected-rate path and inflates the
residual premium. The overlap window ran at a 10y premium of **+1.38 %** against
**+0.46 %** for 1996+, and **+2.60 %** for the 1970–95 regime.

This reads as economically coherent rather than as a quirk of the filter. October 1992
is when the UK *announced* inflation targeting, days after the ERM exit — but
announcement is not re-anchoring, and the gilt curve took roughly three years to price
the new framework as credible. BoE operational independence followed in 1997-05.

**Production sample moved to 1995-11**, for the real (index-linked) curve as well as
the nominal one, so the two legs stay on a common sample and their difference — the
inflation-risk premium — is not a sample artefact.

## Canada — not identified, and not fixable with this history

The fit does not find a monetary-policy regime, and the "break" it nominally produces
should not be reported. Four independent tells, all pointing the same way:

* **It flips 12 times in 487 months** (Germany switches once in 647). Sustained spells
  land in 1988-90, 1992-94, 2000-05 and 2008-10 — a recurring cyclical state, not a
  regime.
* **`theta` is unidentified.** The fitted anchors are **31.6 %** and **14.2 %**, which
  are not plausible Canadian policy-rate levels. The mechanism is visible in the
  parameters: `kappa` for the level factor comes back at **0.0000** in one regime, so
  the level never mean-reverts, `theta` never enters the likelihood, and the optimiser
  is free to put it anywhere.
* **What it split on is volatility** (level `sigma` 83.7 vs 40.7 bp), so labelling a
  regime by "higher `theta_L`" is meaningless here.
* `p_stay` 0.962 / 0.989, well below Germany's 0.999 / 0.998.

**Confirmed by multi-start.** Four restarts on identical data land somewhere different
from the single-start fit — anchors **39.0 % / 10.6 %** against **31.6 % / 14.2 %**, and
a "break" at **2012-07** against **2011-01**, eighteen months apart. `kappa_L` is
0.0000 / 0.0012 in both, so `theta_L_identified` is **false for both regimes** either
way. The restarts also reach a *lower* likelihood (22 989.6 vs 23 203.3), which says the
same thing from another angle: restarts only fire when convergence fails, each perturbs
from the last, and the search wanders because the surface is flat. This is
non-identification, not a near-miss local optimum.

The reason is the sample. The BoC curve begins in **1986** and inflation targeting was
adopted in **February 1991**, leaving ~60 monthly observations before the event under
test — against 11 years of the old regime for Germany and 26 for Japan. Five years
cannot identify a permanent level shift, so the likelihood settles on cyclical
variation instead. The regime-conditional premia the run printed (−10.6 %, −4.2 %) are
artefacts of the 31 % anchor and carry no information.

**Canada's 1991 start is unchanged**, and rests on the policy record. The honest
verdict is *cannot be tested with the data we have* — neither confirmed nor refuted.

## Downstream effects

**The global cycle is essentially unchanged.** Refitting the 1-factor DFM on the new
term premia gives correlation **0.99999** against the old factor on every
term-premium combination (latest G at 10y −0.25 → −0.26), with each market's loading
and global share moving by at most 0.1 pp — Germany holds at 92.2 %. That is a result,
not an absence of one: the DFM standardises each market's series, and Germany's change
is close to a constant offset, which is exactly what a common factor should absorb.
PC1 still explains 92.7 % of cross-market variance at 10y.

**The German 1972+ historical splice stays continuous.** The level-match offsets
recompute at build time; measured at the 1997-08 daily-data boundary the spliced line
moves **−2.7 bp**, i.e. ordinary daily variation rather than a seam.

**A pipeline trap found on the way.** `viz.global_factor` is a separate step, and
`thin_daily` only *reads* whatever factor parquet is on disk. Running
`export_daily → thin_daily` directly produced a page with new country series beside a
cycle fitted on the superseded ones — and it looked convincing: correlation 1.00000,
identical `phi`, identical loadings, which reads as "the cycle is robust to the
re-cut". It was visible only because the DFM's *inputs* were diffed and came back
bit-identical to twelve decimals, which no genuine refit produces. `thin_daily` now
compares the factor's mtime against the canonical dataset's and warns loudly when the
factor is older.

## Two method fixes this exercise forced

**Dating must use sustained spells, not the last month above 50 %.** The original rule
took the final crossing, which counted the UK's 9-month 2008-09 stress excursion as a
regime re-entry and dated the break at **2009-07** — the opposite conclusion, and
wrong. A spell must now run ≥12 months to count as a regime, which leaves Germany and
Canada unchanged and moves the UK to 1995-11.

**Regime-conditional premia must be averaged over the dates that regime prevailed.**
`regime_term_premium` returns the decomposition *as if* the economy were in each regime
on every date. Averaging the high-level regime across the whole sample prices 2020
yields against a 12 % anchor and returned **−1.19 %** for Germany — a number describing
nothing. Over its own dates it is **+0.62 %** (low regime **+1.47 %**).

## Caveats

`success=False` from the optimiser in all three runs. The regime likelihood is flat and
multi-modal — the same caveat recorded for the Japan and US runs — so the evidence is
the smoothed *pattern*, not the optimiser's convergence flag. Germany's is about as
unambiguous as this method produces (1.000/0.000 with a single transition year); the
UK's is noisier before 1977 and rests on the three sustained spells; Canada's fails on
its own diagnostics, which is the finding.

The model sees level and volatility only. "Germany's break is 1984" is a claim about
the *rate* regime, not a claim that monetary union was irrelevant to the risk premium —
that is precisely what re-estimating on the longer sample tests.
