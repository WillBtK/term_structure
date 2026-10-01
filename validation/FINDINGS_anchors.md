# Survey-anchored shifting endpoints — the one-rule design (2026-08-20)

## The rule

**The expected-rate path's far end reverts to a surveyed neutral wherever a
long-horizon rate survey exists, and to the estimated sample mean everywhere else.**
No model r\* appears in any published construction. Anchors apply at decomposition
time (`endpoint=` in `model/term_premium`): the fitted curve and the whole Q-side are
untouched, so no re-estimation is involved and turning an anchor off is one flag.

Two reviews produced this rule:

1. **The Ferreira review.** The original ladder back-cast pre-survey history with a
   `r* + pi^e` bridge. Ferreira's euro-area r\* reads **−0.87% in mid-2008**, implying
   a ~0.7% German nominal neutral while the ECB hiked to 4.25% — worth +1.16pp of 2008
   term premium on its own, from a model path no market participant priced. Verdict:
   surveys only; pre-survey history keeps the yields-only decomposition.
2. **The trader test** (project owner): no current r\* estimate is within a big figure
   of anything traded. Survey objects that professionals actually answer — Consensus
   6-10y, the BoE MPS neutral question, the BoC MPS neutral question — are the anchor;
   academic r\* estimates are at most a displayed cross-check.

## Constructions (all via `model.neutral_anchor.build_survey_neutral`)

| Market | Survey era (frozen Consensus snapshot) | Living carry-forward | After staleness |
|---|---|---|---|
| DE | EA 3m 6-10y, **2016-10+** | ECB SPF longer-term rate (2025+, hand-off −0.01pp) | follows SPF |
| UK | UK 3m 6-10y, **2017-01+** | BoE MPS neutral / 5y Bank Rate median (2022+, monthly xlsx) | follows MPS |
| CA | CA 3m T-bill 6-10y, **2017-01+** | BoC MPS long-term nominal neutral median (2023+, quarterly) | follows MPS |
| JP | JP 3m 6-10y, **2017-01+** | none public (ESP is membership-gated) | **glides to the BoJ-communicated corridor 1.75–2.25%, midpoint 2.0%** across the staleness window (no step) |
| CH | CH 3m 6-10y, **2017-01+** | **KOF pi_5y (public API, no account) + constant real (-0.14pp, 114-month overlap)** | follows KOF inflation; real component held constant |
| US | SPF 2-10y forward: nominal `(10*BILL10 - y1 - y2)/8`, real net of the same-window ATSIX forward — **1992+/2004+, living** | itself (BILL10 Q1-only; SME-bridged, 8 rounds/yr) | — |

Endpoints are RW-smoothed (no release-day staircase, no look-ahead at the edge);
NaN before each market's first survey observation, so pre-survey history is never
rewritten. Every fetcher refreshes weekly via `refresh-anchors.yml`.

## Measured effect on the published 10y term premium

Anchored vs previous published series (baseline: US/DE/UK/CA/CH unanchored; JP the
old bridge design):

| | mean Δ | latest Δ | latest TP | latest expected |
|---|---|---|---|---|
| DE | +0.17pp | **+0.37pp** | 0.59 → 0.96% | 2.73 → 2.36% |
| UK | +0.02pp | **−0.23pp** | 1.69 → 1.45% | 3.47 → 3.70% |
| CA | +0.05pp | **+0.25pp** | 0.78 → 1.03% | 2.94 → 2.70% |
| CH | +0.03pp | **+0.25pp** | −0.29 → −0.04% | 0.77 → 0.52% |
| JP | +0.40pp | −0.04pp | 1.40 → 1.37% | 1.49 → 1.52% |

Notes. The UK is the one market whose survey endpoint sits ABOVE its sample-θ
(3.32% vs 2.91%), so its premium falls — the anchor is not a one-way ratchet. Japan's
+0.40pp mean is the removal of the old Ferreira-bridge back-cast (pre-2017 history
reverts to yields-only); its survey-era values barely move. Cross-sectional sanity of
the underlying surveys: UK,CA ≥ EA ≥ JP ordering holds on 100% of common dates, and
the three markets with living surveys agree with them to −5/−10/−30bp.

## The US anchors — ON under the one rule; the external gates retired in review

**Design review, 2026-08-20 (project owner).** The original US gates made DKW and
Kim-Wright — models — the arbiters of a survey anchor: importing one model's
assumptions to govern another, the Ferreira mistake readmitted, and a standard no
other market was held to. Retired accordingly. The US anchors ship on the same
survey-only basis as the other five markets; DKW and Kim-Wright are displayed
cross-checks, never vetoes. The two failed gate rounds below stand as the record of
what was measured, and as standing cross-checks.

Published effect (10y): nominal TP **1.11 -> 1.63%** (expected 3.63 -> 3.11%),
history from 1992, mean -0.29pp; real TP **1.20 -> 1.45%** (expected 1.18 -> 0.94%),
history from 2004, mean -0.46pp. The real-leg move is the original motivating case:
the 2004+ sample let the expected real rate chase observed yields up 40bp in two
months; the survey control re-attributes that move to the premium. Cross-checks at
switch-on: anchored nominal TP agrees with Kim-Wright's LEVEL to +0.02pp; the survey
real forward sits ~0.4pp above DKW's expected real path (a genuine survey-vs-model
disagreement, displayed to the reader). All other markets bit-identical; global
cycle corr 0.99936 across the change.

### The retired gate record (measured before retirement)

The US surveys are 0-10y AVERAGES (SPF BILL10/CPI10), so the endpoint was solved by
average-matching (`us/anchor.py`): endpoint = θ_LR + (survey − model 10y-avg)/w,
w = 0.565 at the 3.5y-half-life floor. Gates declared before running:

* **Real** (BILL10−CPI10) vs DKW expected real, 10y, 2004+: corr **0.786 → 0.634**,
  mean|gap| **0.32 → 0.53pp**. Worse on both. **Off.**
* **Nominal** (BILL10) vs Kim-Wright 10y TP, 1990+: mean gap **+0.45 → +0.05pp**
  (the level lands on KW) but corr **0.980 → 0.907**. Split; the rule was
  "improves both". **Off.**

**Round 2 (same day): terminal-consistent forward objects.** The near-year T-bill is
itself surveyed (TBILLA/B), so the cyclical front strips out by arithmetic:
`fwd_2_10 = (10*BILL10 - y1 - y2)/8`, with the real leg net of the same-window ATSIX
forward `(10*pi10 - 2*pi2)/8` (Aruoba curve, CPI-basis, already the joint model's
anchor). Window-matched with w = 0.662. Results:

* Real vs DKW: corr 0.786 -> **0.605**, |gap| 0.32 -> **0.60pp**. Fail — the survey
  real forward sits ~0.4pp ABOVE DKW's expected real path; the disagreement is
  between the surveys and DKW, not the object or the transform.
* Nominal vs KW: |gap| 0.62 -> **0.45pp** (mean +0.02 — exactly KW's level), corr
  **0.919** (up from round 1's 0.907, still below unanchored 0.980). Fail on corr.

Both anchors stay OFF; the US publishes yields-only on both curves. Recorded
hypothesis, not a gate change: KW's daily variation is yields-driven like our
unanchored series, so the correlation leg may structurally favour unanchored — if
the criteria are ever revised, it happens in review, never to pass a failed run.

## Provenance and upkeep

Consensus files are a frozen, proprietary one-off (private repo; derived premia only).
Public rungs auto-refresh weekly: ECB SPF (API), BoE MPS (per-round xlsx crawl, the
neutral question preferred, the Bank-Rate-path sheet identified by its MPC-dated rows
after an early fallback bug read a gilt-stock 400 as a rate), BoC MPS (quarterly HTML),
SPF BILL10/CPI10 (Philly Fed xlsx). KOF requires an account (API v2 auth) — wire
`switzerland` `shortrate_ext` to it the day one exists.

## US nominal anchor: SME bridge between SPF rounds, on means (2026-10-01)

**Problem 1: frequency.** The SPF asks BILL10 (10y-average T-bill) in its **Q1 round only**
(Philadelphia Fed file: Q1 filled every year, Q2-Q4 never). TBILLA/B and CPI10 are
quarterly, ATSIX monthly, but the nominal 2-10y forward could only move each February.
It was frozen from Feb-2026 through the Iran conflict and the Fed-chair change.

**Problem 2: medians.** Every survey input was a median, which steps. The SPF BILL10
median sat at 2.50 for 2014-17 and lands on 3.00 in 2011 and 2026. The SME 10y-average
median sat at 3.13 for seven rounds (Jul-25 to Apr-26) while both quartiles moved.

**Fix.**
* SPF: the published cross-forecaster **means** (`mean_<var>_level.xlsx`) for BILL10,
  TBILLA/B and CPI10. On the 2-10y forward, mean minus median averages −0.05pp since 1992,
  sd 0.15pp per round.
* NY Fed Survey of Market Expectations (8 rounds/yr, 2023+): the same 10y-average
  question for fed funds, plus the modal path. It publishes quartiles, not a mean (its
  averaged probability distributions stop at next year-end), so the central estimate is
  the **interquartile mean** (p25+p50+p75)/3. `us.fetch_sme` computes
  `fwd_2_10 = (10*avg10 - near-2y path average)/8` per round.
* `us.anchor.bridged_forward_reads` puts each SME read on the SPF T-bill basis via the
  SPF-minus-SME gap at the latest Q1 (nearest SME round within 60 days; causal): 2024
  −0.26, 2025 −0.15, 2026 −0.23pp. The SPF re-pins the level each February; the SME
  carries it between.
* Per-read noise in `rw_smooth`: SPF 0.15 (unchanged); SME 0.03. The SME forward's
  round-to-round changes are a pure random walk (lag-1 autocorrelation 0.02) and the exact
  Kalman likelihood puts read noise at ≤0.02pp. At 0.15 the smoother discounted the SME
  reads and the anchor stayed within 1bp of February.

**What the SME says (interquartile mean, fed funds basis):**

| SME round | 10y avg | near-2y path | 2-10y fwd |
|---|---|---|---|
| 2026-01-20 | 3.15 | 3.21 | 3.14 |
| 2026-04-20 | 3.17 | 3.31 | 3.13 |
| 2026-06-08 | 3.27 | 3.49 | 3.21 |
| 2026-07-20 | 3.30 | 3.57 | 3.24 |

Since January the near-term path rose ~0.35pp and the 2-10y forward **+10bp** (the median
showed +3bp). The anchor, on the SPF basis, went from 2.94 (Feb) to **3.01**.

**US 10y term premium, month-end (%):**

| | SPF median only (before) | SME bridge, medians | **SME bridge, means (live)** |
|---|---|---|---|
| Nominal, Jun-25 | 1.13 | 1.05 | 1.08 |
| Nominal, Feb-26 | 0.97 | 0.93 | 0.96 |
| Nominal, Sep-26 | 2.14 | 2.11 | **2.08** |
| Real, Jun-25 | 1.21 | 1.08 | 1.07 |
| Real, Sep-26 | 1.99 | 1.89 | **1.87** |

Since February the US nominal premium is 6bp lower than under the frozen anchor and the
real premium 13bp lower. The nominal premium still rose ~1.1pp from Feb-26 to Sep-26:
long-horizon expectations
in both surveys moved by tenths of a point, so the yield rise is almost all premium.

Not used: the FOMC SEP longer-run **median** (FRED FEDTARMDLR). It steps for the same
reason and is not an input anywhere. Refreshed weekly in `refresh-anchors.yml`; the
September SME round posts in early October.
