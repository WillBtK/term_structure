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
| US | — | Philly Fed SPF BILL10/CPI10 (1992+, living) | — **but see the gates below: OFF** |

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

## The US gates — pre-committed, run, FAILED, anchors off

The US surveys are 0-10y AVERAGES (SPF BILL10/CPI10), so the endpoint was solved by
average-matching (`us/anchor.py`): endpoint = θ_LR + (survey − model 10y-avg)/w,
w = 0.565 at the 3.5y-half-life floor. Gates declared before running:

* **Real** (BILL10−CPI10) vs DKW expected real, 10y, 2004+: corr **0.786 → 0.634**,
  mean|gap| **0.32 → 0.53pp**. Worse on both. **Off.**
* **Nominal** (BILL10) vs Kim-Wright 10y TP, 1990+: mean gap **+0.45 → +0.05pp**
  (the level lands on KW) but corr **0.980 → 0.907**. Split; the rule was
  "improves both". **Off.**

Reading: the avg-matching transform pipes survey-release timing into a daily series
(the correlation loss), and the SPF average sits below DKW's expected path. The US
therefore publishes yields-only on both curves; its survey cross-check lives here and
in the anchor metadata rather than in the series. The machinery stays behind one-flag
switches so a terminal-consistent survey object (a true 6-10y reading), if one
appears, is a re-measurement rather than a rebuild.

## Provenance and upkeep

Consensus files are a frozen, proprietary one-off (private repo; derived premia only).
Public rungs auto-refresh weekly: ECB SPF (API), BoE MPS (per-round xlsx crawl, the
neutral question preferred, the Bank-Rate-path sheet identified by its MPC-dated rows
after an early fallback bug read a gilt-stock 400 as a rate), BoC MPS (quarterly HTML),
SPF BILL10/CPI10 (Philly Fed xlsx). KOF requires an account (API v2 auth) — wire
`switzerland` `shortrate_ext` to it the day one exists.
