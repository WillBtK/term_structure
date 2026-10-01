# Survey pass-through: how expected rates move between survey rounds (2026-10-01)

## Question

Every term-premium model fits yields almost exactly, so the premium is decided by the
expected short-rate path. On the object the NY Fed SME asks directly — the expected
average policy rate over years 2-10 — the models disagree by a point:

| Expected avg rate, years 2-10 | 2026-07-20 (SME round) | 2026-09-25 |
|---|---|---|
| SME forecasters (fed funds basis) | 3.24 | — |
| Atlas (T-bill basis) | 3.01 | 3.01 |
| Kim-Wright | 3.90 | 4.11 |
| ACM | 3.98 | 4.38 |

Our level comes from surveys. The open question was how expectations should move
BETWEEN survey rounds. Until now the Atlas held the path beyond two years flat between
reads (it put ~2% of daily 10y moves into expectations since 2022). Kim-Wright puts ~61%.

## Evidence: what surveys actually do

Each survey's round-to-round change regressed on the change in the matching market
forward over the same window (US 2-10y; others 5-10y, the nearest fitted pair to their
6-10y question):

| Survey | n | beta | s.e. |
|---|---|---|---|
| US SME 2-10y forward (2023+) | 24 | 0.07 | 0.02 |
| US SPF 2-10y forward (annual, 1992+) | 34 | 0.15 | 0.08 |
| UK Consensus 6-10y (2017-26) | 38 | 0.18 | 0.07 |
| UK BoE Market Participants Survey, neutral | 26 | 0.16 | 0.13 |
| Euro area Consensus 6-10y | 39 | 0.22 | 0.09 |
| Japan Consensus 6-10y | 38 | 0.08 | 0.11 |
| Canada Consensus 6-10y | 38 | 0.18 | 0.06 |
| **Pooled, survey fixed effects** | **237** | **0.160** | **0.036** |

Dropping any one survey leaves 0.154-0.167. (ECB SPF has 6 changes; excluded.)

Forecasters' long-horizon expectations absorb about a sixth of market forward moves.
Kim-Wright's implied expectations absorb about four times that; a flat path absorbs none.

## Rule (all markets)

The survey smoother (`model.neutral_anchor.rw_smooth`) takes the market forward as a
known input: `x_t = x_{t-1} + 0.16 * dF_t + w_t`. Survey reads still correct the state;
at the edge the anchor is the last filtered read plus 0.16 times the forward's change
since. US: Treasury 2-10y forward (the nominal forward also drives the TIPS anchor,
whose survey leg is a nominal-rate survey). Others: 5-10y. Japan's glide to the BoJ
corridor is replaced by the same rule (the corridor is kept as the no-driver fallback).

## Effect

10y term premium at the latest date, and since 2017 (bp):

| | before | after | change | mean abs change 2017+ | max abs 2017+ |
|---|---|---|---|---|---|
| US nominal | 2.08 | 2.01 | −7.9 | 3.4 | 15.3 |
| US real | 1.87 | 1.79 | −8.0 | 3.6 | 16.0 |
| UK | 1.74 | 1.72 | −1.3 | 1.4 | 8.7 |
| Germany | 1.24 | 1.22 | −1.5 | 1.1 | 5.3 |
| Japan | 1.41 | 1.42 | +0.9 | 0.5 | 2.7 |
| Canada | 1.15 | 1.13 | −2.8 | 1.3 | 5.4 |
| Switzerland | −0.07 | −0.07 | 0.0 | 0.7 | 5.1 |

The rule changes the daily behaviour, not the level: the level is still set by surveys,
and the gap to Kim-Wright/ACM is a level disagreement.

## Sensitivities (footnoted on the Atlas, not shown as a band)

* beta over the single-survey range 0.07-0.22: at most about ±5bp on today's 10y premia
  (US +4.4/−3.0bp; others ≤2.3bp).
* US basis. SPF asks about the 3-month bill, the SME about fed funds; bills usually
  yield less. The Atlas uses the bill basis: the model is fitted to the Treasury curve,
  as are Kim-Wright and ACM. On a fed-funds basis the US 10y premium would be 0.19pp
  lower (nominal) and 0.20pp lower (real).
