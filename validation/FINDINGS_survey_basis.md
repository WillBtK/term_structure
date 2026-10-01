# Survey basis: converting survey rates to each curve's own short rate (2026-10-01)

## The issue

A term premium is the yield minus the expected average of the MODEL's short rate, and
the model's short rate is the short end of the fitted government curve. The long-horizon
surveys that anchor the expected path forecast something else: a money-market rate.

| Market | Survey (level) | Instrument it forecasts |
|---|---|---|
| US | Philadelphia Fed SPF (BILL10, TBILLA/B) | 3-month Treasury bill |
| US | NY Fed SME (timing only; bridged to the SPF) | fed funds target range |
| DE | Consensus euro area 6-10y; then ECB SPF | 3-month Euribor |
| UK | Consensus 6-10y; then BoE MPS | 3-month interbank; Bank Rate |
| JP | Consensus 6-10y | 3-month interbank |
| CA | Consensus 6-10y; then BoC MPS | 3-month T-bill |
| CH | Consensus 6-10y; then KOF | 3-month interbank |

Until now each survey was used as if it were the model's short rate, so any systematic
gap between the instrument and the government curve was booked as term premium.

## Measurement (`validation/survey_basis.py`)

Average of (model's shortest fitted tenor − survey instrument). Consensus markets:
monthly, 2017-2026 (the survey era), OECD 3-month rates via FRED. US: daily, 2016-19
and 2022H2-26 (ample reserves, off the lower bound).

| Market | Model tenor | Gap (pp) | n |
|---|---|---|---|
| Germany | 3m | −0.242 | 116 |
| UK | 6m | −0.225 | 86 |
| Japan | 1y | −0.137 | 115 |
| Canada | 3m | +0.020 | 116 |
| Switzerland | 1y | −0.009 | 116 |
| US vs 3m bill | 3m | +0.126 | 2060 |
| US vs fed funds target midpoint | 3m | +0.152 | — |

Government curves in Germany, the UK and Japan sit below interbank rates (collateral and
safety value; unsecured bank credit). The US Treasury model is fitted to coupon securities
(GSW) and runs above bills in every regime since 1992 (+0.11 to +0.15). Canada's survey
already forecasts the T-bill, and its gap is nil.

## Correction

Each survey read is moved onto the model's basis (`AnchorSpec.basis_pct`;
`us.anchor.SPF_TO_MODEL_PCT`). Living rungs are level-bridged onto the converted survey,
so they inherit it.

10y term premium at the latest date (with the 0.16 pass-through already in place):

| | before | after | change | mean change 2017+ |
|---|---|---|---|---|
| US nominal | 2.01 | 1.90 | −11bp | −10bp |
| US real | 1.79 | 1.68 | −11bp | −11bp |
| UK | 1.72 | 1.85 | +13bp | +12bp |
| Germany | 1.22 | 1.35 | +13bp | +12bp |
| Japan | 1.42 | 1.49 | +7bp | +6bp |
| Canada | 1.13 | 1.12 | −1bp | −1bp |
| Switzerland | −0.07 | −0.06 | 0 | 0 |

## What remains a choice: whose expectations (US)

On the model's basis, the SME panel (dealers and investors) puts the 2-10y expected rate
0.30, 0.19 and 0.27pp above the SPF panel (economists) at the 2024-26 overlaps. That is a
difference in who is asked, not in the instrument. The Atlas keeps the SPF as the level:
one consistent panel back to 1992, no break when the SME starts in 2023, and no rewriting of
thirty years of history from three overlaps. Using the SME level would put the US 10y
premium at 1.69% (option tested: SME level, SPF history shifted by the 0.25 mean gap, or
SME from 2023 only — the same today). Footnoted on the Atlas.
