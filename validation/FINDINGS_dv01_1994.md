# US DV01 outstanding and held by SOMA, extended to 1994 (2026-10-12)

Code: `supply/fetch_us_hist.py`, `supply/us_dv01_hist.py`, spliced in `supply/us_dv01.build`.
Tests: `supply/tests/test_us_dv01_hist.py`.

The series previously began in 2003 because both security-level sources do: the MSPD API
(table 3) starts at 2001-01 and the NY Fed's SOMA-by-CUSIP file at 2003-07-09.

## Outstanding

| months | source | check |
|---|---|---|
| 1994-01 → 1998-03 | auction rebuild: every auction since 1979-11 (Fiscal Data `auctions_query`) summed by CUSIP, plus 15 bonds issued 1955-79 | Jan-1994 statement totals: bills −0.07%, notes −0.18%, bonds +0.03% |
| 1998-04 → 2000-12 | statement PDFs (TreasuryDirect `/ftp/opd/`), Table III parsed | every month ties to the printed class totals (±$4mn rounding) |
| 2001-01 → | MSPD API | — |

* **Parser.** Run on the 2001 statements it reproduces the API's rows exactly: every
  CUSIP and every amount (one bill maturity differs by a day, a typo in the statement).
* **Auction rebuild.** Over 1998-04 → 2000-01, where the statements exist, it carries the
  same CUSIPs month by month, with totals 0.10-0.18% below. The 1994-97 statements are
  scanned images (1993 → Jan 1997) or missing (Feb 1997 → Mar 1998), hence the rebuild.
* **Bonds from before the auction record.** 15 issues (1955-79), from the Jan-1994
  statement (OCR'd, each amount checked as issued − retired = outstanding). The Jan
  1995-97 statements confirm that the four callable ones (8-1/2% 1994-99, 7-7/8% and
  8-3/8% 1995-00, 8% 1996-01) disappear at their first call date. The flower bond
  3-1/2% 1998 steps 253 → 242 → 232 → 229 → 226.
* **Callables priced to call.** Treasury called every callable bond at its first call (the
  MSPD API shows each leaving the month before). MSPD records the final maturity; the
  SOMA file records the call date. Both ladders are now priced to the call. This lowers
  2001-09 outstanding DV01: 2001-03 by about $35mn/bp (−3.1%), 2005 by $26mn/bp (−1.7%),
  2009 by $5mn/bp (−0.2%). Nothing changes from 2010.
* TIPS stay excluded before 2004-01, as in the existing series (no reliable real curve).

## SOMA

| months | source |
|---|---|
| 1994-01 → 1996-05 | Federal Reserve Bulletin table 1.19 (FRASER scans, OCR text layer): month-end holdings by remaining maturity |
| 1996-06 → 2003-06 | H.4.1 maturity distribution, month-end releases (federalreserve.gov archive) |
| 2003-07 → | NY Fed SOMA holdings by CUSIP |

* **Bucket method.** Each remaining-maturity bucket of SOMA's holdings (within 15 days,
  16-90 days, 91 days-1 year, 1-5, 5-10, over 10 years) is priced at the outstanding
  ladder's DV01 per dollar of par in that bucket. Holdings under repo are removed from the
  H.4.1 short buckets. SOMA's TIPS come out of the long buckets before 2004, matching the
  CUSIP series. The release stops footnoting their amount in 2002-12, so it is bridged to
  the CUSIP file's first reading.
* **Validation, 2003-07 → 2007-11 (53 months, like for like).** Bills and nominal coupons
  only, SOMA's TIPS removed from H.4.1 using the CUSIP file.
  * Uncalibrated, the bucket method reads high: +5.5% on average (−0.4% to +9.2%).
  * Within every coupon bucket SOMA's DV01 per dollar runs 0.93-0.97× the market's, because
    SOMA held more seasoned, higher-coupon issues.
  * Calibrated by the overlap mean (×0.951), the error is **0.1% ± 2.7%, range −6.8% to
    +3.9%** (SOMA's TIPS taken out of H.4.1 at accreted principal, as H.4.1 reports them;
    at par, as before 2026-10-13, it was ×0.948 and 0.0% ± 2.2%).
  * The calibration is recomputed at each build and stored in `soma_calibration`.
* **Bulletin vs H.4.1, 1996-06.** The coupon buckets agree to the dollar (91,694 /
  32,941 / 38,721). The short buckets differ only because one is the month-end and the
  other the last Wednesday.
* **Bulletin OCR.** Each month-end is printed in three issues. A reading is kept only if
  its buckets add up to the printed total (within $3mn); all 29 months have one.
* **Seams.**
  * 1996-05 → 06: SOMA 86.9 → 86.6.
  * 2003-06 → 07: SOMA 176.2 → 157.7 (−10.5%), with outstanding −10.6% in the same month
    (the July 2003 sell-off).
* **Monthly flows (added 2026-10-13).**
  * Outstanding: the CUSIP-level monthly flow throughout. Across the 1998-03 → 04 source
    change it uses the auction rebuild at both ends; across 2000-12 → 2001-01 the
    statement and API ladders, which are identical.
  * SOMA through 2003-07: each bucket's holdings beyond what last month's holdings would
    have aged into, with the ageing taken from the outstanding ladder CUSIP by CUSIP, less
    what matured. A test checks that ageing reads zero and a bill redeemed, not rolled,
    reads negative.
  * Pricing (revised 2026-10-13): purchases at the value per dollar of that month's new
    issues in the bucket, maturities at their own. The first version priced purchases at
    the seasoned bucket average and left maturities out (gross, not net). In DV01 that
    barely showed; in market value it read ~$100bn a month high.
  * Against the CUSIP-level monthly flow, 2003-08 → 2007-11 (52 months):
    * market value: correlation 0.996, mean absolute error $0.21bn (monthly sd $3.0bn),
      average 1.71 vs 1.71;
    * DV01: correlation 0.89, mean absolute error $1.25mn/bp (monthly sd $3.1mn/bp),
      average 4.65 vs 5.38 (first version: 0.90, 1.1, 5.0 vs 5.4).
  * The market-value flow is left uncalibrated. The DV01 flow is scaled by the overlap's
    ratio of mean flows, ×1.158 (`soma_flow_calibration`; a ratio of means because a
    monthly flow can be near zero): mean absolute error $1.15mn/bp, average 5.38 vs 5.38.
    Split-sample: fitted on 2003-08 → 2005-09 it is ×1.150 and gives 6.07 vs 6.14 over
    the rest; fitted on 2005-10 → 2007-11, ×1.164 and 4.68 vs 4.63.
* **Market value (added 2026-10-13).** Par × curve-implied invoice price per security,
  FRNs at par, TIPS at par before the real curve.
  * SOMA's stock before 2003-07: every bucket, TIPS in, at the market's value per dollar;
    error against the CUSIP-level stock, 2003-07 → 2007-11, −0.1% ± 0.3% (range −0.8% to
    +0.7%).
  * Net market value of issuance vs the change in par outstanding, 1994-2000 ($bn):
    93/136, 138/181, 103/154, −50/−3, −144/−95, −123/−74, −357/−314. The gap is bills and
    discount coupons sold below par and redeemed at it: interest, not supply.
  * TIPS accretion: MSPD reports accreted principal, so each month's inflation uplift
    was being booked as issuance (in DV01 too). Every flow now strips it: SOMA from its
    published par (SOMA's TIPS now carried at accreted principal, like MSPD, which adds up
    to $82mn/bp to its DV01 stock, 2022-08), MSPD from the median change across TIPS
    (annual accretion 2021 6.2%, 2022 7.7%, 2023 3.3%, in line with lagged CPI). Largest
    change to the monthly DV01 supply flow: $18mn/bp (2022-08).
  * The 3-month SOMA flow remains CUSIP-only (2003-10 →).

## A rejected alternative

Holding SOMA's bucket shares fixed and scaling by total holdings (FRED `FDHBFRBN`) drifts
badly when tested forward from mid-1996: −2% after one year, −6% after two and −12%
after three, as the Fed lengthened its holdings in 1998-2001. The Bulletin buckets
replaced it.

## Levels (annual means, $mn per bp)

| | outstanding | SOMA | SOMA share | 3m net supply |
|---|---|---|---|---|
| 1994 | 974 | 72 | 7.4% | 50 |
| 1996 | 1,094 | 89 | 8.1% | 63 |
| 1998 | 1,266 | 121 | 9.6% | 52 |
| 2000 | 1,102 | 140 | 12.7% | 24 |
| 2002 | 1,049 | 165 | 15.7% | 41 |
| 2004 | 1,394 | 180 | 12.9% | 88 |
