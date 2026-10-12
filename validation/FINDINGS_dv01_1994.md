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
  * Calibrated by the overlap mean (×0.948), the error is **0.0% ± 2.2%, range −5.5% to
    +3.6%**.
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
* **No SOMA flow before 2003-10.** The 3-month SOMA net-purchase flow needs CUSIPs to
  separate purchases from holdings ageing across buckets, so it starts there.

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
