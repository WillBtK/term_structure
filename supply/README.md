# Supply and ownership

Par value is the wrong unit for the supply channel. A bill and a thirty-year bond are
the same number of dollars and completely different quantities of interest-rate risk,
and the whole point of quantitative easing was to buy the second kind. Measured by par
share the Federal Reserve held MORE of the public debt through the 1970s — when its
portfolio was mostly bills held for reserve management — than at the peak of QE. A
par-share measure cannot tell 1975 from 2014, and empirically it does not.

This module measures the quantity that does distinguish them: **duration risk**, in two
units. Ten-year equivalents (Li–Wei 2013, Ihrig et al. 2018) for the *share* held by a
central bank, and DV01 for the *dollar* amount the market has to absorb.

## What is actually in production

Only one output of this module feeds the scenario engine: **`price_insensitive.py`**,
which supplies the ownership driver in `model/tp_drivers.py` (the third US driver
alongside expected inflation and the projected deficit).

The finding that put it there is that only the **price-insensitive** holder compresses
the premium. Foreign official enters at −0.064 (t = −12.6); foreign private enters at
−0.029 (t = −1.2) and does not survive. The official coefficient lands within a whisker
of the Federal Reserve's own (−0.066), which is what you would expect if the mechanism
is duration removed from price-sensitive hands rather than "paper held abroad". So the
driver is not how much is held offshore — it is how much of the stock sits with someone
who does not care what it costs.

Each market's equivalent holder is a different holder (`price_insensitive.py` documents
the four), which is why the driver is built per market rather than transplanted.

**The Fed's balance sheet is carried as a control, not a driver.** If the stated intent
is a neutral balance sheet, the variable is worth roughly zero in projection; the reason
to keep it is that 2009–2026 would otherwise contaminate the inflation coefficients the
tool actually uses. `validate_us_control.py` tests it on those terms — cleaner and more
stable inflation slopes — not on "does supply predict the premium".

## The ladders

Both sources are security-level, giving par amount, coupon and maturity date, which is
everything the duration math needs.

| | source | granularity | span |
|---|---|---|---|
| US outstanding | MSPD table 3 (fiscaldata.treasury.gov) | per CUSIP, monthly | 2003-01 → |
| US held | SOMA holdings (markets.newyorkfed.org) | per CUSIP, weekly → month-end | 2003-07 → |
| JP outstanding | MoF auction results | per issue | 2001 → |
| JP held | BoJ `mei` holdings files | per issue | 2001 → |

Japan needs a join the US does not. The BoJ identifies each holding only by original
tenor and issue number — "10-Year JGB #348" — with no maturity date and no coupon. The
MoF auction workbook supplies both on the same issue number, and summing an issue's
auction allotments also recovers the amount outstanding, so one join gives numerator,
denominator and the pricing inputs.

Raw ladders are cached **unfiltered**, so revising which security classes to keep costs
a reload rather than another download. Refresh is opt-in (`--refresh`): a full SOMA pull
is ~280 requests.

## Ten-year equivalents — the share

```
TYE = Σ_i (D_i · A_i) / D_ref
```

`D_ref` is recomputed per date from the prevailing yield rather than fixed, because a
ten-year note's duration is ~8.9y at 2% and ~7.8y at 6% — holding it constant would put
a spurious yield-driven trend into a series meant to measure quantities.

`us_tye.py` and `jp_tye.py` emit the duration share **and the par share alongside it**.
The divergence between the two is the measurement point, not a diagnostic: SOMA holds
about 14% of the marketable Treasury market by par and about 21% of its duration risk,
the legacy of Operation Twist and of QE concentrated at the long end.

`project_us.py` runs the portfolio forward under a *neutral* balance sheet — par growing
with nominal GDP, purchases matching the market's maturity profile. Convergence from a
portfolio that starts long of the market means shedding duration for as long as the
legacy long holdings take to roll off: a persistent negative flow, which is the variable
the premium responds to.

## DV01 — the dollar amount

`us_dv01.py` prices the **full** marketable ladder — bills, nominal notes and bonds,
TIPS and FRNs — and emits four monthly series in $mn per 1bp, 2003-01 → present:

| column | what |
|---|---|
| `outstanding_dv01_mn` | total DV01 outstanding |
| `soma_dv01_mn` | DV01 held by the Federal Reserve |
| `supply_3m_dv01_mn` | rolling 3-month net new supply |
| `soma_net_3m_dv01_mn` | rolling 3-month net SOMA purchases |

**The flow series are not the change in the stock.** The stock moves every month even
with zero primary-market activity, because every outstanding bond's own duration drifts
with rolldown and with the level of rates. The flows isolate the quantity change:

```
flow = Σ_i [A_i(t) − A_i(t−3mo)] · (DV01 per unit of CUSIP i)
```

priced at the CUSIP's own duration at whichever end of the window it exists. A bond
sitting unchanged contributes exactly zero however much its DV01-per-unit has moved; a
new issue or reopening counts its incremental par at today's duration; a CUSIP that
vanished — matured, or bought back to zero — exits at the duration it had three months
earlier, which is necessarily small, since anything fully gone by then had at most three
months left to run.

That distinction is what makes the QT period readable. Net SOMA purchases stay *positive*
through 2023–2026 (+$25–70mn/bp a quarter) while the SOMA stock falls from $3.9bn/bp to
$2.2bn/bp: the Fed still reinvests up to the runoff cap, and those purchases carry
duration while the bonds rolling off, by definition near maturity, carry almost none.

### Pricing conventions and their limits

* **Curve-implied yield, not a market price.** Each security is priced off the GSW
  nominal zero curve at its own maturity — the same simplification `tenyear.py` already
  uses — not its own quoted yield, which these par-amount snapshots do not carry.
* **TIPS off the real curve** (`us.tips`), since inflation-linked duration is a different
  quantity. NaN and excluded from every sum before 2004-01, where GSW caution against
  their own early TIPS fits.
* **FRNs get a stipulated ~0.125y duration**, not a curve-priced one. A floating coupon
  reprices quarterly, so pricing it as a fixed-rate bond to stated maturity would hand it
  years of spurious duration; MSPD/SOMA do not publish a per-CUSIP reset date, so the
  placeholder stands in for "time to next reset". At under 3% of the ladder and near-zero
  duration either way this cannot move the totals materially — but it is a stipulation,
  not a measurement.
* **MSPD and SOMA snapshot on different days** — Treasury's true month-end against
  whichever Wednesday closed SOMA's reporting month — so the series are aligned on
  calendar month, not exact date. SOMA also publishes ahead of MSPD, so the latest month
  routinely carries a SOMA figure with the outstanding leg still blank.

Corporates are **not** here. This repository holds sovereign security-level detail only;
extending the same arithmetic to IG and HY needs an issue-level corporate source (an
index provider's constituent file, TRACE, Mergent FISD).

## Files

```
fetch_us.py             MSPD + SOMA ladders; *_all variants keep TIPS/FRNs for DV01
fetch_jp.py             BoJ holdings x MoF auction reference
tenyear.py              duration + ten-year-equivalent math (pure, unit-tested)
us_tye.py  jp_tye.py    the duration share, monthly
us_dv01.py              the four DV01 series, monthly
price_insensitive.py    THE PRODUCTION DRIVER: price-insensitive ownership, % of GDP
project_us.py           SOMA duration share under a neutral balance sheet
validate_us.py          pre-registered test of the duration share on the premium
validate_us_control.py  does the balance sheet earn its place as a control?
validate_jp.py          the same test where the prior says it should show up
data/                   cached ladders + derived parquets (gitignored)
```

## Commands

```bash
python -m supply.fetch_us [--refresh]   # report/refresh the cached ladders
python -m supply.us_tye                 # duration share vs par share, by year
python -m supply.us_dv01                # the four DV01 series
python -m supply.validate_us            # the pre-registered test
python -m pytest supply/tests -q
```

`us_dv01` caches the priced per-CUSIP ladders separately from the aggregate, so
re-running after a change to the aggregation step is seconds rather than the ~8 minutes
the full repricing takes.
