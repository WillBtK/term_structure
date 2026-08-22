# Exports for other repos

## `us_term_premia_monthly.csv`

Monthly US term premia for `WillBtK/imf_fiscal_sensitivity`, which regresses yields and
premia on CBO fiscal projections and currently uses ACM.

**Fetch it the same way that repo fetches CBO** — `raw.githubusercontent.com` works,
no new plumbing:

```python
import pandas as pd
URL = ("https://raw.githubusercontent.com/WillBtK/term_structure/main/"
       "exports/us_term_premia_monthly.csv")
tp = pd.read_csv(URL, index_col="month", parse_dates=True)
```

Regenerate with `python -m exports.term_premia_monthly`.

### What it adds over ACM

That repo's handoff records two ceilings ACM imposes rather than the analysis choosing.
Both are filled here.

| columns | what | ACM |
|---|---|---|
| `tp2 tp3 tp5 tp7 tp10 tp20 tp30` (plus `tp0.25 tp0.5 tp1`) | nominal term premium, 1961+ | 1–10y only |
| `rtp2 rtp3 rtp5 rtp7 rtp10 rtp20` | **real** term premium, 2004+ | none |
| `btp2 … btp20` | `tp − rtp`, the inflation-risk premium | structurally impossible |

So results no longer have to stop at 10y or be nominal-only, and the real/breakeven
split of the *premium* — not just of yields — is now an available decomposition rather
than something needing a real-rate term-structure model that repo doesn't have.

### Conventions — chosen so the swap is a one-line change

* **Percent**, matching `acm_tp5` and that repo's GSW yields. The source parquet is decimal.
* **Calendar-month averages of daily observations**, because that repo averages market
  series to the vintage month before its as-of merge. Averaging here leaves that merge
  untouched: replace `acm_tp10` with `tp10` and nothing else moves.
* `n_days` carries the observation count, so a partial final month is visible rather
  than silently thin. Months with fewer than 5 observations are dropped.

### Provenance, and how it differs from ACM

AFNS + Christensen–Rudebusch shadow rate on each market's regime-clean sample, with
Bauer–Rudebusch–Wu bias correction. **Not ACM, and not a drop-in replica of it.** Where
they overlap:

| tenor | months | from | to | corr | mean ours | mean ACM | gap |
|---|---|---|---|---|---|---|---|
| 5y | 783 | 1961-06 | 2026-08 | 0.833 | 1.11 | 0.90 | +0.21 |
| 10y | 661 | 1971-08 | 2026-08 | 0.852 | 1.80 | 1.59 | +0.21 |

Correlated enough that a re-estimation reads as a change from a known baseline; ~21bp
higher on average, so **level coefficients will shift and should not be compared with
published ACM-based ones without saying which premium was used.** `compare_with_acm()`
regenerates that table.

### Two things to watch when re-estimating

* The `rtp`/`btp` columns start in **2004**, so any real-premium regression is
  post-break by construction — which removes the pre/post-2008 comparison for that leg,
  not just shortens it.
* The 20y and 30y nominal columns inherit the GSW composition caveat: `SVENY22`+ only
  begins fitting in 1985-11, so a long-maturity series spanning the early sample mixes
  curve shape with sample composition.
