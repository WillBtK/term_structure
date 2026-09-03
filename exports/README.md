# Exports for other repos

**Clone the repo — do not use a raw URL, for any file in this directory.** This repo
is private, and `raw.githubusercontent.com` returns 404 for private repos regardless
of caller or token in the URL — see the note under `us_term_premia_monthly.csv` below,
which corrects an earlier wrong instruction with exactly this mistake. Standing
constraint: the repo stays private because the Consensus Economics inputs are
proprietary; everything in `exports/` is a **derived** premium/factor, which is why
it's fine to hand out, but the delivery mechanism still has to be a clone.

## `us_term_premia_monthly.csv`

Monthly US term premia for `WillBtK/imf_fiscal_sensitivity`, which regresses yields and
premia on CBO fiscal projections and currently uses ACM.

**Clone the repo — do not use a raw URL.** An earlier version of this file said to
fetch over `raw.githubusercontent.com` "the same way that repo fetches CBO". That was
wrong: the CBO mirrors are public, **this repo is private**, and
`raw.githubusercontent.com` returns 404 for private repos regardless of caller or token
in the URL.

```bash
git clone --depth 1 https://github.com/WillBtK/term_structure /home/user/term_structure
```

then read `exports/us_term_premia_monthly.csv` from the checkout. Regenerate it with
`python -m exports.term_premia_monthly`.

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

## `multi_market_term_premia_monthly.csv` / `global_factor_monthly.csv`

For `WillBtK/imf_fiscal_sensitivity`'s cross-market fiscal-sensitivity design
(`docs/CROSS_MARKET_DESIGN.md`). Regenerate with `python -m
exports.multi_market_term_premia_monthly` / `python -m exports.global_factor_monthly`;
neither runs in CI (same as `us_term_premia_monthly.csv` — a manual regenerate, not a
daily job), so re-run both after any re-estimation before handing off a fresh copy.

**`multi_market_term_premia_monthly.csv`** — long format, one row per (month, country,
tenor): `month, country, tenor, tp, rtp, n_days`. Six markets (US, UK, DE, JP, CH, CA),
each on its **own** `COMMON_GRID`/`REAL_GRID` (they differ — deliberately not forced to
a common set). `rtp` is populated for US and UK only, the only two markets with a
production real curve; blank elsewhere, not a zero. Same conventions as
`us_term_premia_monthly.csv`: percent, calendar-month average of daily observations
from `viz/data_daily.parquet`, months with fewer than 5 observations for that
(month, tenor) dropped. **Japan's YCC-period measurement downweight is already baked
into the canonical parquet** (applied where `japan.production_tp` / `viz.export_daily`
build it, via `japan.config.ycc_downweight`) — this export inherits it as-is; there is
no separate daily-only application to reconcile against.

**`global_factor_monthly.csv`** — one row per (month, country) from the 10y
term-premium DFM (`model/global_factor.py`), the same fit that backs the Atlas's
"Global cycle" panel. Columns: `month, country, G, loading, residual, mean_pct,
sd_pct`. `G` and `residual` are in the model's standardized space; invert with the
carried `mean_pct`/`sd_pct`:

    TP_i,t (%, raw) = mean_pct_i + sd_pct_i * (loading_i * G_t + residual_i,t)

**Switzerland has no rows in this file.** It is excluded from the DFM by design — its
government-bond liquidity is too thin next to the majors, so including it would let its
own noise distort the shared factor for the other five (`model.global_factor.GF_ORDER`)
— not a short-history case. A Swiss series has to come from the plain nominal export
above, read on its own rather than relative to this factor. The other five markets'
loadings/residuals ARE estimated on each market's own full history (US/DE/JP from
1974-09 via the committed pre-modern-curve backfill, UK from 1979-01, CA from 1986-01);
only the standardization mean/sd use the fully-balanced window (all five present, from
1986-01), so an earlier month for a market is decomposed with the same parameters as
every later one, not different ones.

This is the DFM's existing `term_premium@10y` combo (`viz/global_factor_meta.json`'s
`"default"`), re-decomposed here rather than refit, so this file and the Atlas panel
never disagree. Other curve points (5y/20y/30y/5y5y) and other measures (expected rate,
observed yield) are fit separately and are not in this file — ask if one of those is
needed; it is a one-line change to `COMBO` in `exports/global_factor_monthly.py`, not
a re-estimation.
