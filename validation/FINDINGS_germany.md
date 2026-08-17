# Germany (Bund) term premium — findings

> **⚠ Superseded numbers.** Term-premium levels here predate the
> **mean-reversion floor** fix; the corrected numbers (latest 10y ≈ +1.3%) and
> the calibration are in **`FINDINGS_mean_reversion.md`**, which supersedes the
> Bauer-Rudebusch-Wu persistence treatment. The data-layer, ELB, and methodology
> narrative below still stand.

Second country. Same engine as the US (`/model` unforked); only the data layer
(`germany/`) differs. Two deliverables, mirroring the US: a **modern production
series** and a separate **historical exhibit**.

## Data + reconstruction
Deutsche Bundesbank estimated term structure (Svensson method), listed Federal
securities (SDMX flow `BBSIS`, sector S1311/A604). Monthly **1972-09 → 2026-07**
(647 obs), out to 10y. Reconstruction via the shared `us.reconstruct` Svensson
math matches the Bundesbank's own published spot yields to **< 0.6 bp** at every
maturity (0.5-10y). Bund yields went clearly negative 2014-2022 (2015 5y ≈ −0.03%).

## Modern production TP — euro era (1999+), shadow AFNS + BRW

`germany/production_tp.py::run_germany_production`. 331 monthly obs (1999-2026),
converged, **yield-fit MAD 3.49 bp**. BRW bias-correction (auto mode → unrestricted;
the euro-era factors are well-behaved) pushes ρ 0.995 → 0.9999. ELB **ML-estimated
at −0.80%** (validates Praet's −1.0% region; see below).

> Numbers are on the corrected decomposition path (sample-mean θ for the
> expectations leg + observed-based realised premium + auto BRW) adopted pipeline-
> wide from the Japan leg; see `FINDINGS_japan.md` / `FINDINGS_phase5.md`.

**Headline term-premium levels (bias-corrected):**

| | 10y TP | 2y TP |
|---|---|---|
| latest (2026-07) | **+0.13%** | −0.15% |

The German 10y term premium today (~0.13%) is the **lowest of the three built
markets** (US ~0.67%, Japan ~0.90%) — consistent with Bund safe-haven / collateral
demand compressing the premium.

### ELB — the data validates Praet's −1.0%

Unlike the US (where the bound was only identified from above), Bund yields went
clearly negative, so the ELB is **well identified**. Profiling the likelihood over
the ELB (other params at the optimum):

| ELB | negloglik | | ELB | negloglik |
|---|---|---|---|---|
| −2.5% | −14277 | | **−1.0% (Praet)** | **−14363** |
| −1.5% | −14295 | | **−0.75% (argmin)** | **−14369** |
| −1.25% | −14323 | | −0.5% | −12394 |
| | | | 0.0% | +14346 (collapses) |

The data prefers a bound of **−0.75% to −1.0%** (flat, shallow optimum in that
range; sharp deterioration above −0.5%). **Praet's −1.0% is within ~6 log-likelihood
of the −0.75% optimum — confirmed.** We report −1.0% as the calibration with a
−0.5%/−1.5% band; the ML estimate lands in the reversal-rate region, as expected.

### Internal ELB check
The shadow short rate goes clearly negative through the ECB negative-rate era,
reaching a min of **−1.17% (2016-12)** — dipping below the −1.0% bound (the shadow
is unbounded; the ELB floors the *observed* rate), and averaging ~−0.53% over
2019-2021 (vs an actual 3m Bund yield of −0.67%). Away from the bound the shadow
model collapses to plain AFNS. The ELB mechanism behaves as intended.

## Historical / stagflation exhibit — full 1972+ history

`germany/run_history.py` (separate, clearly labeled, *not* joined to the modern
series). Single-regime plain-AFNS on the full 1972+ history + BRW bias-correction
(647 obs, converged):

| 10y term premium (bias-corrected) | Germany | US (for contrast) |
|---|---|---|
| 1973-1985 (stagflation era) | **0.63%** | ~1.3% |
| 1990s | 0.51% | — |
| peak | 2.63% (1976-04) | ~2.9% (mid-1980s) |

The German 1970s term premium was **markedly lower than the US** — consistent with
the Bundesbank's inflation-fighting credibility keeping the inflation-risk premium
compressed even through the oil-shock era. Regime dating (Kim filter over the full
history) is a further lens, appended separately when that (expensive) estimation
completes.

### Atlas overlay — 1972+ monthly (shipped)
`germany/run_history_overlay.py` produces a small committed JSON
(`viz/history/germany_hist_tp.json`) that the Atlas attaches as a labelled
**"Germany — the long view (1972+)"** panel (2y & 10y, monthly), overlaid on the
daily production line for the 1997+ overlap and never spliced onto it. It uses
**plain AFNS + the mean-reversion floor** (3.5y half-life; the same expectations-leg
treatment as the daily series) over the full monthly Bundesbank history — the daily
shadow-rate quadrature is skipped as far slower and equivalent away from the bound.
Result: 10y TP averaged **~2.5%** over the 1973–85 high-inflation era, peaking
**+4.0% (Aug 1981)** and easing to modestly negative latterly. It is a single
full-sample fit with the expected-rate path anchored to a modern long-run mean, so
the deep-past *level* is approximate — read the shape and the cross-country contrast,
not the basis point.

**Level-align splice (default view).** The historical fit uses a full-sample θ that
the 1970s–80s 15–19% rates drag upward, so its expected-rate path is biased high and
its term premium biased **low** vs the modern-sample production series — by a large but
stable offset (overlap-mean gap **+1.89% at 10y, +0.67% at 2y**; std ~0.55% at 10y over
the 1997–2026 overlap). Rather than show the raw gap, the Atlas by default **level-aligns**
the historical series to the production line by that per-tenor overlap-mean offset and
draws one continuous series: production 1997+, level-aligned history before (seam
1997-08). This preserves the 1970s *shape* (the raw fit) and removes only the *level*
bias; on this basis the 1973–85 10y premium averages higher and peaks near **+5.9%**,
broadly in line with the US in the same era. A "Raw exhibit" toggle still shows the
as-estimated series and the gap. The offset is computed in `viz/thin_daily`
(`_attach_germany_history`) from the actual overlap each build (`tp_aligned` /
`align_offset` / `seam`), not hard-coded. Caveat: a constant per-tenor shift; the ~±0.55%
overlap residual means the deep-past level stays indicative, not basis-point.

## Validation summary
No German SF-Fed-equivalent term-premium benchmark exists, so validation rests on:
(1) reconstruction accuracy < 0.6 bp vs published Bundesbank spots; (2) the ELB
profile validating Praet's −1.0%; (3) the internal shadow-rate/ELB behavior above;
(4) sensible modern levels (10y ~0.26%, below the US). External euro-area
directional benchmarks (Wu-Xia / Krippner) were not readily loadable this session.

## Bund liquidity "noise" gauge (market quality)
The German analogue of the US TIPS/UST Hu–Pan–Wang "noise" measure
(`validation/bund_liquidity.py`): per trading day, fit a fixed-λ Nelson–Siegel
curve (OLS) to the yields of every outstanding conventional Federal security and take
the **RMS fitting residual**. A liquid market keeps every issue on a smooth curve
(low noise); illiquidity scatters individual issues (high noise). Built only from
public security-level data — a free daily illiquidity read on the euro benchmark.

**Data** — Deutsche Bundesbank "Prices and yields of listed Federal securities"
(`germany/fetch_bund_prices.py`). The monthly workbooks are linked directly from the
service index page (`…/810710/…`) as `YYYY-MM-excel-data.xlsx`; each workbook has
**one sheet per trading day** (named `DD.MM.YYYY`) with columns
`ISIN | Coupon | Description | Maturity | Residual life | Issue volume | Price |
Yield in % | Dirty price`. The trading date is the **sheet name**, not a column.
The index exposes ~5 recent months, so the cache accumulates month-by-month (deep
history back-fill via the PDF archive is a deferred follow-up). Bundesbank is
egress-blocked from the dev sandbox, so the fetch runs in the daily GitHub job;
confirmed end-to-end via CI probes.

**Key correctness fix** — the workbook mixes nominal Bunds/Bobls/Schätze
(DBR/OBL/BKO) with a few **inflation-linked** issues (`DBRI … I/L`). Linkers quote
**real** yields ~2% below the nominal curve, so a single cross-sectional fit read
them as ~200 bp residuals and the raw "noise" ballooned to **~41 bp** — an artefact,
not illiquidity. `drop_inflation_linked` removes them (by `DBRI`/`I/L` marker),
bringing the gauge to **~5 bp** — the single-digit level expected for the world's most
liquid sovereign curve (diagnosed on 2026-08-14: 71 bonds → RMS 41 bp; 63 nominal →
RMS 5.3 bp; the 3 DBRI linkers carried −189…−202 bp residuals). Tested offline
(`validation/tests/test_bund_liquidity.py`, `germany/tests/test_fetch_bund_prices.py`).

**Wiring** — `build_noise` is refreshed daily (`scheduler/daily_update.py`), staged to
Supabase as source `bund_noise` (`scheduler/supabase_export.py`), attached to
`DE.bund_noise` in `viz/thin_daily.py`, and rendered as a guarded **"Market quality"**
Atlas panel (`viz/artifact_template.html`) mirroring the US TIPS-noise panel, with a
JS-computed callout (mean / peak / latest, in bp).
