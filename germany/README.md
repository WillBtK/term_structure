# /germany — German (Bund) term premium

Second country in the build sequence. Same engine as `/us` (`/model` is not
forked) — only data ingestion differs.

## Data
- **Source:** Deutsche Bundesbank, *Term structure of interest rates on listed
  Federal securities (method by Svensson)* — SDMX REST API, flow `BBSIS`, sector
  S1311 / instrument A604. Six Svensson parameters (BETA0-3 in percent, TAU1-2 in
  years — same convention as GSW). `fetch_bundesbank.py` pulls them and
  reconstructs the common 8-point grid (3m-10y) with the shared `us.reconstruct`
  Svensson math.
- **History:** monthly from **1972-09** (Deutsche-Mark era), out to 10y; daily
  from 1997-08 (`freq='D'`).
- **Cross-check:** reconstruction matches the Bundesbank's own published spot
  yields to **< 0.6 bp** at every maturity (`cross_check_against_published`).
- Egress: needs `www.bundesbank.de` + `api.statistiken.bundesbank.de` allowlisted.

## Effective lower bound
Central value **−1.0%** (ECB reversal-rate estimate; provenance: discussions with
Peter Praet), with a −0.5% to −1.5% sensitivity band (`config.ELB`, `ELB_BAND`).
Unlike the US, Bund yields went clearly negative (2014-2022), so the ELB is **well
identified by ML** — we estimate it as a check on the −1.0% calibration.

## Two deliverables (mirroring the US)
- **Modern production TP** — `germany/production_tp.py::run_germany_production`:
  shadow-rate AFNS + Bauer-Rudebusch-Wu bias-correction on the **euro era
  (1999+)**. The modern sample keeps Deutsche-Mark-era dynamics out of today's
  estimate (the German analogue of the US 1987+ choice). Headline,
  cross-country-comparable series. See `validation/FINDINGS_germany.md`.
- **Historical / stagflation exhibit** — `germany/run_history.py`: regime-switching
  regime dating + a single-regime bias-corrected level over the full 1972+ history
  (DM + euro eras). Separate and clearly labeled; not joined to the modern series.

## Validation
No German SF-Fed-equivalent term-premium series exists, so validation is
**reconstruction accuracy** (< 0.6 bp vs published spots), **internal** (shadow
rate goes negative during the ECB negative-rate era and collapses to plain AFNS
away from the bound; shadow vs the actual short rate), and **ML-ELB vs the −1.0%
calibration**. External directional benchmarks (euro-area Wu-Xia / Krippner) are
used where reachable.
