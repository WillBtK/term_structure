# Japan expected-rate neutral anchor — input files

Drop the survey / neutral-rate CSVs here. They are read by `japan/neutral_anchor.py`
and used as a **shifting endpoint** for Japan's expected-rate path (fixing the
deflation-era θ-anchor that overstated the modern term premium). The pipeline is
**inert without these files** — Japan simply falls back to the yields-only anchor —
so nothing here is required for the build to run.

Each file is a two-column CSV: **`date,value`**, `date` as `YYYY-MM-DD` (any regular
date works — the release day, or a period-end), `value` in **percent** (e.g. `1.75`).
Releases are intermittent/irregular; the loader carries the last observation **forward**
(never interpolates toward a future release) and aligns onto the model's dates.

| File | Series | Source | Span | Role |
|---|---|---|---|---|
| `consensus_shortrate_6_10y.csv` | Consensus Economics **3-month interbank rate, 6–10y-ahead** forecast | Consensus (proprietary) | 2017+ | **Primary** nominal anchor, used directly within a staleness window |
| `consensus_cpi_6_10y.csv` | Consensus Economics **CPI, 6–10y-ahead** forecast | Consensus (proprietary) | 1990+ | Inflation leg π^e of the public bridge |
| `ferreira_rstar_jp.csv` | **Longer-run real neutral rate, Japan** | Fed FEDS Note "Real-Time Global Longer-Run Neutral Rates" (Davis-Ferreira / Ferreira-Shousha) | 1960+ semi-annual | Real leg r* of the public bridge |
| `tankan_infl_5y.csv` | **Tankan 5-years-ahead** enterprise inflation expectation | Bank of Japan, "Inflation Outlook of Enterprises" | 2014+ quarterly | Public π^e fallback that maintains the anchor forward when Consensus lapses |

## How they combine (degradation ladder — see `neutral_anchor.py`)
1. **Consensus 3m-interbank 6–10y** directly (2017+), carried forward up to ~1.5 releases.
2. **Public bridge** `n = r*(Ferreira) + π^e`, where `π^e` = Consensus CPI (1990+),
   extended past its last release by Tankan (level-bridged), else Tankan, else the BoJ
   2% target; the bridge is level-calibrated to the Consensus short-rate over the overlap.
3. **Carry-forward** between releases.
4. **No files** → yields-only (constant-θ) fallback, logged.

## Provenance / licensing
Consensus Economics is **proprietary** — commit the two `consensus_*.csv` only to this
**private** repo, and we publish only the derived term premium, never the raw survey.
Ferreira (Fed) and Tankan (BoJ) are public. Committing the CSVs keeps the atlas
self-contained and reproducible without any live or subscription access.
