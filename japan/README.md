# /japan — Japanese (JGB) term premium

Third country in the build sequence. Same engine as `/us` and `/germany`
(`/model` is **not** forked) — only data ingestion and two Japan-specific
handling steps (the ELB and the YCC downweight) differ.

## Data
- **Source:** Ministry of Finance (MoF), *Interest Rate (JGB) — historical data*
  (`jgbcm_all.csv`). Yields are published **directly by maturity** (unlike the
  GSW/Bundesbank Svensson parameters), so there is **no curve reconstruction** —
  `fetch_mof.py` parses, converts Japanese-era dates and percent→decimal, and
  subsets to the grid. Shift-JIS encoded.
- **Grid:** `[1, 2, 3, 5, 7, 10, 20, 30]` years (`config.COMMON_GRID`). No 3m/6m
  (MoF publishes ≥1y here; sub-1y T-bills are a documented refinement). 15/25/40y
  are held out as thin/redundant.
- **Ragged history.** Availability differs by tenor (user-confirmed vs MoF):
  1-9y from 1974-09-24; **10y from 1986-07-05**; 20y from 1986-12-01;
  15y 1991-08-30; 30y 1999-09-02; 25y 2004-03-22; 40y 2007-11-06. We start the
  production sample at **1986-07** (10y available), which covers the **1989-90
  bubble peak/collapse** and the **1998 Trust Fund Bureau (VaR) shock** the user
  asked for. The long end is genuinely ragged (20y joins 1986-12, 30y 1999-09) —
  handled by the EKF's per-date maturity masking, as for the early US long end.
  Extensible back to 1974 using ≤9y.
- Egress: needs `www.mof.go.jp` allowlisted (self-healing to the raw cache).

## Effective lower bound
NIRP took the BoJ policy rate to **−0.10%** (2016-02) and short JGB yields floored
around −0.1% to −0.35%. The charter's −0.75% is a *reversal-rate* concept, deeper
than the revealed floor — so we **start near the NIRP rate (−0.20%) and estimate
the ELB by ML** (`config.ELB`, `ELB_BAND`). Japan is the best-identified ELB of
the four markets: rates sat at the bound for years.

## QQE / Yield Curve Control — the Japan-specific challenge
Under **Yield Curve Control** (2016-09-21 → 2024-03-19) the BoJ *price-pegged* the
10y JGB. A pegged yield is an administered price, not a market clearing price, so
it breaks the no-arbitrage cross-section the AFNS relies on — feeding it in raw
would drag the estimated expected-rate path toward the peg and corrupt the term
premium at **every** tenor.

Handling (chosen with the user — "downweight pegged maturities"):
- **Downweight, don't drop.** `config.ycc_downweight(dates, grid)` inflates the
  *measurement variance* on the pegged tenors (10y and its arbitrage-tied 7y
  neighbour) over the YCC window — heaviest while the band was tight (±0.10%),
  tapering as it widened (±0.25→0.50→1.0%) and increasingly market-driven. The
  filter then effectively ignores the peg as a market price when inferring the
  latent factors, while the long end (20y/30y — never pegged) still identifies the
  curve. This flows through `run_shadow`/`run_production_tp` via `var_mult`.
- **Report TP against observed yields at all tenors.** The downweight only affects
  *estimation* of the expected-rate path. The term premium is still reported as
  `observed-tracking fitted yield − expected floored short rate` at every tenor,
  including the pegged 10y — i.e. the **realised, policy-suppressed premium**. As
  the user expected, this is **negative out to 10y under QQE/YCC and normalises**
  as the band widened, YCC ended, and the BoJ shifted toward QT.

## Two deliverables (mirroring the US / Germany)
- **Modern production TP** — `japan/production_tp.py::run_japan_production`:
  shadow-rate AFNS + Bauer-Rudebusch-Wu bias-correction on the ragged 1986+ MoF
  panel, ML-estimated ELB, YCC downweight. Headline, cross-country-comparable
  series (term premium at all tenors). See `validation/FINDINGS_japan.md`.
- **Historical exhibit** — the 1974+ (≤9y) extension for the bubble era is a
  documented extension of the same driver (start `1974-09-24`, grid ≤9y).

## Validation
No Japanese SF-Fed-equivalent term-premium series exists, so validation is
**internal** (shadow rate goes negative through the NIRP/YCC era and collapses to
plain AFNS away from the bound; long-end 20y/30y stay market-driven under YCC),
**ML-ELB vs the NIRP −0.10% anchor**, the **YCC downweight test** (downweighting
≈ dropping the pegged datum, `model/tests/test_ekf.py`), and **directional** vs
the Krippner Japan shadow short rate where reachable
(`validation/shadow_validation.py`, `economy="JP"`).
