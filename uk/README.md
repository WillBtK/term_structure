# /uk — UK (gilt) term premium + the real curve

Fourth country. Same engine as US/Germany/Japan (`/model` unforked); only the data
layer differs. The UK leg also carries the **real (index-linked) curve** and the
**inflation-risk-premium** decomposition (shared with the US — see below).

## Data — Bank of England Government Liability Curve (GLC)
- **Source:** BoE published fitted government yield curves (VRP spline), as *spot
  rates directly* — no parametric reconstruction (like Japan's MoF). Three curves
  from the same month-end archive: **nominal** (gilts), **real** (index-linked),
  **implied inflation** (breakeven). `fetch_boe.py` downloads the era-split Excel
  workbooks, parses the `4. spot curve` sheet, concatenates, percent→decimal.
  Self-healing download URL (discovered from the yield-curves page).
- **Coverage:** nominal month-end from **1970** (grid 0.5–20y; 20y ragged early,
  extends to 40y from 2016); real/inflation from **1985**, starting at 3y (no
  RPI-lag short end → real grid `[3,5,7,10,20]`).
- Needs `bankofengland.co.uk` egress (already enabled).

## Effective lower bound
UK is the **mildest ZLB** of the four: Bank Rate floored at 0.10% (2020-21) and
short gilt yields barely went negative, so the ELB is weakly identified. We **fix**
it at −0.35% (like the US) rather than ML-estimate; the shadow-rate machinery
barely binds. The **real** curve has no ELB (no "real cash") → plain AFNS.

## Deliverables
- **Nominal production TP** — `uk/production_tp.py::run_uk_nominal`: shadow AFNS +
  BRW on the **inflation-targeting era (1992-10+)**, the regime-clean modern
  sample (UK analog of US-1987 / Germany-1999). The pre-1992 stagflation era is a
  separate historical exhibit.
- **Real production TP** — `run_uk_real`: plain-AFNS real term premium on the
  index-linked curve (no ELB).
- **Inflation-risk premium** — `us.inflation.inflation_decomposition` (country-
  agnostic): IRP = nominal TP − real TP; breakeven = nominal − real yield; implied
  expected inflation = breakeven − IRP. Carries the **RPI indexation-lag** and
  **liquidity-differential** caveats. See `validation/FINDINGS_inflation.md`.

## Validation
No UK SF-Fed-equivalent TP benchmark; validation rests on BoE data coverage,
yield-fit MAD, sensible nominal levels vs the other three markets, and the
breakeven curve tracking realised RPI/CPI inflation and known episodes (2022
inflation spike). External directional check vs the Krippner UK shadow rate where
reachable (`economy="UK"`).
