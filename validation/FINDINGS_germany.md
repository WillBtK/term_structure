# Germany (Bund) term premium — findings

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
converged, **yield-fit MAD 3.42 bp**, λ=0.413. BRW bias-correction pushes ρ
0.988 → 0.9999. ELB fixed at Praet's −1.0% (validated below).

**Headline term-premium levels (bias-corrected):**

| | 10y TP | 2y TP |
|---|---|---|
| latest (2026-07) | **+0.26%** | +0.04% |
| mean 2016-2019 | +0.19% | — |

The German 10y term premium today (~0.26%) is **below the US (~0.75%)** —
consistent with Bund safe-haven / collateral demand compressing the premium.

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
series): a single-regime bias-corrected level and regime-switching regime dating
over the Deutsche-Mark + euro eras. Results appended when the (expensive Kim-filter)
regime estimation completes.

## Validation summary
No German SF-Fed-equivalent term-premium benchmark exists, so validation rests on:
(1) reconstruction accuracy < 0.6 bp vs published Bundesbank spots; (2) the ELB
profile validating Praet's −1.0%; (3) the internal shadow-rate/ELB behavior above;
(4) sensible modern levels (10y ~0.26%, below the US). External euro-area
directional benchmarks (Wu-Xia / Krippner) were not readily loadable this session.
