# Joint nominal/real AFNS — US spike findings (vs DKW)

**Question.** Does a single-kernel *joint* nominal/real AFNS extract a better real term
premium and inflation-risk premium (IRP) than the current "two separate AFNS,
subtract" approach (`us/inflation.py`; IRP = nominal TP − real TP)?

**Setup.** `model/joint_afns.py` — 4-factor state [L_N, L_R, S, C], nominal short
= L_N+S, real short = L_R+S, shared slope/curvature and NS decay; two-block EKF
(nominal Krippner-floored, real plain affine), per-curve measurement error, diagonal
K^P with the ~3.5y mean-reversion floor on the P legs. Estimated on the matched US
GSW-nominal + TIPS-real monthly panel, **2004+ (271 months)**. Benchmarked against
**DKW** (D'Amico-Kim-Wei, the Fed Board's joint model *with* a TIPS liquidity factor
and surveys) via `validation/benchmarks.py`.

## Result (10y, latest; correlation vs DKW over the sample)

| 10y quantity | Joint | Separate-AFNS | DKW | corr(joint, DKW) |
|---|---|---|---|---|
| nominal term premium | +1.51% | +1.18% | +0.90% | — |
| **real term premium** | +1.68% | +1.25% | +0.79% | **0.84** |
| **inflation-risk premium** | −0.17% | −0.06% | +0.11% | **−0.14** |
| expected inflation | +2.59% | — | +2.80% | 0.38 |

(5y: real-TP corr 0.79, IRP corr −0.01, E[infl] corr 0.44. `success=False` from
Nelder-Mead — as everywhere in this project the flat QML surface means we use the
smoothed fit, not formal convergence; loglik 14723.)

## What works
- **The framework is coherent.** One no-arbitrage kernel prices both curves; the
  accounting identity **breakeven = expected inflation + IRP** holds exactly (tested),
  removing the matched-sample fudge the subtract approach needs.
- **Per-curve measurement error is correctly identified**: `h_r ≈ 31 bp` vs
  `h_n ≈ 3.6 bp` — the model sees TIPS as ~9× noisier than nominals, i.e. it *detects*
  the TIPS illiquidity even though it has no factor to price it.
- **Real term premium co-moves well with DKW** (corr 0.84 at 10y, 0.79 at 5y).
- **Expected inflation is sensible** — ~2.4–2.6%, near target + a small wedge (DKW ~2.8%).

## What does not (yet) work — the decisive finding
- **The IRP is not improved.** Its correlation with DKW is ≈ 0 (−0.14 at 10y, −0.01
  at 5y): the joint IRP is dominated by identification noise, exactly the
  weakly-identified cross-curve object the pre-build exploration flagged. The joint
  kernel *alone* does not deliver a trustworthy IRP.
- **Real TP runs high** (+1.68% vs DKW +0.79%). DKW's real TP is lower precisely
  because its **TIPS liquidity factor** strips the illiquidity that our yields-only
  model instead loads onto the real term premium (and hence, by difference, distorts
  the IRP). Our own `h_r`=31 bp is the smoking gun that this liquidity is present and
  unmodelled.

## Verdict — conditional GO; do NOT extend to the UK yet
The joint model is the right foundation and a genuine improvement in *coherence* over
subtract (single kernel, exact breakeven identity, correct liquidity *detection*,
strong real-TP co-movement). But **yields-only, it does not beat subtract on the IRP**
— the number the joint model was meant to fix. The missing ingredient is DKW's:
a **TIPS liquidity factor** (a 5th state loading on TIPS yields only), which should
pull the real TP down toward DKW and clean up the IRP.

Recommended next step (US, before any UK work): add the liquidity factor and re-validate
the IRP against DKW. Only once the US IRP tracks DKW should the joint model go to the
**UK**, which additionally needs an explicit **RPI→CPI** wedge and an **LDI/convenience**
factor (per `FINDINGS_inflation.md`) — i.e. more identification burden, not less.

Until then, the production IRP stays the documented separate-AFNS "subtract" number
(US clean, UK caveated); the joint model lives in `model/joint_afns.py` / `us/joint_us.py`
as the validated foundation for the liquidity-factor extension.
