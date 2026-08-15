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

---

## Update — the TIPS liquidity factor (5-factor joint model)

Added a 5th latent factor loading on the **real (TIPS)** measurement only, via a
maturity-decaying loading β(τ)=(1−e^{−γτ})/(γτ) (distinct from the flat real level, so
identified). It captures TIPS illiquidity; the **clean** real yield/TP excludes it and
the liquidity wedge is reported separately. Re-estimated on the same US 2004+ panel.

| 10y quantity | 4-factor | **5-factor (+liq)** | DKW | 5f corr vs DKW |
|---|---|---|---|---|
| loglik | 14723 | **15980** | — | — |
| real term premium | +1.68% | **+0.06%** | +0.79% | **0.93** |
| inflation-risk premium | −0.17% | +0.38% | +0.11% | **−0.65** |
| expected inflation | +2.59% | +2.15% | +2.80% | **0.86** |
| TIPS liquidity premium | — | +0.10% | +0.44% | **0.65** |

**What the liquidity factor fixes (real improvements):**
- **Much better fit** (loglik 14723 → 15980) — TIPS illiquidity was a large, real,
  previously-unmodelled source of misfit (the 4-factor `h_r`=31 bp was the tell).
- **Real-TP co-movement rises to 0.93** (from 0.84) and it is pulled *down* off the
  inflated 4-factor level toward DKW's neighbourhood.
- **Expected-inflation co-movement jumps to 0.86** (from 0.38), level ~2.2% (DKW 2.8%).
- **A recovered liquidity series that tracks DKW's** (corr 0.65) — spiking in
  2008/2020 as it should.

**What it still does not fix (the honest wall):**
- **The IRP is still wrong** — the 10y IRP now *negatively* correlates with DKW
  (−0.65). The IRP is the small residual difference of two large components (nominal
  TP, clean real TP), and without an external anchor the model cannot pin the *level
  split* between the real term premium and the liquidity premium: it overshoots
  (clean real TP → ~0, below DKW; liquidity premium ~0.1% vs DKW's ~0.44%). Good
  co-movement, wrong decomposition of the level.

**Refined verdict — GO on the model, but the IRP needs a survey anchor before
production.** The joint + liquidity model is a clear, coherent improvement (fit,
real-TP and expected-inflation co-movement, a DKW-tracking liquidity series). The
remaining gap is exactly what DKW has and we do not: **survey inflation expectations**
(SPF / Blue Chip) to anchor the level split and identify the IRP. That is a **US-only
overlay** (the portability caveat in `FINDINGS_phase3.md` Strand E) — acceptable for a
US inflation product. Sequencing: add the survey anchor and re-validate the US IRP vs
DKW; only then take the joint model to the **UK** (which additionally needs RPI→CPI and
an LDI/convenience factor). The production IRP stays the separate-AFNS number until the
US joint IRP validates.

---

## Update — the ATSIX survey anchor (expected-inflation identification)

Blue Chip is paywalled, so we anchor to the Philadelphia Fed's **ATSIX** (Aruoba Term
Structure of Inflation Expectations) instead: free, monthly (1998+), a full term
structure of *survey-based* expected inflation that itself blends SPF + Blue Chip +
Michigan through a Nelson-Siegel state-space model — i.e. the same survey information
DKW use, delivered ready-to-use. `infexp{h}` is the horizon-average expected inflation,
matching the model's `expected_inflation(τ)` object; horizons map exactly to the grid
(2y=`infexp24` … 10y=`infexp120`). It enters as **extra EKF measurement rows** on the
model's expected-inflation term structure (survey error SD `h_s`, analytic Jacobian
`d E[max(s,ELB)]/dμ = Φ(d)`, verified vs finite differences to 7e-12). Estimated on the
same US 2004+ panel; `atsix_panel` covered all 271 months.

### Result at 10y (correlation over the sample; E[infl] vs ATSIX, the anchor target)

| 10y quantity | 4f | 5f (+liq) | **4f +survey** | **5f (+liq) +survey** | DKW |
|---|---|---|---|---|---|
| real term premium corr(DKW) | 0.84 | 0.93 | 0.97 | **0.98** | — |
| real term premium (latest) | +1.68% | +0.06% | +1.05% | +1.28% | +0.79% |
| TIPS liquidity prem corr(DKW) | — | 0.65 | — | **0.77** | — |
| expected inflation corr | 0.38(DKW) | 0.86(DKW) | 0.71(ATSIX) | 0.67(ATSIX) | — |
| inflation-risk prem corr(DKW) | −0.14 | −0.65 | −0.68 | **−0.21** | — |
| inflation-risk prem (latest) | −0.17% | +0.38% | −0.20% | **−0.03%** | +0.11% |

`h_s` converged to **~6 bp** (5f) — the optimizer *chose* to trust the survey tightly.
At 5y the survey pins E[infl] to ATSIX at **corr 0.95**.

### What the survey fixes (real improvements)
- **Expected inflation is now anchored, by construction**: E[infl] tracks ATSIX at
  0.95 (5y) / 0.67 (10y), mean gap ≈ 0. (Corr vs *DKW* is lower, ~0.5, only because
  ATSIX and DKW's own expected-inflation series genuinely differ — ATSIX is smoother;
  we chose ATSIX as the anchor, so we track ATSIX.)
- **Real term premium co-moves with DKW at 0.98** (5f+survey), up from 0.84 (4f) /
  0.93 (5f) — the best of any configuration — and its *level* is pulled to within
  ~0.1 pp of DKW.
- **The TIPS liquidity factor is better identified with the anchor present**: liquidity
  corr(DKW) rises to **0.77** (from 0.65), spiking in 2008/2020 as it should. The
  survey frees the yields to load illiquidity onto the liquidity factor rather than the
  real level.
- **The IRP level lands in DKW's neighborhood**: 5f+survey 10y IRP −0.03% vs DKW
  +0.11% (was +0.38% for 5f-no-survey, −0.17% for 4f) — a small, near-zero premium, as
  DKW's is.

### The honest wall — the IRP's *dynamics*, and why expectations data can't fix it
- **The IRP's high-frequency correlation with DKW stays weak** (−0.21 at 10y, −0.06 at
  5y for 5f+survey; the 4-factor case is worse, −0.68). The IRP is the small residual
  of two large components (nominal TP, clean real TP), so small errors in each dominate
  it.
- **The structural reason is decisive and worth stating**: the TIPS liquidity premium
  is a **Q-side (pricing) wedge** on observed real yields — `observed real yield =
  clean real yield + liquidity`, one equation with two unknowns per date/tenor. A
  survey on *expectations* (ATSIX InfExp or Real, FRB/US PTR/RSTAR — all P-side) pins
  the expected-rate paths but **cannot separate two pricing-side objects**. The
  4-factor+survey case makes this visible: with no liquidity factor the IRP =
  observed-breakeven − E[infl] is fully identified, but the observed breakeven carries
  the TIPS-illiquidity spikes (2008, 2020) — which crater the IRP exactly when DKW's
  liquidity-cleaned IRP does not — hence the *negative* correlation. That is the
  liquidity signature, direct.
- The 5-factor liquidity factor mitigates this (IRP corr −0.65 → −0.21, level → DKW's
  neighborhood) because it *does* carry a pricing-side loading, but yields+survey alone
  identify it only weakly. Fully pinning the IRP/liquidity split needs a **direct TIPS
  liquidity observable** (on/off-the-run TIPS spread, TIPS-vs-inflation-swap basis, or
  DKW's own liquidity series as a prior — the last legitimate for production, circular
  for validation), *not* more expectations data.

*(Aside considered and rejected as non-additive: the FRB/US `LONGBASE` dataset has
`PTR` (10y survey inflation) and `RSTAR`/`RRTR` (equilibrium real rate). These anchor
the same P-side that ATSIX already covers — more redundant expectations, plus a
quarterly dependency to maintain — and, being P-side, cannot touch the liquidity split.
Skipped.)*

### Refined verdict — survey anchor is a clear GO; ship the level, caveat the dynamics
The ATSIX anchor materially improves the joint model: it identifies expected inflation
(→ ATSIX), delivers the best real-TP co-movement with DKW (0.98) and a well-tracking
liquidity series (0.77), and brings the IRP **level** into DKW's neighborhood. For a US
inflation product this is production-usable for **expected inflation, the real term
premium, and a defensible IRP level**. The IRP's **time-variation** remains
under-identified (corr ~−0.2 vs DKW) and is gated on a direct liquidity observable, not
survey data — that is the one remaining ingredient before the joint IRP dynamics match
DKW. UK extension stays deferred (RPI→CPI + LDI/convenience factor on top of this).
Config: `python -m us.joint_us --liq --survey` (best), warm-started from `--liq`.
