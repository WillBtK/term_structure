# Global term-premium factor — findings

**Question.** Is there a common *global* term-premium cycle across the five bond
markets, and is its complement — each market's **local** term premium (the residual) —
a portfolio-relevant relative-value signal? This is the one genuinely novel,
portfolio-facing object we can build on top of the five single-curve TP series (the rest
of the project is a careful *implementation* of AFNS + shadow rate). Distinct from Del
Negro et al. "Global trends in interest rates", which decomposes the secular *r\** trend
(slow, not tradeable); this is the *cyclical* risk-compensation factor.

**Approach.** A reduced-form **dynamic factor model** (`model/global_factor.py`) on the
five markets' 10y term-premium series (from `viz/data_daily.parquet`) — it does **not**
re-estimate any yield curve:

    z_{i,t} = λ_i·G_t + e_{i,t}  (diagonal R);   G_t = φ·G_{t-1} + u_t,  q = 1−φ² (Var G =1)

standardized per market over the balanced overlap; Kalman filter + RTS smoother + EM,
ragged-aware (markets enter at different dates), sign-anchored so G loads positively on
the US. Local residual = `TP_i − (α_i + β_i·G_t)`.

## Result 1 — the global factor is real and dominant

DFM on the 10y panel (full history 1961+, balanced 5-market cross-section from 1997-08):

| market | β (raw) | global variance share |
|---|---|---|
| US | +0.011 | 83% |
| UK | +0.013 | 86% |
| DE | +0.019 | ~100% |
| JP | +0.010 | 90% |
| CH | +0.008 | 77% |

**~87% of each market's 10y term-premium variation is a single common factor.** The DFM
factor matches a static PCA first component at **0.985** (robustness), and `G_t`
correlates **+0.96 with the US 10y term premium** — i.e. the global bond-market term
premium is, unsurprisingly, US-anchored. (`φ ≈ 1.0000` at daily frequency — the factor
is effectively a random walk / slow common *level*; the AR(1) dynamics are near-trivial
at daily frequency and irrelevant to the decomposition, which rests on the loadings.)

**Construct validity** — what G co-moves with:
- `corr(G, US 10y TP) = +0.96` — it is the common bond cycle.
- `corr(G, broad dollar [DTWEXBGS]) = −0.60` — a strong dollar coincides with a
  compressed global term premium (an economically sensible, non-trivial link).
- `corr(G, VIX) ≈ 0` — **not** an equity-vol story; this is a rates/dollar object, not a
  risk-off proxy. (Conceptually the term-premium analogue of the Miranda-Agrippino–Rey
  "global financial cycle", but distinct from equity-vol GFC measures.)

## Result 2 — the local residual is a portfolio-useful relative-value signal

Relative-value diagnostic (`validation/global_factor_rv.py`): construct a duration-hedged,
cross-sectionally-demeaned (≈ currency-neutral) monthly excess return from the yields
alone (`carry − duration·Δy`, demeaned across markets), and test whether the local
residual (positive = *cheap* vs the global cycle) predicts next-month relative bond
returns. **Balanced 1997-08 → 2026-07, monthly, 5-name cross-section:**

| signal | IR | hit-rate | pooled rank IC | cum. relative P&L |
|---|---|---|---|---|
| **local residual** | **0.59** | 52% | **0.10** | **+41%** |
| naive carry (term spread) | 0.28 | 54% | 0.05 | +22% |

Pooled OLS of next-month relative return on both signals (cross-sectionally demeaned):

    rx_{t+1} = +0.37·residual (t = +3.6)  +  0.01·carry (t = +0.3)

**The local residual roughly doubles carry's IR and information coefficient, and
*subsumes* carry** — once the residual is in the regression, carry adds nothing
(t = +0.3). So "cheap vs the global term-premium cycle" carries genuine relative-value
information beyond the naive term-spread carry that global-bond investors typically use.

## Verdict — GO (a real, portfolio-relevant result)
Unlike the joint-model IRP (data-blocked) and the breakeven split (too noisy to ship),
this validates cleanly: a dominant global term-premium factor (87% common, 0.96 with US
TP, −0.60 with the dollar) and a local residual that is a statistically strong
(t = 3.6), economically sensible relative-value signal beating and subsuming carry. This
is the most defensible "innovation-flavoured" output in the project — and it answers the
portfolio question in the affirmative.

## Honest caveats (do not oversell)
- **Diagnostic, not a strategy.** The backtest uses synthetic yields-only excess returns
  (`carry − duration·Δy`, no rolldown, no real total-return data), cross-sectional demean
  as a *first-order* currency/level hedge (exact only for equal durations; residual
  leakage from duration dispersion is ~1bp vs a ~4.5% raw shock), a 5-name cross-section,
  monthly rebalancing, **no transaction costs, no real hedging**. Read IR/IC as "does the
  residual carry information?", not as a net Sharpe.
- **Params are full-sample**; the *signal* uses the real-time **filtered** factor (no
  state look-ahead), but the loadings carry mild in-sample look-ahead — a rolling re-fit
  would be the production-grade version.
- **The residual inherits each market's TP estimation noise** (mean-reversion-floor
  assumptions, extrapolated pre-sample params).
- **G is well-identified only where markets overlap** (US-dominated pre-1988; balanced
  cross-section from 1997-08). `G` is standardized/latent — interpret movements and
  correlations, not the absolute level.
- **10y only, single global factor** — a curve-wide global level/slope (run at 2y/5y/30y)
  is the natural extension; deliberately not a joint multi-tenor factor here.

## Status & next (research-first)
Shipped as research: `model/global_factor.py` (estimator, 7 tests), `validation/
global_factor_rv.py` (RV + construct-validity, 4 tests), this findings doc. **Not yet
wired into the Atlas / daily pipeline / Supabase** — that productionization (a "Global
cycle" panel with the factor + per-market local-residual RV chart; a weekly EM fit /
daily filter-forward in the scheduler; a Supabase table) is gated on this GO verdict and
is the recommended next step.
