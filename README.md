# Cross-Country Term Premium Estimation

## Objective

Estimate government bond term premia consistently across the **United States,
Germany, Japan, the United Kingdom, Switzerland and Canada**, using a single model
specification across all six markets, while correctly handling periods of zero or
negative short-term rates (Japan since the mid-1990s; Germany/euro area since 2014;
Switzerland since 2015; Canada grazed its ~0.25% lower bound in 2009–10 and 2020–21).

One model, one codebase, six markets. **Do not fork the model per country** — only the
data ingestion and each market's sample window differ.

## What this produces

| deliverable | what it is |
|---|---|
| **The Atlas** (`/viz`) | self-contained HTML: the daily term-premium dataset, decompositions, cross-checks, the global cycle |
| **The scenario sheet** (`/scenarios`) | macro paths in → ten-year yield and bond return out, four markets |
| **Exports** (`/exports`) | derived monthly series handed to other repos (clone — this repo is private) |
| **Decks** (`/presentation`) | the twelve-slide deck and the two-page executive summary |

Everything downstream rests on one object: a daily, six-market decomposition of the
nominal yield into an **expected-rate path** and a **term premium**.

---

## The model

**Arbitrage-free Nelson-Siegel (AFNS)**, extended with the **Christensen-Rudebusch
shadow-rate specification** for the zero/negative lower bound. Independent-factor
spec (diagonal `K^P`, diagonal `Σ`) — the same specification the SF Fed publishes.

- **Production engine: the Python EKF** (`model/ekf.py`). It is the workhorse for
  everything shipped because it exposes `K^P` (needed for bias correction), handles
  ragged early maturities by masking per date, supports per-date/per-maturity
  measurement variance (Japan's YCC downweight), and can estimate the ELB.
- **`TermStructureModels.jl`** (MIT; JSZ constrained by the AFNS restriction of
  Christensen-Diebold-Rudebusch 2011) is retained as the *validated plain-AFNS
  benchmark* the Python path was checked against — not the daily path. See
  `model/README.md`, `validation/FINDINGS_phase1e.md`.
- Shadow-rate extension follows Christensen, Jens H. E., and Glenn D. Rudebusch. 2015.
  "Estimating Shadow-Rate Term Structure Models with Near-Zero Yields." *Journal of
  Financial Econometrics* 13 (2): 226–59. The option-based (Krippner) forward is in
  `model/shadow_rate.py`.

We initially built on the Python repo `snejens/arbfree_dynamic_nelson_siegel` and
replaced it: it tracked the SF Fed benchmark far worse (2y term-premium correlation
0.43 vs 0.85) and carried no license.

## How the term premium is actually computed

The premium is a residual — what the market pays above the expected path — so it is
only as good as the expectations leg. Four things go into it, and three of them exist
because the raw yields-only estimate of that leg is *wrong in a known direction*.

**1. The fitted yield (Q side).** The arbitrage-free AFNS yield,
`y(τ) = B(τ)'x − C(τ)/τ`, with the shadow-rate floor applied to the forward curve
(`model/shadow_rate.py`: `f(τ) = ELB + (f_s(τ)−ELB)Φ(d) + ω(τ)φ(d)`, a Gaussian call
struck at the ELB). Away from the bound this collapses to plain AFNS.

**2. The expectations leg (P side).** The average expected **floored** short rate,
`model/term_premium.expected_floored_short_rate_yields`. With `r = max(s, ELB)` and
the shadow rate `s = ρ'x` Gaussian under P,

```
E^P_t[max(s_{t+u}, ELB)] = ELB + (μ_u − ELB)·Φ(d) + σ^P_u·φ(d),   d = (μ_u − ELB)/σ^P_u
ỹ_lb(τ) = (1/τ) ∫₀^τ E^P[max(s,ELB)] du
```

the P-measure analogue of the Krippner Q-forward. As `ELB → −∞` it collapses to the
plain-AFNS expectation, so the shadow premium and the plain premium coincide away from
the bound.

**3. `term premium = fitted yield − expectations leg`.** Per date, per maturity, on
each market's common grid. The identity `observed = expected + term premium` holds to
0.0 bp per observation in the published dataset.

### The expectations leg is not the raw ML output

A yields-only affine model's expected-rate/premium split hinges on the P-dynamics —
weakly identified, and near a unit root in any modern sample. **Neither the ML `θ` nor
the ML `K^P` is used as-is for the expectations leg.** Three deliberate substitutions,
all applied at decomposition time in `run_production_tp`:

| | production choice | what it fixes |
|---|---|---|
| **long-run mean `θ`** | **sample mean of the filtered factors**, not the ML transition `θ` | the ML `θ` is unidentified when the sample is near-unit-root, and can be nonsensical — Japan's implied a **~17% long-run short rate**. It is fine for the transition/filter; it must not drive the expected path |
| **mean reversion `K^P`** | **floored** at a ~3.5-year half-life (`mean_reversion_floor`) | ML lands at a near-unit root, collapsing the expected path onto *today's* short rate |
| **where it reverts to** | the **survey endpoint** `n(t)`, where a market has one | a regime-clean sample mean can still be stale (secular r\* drift) |

The Q side — the fitted curve, the ELB, the shadow short rate — is untouched by all
three, so applying them requires **no re-estimation**.

**The mean-reversion floor is the production decomposition**
(`run_production_tp(persistence="mrfloor")`, the default). The symptom that forced it:
yields-only ML put the level factor's half-life effectively at infinity, so
`E[avg short rate over 10y]` tracked the policy rate almost one-for-one and the 10y
premium **collapsed negative through the 2022–23 hiking cycle** — an artefact, not a
signal (the SF Fed Christensen-Rudebusch series stayed near +1% throughout). The floor
lifts every factor's mean-reversion rate to at least `κ_min`, calibrated on the US
against SF Fed to a ~3.5-year half-life; speeds already above it are untouched.

**Bauer-Rudebusch-Wu bias correction** (`model/bias_correction.py`, a Kilian bootstrap
of the VAR(1) persistence — yields-only, so portable) was the *original* fix for the
same problem and is why the short-end SF-Fed correlation went from ~0.14 to ~0.83. It is
now **superseded, not stacked**: `persistence="brw"` is a mutually exclusive comparison
branch, retained because it is the literature's answer, and it *worsens* the
policy-cycle over-tracking (it raises persistence, which is the wrong direction for this
symptom). Production runs `mrfloor`.

It is a **post-estimation decomposition change**: the shadow-rate AFNS fit, the ELB and
the shadow short rate itself are all unchanged, so nothing needs re-estimating when it
is applied. The authoritative numbers and the US-vs-SF-Fed calibration are in
**`validation/FINDINGS_mean_reversion.md`**.

> **Reading the older findings.** Term-premium levels quoted in the per-country
> `FINDINGS_*.md` notes predate this fix. Where a per-country note and this README
> disagree on a level, this README and `FINDINGS_mean_reversion.md` are current.

## Surveys: the shifting endpoint

Surveys are **in the core**, at decomposition time — not as a US-only overlay, and not
inside the estimation. The mechanism is a **shifting endpoint** (Kozicki-Tinsley,
Kim-Orphanides): a per-date long-run nominal short rate `n(t)` that the expected-rate
path reverts to, passed to `expected_floored_short_rate_yields(..., endpoint=)`.

Why it is needed is distinguishable from the regime problem:

* a **regime break inside the sample** → fixed by re-cutting the estimation window
  (below), not by an anchor;
* **secular drift in r\* within one regime** → *not* fixed by re-cutting, because a
  sample can be regime-clean and its mean still stale. Japan's 2000+ deflation sample
  implies a ~0% long-run short rate against a survey neutral near 2%; Germany's
  regime-clean 1984+ sample implies 2.84% against a euro-area neutral nearer 2.5%.

Anchoring re-targets **only where the path mean-reverts**. The fitted curve and the
entire Q side are untouched, so **no re-estimation is required**.

### The degradation ladder

Built so a market can be anchored with a **one-off, unmaintainable** proprietary survey
and stay live indefinitely afterwards:

1. a direct **6-10y-ahead short-rate survey**, used as-is within a staleness window;
2. a **public bridge** `n = r* + π^e`, level-calibrated to the direct survey over their
   overlap so the rungs join without a step;
3. carry-forward between irregular releases;
4. nothing usable → NaN → the model falls back to constant-`θ`, i.e. today's
   yields-only behaviour. **The pipeline never breaks on missing data.**

Rung 1 is the proprietary, frozen series; rungs 2–4 are public and auto-refreshable.
That asymmetry is the point: the expensive input can lapse and the anchor still tracks.

| market | survey era (frozen snapshot) | living carry-forward |
|---|---|---|
| US | SPF 2-10y forward, nominal `(10·BILL10 − y1 − y2)/8`, real net of the ATSIX forward — **1992+/2004+, living and maintained** | itself (BILL10 Q1-only; bridged by the NY Fed SME, 8 rounds/yr) |
| DE | EA 3m 6-10y, 2016-10+ | ECB SPF longer-term rate (hand-off −0.01pp) |
| UK | UK 3m 6-10y, 2017-01+ | BoE MPS neutral / 5y Bank Rate median (monthly) |
| CA | CA 3m T-bill 6-10y, 2017-01+ | BoC MPS long-term nominal neutral median (quarterly) |
| CH | CH 3m 6-10y, 2017-01+ | KOF π₅ᵧ (public API) + constant real (−0.14pp, 114-month overlap) |
| JP | JP 3m 6-10y, 2017-01+ | none public (ESP is membership-gated) — **glides to the BoJ-communicated 1.75–2.25% corridor, midpoint 2.0%**, across the staleness window, with no step |

Effect on the 10y premium at adoption: DE +0.37pp, CA +0.25pp, CH +0.25pp,
UK −0.23pp, JP −0.04pp. See **`validation/FINDINGS_anchors.md`**.

> **Proprietary inputs stay private.** The Consensus Economics rung-1 files are
> committed only to this private repo. Publish the derived premium, never the raw
> survey — this is why the repo is private and why `/exports` is clone-only.

## Regime detection

Used for two distinct jobs. The first is production-critical.

### 1. Setting each market's estimation window

A structural break inside the estimation window corrupts `θ` — the long-run anchor the
expected-rate path reverts to — so **every market is fit on a regime-clean modern
sample**. Where the history allows it, that start date is *dated by a fitted model*
rather than asserted from the policy record: a 2-regime Markov-switching AFNS (Kim 1994
filter; per-regime `θ`/`K^P`/`Σ`, common `λ`), estimated by QML on the full published
history, then smoothed to say which regime each month was in. Dating is on the **level**
regime, because `θ` is what a mis-cut sample corrupts.

| market | production sample | how the start was set |
|---|---|---|
| US | **1987-01+** | Kim filter on 1961+ confirms high-volatility confined to 1973–75 and 1979–82 |
| DE | **1984-01+** | break dated **1983-11** (anchor 11.94% → 2.52%, *one* switch in 647 months). Monetary union is **invisible** to the rate regime; the old 1999 start discarded 15 years of same-regime data |
| JP | **2000-05+** | break dated **May 2000**. 1986 — the first published-10y month — sat 14 years *inside* the old high-level regime |
| UK | **1995-11+** | last sustained high-anchor spell ends 1995-10, i.e. the gilt curve re-anchored ~3 years *after* inflation targeting was announced; the old 1992-10 start straddled the break by 37 months |
| CH | **1988-01+** | start of the published curve |
| CA | **1991-01+** | inflation targeting. **Not identified** by the filter — the curve begins only 5 years before the policy change and anchors come back unidentified (31–39%), moving between restarts, so this date rests on the policy record |

Two of the three tested markets **moved** as a result. See
**`validation/FINDINGS_regime_dating.md`**, `FINDINGS_japan_regimes.md`.

### 2. The historical / stagflation exhibit

The same machinery, run on the full 1961+ US history, answers "how high can the premium
go in a stagflation regime" — kept as a **separate, clearly labelled exhibit** and never
joined to the modern series (see below).

**Kim-filter validation.** Before any use, the filter was checked against an exact
path-enumeration filter (collapsing correct to ~1e-4 loglik / ~1e-5 regime probability),
against degenerate identical-regimes (flat at the stationary distribution), for
relabel-invariance, and against the Ang-Bekaert (2002) / Levant-Ma (2017) empirical
pattern (Volcker P(high-vol) ≈ 0.98 — captured *within* the 2-regime model, so no third
regime is needed). See `validation/FINDINGS_kim_validation.md`.

## The effective lower bound

The ELB is a **calibrated input, not an estimated parameter** where yields never
approached it — standard practice (Christensen-Rudebusch note it can be replaced with
"some appropriate, possibly time-varying, negative epsilon"; Bauer-Rudebusch 2016 and
Krippner 2015a both document that shadow-rate estimates are highly sensitive to this
choice). Treat it as a sensitivity parameter, not a fact.

| market | production ELB | band | basis |
|---|---|---|---|
| US | −0.35% | fixed | currency-storage-cost floor; ML weakly identified (barely touched the bound) |
| UK | −0.35% | fixed | as US; weakly identified |
| Germany | **−1.00%** | −1.5% … −0.5% | ML-identified; validates the Praet/ECB reversal-rate region |
| Japan | **−0.20%** | −0.4% … 0.0% | ML-identified; lands at the NIRP policy rate (−0.10%) |
| Switzerland | −0.75% | −1.0% … −0.5% | the revealed SNB floor |
| Canada | 0.00% | −0.5% … +0.25% | ML-refined but weakly identified — Canada only grazed the bound |

Use **one constant bound per market for the full sample**. Away from the bound the
shadow-rate model collapses to standard AFNS regardless of the bound chosen, so a
constant bound runs across the whole sample without distorting positive-rate periods and
never needs "switching on" at a date. If a time-varying bound ever becomes unavoidable
(shadow rate pinned to the floor with no room to reflect subsequent tightening), see
Kortela (2016), BoF DP 19/2016.

**ML estimation** (`ekf.estimate(..., estimate_elb=True)`) is the primary method where
rates went clearly negative (DE, JP) and the reporting basis is the calibration +
sensitivity band elsewhere. The revealed bound abroad (SNB/ECB/BoJ reached roughly
−0.75/−0.50/−0.10% without triggering large-scale cash conversion) is a better anchor
than a retail currency-storage calculation: the binding floor is the cost of
warehousing *institutional-scale* cash, which "digital cash" makes more, not less,
binding. Run each market at 2–3 values in its band and confirm the profile is stable in
shape before treating a run as final. See `validation/FINDINGS_phase3.md`.

## Pricing dislocation: the Hu-Pan-Wang noise gauges

Per trading day, fit a smooth curve to the yields of the **individual outstanding
securities** and take the residual dispersion (Hu, Pan & Wang 2013, *J. Finance*). When
security-specific liquidity is scarce, individual issues scatter further from any smooth
curve, so the "noise" rises. **Three** gauges, all free, all from public data:

| gauge | cross-section | source | daily series stored at |
|---|---|---|---|
| **US nominal** (`validation/ust_liquidity.py`) | coupon Treasuries | TreasuryDirect **FedInvest** end-of-day *prices* (`us/fetch_fedinvest.py`) | **`us/data_store/ust_noise_daily.parquet`** (versioned) |
| **US TIPS** (`validation/tips_liquidity.py`) | outstanding TIPS | FRED individual TIPS (`DTP*`, category 82), official API | `validation/data/tips_noise_daily.parquet` (cache) |
| **Bund** (`validation/bund_liquidity.py`) | Bunds/Bobls/Schätze | Bundesbank per-ISIN workbooks (`germany/fetch_bund_prices.py`) | `validation/data/bund_noise_daily.parquet` (cache) |

The **US nominal gauge is HPW's original object** — their measure is a nominal Treasury
one, and the other two are variations on it. It was the last one built because no free
source publishes per-security Treasury *yields*: FedInvest publishes **prices**, so the
price → yield step is ours (`model/bond_math.py` — coupon schedule, actual/actual
accrual, and inversion to the semiannual bond-equivalent basis). It carries two residual
definitions per day:

* **`noise_gsw`** (headline) — RMS deviation from the *published* GSW Svensson curve.
  The accurate one: on a synthetic cross-section priced exactly off a known curve (true
  noise zero) it reads **0.2 bp**, because it fits nothing and so has no functional-form
  error of its own. Also the stricter reading — GSW deliberately exclude on-the-run and
  first-off-the-run issues, so a rich on-the-run shows up as a real residual rather than
  being absorbed.
* **`noise`** (comparability) — residual standard error of our own Svensson fit,
  profiled over a grid of decay pairs. Fitting the level ourselves absorbs any constant
  offset between FedInvest end-of-day prices and the quotes GSW was built from, so this
  is the column that lines up with the Bund and TIPS gauges.

The shared fit is `validation/noise_core.residual_se` — one implementation for all
three, with the coupon regressed out and a degrees-of-freedom correction.

**Why they exist.** The joint model's inflation-risk premium needs the TIPS **liquidity
premium** identified separately from the real term premium, and that is a Q-side
(pricing) object survey data cannot pin. DKW publish such a series, but using it to
identify our model and then validating against DKW would be **circular**. These are free
of that. The TIPS gauge peaks at ~45 bp in Dec-2008 (the post-Lehman freeze) and sits
~8 bp now.

Limitation, stated honestly: this is a **crisis-frictions** measure, not the persistent
liquidity *level*, so it does not by itself close the IRP gap (below).

### Where the data lives, and why the three differ

The storage split is driven by whether history can be **re-fetched**:

* **US nominal** — FedInvest serves any past date, so the raw price panel
  (`validation/data/ust_individual_prices.parquet`, ~25 MB) is a **rebuildable,
  gitignored cache** and history is back-fillable over any window. Only the derived
  daily series is **versioned** (`us/data_store/`, explicitly un-ignored), so the daily
  CI job appends new days instead of re-crawling a decade every run.
* **Bund** — the Bundesbank index exposes only the most recent ~5 monthly workbooks, and
  its per-ISIN yield archive drops a security ~4 years after maturity. History therefore
  **cannot be re-fetched, only retained**, so the raw per-ISIN cross-section itself is
  the versioned store (`germany/data/bund_prices/individual_yields.parquet`); each run
  unions the freshly scraped months in and CI commits it.
* **TIPS** — FRED serves full history on demand, so both the raw and derived files are
  ordinary caches.

Note `treasurydirect.gov` is **egress-blocked from the dev sandbox**, so the US nominal
fetcher is exercised on the GitHub runner (`python -m us.fetch_fedinvest --probe` dumps
the endpoint that answers, the parsed columns and the reachable history depth).

---

## Two deliverables: the modern production TP and the historical exhibit

There is **no single 1961-present term-premium series** — deliberately. One model
spanning 1961-today estimates one `θ` and `K^P` over the whole span, pulling them toward
the 1970s–80s high-inflation world and **contaminating *today's* premium**. That is the
long-sample side of the sample-sensitivity tradeoff; the short-sample side is
finite-sample persistence bias. So the output splits by what each is good for.

**Modern / production TP** — `us/production_tp.py` and each market's equivalent. The
headline, cross-country-comparable series: shadow-rate AFNS on the regime-clean modern
sample, decomposed with the sample-mean `θ`, the mean-reversion floor and the survey
endpoint. This is "the term premium today" and the series everything downstream consumes.

**Historical / stagflation exhibit** — the 1961+ regime-switching and single-regime work,
kept separate and clearly labelled. Regime *dating* (1973–75, 1979–82) plus a
single-regime 1961+ bias-corrected fit putting the 1970s 10y premium at a moderate
**~1.3%** (the premium peaked later, early-1980s to early-1990s). Japan has its own:
the 1974–2000 high-inflation era, ~+3.4% mean / +4.9% peak 10y TP.

**Why not a 2008 splice.** An earlier design spliced a pre-2008 regime model to the
post-2008 shadow model at 2008-12-31. It produced a 243 bp discontinuity at the seam
*and*, via the pre-2008 leg, still risked contaminating the modern estimate. The splice
code is retained in `us/splice.py` as an internal option — it is **not the product**.

## The inflation-linked leg (US + UK)

**Real** term premia (US TIPS via `us/tips.py`, GSW feds200805, 2004+; UK index-linked
gilts, 1985+), estimated with plain AFNS — real rates have **no ELB** ("no real cash";
the bound is pushed to −100% so it never binds). Both are solid and **both are
published**: the Atlas puts the nominal and inflation-protected premia side by side over
their overlap.

What the Atlas deliberately **does not** publish is the further split of the breakeven
into expected inflation + an inflation-risk premium. The breakeven itself
(nominal − real) is a clean market number, but the "two separate AFNS, subtract" split
(`us/inflation.py`) is **too noisy to headline**: the implied 10y expected-inflation
series swings from −1.3% to +5.4% (std 1.27) even though its latest and average values
land near target. It is the small difference of two independently estimated premia and
inherits both models' noise. The UK split is worse (RPI-CPI basis, LDI-driven negative
real premia). See `validation/FINDINGS_inflation.md`.

### Joint nominal/real model — research spike, not shipped

A single-kernel joint AFNS (`model/joint_afns.py`, `us/joint_us.py`) prices both curves
at once to try to extract a better IRP than "subtract", with a TIPS liquidity factor, an
ATSIX survey anchor on expected inflation, and the HPW noise proxy. Outcome vs the Fed
Board's DKW model:

- **real term premium** co-moves with DKW at **0.98**, and **expected inflation** tracks
  ATSIX — both usable;
- **the IRP is not resolved.** Its high-frequency correlation with DKW stays weak,
  because separating the TIPS liquidity *level* from the real term premium needs a direct
  liquidity observable — the TIPS–inflation-swap basis, which is **paid**.

So it stays a documented spike, not surfaced in the Atlas. `validation/FINDINGS_joint_us.md`.
**UK joint model: on hold** — runnable but not validatable to the US standard (no ATSIX
analogue, an RPI→CPI wedge, a larger unidentified LDI/convenience premium, no public
benchmark).

## The global cycle

`model/global_factor.py` decomposes the markets' premia into one common cyclical factor
plus market-specific local residuals — a reduced-form 1-factor **dynamic factor model**
(Kalman filter + RTS smoother + EM, ragged-aware) on the already-estimated TP series. It
does **not** re-estimate any curve.

```
z_i,t = λ_i·G_t + e_i,t        G_t = φ·G_{t−1} + u_t,   Var(G)=1
```

- **Five markets** — US, DE, JP, UK, CA. **Switzerland is excluded by design**: its
  government-bond liquidity is too thin next to the majors, so it enters as noise.
- Month-end, not daily: daily TP changes are dominated by curve-fitting noise, the daily
  AR(1) sits at ~0.999 (effectively a unit root, badly conditioned), and the deep history
  is natively monthly anyway.
- At the 10y term-premium point the first principal component is **93%** of the
  variation, balanced from 1986-01 (when Canada joins).
- Fit per **(measure × curve point)** — actual yield / expected rate / term premium, at
  5y, 10y, 20y, 30y and the **5y5y forward** — so the Atlas panel can move along the
  curve. The forward is derived from the two zero legs, and because that transform is
  linear the three measures stay additive there without re-estimation.

The **local residual** is the relative-value read: which curve is cheap or rich against
the common cycle. `validation/FINDINGS_global_factor.md` carries the honest verdict,
including the relative-value backtest.

## The supply channel — duration, not par (`/supply`)

The premium is compensation for holding duration risk, so the quantity of that risk in
private hands is a candidate driver — the preferred-habitat reading (Vayanos-Vila).
Measuring it in **par** cannot work: by par share the Federal Reserve held more of the
public debt in the 1970s, running a bills portfolio for reserve management, than at the
peak of QE. `/supply` therefore measures duration, in two units.

**Ten-year equivalents** (`Σ D_i·A_i / D_ref`) give the *share* a central bank holds —
SOMA runs about 14% of the Treasury market by par against about 21% of its duration risk.
**DV01** (`supply/us_dv01.py`) gives the *dollar* amount: the stock outstanding, the
slice held by SOMA, and the rolling three-month flow of each. The flow series are
deliberately **not** the change in the stock — they are per-CUSIP quantity changes priced
at each security's own duration, so rolldown and rate moves on unchanged positions
contribute nothing.

What reaches production is narrower than what is measured. Only the **price-insensitive**
holder compresses the premium — foreign official at −0.064 (t = −12.6) against foreign
private at −0.029 (t = −1.2), the official coefficient within a whisker of the Fed's own
— so `supply/price_insensitive.py` supplies the ownership driver in `model/tp_drivers.py`,
and the Fed's balance sheet is carried as a **control** (keeping QE-era observations from
contaminating the inflation coefficients) rather than as a forecastable driver. See
**`supply/README.md`**.

## From premia to a view (`/scenarios`)

The scenario engine takes annual macro paths — policy rate, inflation, growth — and
returns a ten-year yield and a total return at three- and ten-year horizons for the US,
UK, Germany and Japan. The premium leg is driven by expected inflation and the volatility
it brings, the projected deficit (cyclically adjusted: a growth view moves the
deficit row but not the premium), and price-insensitive ownership; the spread markets are
priced as **the US plus a difference** (relative inflation, relative ownership, relative
fiscal position) on regime-fitted betas. Both legs converge from **today's market
pricing** rather than jumping to fair value, at a learning speed fitted from event-study
episodes. Full spec in **`scenarios/README.md`**.

---

## Data sources, per market

| market | source | method | what to pull |
|---|---|---|---|
| US | Federal Reserve Board (GSW) | Svensson, parametric | daily params (beta0-3, tau1-2); reconstruct the grid on demand |
| Germany | Bundesbank | Svensson, parametric | daily params since Aug 1997 (monthly since Sept 1972) |
| UK | Bank of England | VRP spline, non-parametric | published grid, nominal + real, daily since 1979 (real 1985), out to 20y+ |
| Japan | MoF / BoJ | spline (per BIS Papers no. 25) | MoF JGB spot yields, ragged 1974+ (10y from 1986) |
| Switzerland | SNB | `rendoblim` Svensson spot curve | daily CHF spot rates 1988+ (ragged long end from 1997-12) |
| Canada | Bank of Canada | fitted zero-coupon curve | daily CAD zeros 0.25–30y, 1986+ (30y from 1991) |

US real (TIPS, GSW feds200805, 2004+) and UK real (index-linked gilts, 1985+) feed the
inflation-linked leg. Benchmarks, surveys and the liquidity gauge come via `/validation`;
survey anchor inputs live per-market under `*/data/anchors/`.

Common grid, by market — deliberately **not** forced to a shared set:

```
US / DE / CA   0.25 0.5 1 2 3 5 7 10 20 30
UK             0.5 1 2 3 5 7 10 20            (real: 3 5 7 10 20)
JP / CH        1 2 3 5 7 10 20 30
```

## Validation benchmarks

- **US plain-AFNS baseline**: SF Fed "Treasury Yield Premiums" — same model family,
  same GSW input. This validated the AFNS implementation itself before the shadow-rate
  extension, and it is what `κ_min` was calibrated against.
- **US external models** (`validation/benchmarks.py`): **ACM** (NY Fed, 5-factor),
  **Kim-Wright** (Fed Board, FRED `THREEFYTP1..10` — *not* zero-padded), **DKW**
  (D'Amico-Kim-Wei; also supplies real-TP, IRP and TIPS-liquidity references), and
  **C&R**. The Atlas "Cross-check" panel overlays ours on all four: the spread between
  independent models is the honest uncertainty band on "the" US term premium.
- **Cross-market directional check**: Krippner's published series. Methodology differs
  (three-factor, OIS-augmented, survey-based natural rate) — use for plausibility on
  level and turning points, not as a target.
- Expect the shadow-rate model to **diverge** from plain-AFNS and from Krippner inside
  each market's ZLB/negative window. That divergence is expected, not a bug.
- **ATSIX** (`load_atsix`): the Philadelphia Fed *Aruoba Term Structure of Inflation
  Expectations* — free, monthly (1998+), blends SPF + Blue Chip + Michigan, so it carries
  the otherwise-paywalled Blue Chip signal. US-only; anchors the joint model's expected
  inflation.
- **FRED access** goes through `validation/fred_api.py`: the official API when
  `FRED_API_KEY` is set, else the keyless `fredgraph.csv` fallback. The official path is
  materially more reliable through the egress proxy and exposes series metadata.

Each loader mirrors `sf_fed.py`: env-overridable URL, parquet cache, cached-fallback on
egress failure.

## Current production numbers

**Nominal 10y term premium**, latest monthly average (Aug 2026; CH Jul 2026):

| CH | DE | CA | UK | JP | US |
|---|---|---|---|---|---|
| −0.08% | +0.88% | +1.05% | +1.35% | +1.36% | +1.64% |

**Real 10y term premium**: US +1.50%, UK +1.30%.

Switzerland lowest (deepest safe-haven, still slightly negative); the US highest, on
fiscal supply and the retreat of the price-insensitive bid. Regenerate with
`python -m exports.multi_market_term_premia_monthly`; the daily series is in the Atlas.

## Estimation cadence (operational design)

Do not force a full nonlinear MLE re-optimization every day — convergence risk is real
for shadow-rate models. Instead:

- re-estimate state-space parameters on a **slower cadence** (weekly/monthly);
- run the filter **forward daily** with fixed parameters;
- **warm-start** each re-estimation from the previous parameter vector;
- log the likelihood at convergence and **fall back to the last valid parameter set**
  on failure, rather than publishing a broken estimate.

**This is what makes daily publishing cheap.** Only the estimation is slow — minutes
cold, faster warm-started — and it is amortized in the background. The daily operation
is a single EKF pass with cached parameters:

| operation | cost per market |
|---|---|
| daily filter-forward, 12-month window | **~20 ms** |
| full-history monthly re-filter + decompose (~40y) | ~0.5 s |
| full-history *daily* re-filter + decompose (~28y) | ~8 s |

Implementation: `ProductionParams` (the cached, JSON-serializable parameter set) +
`filter_forward` (the fast daily decomposition). Estimation `restarts` defaults to **1**;
multi-start is opt-in for cold fits.

**Implemented in `/scheduler`.** `daily_update.py` is the fast path (refresh all source
caches and benchmarks → `export_daily` → `global_factor` → `thin_daily` →
`build_artifact`); `reestimate.py` is the slow warm-started re-estimation;
`supabase_export.py` upserts the Supabase tables and publishes the Atlas to a public
bucket while this repo stays private.

**Two runs a day**, each placed just *after* a publication the other misses:
**00:45 UTC** (clears the MoF JGB post at 09:30 JST) and **12:15 UTC** (clears the BoE
GLC at noon London). The single midnight run it replaced sat 30 minutes *before* the MoF
post, so Japan was a full day stale every day. Both runs are idempotent; cron is
fixed-UTC, so they sit ~1h late in winter, with slack chosen to absorb that plus
scheduler jitter. Nowcast extensions (`us/fetch_h15.py` H.15 CMT, and the Canada
benchmark equivalent, with a roll guard) carry each curve to today when the fitted-curve
publisher lags. See `scheduler/README.md`.

## Build status

All six markets are **done** and in production. Per-market notes:

1. **US** — modern production TP (1987+ shadow AFNS) validated vs SF Fed, and the market
   `κ_min` is calibrated on; separate 1961+ historical/regime exhibit. `FINDINGS_phase5.md`.
2. **Germany** — Bundesbank Svensson, reconstruction < 0.6 bp vs published spots; ELB
   well identified, validating the −1.0% (Praet) calibration. Refit on the regime-clean
   **1984+** sample. `FINDINGS_germany.md`, `FINDINGS_regime_dating.md`.
3. **Japan** — MoF JGB, ragged panel; ML ELB −0.20%; **YCC measurement-downweight** on the
   pegged 7y/10y (2016-09 → 2024-03, tapering as the band widened). Realised 10y premium
   is **negative through QQE/YCC** (min −0.44% in 2020) and normalises post-YCC. This leg
   surfaced decomposition fixes applied pipeline-wide. `FINDINGS_japan.md`,
   `FINDINGS_japan_regimes.md`.
4. **UK** — BoE GLC (1970→2026), mild fixed ELB (weakly identified). Regime-clean
   **1995-11+**, applied to the index-linked curve too, so the inflation-risk premium is
   not a sample artefact. `FINDINGS_uk.md`.
5. **Switzerland** — SNB `rendoblim`, 1988+, ELB −0.75% (the revealed floor). Lowest
   premium of the six.
6. **Canada** — BoC fitted zero curve, 1991+ inflation-targeting sample; ELB near 0% and
   weakly identified. Included in the global-cycle factor as a deep, liquid G7 market.

## Repo structure

```
/model          # shared AFNS + CR shadow-rate + EKF; bias correction, regime switching,
                #   the neutral anchor, the global factor, the driver model
/us             # GSW nominal + TIPS real fetchers, production TP, joint spike, anchors
/germany /japan /uk /switzerland /canada
/macro          # driver inputs: core CPI, fiscal projections, spliced TP history
/supply         # duration risk: MSPD/SOMA + BoJ/MoF ladders, ten-year equivalents,
                #   DV01 (us_dv01.py), and the price-insensitive ownership driver
/scenarios      # macro paths in -> ten-year yield and bond return out (the sheet)
/validation     # benchmarks (SF Fed / ACM / Kim-Wright / DKW), ATSIX, the HPW TIPS
                #   liquidity gauge, the FRED API helper, and every FINDINGS_*.md
/scheduler      # cadence: weekly re-estimation, daily filter-forward, Supabase export
/viz            # canonical daily dataset -> thinned payload -> self-contained Atlas
/presentation   # the deck and the two-page executive summary (+ PDF/PPTX exporters)
/exports        # derived series handed to other repos (clone, not raw URL — repo is private)
```

## Session guidance

- Work through one market at a time; do not attempt all six unsupervised in one session.
- Flag any convergence failure, implausible shadow-rate path, or benchmark divergence
  **explicitly** rather than silently proceeding.
- The ELB values above are provisional and open to revision on sensitivity testing — do
  not treat them as fixed facts.
- Derived caches (`viz/*.parquet`, `supply/data/*`) are gitignored and regenerable; a
  fresh container has the repo but not the caches. Rebuild via `/scheduler`, or the
  per-module command in that module's README.
