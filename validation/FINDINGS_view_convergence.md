# How fast does the market adopt your view? Estimating λ

`scenarios.engine.VIEW_CONVERGENCE` (λ = 0.8) governs how fast a user's policy path
displaces the curve's own forward expectation:

```
Pbar(h) = mkt(h) + [your(h) − mkt(h)] · (1 − λ^h)
```

Unlike `RESIDUAL_PERSISTENCE`, which is measured directly out of sample, λ describes
convergence to a forecast with no historical track record. Two proxies were run.
Reproduce with `python -m validation.view_convergence`.

## A. The direct test cannot identify λ — and that is the finding

The SPF asks annually for the average **3-month bill rate expected over the next ten
years**: the same object our expectations leg projects, forecast independently and
dated. Matched to the curve's expected-rate reading at each vintage, 1992–2026, n = 35.

**The two essentially never disagree.** Mean gap −0.00pp, **standard deviation 0.21pp**,
and the gap exceeds 25bp in **1 year out of 35**.

With no material disagreement in the sample there is nothing to identify a convergence
speed from, and the regressions say so: coefficients on the SPF view carry standard
errors of 0.28–0.63 on a parameter bounded in [0, 1], t-statistics of 0.1–1.5, and
signs that flip with horizon (share-to-view of 5%, −6%, 108%, 146% at h = 1, 2, 3, 5).
**λ is not estimable this way.**

This is worth stating positively rather than as a failure. It calibrates how strong a
view really is: if professional forecasters and the curve agree to within about 20bp on
the ten-year average short rate, a scenario path sitting 40bp from the curve is already
a two-sigma claim by the standards of the last thirty years.

## B. The indirect test gives an upper bound of λ ≈ 0.86–0.92

Longer sample, perfect-foresight view. For each ten-year window 1975–2016, compare the
curve's error about the realised average short rate h years before the window starts
with its error at the window start:

| h | β = λ^h | implied λ |
|---|---|---|
| 1 | 0.86 | 0.86 |
| 2 | 0.77 | 0.88 |
| 3 | 0.71 | 0.89 |
| 5 | 0.65 | 0.92 |

Stable across horizons, which is reassuring, but **three caveats make this a bound, not
an estimate**:

1. **It overstates persistence.** Shocks landing *inside* the window are in both errors
   and are never forecastable at any horizon, so they push β toward 1. True λ is below
   these numbers, by an unknown amount.
2. **The reported standard errors (0.01) are meaningless.** 493 monthly windows of ten
   years each over a 41-year sample is roughly **four independent observations**.
3. **It is essentially one macro episode.** The curve's ten-year errors have a standard
   deviation of 2.5–3.2pp and are dominated by systematically over-predicting rates
   through the great disinflation. This measures how fast the curve gave up on that,
   not a general convergence law.

## Verdict

**λ = 0.8 is disciplined by the evidence but not pinned by it.** It sits just inside the
upper bound from test B, in the slightly-faster-convergence direction — which is the
right side to err on, given that B is biased toward persistence.

The honest statement to a committee is that this is a *reasoned setting inside a wide,
weakly-identified band*, not a fitted constant — which is why it is a control on the
scenario sheet rather than a number buried in the code. A user who believes their view
gets recognised quickly can set λ = 0.7; one who thinks the market is stubborn can set
0.9. Both are inside what the data can distinguish.

---

## Correction — one view must price at one speed

The parameter above governs how fast a stated RATE view reaches the expectations leg.
It did not govern the premium, and that asymmetry was a defect.

**What the code did.** `Pbar(h)` moved from the market toward the user's path at
`1 - lambda**h` — a fifth of a stated view after a year, half after three. But
`drivers_at()` handed the user's inflation path to the premium models raw, so the
premium's fair value responded in full from year one.

| horizon | weight on the view, policy leg | weight on the view, premium |
|---|---|---|
| 1y | 0.20 | 1.00 |
| 3y | 0.49 | 1.00 |
| 10y | 0.89 | 1.00 |

**Why that is indefensible.** A view is one object. "Policy tightens and inflation
comes down with it" is a single forecast, not two independent ones, and there is no
evidence — none was ever offered — for the market believing one half of it faster than
the other. Split across two dials, the net ten-year yield moved on the gap between the
dials rather than on the economics, and the sign of that gap depends on which way the
two halves of the view point. A tool whose answer can be perverse for reasons the user
cannot see or defend is not usable in front of a committee.

**The fix.** `DriverModel.scenario` and `RelativeModel.scenario` take a `view_weight`,
and the engine passes `1 - lambda**h` — the same number the expectations leg uses. The
premium is now

    TP(h) = TP_today + [fair(h) - fair(0)] * (1 - lambda**h) + residual * (rho**h - 1)

so at any horizon the model shows the same fraction of the user's world in both legs.
`view_weight=1.0` reproduces the previous result exactly, which is what makes this a
generalisation rather than a change of model.

**rho is untouched, and that is deliberate.** The residual is not a view — it is the
part of today's premium the drivers cannot explain. How fast a mispricing mean-reverts
and how fast a forecast is believed are different questions, and only the first has
evidence behind it.

**Two things fell out.**

*The spread markets need the move split before it is weighted.* The premium is linear,
so `premium(s) - premium(t) = beta*(tp_us_s - tp_us_t) + [own relative drivers]`. The
US premium arriving through `tp_us_scenario` has already been blended, so weighting the
whole move would damp that channel twice and make every spread market converge more
slowly than the US for no reason. The beta channel passes at full weight; only the
market's own relative drivers carry the weight.

*The h=0 anchoring is now a property of the model rather than a convention of one
caller.* Unweighted, `scenario()` at h=0 returned today's premium PLUS the whole move
in fair value, because `drivers_at(.., 0)` reads the user's full ten-year forward
average. The engine hid this by writing `tp[0] = tp_today` at the call site instead of
calling the model. With the weight it is zero at h=0 and the identity holds by
construction.

**Size of the effect.** Small for realistic scenarios and large for extreme ones,
because it scales with `(1 - w) * (fair(0) - fair(h))`. A hike-then-cut disinflation
(policy +150bp for three years then below the market, inflation back to target) moves
the US ten-year by **6bp at three years and -1bp at ten**. A deliberately extreme test —
a permanent +2pp shift in long-run expected inflation — moves it by about a percentage
point at three years. An earlier note in this repo quoted the extreme figure as though
it were typical; it is not.

Verified in `model/tests/test_view_weight.py`, and the browser sheet re-checked against
the Python engine at 0.17bp worst disagreement.
