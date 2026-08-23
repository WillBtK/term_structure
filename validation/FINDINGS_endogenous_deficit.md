# The endogenous deficit: growth moves the fiscal input

*August 2026. Code: `macro/fiscal_rules.py`, `macro/weo.py`, the `growth_defaults`
wiring in `scenarios/engine.py`, and the derived deficit row in the scenario sheet.*

The defect this closes: a slower-growth scenario used to leave the deficit at the
official projection — the one inconsistency everyone in the room notices. The deficit
is now a **derived quantity with an override**: when a scenario states a growth view
and no fiscal view,

```
deficit(t) = official(t) − ε · gap(t)
gap(t)     = 0.7 · gap(t−1) + [your growth(t) − source baseline growth(t)]
```

An explicit deficit path switches the rule off entirely — a stated fiscal view is the
whole fiscal path, never rule-plus-override.

## Each market's own institution supplies ε

| | ε | source |
|---|---|---|
| US | 0.36 | CBO *rules of thumb* workbook (Feb 2026), reproduced at year 5 |
| UK | 0.70 | OBR cyclical-adjustment coefficient (Briefing Paper No. 4) |
| DE | 0.504 | EC semi-elasticity (Mourre et al.), recovered exactly from AMECO |
| JP | 0.45 | OECD (Price, Dang & Guillemette); Cabinet Office rejected — below |

**US.** CBO's own interactive workbook ("How Changes in Economic Conditions Might
Affect the Federal Budget: 2026 to 2036") scales its published ±0.1pp scenarios
linearly in the user's delta, valid ±0.5pp — CBO itself endorses the linear
heuristic. Its deficit deltas per sustained 1pp/yr shock, % of GDP (calendar years,
Feb 2026 baseline — the same vintage as our deficit default):

| shock | yr 5 (2031) | yr 10 (2036) |
|---|---|---|
| productivity growth 1pp slower | +1.0 | +2.3 |
| labor-force growth 1pp slower | +0.5 | +1.3 |
| interest rates 1pp higher | +0.9 | +1.3 |
| inflation 1pp higher, rates fixed | **−0.5 (improves)** | −0.9 |

ε = 0.36 makes the rule reproduce the +1.0pp year-five point under the 30% closure
mechanic (gap(5) = Σ 0.7^j = 2.77; 1.0/2.77 = 0.36). The CBO scenarios are
supply-side, so this sits below the OECD *cyclical* semi-elasticity (~0.45–0.5) —
deliberately conservative given the +52bp/point coefficient downstream.

**UK.** The OBR's March 2025 EFO ready-reckoner workbook is the CBO tool's exact
counterpart but stream-level (~30 determinants through HMRC/DWP model elasticities);
OBR's own notes say it is not an integrated macro-fiscal model. The aggregate object
is their published cyclical-adjustment coefficient: a 1% output gap improves
borrowing ~0.7% of GDP after two years (0.5 in-year + 0.2 lagged). The workbook is
the recorded stream-level upgrade path. (The macro model code published alongside —
`Macroeconomic_model_code_March_2025.txt` — is NOT extractable for this purpose: its
receipts and PSNB equations are accounting identities over ~100 exogenous tax
streams whose elasticities live in satellite models.)

**DE.** AMECO carries the full cyclical-adjustment decomposition, and dividing the
cyclical component of the balance (`UBLGCP`) by the output gap (`AVGDGP`) returns
**0.504 in every year 2015–2027** of the current vintage — the EC's official
semi-elasticity, recovered from the same file that supplies the deficit baseline.
Notably the cyclical revenue *ratio* is ≈0 (unit revenue elasticity) and virtually
all of the 0.504 sits on the expenditure ratio — the opposite composition to the US,
where CBO's action is mostly revenue.

**JP — why the Cabinet Office's own number is rejected.** The CAO workbook carries
three scenarios; case ① runs growth ~1.4pp/yr above baseline, a +14.6% GDP level gap
by FY2040. The deficit response: primary balance +1.5pp, net interest −1.0pp
(rates follow growth, r-tracks-g, on a 180%-of-GDP stock), overall **+0.6pp** — an
implied all-in sensitivity of **~0.04**, an order of magnitude below the OECD's.
That is the right answer to a *different* question: permanent trend growth with no
cyclical gap and an endogenous rate response. The closure mechanic is what
reconciles the two: deviations that close are cyclical and price at 0.45; a
permanent trend view decays out of the gap instead of compounding at 0.45 forever,
which approximates the CAO's ≈0 without a second parameter.

## What is deliberately excluded

* **Inflation.** CBO settles it empirically: inflation *alone* improves the US
  deficit (~0.5pp per 1pp at year five) but inflation *with* Fisher-consistent rates
  worsens it — the sign is entirely a function of the rate response, which is this
  model's own output. And inflation already enters the premium through the
  expected-inflation and uncertainty channels. Wiring it would be a double count
  wearing a real-rate coefficient.
* **The rates feedback.** CBO: 1pp on rates → ~0.9pp of GDP at year five. Against
  our +52bp/point coefficient that is a loop gain of ~0.45 (1pp deficit → 52bp
  premium → wider deficit → …, geometric, convergent). The CAO's coupling (~1.0pp of
  GDP per ~1.2pp of rates by 2040) is the Japanese calibration of the same loop.
  Documented, not wired: wiring it makes the model an input to itself.
* **Policy-driven growth.** The rule assumes growth deviations are economy-driven —
  the CBO scenarios are supply-side by construction. Deficit-financed stimulus is a
  *fiscal* view: it is entered in the deficit row directly, which turns the rule off.
  The sheet's footnote says exactly this.

## The growth baseline is the deficit source's own

Growth deviations are measured against **the growth path the deficit projection was
conditioned on** — each institution's own forecast, fetched from the same source
family as the balance:

| | growth baseline | route |
|---|---|---|
| US | CBO economic projections, Feb 2026 (`real_gdp_pct_change`) | us-cbo/cbo-data GitHub mirror, same vintage pin as the deficit |
| UK | OBR long-term economic determinants, March 2026 EFO | obr.uk/data, real GDP growth by fiscal year to 2075 |
| DE | AMECO (growth from the `OVGD` level series) + WEO beyond | same splice discipline as the deficit path |
| JP | Cabinet Office baseline case, same workbook as the balance | 計数表（長期）, to FY2040 |

Measuring against any *other* source's growth silently converts that source's
disagreement with the deficit institution into "no view." The differences are not
small: the CAO's long-run growth is 0.4% against the WEO's 0.6% — under the
WEO-baseline design, *agreeing with the Cabinet Office* registered as ~0.2pp/yr of
growth pessimism, a spurious ~0.3pp of deficit at the steady-state gap; Germany's
AMECO near years run 0.2–0.3pp below WEO's. Equivalently: the sheet's untouched
grid must show the official deficit, and that is only consistent if the growth row
opens on the numbers the deficit was built on. The baseline is rounded to 2dp at
the source so the displayed grid default IS the baseline exactly.

## WEO's remaining role

WEO (`macro.weo`) extends Germany (deficit and growth) beyond AMECO's two-year
horizon and is the fallback when a committed baseline is missing. It was the one
source with growth, inflation and the balance together for all four markets. The
working route is the SDMX 2.1 API on `api.imf.org` (dataflow `IMF.RES,WEO`, key
`COUNTRY.INDICATOR.A`, indicators `NGDP_RPCH` / `PCPIPCH` / `GGXCNL_NGDP`); the
Datamapper front door on `www.imf.org` sits behind Akamai bot protection and is not
reachable from this environment.

WEO also extends **Germany's deficit path**: AMECO's two-year horizon is spliced
with the WEO shape, level-matched at the last common year (offset −0.17pp at the
current seam), taking DE to t+5 and moving the five-year-ahead reading from a flat
4.06 extension to 3.49.

WEO deficits are NOT adopted for US/UK/JP: the +51.9bp coefficient was measured on
the CBO-family current-law basis, and WEO's staff-judgement basis is a different
object. The divergences at t+5 are the sensitivity: US, WEO 7.39 vs CBO 6.64
(−0.75pp ≈ 39bp of premium); JP, WEO deteriorating to 4.04 while the CAO improves
(~1.3pp ≈ 68bp). Germany is the easy case — WEO, AMECO and the OECD EO agree within
~0.5pp.

Alternate sources probed and set aside: OECD Economic Outlook via SDMX works and
archives **five past vintages as dataflows** (`DF_EO_114`–`118` + live) — the
recorded route to the UK/DE vintage panels; the OECD long-term baseline's fiscal
series is an *assumed* debt-stabilising primary balance (circular, rejected); the EC
Debt Sustainability Monitor reaches 2035 but is an annual PDF whose long end is
mechanical debt dynamics our fiscal channel would double-count.

## Verification

* `macro/tests/test_fiscal_rules.py`: sign, symmetry, decay, the CBO year-five
  reproduction, per-market ε, fractional scenario keys, year-0 invariance.
* `scenarios/tests/test_engine.py`: a growth view alone moves the fiscal driver; an
  explicit deficit override disables the rule; behaviour without `growth_defaults`
  is bit-identical to before.
* Sheet↔engine agreement with the rule active: **0.245bp** worst gap across all
  markets, both horizons, all outputs (previous band ~0.2bp). The sheet's derived
  row was verified cell-perfect against ε per market, and the CSV export omits the
  deficit column while derived so the engine re-derives the identical full path
  rather than freezing a six-year truncation.
* Full suite: 292 passing.
