# The spread markets' fiscal sensitivity (2026-09)

**Decision.** Germany, the UK and Japan no longer borrow the US fiscal coefficient
(+0.519pp of 10y premium per +1pp of GDP of five-year-ahead projected deficit). They
carry the pooled cross-market estimate, **+0.235**, applied to the deficit relative to
the US, bounded so that neither partial effect can turn negative
(`model/tp_relative.py`: `SPREAD_FISCAL_COEF`, `spread_fiscal_coef`).

## Why the borrowed number had to go

It was right for the US and wrong to transfer. In the scenario baseline it was 85% of
the German own-driver channel, taking the Bund 10y premium from 0.96 today to **0.10**
at ten years while the US fell only 1.63 → 1.35 — the spread nearly doubling. A
sensitivity of that size had never been estimated for any spread market; it was
simply the US's.

## Where +0.235 comes from

The `imf_fiscal_sensitivity` repo's **deep panel** (branch
`claude/fiscal-data-pipeline-validation-wz3rbx`, not yet merged there): the 10y term
premia of US/UK/DE/JP/CA **from this repo**, regressed on the WEO five-year-ahead
structural balance vintage public each month, 2007-11..2026-06. Reported in bp per
+1pp of GDP of deficit.

| 10y | estimate | t |
|---|---|---|
| **Pooled, level, country effects** | **+23.5** | **7.9** |
| Pooled, two-way effects | +5.4 | 1.7 |
| Pooled, relative to global factor | +3.4 | 1.6 |
| DE, own, relative to global factor | −4.3 | −1.6 |
| UK, own, relative to global factor | +2.9 | 0.6 |
| JP, own, relative to global factor | +2.7 | 0.9 |
| US (that repo's own CBO-vintage study, post-2008) | +65.6 | — |

### Why not the per-market estimates

None of the three is significant at 10y, and Germany's is **negative**. In a scenario
tool a negative coefficient says a German deficit of 10% of GDP *lowers* Bund yields.
That is an identification failure, not a finding:

- **Safe haven.** German deficits widened in 2009 and 2020 exactly as Bunds rallied on
  flight to quality; a structural balance strips only part of that cycle, and Germany is
  the purest safe haven of the five.
- **Timing.** WEO vintages publish in April and October. The market reprices in days;
  the regressor catches up months later, so the event is lost in a monthly regression.
- **Netting.** Germany is one of the five markets that build the global factor, so part
  of a German shock is removed as "global" before the German coefficient sees it.

### The one clean German experiment

The debt-brake reform, announced 4–5 March 2025. This repo's own series:

| | Bund 10y | expected path | term premium | US term premium |
|---|---|---|---|---|
| 4 Mar 2025 | 2.50 | 2.33 | 0.17 | 1.12 |
| 14 Mar 2025 | 2.93 | 2.29 | 0.64 | 1.22 |
| change | **+0.43** | −0.04 | **+0.47** | +0.10 |

The whole move was term premium, +37bp of it beyond the US premium's own move. Against
a five-year-ahead deficit revised up by roughly 1.5–2pp of GDP (today's AMECO path is
~3.5–4% against ~2% before the reform — an approximation: there is no AMECO vintage
archive here to pin it down), that is **+20 to +30bp per point**, bracketing the pooled
+23.5. The model with +0.235 attributes 35–47bp to such a revision; observed 37–47bp.

## Why the relative deficit, not the own deficit

The coefficient on the relative (own − US) deficit fixes both partial effects:

| | own deficit +1pp | US deficit +1pp |
|---|---|---|
| general | `c` | `beta × 0.519 − c` |
| DE (beta 0.964) | +0.235 | **+0.265** |
| UK (beta 0.888) | +0.235 | +0.226 |
| JP (beta 0.482) | +0.235 | +0.015 |

US fiscal news reaches the Bund at about half the US's own response — close to what
the twelve-month-CHANGES beta (~0.6, see `REGIME_START`) implies, 0.31 — and barely
reaches Japan, consistent with its near-decoupling. Applied to the own deficit alone,
US deficits would instead pass through the LEVELS beta and move Bunds almost
one-for-one with Treasuries. `spread_fiscal_coef` clips `c` to `[0, beta × 0.519]`
so a future refit with a low beta cannot make US deficits lower a premium; nothing
binds today (Japan has 0.015 of room).

## What it does to the scenarios

Committed sheet payload (as of 2026-08-19), 10y premium and yield, old → new. The US
is unchanged by construction.

| scenario | mkt | TP 3y | TP 10y | 10y yield at 10y |
|---|---|---|---|---|
| Baseline | US | 1.46 → 1.46 | 1.35 → 1.35 | 4.97 → 4.97 |
| Baseline | UK | 1.35 → 1.28 | 1.25 → 1.14 | 4.29 → 4.18 |
| Baseline | DE | 0.62 → 0.73 | 0.10 → 0.46 | 1.90 → 2.26 |
| Baseline | JP | 1.06 → 1.03 | 0.86 → 0.70 | 2.59 → 2.43 |
| Goldilocks | US | 0.70 → 0.70 | 0.37 → 0.37 | 4.37 → 4.37 |
| Goldilocks | UK | 0.62 → 0.71 | 0.19 → 0.33 | 3.94 → 4.08 |
| Goldilocks | DE | -0.06 → 0.14 | -0.44 → -0.05 | 2.06 → 2.45 |
| Goldilocks | JP | 1.15 → 1.01 | 1.04 → 0.75 | 2.64 → 2.35 |
| High Growth High Inflation | US | 1.70 → 1.70 | 1.51 → 1.51 | 6.51 → 6.51 |
| High Growth High Inflation | UK | 1.15 → 1.22 | 0.87 → 0.97 | 5.12 → 5.22 |
| High Growth High Inflation | DE | 1.05 → 1.03 | 0.61 → 0.73 | 3.61 → 3.73 |
| High Growth High Inflation | JP | 1.22 → 1.13 | 0.98 → 0.76 | 2.98 → 2.76 |
| Stagflation | US | 4.23 → 4.23 | 4.91 → 4.91 | 10.34 → 10.34 |
| Stagflation | UK | 3.08 → 2.86 | 3.51 → 3.20 | 8.23 → 7.92 |
| Stagflation | DE | 2.71 → 2.86 | 2.93 → 3.31 | 6.89 → 7.27 |
| Stagflation | JP | 0.58 → 0.62 | 0.93 → 0.84 | 2.93 → 2.84 |
| Hard Landing | US | 1.85 → 1.85 | 1.53 → 1.53 | 2.53 → 2.53 |
| Hard Landing | UK | 2.63 → 2.23 | 2.29 → 1.87 | 2.30 → 1.87 |
| Hard Landing | DE | 1.17 → 1.33 | 0.75 → 1.09 | 0.75 → 1.09 |
| Hard Landing | JP | 1.13 → 1.21 | 1.07 → 1.03 | 1.07 → 1.03 |

Checked for sense, market by market:

- **Germany** now ends the baseline at 0.46 rather than 0.10, and the 10y spread to the
  US at about 90bp rather than 126bp (today 67bp). Netted across both channels, the
  inflation kink contributes nothing (−0.01); the remaining decline is the US premium's
  own fall passing through beta, and the fiscal channel (−0.19): a German deficit that
  consolidates from 3.5% to 2.3% of GDP while the US path widens. That consolidation is
  itself cyclical — see "Cyclical vs structural deficits".
- **Hard Landing, Germany** rises 0.75 → 1.09 and sits above its own baseline. Netting
  each driver across BOTH channels (through beta, and the relative term) the effects on
  the Bund at ten years are: inflation **+0.01**, inflation uncertainty +0.10, fiscal
  **+0.36** (Baseline: −0.01, +0.03, −0.19). The gap to Baseline is fiscal: the US
  deficit jumps from 5.9% to 7.1% of GDP on the recession, lifts the US premium, and
  spills into Bunds. See "Cyclical vs structural deficits" below. *(An earlier version
  of this note blamed the inflation kink; that read the relative inflation term without
  netting it against the beta channel, which carries the same move in reverse.)*
- **UK** falls 0.1–0.4 across scenarios, most in Hard Landing (2.29 → 1.87), where a
  cyclically widening UK deficit (the highest semi-elasticity of the four, 0.70) had
  been priced at the US rate.
- **Japan** falls modestly everywhere; its baseline deficit path rises less than the
  US's and the borrowed coefficient had been lifting it on that gap.

## Open, not changed here

- **The inflation kink is neutral for the spread markets.** Expected inflation enters
  as `max(pi − 2, 0)` on each side before differencing, so the US inflation premium
  stays with the US: it passes into Bunds through beta and comes straight back off
  through the relative term (net −0.01 Baseline, +0.01 Hard Landing). Responding to
  own inflation only, with US inflation through beta, was tested and rejected: the model
  is linear, so it double-counts any common shock — Stagflation takes the Bund premium
  to 6.46% and the JGB to 4.00%. Every preset is a common shock, which is the case the
  relative spec gets right. Germany sitting exactly on the hinge in the Baseline (2.0%)
  is the intended asymmetry, not an artefact: 2.2% would add 20bp, 1.8% nothing.
- **Cyclical vs structural deficits — resolved.** The premium now prices the cyclically
  adjusted deficit; see `FINDINGS_structural_deficit.md`. Hard Landing's Bund premium
  falls from 1.09 to 0.87, and the German baseline no longer consolidates in the
  premium on the strength of its growth row.
- **Upgrade path** unchanged: AMECO's twice-yearly vintages and OBR's historical
  forecasts database would let each market identify its own coefficient.
