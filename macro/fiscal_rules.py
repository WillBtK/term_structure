"""The endogenous deficit: each market's official rule of thumb for growth.

A slower-growth scenario that leaves the deficit untouched is the one inconsistency
everyone in the room notices. This module closes it: when a scenario states a growth
view and no explicit fiscal view, the deficit becomes a derived quantity --

    deficit(t) = official(t) - epsilon * gap(t)
    gap(t)     = (1 - GAP_CLOSURE) * gap(t-1) + [user growth(t) - baseline growth(t)]

The gap cumulates growth deviations into an output-level gap and closes it at
~30%/yr, so a one-off growth miss washes out and only a sustained view leaves a
mark. ``epsilon`` is each market's own institution's arithmetic, not one generic
multilateral number:

    US  0.36  calibrated so the rule reproduces the CBO's published rules of thumb
              ("How Changes in Economic Conditions Might Affect the Federal Budget",
              Feb 2026 workbook): a sustained 1pp/yr-slower-growth path widens the
              deficit 1.0pp of GDP at year five; under the 30% closure mechanic the
              gap at year five is 2.77, and 1.0/2.77 = 0.36. The CBO scenario is
              supply-side (no automatic-stabilizer surge), so this sits below the
              OECD cyclical semi-elasticity of ~0.45-0.5 -- deliberately
              conservative given the +52bp/point coefficient downstream.
    UK  0.70  the OBR's published cyclical-adjustment coefficient: a 1% output gap
              improves borrowing by ~0.7% of GDP after two years (0.5 in-year +
              0.2 lagged; OBR Briefing Paper No. 4). The March 2025 EFO
              ready-reckoner workbook is the stream-level upgrade path.
    DE  0.504 the European Commission's official budget-balance semi-elasticity
              (Mourre et al.), recovered exactly from AMECO's own decomposition:
              cyclical component / output gap = 0.504 in every year 2015-2027 of
              the current vintage.
    JP  0.45  OECD (Price, Dang & Guillemette). The Cabinet Office's own
              cross-scenario delta is NOT used: its growth-strategy cases move
              trend growth with rates following (r tracks g), and the implied
              all-in sensitivity is ~0.04 -- the right answer to a different
              question. The closure mechanic is what reconciles the two: a
              deviation that closes is cyclical and prices at 0.45; a permanent
              trend view decays out of the gap rather than compounding forever.

Growth deviations are measured against the DEFICIT SOURCE's own growth baseline
(``macro.fiscal_projections.growth_baseline``: CBO, OBR, AMECO extended with WEO,
Cabinet Office) -- the growth path each official deficit projection was conditioned
on. Measuring against any other source's growth silently converts that source's
disagreement with the deficit institution into "no view" (the CAO's 0.4% long-run
growth vs the WEO's 0.6% is ~0.3pp of spurious gap under the OECD elasticity). WEO
(``macro.weo``) is the fallback when no committed baseline exists.

What the rule deliberately does NOT do:

* **Inflation.** Excluded. Its budget effect nets to a small improvement only while
  interest rates are held fixed (CBO: +0.1pp inflation alone narrows the deficit
  ~0.05pp of GDP at year five, but inflation *with* Fisher pass-through widens it);
  the sign depends entirely on the rate response, which is this model's own output.
  Wiring it would double-count inflation through the fiscal channel on top of the
  expected-inflation and uncertainty channels.
* **Interest costs.** The rates feedback (CBO: 1pp on rates -> ~0.9pp of GDP at
  year five; loop gain ~0.45 against our +52bp/point coefficient) is documented in
  the FINDINGS, not wired -- wiring it makes the model an input to itself.
* **Policy-driven growth.** The rule assumes growth deviations are economy-driven
  (the CBO scenarios are supply-side by construction). A stimulus scenario is a
  fiscal view: enter the deficit directly, which switches the rule off entirely --
  the override is the whole fiscal path, never rule-plus-override.
"""
from __future__ import annotations

SEMI_ELASTICITY = {"US": 0.36, "UK": 0.70, "DE": 0.504, "JP": 0.45}
GAP_CLOSURE = 0.30      # share of the output gap closing per year
HORIZON = 15            # years ahead the derived path is produced for


def _flat(path: dict, k: int) -> float:
    """Last value at or before k, engine convention: the last entered year holds."""
    keys = [x for x in path if x <= k]
    return float(path[max(keys)]) if keys else float(path[min(path)])


def adjusted_deficit(country: str, growth: dict, growth_baseline: dict,
                     official: dict) -> dict:
    """The derived deficit path, keyed like ``official`` (years ahead).

    ``growth`` and ``growth_baseline`` are {years ahead: percent}; both extend flat
    beyond their last key (the engine's hold-flat convention), so a growth view held
    into the long run keeps feeding the gap and the closure sets its steady state.
    An empty ``growth`` returns ``official`` unchanged.
    """
    if not growth or not growth_baseline or not official:
        return dict(official)
    eps = SEMI_ELASTICITY[country]
    out, gap = {}, 0.0
    last = max(max(official), HORIZON)
    for k in range(1, last + 1):
        gap = (1.0 - GAP_CLOSURE) * gap + (_flat(growth, k) - _flat(growth_baseline, k))
        if k in official or k <= HORIZON:
            out[k] = _flat(official, k) - eps * gap
    # year 0 (the current, partly-elapsed year) is never adjusted: the gap needs a
    # full year of deviation to exist, and horizon averages sample from year 1 anyway.
    if 0 in official:
        out[0] = official[0]
    return dict(sorted(out.items()))


def growth_defaults(markets=("US", "UK", "DE", "JP")) -> dict:
    """Each market's growth baseline -- the DEFICIT source's own growth projection
    (``macro.fiscal_projections.growth_baseline``: CBO, OBR, AMECO+WEO, Cabinet
    Office), so the gap is measured against the assumptions the official deficit was
    conditioned on. Keyed exactly like scenario paths after ``load_scenario``
    conversion: calendar year y sits at horizon ``(y+1) - today``. Keying both sides
    the same way is what keeps the deviation aligned year by year -- an integer-keyed
    baseline against fractional-keyed scenarios compared each year's entry against
    the NEXT year's baseline."""
    import pandas as pd

    from macro.fiscal_projections import growth_baseline
    now = pd.Timestamp.today()
    frac = now.year + (now.dayofyear - 1) / 365.25
    out = {}
    for c in markets:
        try:
            out[c] = {round((y + 1) - frac, 3): v
                      for y, v in growth_baseline(c).items()}
        except Exception:                                           # noqa: BLE001
            pass
    return out
