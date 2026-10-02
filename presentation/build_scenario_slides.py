"""Refresh the scenario numbers on the deck's scenario-framework slides (``const X``).

Those slides show what the scenario sheet itself computes: today's premium and yield,
each preset's ten-year premium and yield at the three- and ten-year horizons, the
shipped fiscal sensitivities, and the hard-landing decomposition. They are read off the
sheet by evaluating it in headless Chromium -- the same functions the browser runs --
so the deck can never disagree with the sheet it describes.

Only the LIVE parts of ``X`` are rewritten. The dated methodology record stays as it was
written: the March 2025 Bund window (``bund_mar25``), slide 12's "before" column
(``hl.before``) and Germany's earlier steps (``de_journey``).

    python -m presentation.build_scenario_slides          # refresh and splice
    python -m presentation.build_scenario_slides --check  # compute and print only

Run after the sheet's payload or presets change (``scenarios.build_sheet``,
``scenarios.load_presets``). Needs Playwright; uses the pre-installed Chromium.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHEET = ROOT / "scenarios" / "sheet" / "index.html"
DECK = ROOT / "presentation" / "index.html"
CHROMIUM = os.environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium")
MARKETS = ("US", "UK", "DE", "JP")

# Evaluated inside the sheet page: every number comes from the sheet's own functions.
_JS = """() => {
  const out = {asof: D.asof, scen: {}, today: {}, models: {}, hl: {}, cmp: {}};
  for(const c of ['US','UK','DE','JP']) out.today[c] = {tp: D.markets[c].tp_today, y: D.markets[c].y_today};
  for(const c of ['UK','DE','JP']){ const m = D.markets[c].model;
    out.models[c] = {beta: m.beta_us, own: m.adjust.fiscal,
                     spill: m.beta_us*D.markets.US.model.adjust.fiscal - m.adjust.fiscal}; }
  out.models.US = {own: D.markets.US.model.adjust.fiscal};
  for(const s of SCEN_NAMES){ activate(s); out.scen[s] = {};
    for(const c of ['US','UK','DE','JP'])
      out.scen[s][c] = project(c,[3,10]).map(x=>({tp:+x.tp.toFixed(2), y:+x.y.toFixed(2)})); }
  // Baseline vs Hard Landing at the ten-year horizon: the fiscal channel and the premium
  for(const s of ['Baseline','Hard Landing']){ activate(s);
    const mu = D.markets.US.model, us0 = D.markets.US.drivers, usH = driversAt('US', scenarioFor('US'), 10);
    const vwU = 1 - Math.pow(lamFor(SCEN,'US'), 10);
    const usPart = {inflation: mu.coef.expected_inflation*(kink(usH.expected_inflation)-kink(us0.expected_inflation))*vwU,
                    uncertainty: mu.coef.inflation_uncertainty*(usH.inflation_uncertainty-us0.inflation_uncertainty)*vwU,
                    fiscal: mu.adjust.fiscal*(usH.fiscal-us0.fiscal)*vwU};
    const m = D.markets.DE.model, own0 = D.markets.DE.drivers, ownH = driversAt('DE', scenarioFor('DE'), 10);
    const vw = 1 - Math.pow(lamFor(SCEN,'DE'), 10), r0 = relDrivers(own0, us0), rH = relDrivers(ownH, usH);
    const g = k => (m.coef[k] ?? m.adjust[k] ?? 0);
    const de = {}; for(const k in usPart) de[k] = m.beta_us*usPart[k] + g(k)*((rH[k]??m.base[k])-(r0[k]??m.base[k]))*vw;
    out.cmp[s] = {US: usPart, DE: de,
                  tp: Object.fromEntries(['US','UK','DE','JP'].map(c=>[c, project(c,[10])[0].tp]))};
  }
  out.hl = {US: out.cmp['Hard Landing'].US};
  activate('Baseline');
  return out; }"""


async def _evaluate(sheet: Path) -> dict:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROMIUM)
        pg = await b.new_page()
        errs: list = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(f"file://{sheet.resolve()}")
        await pg.wait_for_function("typeof project === 'function' && typeof D === 'object'")
        out = await pg.evaluate(_JS)
        await b.close()
    if errs:
        raise RuntimeError(f"sheet raised in the browser: {errs[:3]}")
    return out


def live_block(n: dict) -> dict:
    """The parts of ``X`` that track the sheet."""
    r2 = lambda v: round(float(v), 2)
    c = n["cmp"]
    now = {m: {"fb": r2(c["Baseline"][m]["fiscal"]), "fh": r2(c["Hard Landing"][m]["fiscal"]),
               "tb": r2(c["Baseline"]["tp"][m]), "th": r2(c["Hard Landing"]["tp"][m])}
           for m in ("US", "DE")}
    return {
        "asof": n["asof"],
        "fiscal": {"US": {"own": round(n["models"]["US"]["own"] * 100, 1)},
                   **{m: {"own": round(n["models"][m]["own"] * 100, 1),
                          "spill": round(n["models"][m]["spill"] * 100, 1),
                          "beta": r2(n["models"][m]["beta"])} for m in ("UK", "DE", "JP")}},
        "today": {m: {"tp": r2(n["today"][m]["tp"]), "y": r2(n["today"][m]["y"])} for m in MARKETS},
        "scen": n["scen"],
        "hl_now": now,
        "us_infl": r2(n["hl"]["US"]["inflation"]), "us_unc": r2(n["hl"]["US"]["uncertainty"]),
    }


_X_LINE = re.compile(r"^const X = (\{.*\});$", re.M)


def splice(live: dict, deck: Path = DECK) -> dict:
    src = deck.read_text(encoding="utf-8")
    m = _X_LINE.search(src)
    if not m:
        raise ValueError("no `const X = {...};` line in the deck")
    x = json.loads(m.group(1))
    x["asof"], x["fiscal"], x["today"], x["scen"] = (live["asof"], live["fiscal"],
                                                     live["today"], live["scen"])
    x["hl"]["now"] = live["hl_now"]
    x["hl"]["us_infl"], x["hl"]["us_unc"] = live["us_infl"], live["us_unc"]
    line = "const X = " + json.dumps(x, separators=(",", ":")) + ";"
    deck.write_text(src[:m.start()] + line + src[m.end():], encoding="utf-8")
    return x


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compute and print; do not splice")
    a = ap.parse_args()
    live = live_block(asyncio.run(_evaluate(SHEET)))
    print(f"sheet as of {live['asof']}")
    print("today      " + "  ".join(f"{m} {live['today'][m]['tp']:.2f}/{live['today'][m]['y']:.2f}"
                                    for m in MARKETS))
    for s, v in live["scen"].items():
        print(f"{s:27}" + "  ".join(f"{m} {v[m][0]['tp']:.2f}|{v[m][1]['tp']:.2f}" for m in MARKETS))
    print("fiscal", live["fiscal"]); print("hard landing vs baseline (10y)", live["hl_now"])
    if not a.check:
        splice(live)
        print(f"spliced -> {DECK.name}")


if __name__ == "__main__":
    _main()
