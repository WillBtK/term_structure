"""Parity: scenarios.engine against the scenario sheet on the Baseline, year-end horizons.

Feeds the engine the exact paths the sheet runs (policy, inflation with the current
year's remaining months, growth, the priced structural deficit) and reports the largest
gap in the yield, the premium and the rate leg. Last run: 0.05bp or less in every market.

    python -m validation.sheet_engine_parity
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
from playwright.sync_api import sync_playwright
from scenarios.engine import Scenario, run, latest_states
from model.tp_drivers import build_panel, fit_market
from model.tp_relative import build_pair, fit_relative
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium"); pg = b.new_page()
    pg.goto("file:///home/user/term_structure/scenarios/sheet/index.html"); pg.wait_for_function("typeof project==='function'")
    J = pg.evaluate("""()=>{activate('Baseline'); const o={hs: YE_YEARS.map(hz), m:{}};
      for(const c of MK){ const sc=scenarioFor(c);
        const r = project(c, YE_YEARS.map(hz));
        o.m[c]={policy: sc.policy_rate, infl: sc.inflation, growth: sc.real_growth,
                priced: structuralDeficit(c, sc.deficit), proj: D.markets[c].deficit_proj,
                y: r.map(x=>x.y), tp: r.map(x=>x.tp), pbar: r.map(x=>x.pbar),
                d0: driversAt(c, sc, 0), M: {y: D.markets[c].y_today, tp: D.markets[c].tp_today, drv: D.markets[c].drivers}}; }
      return o;}"""); b.close()
panels = {c: build_panel(c) for c in ("US","UK","DE","JP")}
us = fit_market("US", panels["US"]); spread = {c: fit_relative(c, build_pair(c, panels), us) for c in ("UK","DE","JP")}
states = latest_states()
f = lambda d: {float(k): float(v) for k, v in d.items()}
scen = {c: Scenario(c, policy_rate=f(m["policy"]), inflation=f(m["infl"]), real_growth=f(m["growth"]),
                    deficit=f(m["priced"]), deficit_priced={int(float(k)): float(v) for k, v in m["priced"].items()},
                    calendar=True, is_view=True) for c, m in J["m"].items()}
res = run(us, spread, states, scen, deficit_defaults={c: {int(k): float(v) for k, v in m["proj"].items()} for c, m in J["m"].items()},
          view_convergence=0.8, horizons=J["hs"])
for c in res:
    m = J["m"][c]; st = states[c]
    print(c, f"state y {st.y_today:.3f}/{m['M']['y']:.3f} tp {st.tp_today:.3f}/{m['M']['tp']:.3f}")
    for k in ("yield_10y", "tp", "policy_avg"):
        e = [getattr(r, k) for r in res[c]]; sh = m[{"yield_10y": "y", "tp": "tp", "policy_avg": "pbar"}[k]]
        print(f"   {k}: max |engine - sheet| {max(abs(a - b) for a, b in zip(e, sh)) * 100:.2f}bp")
