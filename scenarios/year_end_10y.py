"""Baseline year-end ten-year yields, end-2026 to end-2035, on the proposed construction
(validation/FINDINGS_yield_forecast_backtest.md): rate leg converging from today's priced
level to the house policy path at 1 - 0.8^h; US premium = fair value on in-house drivers +
0.5^h of today's deviation; spread markets move with the US premium at beta and their own
drivers, the own inflation channel scaled to its measured response. Reads the sheet.

    python -m scenarios.year_end_10y
"""
import json
from playwright.sync_api import sync_playwright
JS = r"""() => {
  activate('Baseline');
  const BASIS = {US: 0.277, UK: 0, DE: 0, JP: 0};       // policy rate -> curve short rate (US: house rate = FF lower bound; curve = midpoint + 0.152)
  const S_INFL = {UK: 0.15, DE: 0.75, JP: 0.0};          // measured premium response
  const RHO_US = 0.5;
  const avg = (path, h, span=10) => { let s=0,n=0; for(let x=h+0.125; x<h+span; x+=0.25){ s+=step(path,x); n++; } return s/n; };
  const drv = (c, sc, h) => Object.assign(driversAt(c, sc, h), {expected_inflation: avg(sc.inflation, h)});
  const usS = scenarioFor('US'), U = D.markets.US, us0 = drv('US', usS, 0);
  const usRes = U.tp_today - usPremium(us0);
  const tpUS = h => usPremium(drv('US', usS, h)) + Math.pow(RHO_US, h) * usRes;
  const out = {asof: D.asof, year_frac: D.year_frac, rows: {}};
  for(const c of ['US','UK','DE','JP']){
    const M = D.markets[c], sc = scenarioFor(c), lam = lamFor(SCEN, c);
    const E0 = M.y_today - M.tp_today, own0 = drv(c, sc, 0);
    const r = {today: M.y_today, tp_today: M.tp_today, E0, lam, cols: {}};
    for(let Y=2026; Y<=2035; Y++){
      const h = hz(Y), w = 1 - Math.pow(lam, h);
      const V = avg(sc.policy_rate, h) + BASIS[c];
      const P = E0 + w * (V - E0);
      let tp;
      if(c === 'US') tp = tpUS(h);
      else {
        const m = M.model, usH = drv('US', usS, h), ownH = drv(c, sc, h);
        const total = spreadPremium(c, usH, ownH, U.tp_today) - spreadPremium(c, us0, own0, U.tp_today);
        const pick = (a, b) => Object.assign({}, a, {expected_inflation: b.expected_inflation, inflation_uncertainty: b.inflation_uncertainty});
        const infl = spreadPremium(c, pick(us0, usH), pick(own0, ownH), U.tp_today) - spreadPremium(c, us0, own0, U.tp_today);
        tp = M.tp_today + m.beta_us * (tpUS(h) - U.tp_today) + total - (1 - S_INFL[c]) * infl;
      }
      r.cols[Y] = {h, y: P + tp, P, tp, V};
    }
    out.rows[c] = r;
  }
  return out; }"""
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium"); pg = b.new_page()
    pg.goto("file:///home/user/term_structure/scenarios/sheet/index.html")
    pg.wait_for_function("typeof project === 'function' && typeof D === 'object'")
    o = pg.evaluate(JS); b.close()

Y = range(2026, 2036)
print("asof", o["asof"])
for lab, key in (("10y", "y"), ("rate leg", "P"), ("premium", "tp"), ("house 10y avg policy", "V")):
    print(f"\n{lab}\n{'':4}{'today':>7}" + "".join(f"{y:>7}" for y in Y))
    for c, r in o["rows"].items():
        first = {"y": r["today"], "P": r["E0"], "tp": r["tp_today"], "V": float("nan")}[key]
        print(f"{c:4}{first:7.2f}" + "".join(f"{r['cols'][str(y)][key]:7.2f}" for y in Y))
