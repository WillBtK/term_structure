"""Year-end ten-year yields, end of this year to ten years out, read off the scenario
sheet itself (scenarios/sheet/index.html, ``project``) so the numbers can never disagree
with what the sheet shows. Construction and evidence:
validation/FINDINGS_yield_forecast_backtest.md.

    python -m scenarios.year_end_10y                      # Baseline, unrounded
    python -m scenarios.year_end_10y --scenario all --round --csv out.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHEET = ROOT / "scenarios" / "sheet" / "index.html"
CHROMIUM = "/opt/pw-browsers/chromium"
NAMES = {"US": "United States", "UK": "United Kingdom", "DE": "Germany", "JP": "Japan"}

_JS = """(name) => { activate(name);
  const out = {asof: D.asof, years: YE_YEARS, rows: {}};
  for(const c of MK){ const r = project(c, YE_YEARS.map(hz));
    out.rows[c] = {today: D.markets[c].y_today, y: r.map(x => x.y), pbar: r.map(x => x.pbar),
                   tp: r.map(x => x.tp)}; }
  return out; }"""


def evaluate(names: list) -> dict:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROMIUM)
        pg = b.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"file://{SHEET}")
        pg.wait_for_function("typeof project === 'function' && typeof D === 'object'")
        if names == ["all"]:
            names = pg.evaluate("SCEN_NAMES")
        out = {n: pg.evaluate(_JS, n) for n in names}
        b.close()
    if errs:
        raise RuntimeError(f"sheet raised: {errs[:3]}")
    return out


def table(res: dict, rnd: bool) -> list:
    q = (lambda v: round(v * 4) / 4) if rnd else (lambda v: v)
    lines = ["Country,Today (" + res["asof"] + ")," + ",".join(str(y) for y in res["years"])]
    for c, r in res["rows"].items():
        lines.append(",".join([NAMES[c], f"{r['today']:.2f}"] + [f"{q(v):.2f}" for v in r["y"]]))
    return lines


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="Baseline", help="a scenario name, or 'all'")
    ap.add_argument("--round", action="store_true", help="round to the nearest 0.25%%")
    ap.add_argument("--csv", help="write the table(s) here")
    a = ap.parse_args()
    res = evaluate([a.scenario])
    out = []
    for n, r in res.items():
        out += [f"# {n}"] + table(r, a.round) + [""]
    text = "\n".join(out)
    print(text)
    if a.csv:
        Path(a.csv).write_text(text)


if __name__ == "__main__":
    _main()
