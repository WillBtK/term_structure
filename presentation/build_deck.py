"""Assemble the data payload for the presentation deck.

Every number on a slide traces to this builder and through it to the repo's own
models and caches -- nothing is typed into the deck by hand. Output:
``presentation/data/deck_payload.json`` (~30KB), inlined into ``index.html`` the
same way the scenario sheet's payload is.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "deck_payload.json"
MARKETS = ("US", "UK", "DE", "JP")


def monthly(s: pd.Series, start: str | None = None, nd: int = 2) -> dict:
    a = s.dropna()
    if start:
        a = a.loc[start:]
    if a.empty:
        return {"t0": None, "v": []}
    a = a.resample("MS").mean()
    a = a.loc[a.first_valid_index():a.last_valid_index()]
    t0 = a.index[0]
    return {"t0": round(t0.year + (t0.month - 1) / 12.0, 4),
            "v": [None if pd.isna(v) else round(float(v), nd) for v in a]}


def build() -> dict:
    warnings.filterwarnings("ignore")
    from model.tp_drivers import build_panel, fit_market
    from model.tp_relative import build_pair, fit_relative

    panels = {c: build_panel(c) for c in MARKETS}
    us = panels["US"]
    us_m = fit_market("US", us)
    spread = {c: fit_relative(c, build_pair(c, panels), us_m)
              for c in MARKETS if c != "US"}

    # --- Act I: the object -------------------------------------------------------
    # Yield and premium must share one monthly convention or their difference -- the
    # expectations leg -- picks up an artefact. Both come from macro.tp_history, which
    # is month-END throughout and history-extends Germany on both series.
    from macro.tp_history import long_yield_monthly, spliced_tp
    ten = {c: long_yield_monthly(10.0)[c] for c in MARKETS}
    # The DISPLAY series come from the spliced monthly history, not the daily panel.
    # The daily Bund curve starts 1997-08, but Germany has a committed monthly exhibit
    # back to 1972-09 which macro.tp_history level-matches onto the production series at
    # a single month-end -- preserving the 1970s shape and removing only the level
    # offset that the historical fit's full-sample mean introduces. Showing four markets
    # from 1997 when three of them run to the 1970s understates what we have.
    hist = spliced_tp(10.0)
    tp = {c: hist[c].dropna() for c in MARKETS}
    y_us = ten["US"].dropna()
    exp_us = (y_us - tp["US"]).dropna()          # expectations leg = yield - premium
    today = {c: {"y": round(float(ten[c].dropna().iloc[-1]), 2),
                 "tp": round(float(tp[c].iloc[-1]), 2)} for c in MARKETS}

    # --- Act II: common, but never naively --------------------------------------
    # PC1 share on the balanced four-market panel, levels and 5y-rolling on changes
    # PC1 stays on the balanced overlap -- a common factor needs all four present
    panel4 = pd.concat({c: tp[c] for c in MARKETS}, axis=1).loc["1997-08":].dropna()
    z = (panel4 - panel4.mean()) / panel4.std()
    pc1_level = float(np.linalg.eigvalsh(z.corr())[-1] / 4)
    dz = panel4.diff().dropna()
    dzz = (dz - dz.mean()) / dz.std()
    pc1_chg = float(np.linalg.eigvalsh(dzz.corr())[-1] / 4)
    roll = []
    idx = panel4.index
    for i in range(60, len(panel4)):
        w = dzz.iloc[max(0, i - 60):i]
        roll.append((idx[i], float(np.linalg.eigvalsh(w.corr())[-1] / 4)))
    pc1_roll = pd.Series({t: v for t, v in roll})
    # JP ownership relative to the US, the divergence engine
    rel_own_jp = (panels["JP"]["price_insensitive"]
                  - us["price_insensitive"]).dropna()
    # naive (beta-only) vs full R2, full sample and 2010+ -- computed, not typed
    def _r2(y: pd.Series, yhat: pd.Series) -> float:
        a = pd.concat([y, yhat], axis=1).dropna()
        ss = ((a.iloc[:, 0] - a.iloc[:, 1]) ** 2).sum()
        return float(1 - ss / ((a.iloc[:, 0] - a.iloc[:, 0].mean()) ** 2).sum())
    naive_full = {}
    for c in ("UK", "DE", "JP"):
        m = spread[c]
        pair = build_pair(c, panels)
        naive_fit = pair["us_tp"] * np.polyfit(pair["us_tp"].values,
                                               pair["tp"].values, 1)[0] \
            + np.polyfit(pair["us_tp"].values, pair["tp"].values, 1)[1]
        full_fit = pd.Series(m.intercept + m.beta_us * pair["us_tp"], index=pair.index)
        for k in m.coef:
            col = f"rel_{k}"
            if col in pair.columns:
                full_fit = full_fit + m.coef[k] * pair[col]
        naive_full[c] = {
            "full_sample": [round(_r2(pair["tp"], naive_fit), 2),
                            round(_r2(pair["tp"], full_fit), 2)],
            "post2010": [round(_r2(pair["tp"].loc["2010":], naive_fit.loc["2010":]), 2),
                         round(_r2(pair["tp"].loc["2010":], full_fit.loc["2010":]), 2)]}

    # --- Act III: what drives the US premium ------------------------------------
    # the fit's own design matrix (kink applied there); fiscal is an adjust channel
    # measured from TODAY's base, so it contributes ZERO to the historical fair value
    from model.tp_drivers import design
    Xd = design(us, list(us_m.drivers)).dropna()
    fair_full = us_m.intercept + sum(us_m.coef[k] * Xd[k] for k in us_m.drivers)
    # inflation-only refit on the same sample
    a = pd.concat([us["tp"], us["expected_inflation"]], axis=1).dropna()
    b1, b0 = np.polyfit(a["expected_inflation"].values, a["tp"].values, 1)
    fair_infl = b0 + b1 * a["expected_inflation"]
    r2_infl = _r2(a["tp"], fair_infl)
    r2_full = float(us_m.r2)
    # the QE-decade band: inflation-only over-prediction vs the full model's fit
    qe = slice("2009", "2021")
    resid_i = (a["tp"] - fair_infl)
    resid_f = (us["tp"] - fair_full).dropna()
    qe_bias_infl = float(resid_i.loc[qe].mean())          # negative = over-predicts
    rmse = lambda s: float(np.sqrt((s.dropna() ** 2).mean()))
    rmse_cut = {
        "pre2009": round(1 - rmse(resid_f.loc[:"2008"]) / rmse(resid_i.loc[:"2008"]), 2),
        "qe": round(1 - rmse(resid_f.loc[qe]) / rmse(resid_i.loc[qe]), 2),
        "qt": round(1 - rmse(resid_f.loc["2022":]) / rmse(resid_i.loc["2022":]), 2)}

    # Okun's mirage: level vs 10y-window vol, six markets' worth already reduced to
    # the fitted annual panel in model.okun -- rebuild the scatter from US core CPI
    from model.okun import fit_hockey_stick
    from validation.okun_mirage import annual as okun_annual
    from validation.okun_mirage import build_panel as okun_panel
    A = okun_annual(okun_panel())
    pts = [[round(float(r["level"]), 2), round(float(r["vol"]), 2)]
           for _, r in A.iterrows()]
    ok = fit_hockey_stick(A["level"], A["vol"])
    okun = {"pts": pts, "kink": round(float(ok.kink), 2),
            "a": round(float(ok.floor), 3), "b": round(float(ok.slope), 3),
            "r2": round(float(ok.r2), 3)}

    # fiscal: the CBO 5y-ahead vintage series and the break
    from macro.fetch_cbo import load as cbo_load
    cbo = cbo_load()
    fiscal_hist = [[round(d.year + (d.month - 1) / 12.0, 2), round(float(v), 2)]
                   for d, v in cbo.items()]

    # fair value today: the engine's live driver readings, priced by premium(),
    # decomposed into per-driver contributions on the transformed inputs
    from scenarios.engine import latest_states
    st = latest_states(("US",))["US"]
    d_today = dict(st.drivers_today)
    fair_today = us_m.premium(**d_today)
    kinked = dict(d_today)
    kinked["expected_inflation"] = max(kinked["expected_inflation"] - 2.0, 0.0)
    waterfall = ([["Intercept", round(float(us_m.intercept), 2)]]
                 + [[k, round(float(us_m.coef[k] * kinked[k]), 2)]
                    for k in us_m.drivers]
                 + [["fair value", round(float(fair_today), 2)]])
    tp_state = round(float(st.tp_today), 2)

    # --- Q5: bonds stop hedging — the one OBSERVABLE link in the chain ----------
    # Brixton et al. (AQR 2023): growth vol pushes the stock-bond correlation down
    # (bonds hedge), inflation vol pushes it up (bonds stop hedging). The ratio form
    # is what we use -- see validation/FINDINGS_sbc.md.
    from validation.sbc_macro import build_panel as sbc_panel
    sp = sbc_panel()
    sbc_pts = [[round(float(r["sigma_pi"] / r["sigma_g"]), 3), round(float(r["sbc"]), 3),
                r["country"]] for _, r in sp.iterrows()
               if r["sigma_g"] and np.isfinite(r["sigma_pi"] / r["sigma_g"])
               and r["sigma_pi"] / r["sigma_g"] < 4]
    x = np.array([q[0] for q in sbc_pts]); yv = np.array([q[1] for q in sbc_pts])
    sb, sa = np.polyfit(x, yv, 1)
    # The US series comes from SHILLER, not the six-market panel. That panel starts in
    # 1981 and so contains one inflation regime change; a relationship shown on one
    # turning point is an anecdote. Shiller's monthly data runs from 1871, which puts
    # the Great Inflation, the disinflation and two earlier inflations in the picture.
    from macro import shiller
    us_sbc = shiller.stock_bond_correlation()
    eras = shiller.regime_means(us_sbc)
    sbc = {"pts": sbc_pts, "slope": round(float(sb), 2), "a": round(float(sa), 3),
           "us": monthly(us_sbc, nd=3),
           "eras": [[r["era"], r["regime"], r["corr"]] for _, r in eras.iterrows()],
           "us_from": int(us_sbc.index[0].year), "window_y": shiller.WINDOW // 12,
           "n": len(sbc_pts), "markets": int(sp["country"].nunique())}

    # --- Q6: the official bid, US ------------------------------------------------
    own_us = monthly(us["price_insensitive"], nd=1)

    # --- Q7: growth -> deficit, the sensitivity chain ----------------------------
    from macro.fiscal_rules import GAP_CLOSURE, SEMI_ELASTICITY
    from macro.fiscal_projections import growth_baseline, path as fiscal_path
    from macro.fiscal_rules import adjusted_deficit
    off = fiscal_path("US")
    base_g = {k: v for k, v in enumerate(
        [growth_baseline("US").get(pd.Timestamp.today().year + k, 1.8)
         for k in range(0, 12)])}
    slow = {k: v - 1.0 for k, v in base_g.items()}
    chain = {"official": [round(off.get(k, list(off.values())[-1]), 2) for k in range(1, 11)],
             "slower": [round(v, 2) for k, v in
                        sorted(adjusted_deficit("US", slow, base_g, off).items())
                        if 1 <= k <= 10],
             "eps": SEMI_ELASTICITY, "closure": GAP_CLOSURE}

    # --- Q6: the fiscal break, tested rather than asserted ------------------------
    from model.tp_drivers import fiscal_break_search
    brk = fiscal_break_search(us)

    # --- Q6: the ownership channel, as a COEFFICIENT ------------------------------
    # "a very large buyer who does not care about price" is only an argument if the
    # buyer's size maps to a number of basis points. It does: the driver enters the US
    # fit at coef["price_insensitive"] per percentage point of GDP held.
    pi = us["price_insensitive"].dropna()
    own_bp = -float(us_m.coef["price_insensitive"]) * 100      # bp per pp of GDP, +ve
    pi_lo, pi_hi, pi_now = float(pi.min()), float(pi.max()), float(pi.iloc[-1])
    own = {"bp_per_pp": round(own_bp, 1),
           "lo": round(pi_lo, 1), "hi": round(pi_hi, 1), "now": round(pi_now, 1),
           "hi_date": str(pi.idxmax().date()), "lo_date": str(pi.idxmin().date()),
           # what the build-up took out, and what the unwind has put back
           "buildup_bp": round((pi_hi - pi_lo) * own_bp),
           "unwind_bp": round((pi_hi - pi_now) * own_bp)}

    # --- Act V: what it says now -------------------------------------------------
    tp_now = float(tp_state)
    resid_now = tp_now - float(fair_today)
    sigma = float(us_m.rmse)          # the fit's own in-sample dispersion
    rho_paths = {str(r): [round(float(fair_today + resid_now * r ** h), 2)
                          for h in range(11)] for r in (1.0, 0.9, 0.0)}

    return {
        "asof": str(panel4.index[-1].date()),
        "today": today,
        "tp": {c: monthly(tp[c]) for c in MARKETS},
        "y_us": monthly(y_us), "exp_us": monthly(exp_us),
        "pc1": {"level": round(pc1_level, 2), "chg": round(pc1_chg, 2),
                "roll": monthly(pc1_roll)},
        "rel_own_jp": monthly(rel_own_jp, nd=1),
        "naive_full": naive_full,
        "fair_full": monthly(fair_full), "fair_infl": monthly(fair_infl),
        "r2": {"infl": round(r2_infl, 3), "full": round(r2_full, 3)},
        "qe": {"bias_infl": round(qe_bias_infl, 2), "rmse_cut": rmse_cut},
        "okun": okun,
        "fiscal_hist": fiscal_hist,
        "fiscal": {"coef": round(float(us_m.adjust["fiscal"] * 100), 1),
                   "break_year": 2008},
        "waterfall": waterfall,
        "models": {c: {"beta": round(float(spread[c].beta_us), 2),
                       "r2": round(float(spread[c].r2), 2)} for c in spread},
        "now": {"tp": round(tp_now, 2), "fair": round(float(fair_today), 2),
                "resid": round(resid_now, 2), "sigma": round(sigma, 2),
                "z": round(resid_now / sigma, 1)},
        "rho_paths": rho_paths,
        "sbc": sbc, "own_us": own_us, "chain": chain,
        "brk": brk, "own": own,
    }


PAGES = (HERE / "index.html", HERE / "summary.html")
_MARK = "const P = "


def splice(payload: dict, pages=PAGES) -> list[Path]:
    """Inline the payload into the deck pages.

    This step used to be manual, and it drifted: the pages carried a payload built
    before the last three data changes, so every new field read as ``undefined`` and
    three slides threw. A build that writes the JSON but not the page it is read from
    has not finished. Rewrites only the ``const P = {...};`` line, so the page's own
    markup is untouched.
    """
    line = _MARK + json.dumps(payload) + ";"
    done = []
    for page in pages:
        if not page.is_file():
            continue
        src = page.read_text(encoding="utf-8").split("\n")
        hit = [i for i, l in enumerate(src) if l.startswith(_MARK)]
        if not hit:
            logger.warning("%s has no `%s` line to splice", page.name, _MARK.strip())
            continue
        src[hit[0]] = line
        page.write_text("\n".join(src), encoding="utf-8")
        done.append(page)
    return done


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    for page in splice(payload):
        print(f"spliced -> {page.name} ({page.stat().st_size:,} bytes)")
    print(f"asof {payload['asof']}  PC1 {payload['pc1']['level']:.2f}  "
          f"R2 infl {payload['r2']['infl']:.3f} full {payload['r2']['full']:.3f}  "
          f"gap {payload['now']['resid']:+.2f} ({payload['now']['z']:.1f} sigma)  "
          f"okun kink {payload['okun']['kink']}")
    print(f"wrote {OUT} ({len(OUT.read_text())} bytes)")


if __name__ == "__main__":
    _main()
