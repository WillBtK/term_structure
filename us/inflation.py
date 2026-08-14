"""Breakeven inflation and the inflation-risk premium (separate-AFNS approach).

With a nominal term premium (`us.production_tp`) and a real term premium
(`us.real_tp`) estimated independently on the same maturities/dates, the standard
Fisher decomposition of the yield curve gives, at each maturity tau:

    nominal yield = E[avg nominal short] + nominal TP
    real yield    = E[avg real short]    + real TP
    breakeven = nominal yield - real yield = E[avg inflation] + (nominal TP - real TP)

so the **inflation-risk premium** is

    IRP(tau) = nominal TP(tau) - real TP(tau)

and the (model-implied) **expected average inflation** is the residual

    E[avg inflation](tau) = breakeven(tau) - IRP(tau).

Caveat (carried from the GSW TIPS header): TIPS are less liquid than nominals, so
the observed breakeven — and hence this IRP — also embeds a **TIPS-nominal
liquidity differential** (starkly visible in the 2008 crisis, when the 2y
breakeven went to ~ -4% on a liquidity flight, not a genuine deflation forecast).
Separating it needs a liquidity factor (D'Amico-Kim-Wei) — a documented
extension; here the IRP is reported inclusive of it, and flagged.

This is the "two separate AFNS, subtract" approach (chosen with the user); a joint
no-arbitrage nominal-real AFNS is the rigorous alternative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inflation_decomposition(nominal_tp: pd.DataFrame, real_tp: pd.DataFrame) -> pd.DataFrame:
    """Combine nominal and real production TP frames into the inflation split.

    Both inputs are canonical `_assemble_tp` frames (columns include ``date``,
    ``maturity``, ``term_premium`` [realised, observed-based], ``observed_yield``).
    Returns a tidy frame on the shared (date, maturity):
    ``(date, maturity, nominal_tp, real_tp, inflation_risk_premium, breakeven,
    expected_inflation)`` — all decimal.
    """
    keep = ["date", "maturity", "term_premium", "observed_yield"]
    n = nominal_tp[keep].rename(columns={"term_premium": "nominal_tp",
                                         "observed_yield": "nominal_yield"})
    r = real_tp[keep].rename(columns={"term_premium": "real_tp",
                                      "observed_yield": "real_yield"})
    m = n.merge(r, on=["date", "maturity"], how="inner").dropna()
    m["inflation_risk_premium"] = m["nominal_tp"] - m["real_tp"]
    m["breakeven"] = m["nominal_yield"] - m["real_yield"]
    m["expected_inflation"] = m["breakeven"] - m["inflation_risk_premium"]
    cols = ["date", "maturity", "nominal_tp", "real_tp", "inflation_risk_premium",
            "breakeven", "expected_inflation"]
    return m[cols].sort_values(["maturity", "date"]).reset_index(drop=True)


def summarize(decomp: pd.DataFrame, maturity: float = 10.0) -> dict:
    """Latest-date summary at ``maturity`` (decimal) for quick reporting."""
    d = decomp[decomp.maturity == maturity].dropna()
    if d.empty:
        return {}
    last = d.sort_values("date").iloc[-1]
    return {
        "date": str(last.date.date()), "maturity": maturity,
        "nominal_tp": float(last.nominal_tp), "real_tp": float(last.real_tp),
        "inflation_risk_premium": float(last.inflation_risk_premium),
        "breakeven": float(last.breakeven),
        "expected_inflation": float(last.expected_inflation),
    }
