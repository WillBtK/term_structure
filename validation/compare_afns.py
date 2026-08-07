"""Compare our plain-AFNS output against the SF Fed Treasury Yield Premiums.

The comparison core here is pure and unit-tested: it takes two *canonical*
tidy frames and produces per-maturity metrics, a pass/fail verdict, and (when
matplotlib is available) overlay plots.

Canonical frame schema (both ``ours`` and ``benchmark``)
--------------------------------------------------------
A long-format ``pandas.DataFrame`` with columns:

    date          datetime64
    maturity      float (years)
    term_premium  float (decimal)
    expected_rate float (decimal)   # risk-neutral / expected-average-short-rate
    fitted_yield  float (decimal)

``sf_fed.load_sf_fed_series`` returns the benchmark in this schema. Producing
``ours`` from the vendored engine (Kalman/MLE on the US GSW panel, decomposed
into the risk-neutral path and the term premium) is the model-integration step
that runs in a full-stack, egress-open environment — see README.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from .config import (
    GATE_MATURITIES,
    MAX_ABS_BIAS_BP_TP,
    MIN_CORRELATION_TP,
    QUANTITIES,
    REPORTS_DIR,
)

__all__ = ["align", "metrics", "evaluate_gate", "plot_overlays", "run_comparison"]

logger = logging.getLogger(__name__)

_KEYS = ["date", "maturity"]


def _validate_frame(df: pd.DataFrame, name: str) -> None:
    required = set(_KEYS) | set(QUANTITIES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} frame is missing columns: {sorted(missing)}")


def align(ours: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the two series on (date, maturity).

    Returns a frame with, per quantity, ``<q>_ours`` and ``<q>_bench`` columns.
    """
    _validate_frame(ours, "ours")
    _validate_frame(benchmark, "benchmark")
    o = ours[_KEYS + QUANTITIES].copy()
    b = benchmark[_KEYS + QUANTITIES].copy()
    for f in (o, b):
        f["date"] = pd.to_datetime(f["date"])
        f["maturity"] = f["maturity"].astype(float)
    merged = o.merge(b, on=_KEYS, suffixes=("_ours", "_bench"))
    if merged.empty:
        logger.warning("No overlapping (date, maturity) rows between the series.")
    return merged


def metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-(maturity, quantity) comparison metrics.

    Columns: maturity, quantity, n, rmse_bp, mean_bias_bp, mean_abs_diff_bp,
    correlation. Differences are ours - bench, reported in basis points.
    """
    rows = []
    for maturity, grp in merged.groupby("maturity"):
        for q in QUANTITIES:
            a = grp[f"{q}_ours"].to_numpy(dtype=float)
            b = grp[f"{q}_bench"].to_numpy(dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            n = a.size
            if n == 0:
                continue
            diff_bp = (a - b) * 1e4  # decimal -> basis points
            corr = float(np.corrcoef(a, b)[0, 1]) if n >= 2 and a.std() > 0 and b.std() > 0 else np.nan
            rows.append(
                {
                    "maturity": float(maturity),
                    "quantity": q,
                    "n": int(n),
                    "rmse_bp": float(np.sqrt(np.mean(diff_bp**2))),
                    "mean_bias_bp": float(np.mean(diff_bp)),
                    "mean_abs_diff_bp": float(np.mean(np.abs(diff_bp))),
                    "correlation": corr,
                }
            )
    return pd.DataFrame(rows).sort_values(["quantity", "maturity"]).reset_index(drop=True)


def evaluate_gate(metric_df: pd.DataFrame) -> Dict:
    """Apply the term-premium "close tracking" criteria.

    Gate: at each maturity in GATE_MATURITIES, term-premium correlation >=
    MIN_CORRELATION_TP and absolute level bias |mean_bias_bp| <=
    MAX_ABS_BIAS_BP_TP. Correlation captures shape/co-movement; the bias bound
    captures the level. (The premium is weakly identified, so we require good
    tracking and a bounded level gap, not the fitted-yield-grade tolerance.)
    Returns a dict with ``passed`` (bool) and a per-maturity ``details`` list.
    """
    details = []
    passed = True
    tp = metric_df[metric_df["quantity"] == "term_premium"]
    for mat in GATE_MATURITIES:
        row = tp[tp["maturity"] == mat]
        if row.empty:
            details.append({"maturity": mat, "status": "MISSING", "reason": "no overlap"})
            passed = False
            continue
        r = row.iloc[0]
        corr_ok = bool(np.isfinite(r["correlation"]) and r["correlation"] >= MIN_CORRELATION_TP)
        bias_ok = bool(abs(r["mean_bias_bp"]) <= MAX_ABS_BIAS_BP_TP)
        ok = bool(corr_ok and bias_ok)
        passed &= ok
        details.append(
            {
                "maturity": mat,
                "status": "PASS" if ok else "FAIL",
                "correlation": float(r["correlation"]),
                "mean_bias_bp": float(r["mean_bias_bp"]),
                "mean_abs_diff_bp": float(r["mean_abs_diff_bp"]),
                "corr_ok": corr_ok,
                "bias_ok": bias_ok,
            }
        )
    return {"passed": bool(passed), "details": details}


def plot_overlays(merged: pd.DataFrame, out_dir=REPORTS_DIR) -> list:
    """Write term-premium overlay plots per gate maturity. Optional (matplotlib).

    Returns the list of written file paths (empty if matplotlib is unavailable).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping overlay plots.")
        return []

    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mat in GATE_MATURITIES:
        grp = merged[merged["maturity"] == mat].sort_values("date")
        if grp.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(grp["date"], grp["term_premium_ours"] * 100, label="ours (AFNS)")
        ax.plot(grp["date"], grp["term_premium_bench"] * 100, label="SF Fed", linestyle="--")
        ax.set_title(f"Term premium — {mat:g}y")
        ax.set_ylabel("percent")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / f"term_premium_{mat:g}y.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def run_comparison(
    ours: pd.DataFrame,
    benchmark: pd.DataFrame,
    make_plots: bool = True,
    out_dir=REPORTS_DIR,
) -> Dict:
    """Full comparison: align, metrics, gate, optional plots, write report.

    Returns a dict: ``{metrics, gate, plots, report_path}``.
    """
    from pathlib import Path

    merged = align(ours, benchmark)
    metric_df = metrics(merged)
    gate = evaluate_gate(metric_df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_path = out_dir / "comparison_metrics.csv"
    metric_df.to_csv(metric_path, index=False)

    plots = plot_overlays(merged, out_dir) if make_plots else []

    report_path = out_dir / "validation_report.md"
    _write_report(report_path, metric_df, gate, plots)
    return {"metrics": metric_df, "gate": gate, "plots": plots, "report_path": report_path}


def _write_report(path, metric_df: pd.DataFrame, gate: Dict, plots: list) -> None:
    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"
    lines = [
        "# US plain-AFNS vs SF Fed Treasury Yield Premiums",
        "",
        f"**Term-premium gate: {verdict}**",
        "",
        f"Criteria (close tracking): term-premium correlation ≥ {MIN_CORRELATION_TP} "
        f"and |level bias| ≤ {MAX_ABS_BIAS_BP_TP:g} bp at "
        f"{', '.join(f'{m:g}y' for m in GATE_MATURITIES)}.",
        "",
        "## Gate detail",
        "",
    ]
    for d in gate["details"]:
        if d["status"] == "MISSING":
            lines.append(f"- {d['maturity']:g}y: MISSING ({d['reason']})")
        else:
            lines.append(
                f"- {d['maturity']:g}y: {d['status']} — corr={d['correlation']:.4f}, "
                f"bias={d['mean_bias_bp']:+.1f}bp"
            )
    try:
        table = metric_df.to_markdown(index=False)
    except ImportError:  # 'tabulate' not installed
        table = "```\n" + metric_df.to_string(index=False) + "\n```"
    lines += ["", "## All metrics", "", table, ""]
    if plots:
        lines += ["## Plots", ""] + [f"- `{p.name}`" for p in plots] + [""]
    path.write_text("\n".join(lines))
