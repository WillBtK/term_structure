"""Tests for the pure comparison core (no network, no model run)."""
import numpy as np
import pandas as pd
import pytest

from validation import config
from validation.compare_afns import align, evaluate_gate, metrics
from validation.sf_fed import ColumnSpec, normalize


def _make_frame(dates, maturities, tp, exp, fit):
    rows = []
    for d in dates:
        for m in maturities:
            rows.append(
                {
                    "date": d,
                    "maturity": m,
                    "term_premium": tp(d, m),
                    "expected_rate": exp(d, m),
                    "fitted_yield": fit(d, m),
                }
            )
    return pd.DataFrame(rows)


def test_perfect_match_passes_gate():
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    mats = [5.0, 10.0]
    rng = np.random.default_rng(0)
    tp_map = {(d, m): float(rng.normal()) / 100 for d in dates for m in mats}
    tp = lambda d, m: tp_map[(d, m)]
    exp = lambda d, m: 0.02
    fit = lambda d, m: tp_map[(d, m)] + 0.02
    ours = _make_frame(dates, mats, tp, exp, fit)
    bench = ours.copy()
    merged = align(ours, bench)
    md = metrics(merged)
    gate = evaluate_gate(md)
    assert gate["passed"] is True
    tp_metrics = md[md["quantity"] == "term_premium"]
    assert tp_metrics["mean_abs_diff_bp"].max() == pytest.approx(0.0, abs=1e-9)
    assert tp_metrics["correlation"].min() == pytest.approx(1.0, abs=1e-9)


def test_large_divergence_fails_gate():
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    mats = [5.0, 10.0]
    rng = np.random.default_rng(1)
    base = {(d, m): float(rng.normal()) / 100 for d in dates for m in mats}
    ours = _make_frame(dates, mats, lambda d, m: base[(d, m)], lambda d, m: 0.02,
                       lambda d, m: base[(d, m)] + 0.02)
    # Benchmark: add a 50bp constant offset -> mean|diff| ~50bp >> 15bp gate.
    bench = _make_frame(dates, mats, lambda d, m: base[(d, m)] + 0.0050, lambda d, m: 0.02,
                        lambda d, m: base[(d, m)] + 0.02)
    gate = evaluate_gate(metrics(align(ours, bench)))
    assert gate["passed"] is False
    # correlation still high (constant offset) but diff gate fails.
    for d in gate["details"]:
        assert d["status"] == "FAIL"
        assert d["diff_ok"] is False


def test_align_requires_columns():
    good = _make_frame(pd.date_range("2020-01-01", periods=2, freq="MS"), [5.0],
                       lambda d, m: 0.0, lambda d, m: 0.0, lambda d, m: 0.0)
    bad = good.drop(columns=["term_premium"])
    with pytest.raises(ValueError):
        align(bad, good)


def test_gate_missing_maturity_is_fail():
    dates = pd.date_range("2015-01-01", periods=30, freq="MS")
    # Only 5y present; 10y missing from the overlap -> gate must fail.
    ours = _make_frame(dates, [5.0], lambda d, m: 0.0, lambda d, m: 0.0, lambda d, m: 0.0)
    bench = ours.copy()
    gate = evaluate_gate(metrics(align(ours, bench)))
    assert gate["passed"] is False
    statuses = {d["maturity"]: d["status"] for d in gate["details"]}
    assert statuses[10.0] == "MISSING"


def test_sf_fed_normalize_wide_to_tidy():
    # Minimal wide table -> canonical tidy frame, percent -> decimal.
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "tp5": [0.50, 0.60],      # percent
            "tp10": [1.00, 1.10],
            "er5": [1.50, 1.40],
            "er10": [1.80, 1.70],
            "fit5": [2.00, 2.00],
            "fit10": [2.80, 2.80],
        }
    )
    spec = ColumnSpec(
        date_column="Date",
        values_in_percent=True,
        term_premium={5.0: "tp5", 10.0: "tp10"},
        expected_rate={5.0: "er5", 10.0: "er10"},
        fitted_yield={5.0: "fit5", 10.0: "fit10"},
    )
    tidy = normalize(raw, spec)
    assert set(tidy["maturity"]) == {5.0, 10.0}
    row = tidy[(tidy["maturity"] == 10.0) & (tidy["date"] == pd.Timestamp("2020-01-31"))].iloc[0]
    assert row["term_premium"] == pytest.approx(0.01)      # 1.00% -> 0.01
    assert row["fitted_yield"] == pytest.approx(0.028)


def test_sf_fed_normalize_empty_spec_raises():
    raw = pd.DataFrame({"Date": ["2020-01-31"], "tp5": [0.5]})
    with pytest.raises(ValueError):
        normalize(raw, ColumnSpec(date_column="Date"))
