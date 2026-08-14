"""Tests for the TIPS real-curve reconstruction + inflation decomposition (no network)."""
import numpy as np
import pandas as pd

from us import config as cfg
from us.reconstruct import svensson_yield
from us.tips import reconstruct_real, parse_tips_csv
from us.inflation import inflation_decomposition


def _params():
    # two dates; the first has only 2-10y TIPS (no 20y), the second the full 2-20y.
    idx = pd.to_datetime(["2005-06-30", "2015-06-30"])
    df = pd.DataFrame({
        "BETA0": [2.0, 1.0], "BETA1": [-1.0, -2.0], "BETA2": [0.5, 1.0],
        "BETA3": [0.2, 0.1], "TAU1": [1.5, 2.0], "TAU2": [8.0, 10.0],
        "TIPSY02": [1.0, -0.5], "TIPSY05": [1.2, -0.3], "TIPSY10": [1.6, 0.1],
        "TIPSY20": [np.nan, 0.6],           # 20y missing on the first date
        "BKEVEN02": [2.3, 1.8], "BKEVEN10": [2.5, 2.0],
    }, index=idx)
    df.index.name = "date"
    return df


def test_reconstruct_real_matches_svensson_and_units():
    p = _params()
    out = reconstruct_real(p, grid=np.array([5.0, 10.0]), mask_by_availability=False)
    # decimal units, matches the shared Svensson formula
    exp5 = np.asarray(svensson_yield(5.0, p.BETA0, p.BETA1, p.BETA2, p.BETA3, p.TAU1, p.TAU2)) / 100.0
    assert np.allclose(out[5.0].to_numpy(), exp5)
    assert out.abs().to_numpy().max() < 0.2          # ~ percent-scale in decimal


def test_real_curve_ragged_both_ends():
    p = _params()
    out = reconstruct_real(p, grid=np.array([1.0, 2.0, 10.0, 20.0]))
    # 1y is below the shortest traded TIPS (2y) on both dates -> NaN (no short end)
    assert out[1.0].isna().all()
    # 20y present only on the second date (TIPSY20 published then)
    assert np.isnan(out.loc[pd.Timestamp("2005-06-30"), 20.0])
    assert np.isfinite(out.loc[pd.Timestamp("2015-06-30"), 20.0])
    # 2y and 10y available on both
    assert out[2.0].notna().all() and out[10.0].notna().all()


def test_parse_tips_csv_shape():
    text = (
        "Note: research product\n"
        "Date,BETA0,BETA1,BETA2,BETA3,TAU1,TAU2,TIPSY02,TIPSY10,BKEVEN02,BKEVEN10\n"
        "2010-01-04,1.5,-1.0,0.5,0.1,2.0,9.0,0.80,1.40,2.10,2.30\n"
        "2010-01-05,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n"
    )
    df = parse_tips_csv(text)
    assert list(df.index) == [pd.Timestamp("2010-01-04")]   # all-NA param row dropped
    assert df.loc[pd.Timestamp("2010-01-04"), "TIPSY10"] == 1.40


def test_inflation_decomposition_identities():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    def frame(tp, y):
        rows = []
        for mat in (5.0, 10.0):
            rows.append(pd.DataFrame({"date": dates, "maturity": mat,
                                      "term_premium": tp[mat], "observed_yield": y[mat]}))
        return pd.concat(rows, ignore_index=True)
    nom = frame({5.0: [0.010, 0.011], 10.0: [0.015, 0.016]},
                {5.0: [0.020, 0.021], 10.0: [0.025, 0.026]})
    real = frame({5.0: [0.004, 0.005], 10.0: [0.006, 0.007]},
                 {5.0: [-0.002, -0.001], 10.0: [0.003, 0.004]})
    d = inflation_decomposition(nom, real)
    # IRP = nominal TP - real TP
    assert np.allclose(d.inflation_risk_premium, d.nominal_tp - d.real_tp)
    # breakeven = nominal yield - real yield
    row = d[(d.maturity == 5.0) & (d.date == dates[0])].iloc[0]
    assert np.isclose(row.breakeven, 0.020 - (-0.002))
    # expected inflation = breakeven - IRP
    assert np.allclose(d.expected_inflation, d.breakeven - d.inflation_risk_premium)
