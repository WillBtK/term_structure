"""Offline tests for the Bund liquidity 'noise' core (no network / no fetch)."""
import numpy as np
import pandas as pd

from validation.bund_liquidity import (noise_from_panel, drop_inflation_linked,
                                        _ns_design, _rms_residual)


def _panel_on_curve(date, mats_years, betas, lam=0.5, noise_sd=0.0, seed=0):
    """Build a per-security panel whose yields lie on a fixed-λ NS curve (betas)
    plus optional iid noise of sd ``noise_sd`` (decimal)."""
    rng = np.random.default_rng(seed)
    tau = np.asarray(mats_years, float)
    y = _ns_design(tau, lam) @ np.asarray(betas, float)
    y = y + rng.normal(0.0, noise_sd, size=tau.size)
    d = pd.Timestamp(date)
    return pd.DataFrame({
        "date": d,
        "maturity": [d + pd.Timedelta(days=int(round(t * 365.25))) for t in tau],
        "yield": y,
    })


def test_clean_curve_has_near_zero_noise():
    mats = np.linspace(1.5, 25, 20)
    df = _panel_on_curve("2020-06-15", mats, betas=[0.01, -0.005, 0.002], noise_sd=0.0)
    out = noise_from_panel(df)
    assert len(out) == 1
    assert out["noise"].iloc[0] < 1e-6          # on-curve -> ~0 residual
    assert out["n_bonds"].iloc[0] == len(mats)


def test_injected_noise_is_recovered():
    mats = np.linspace(1.5, 25, 40)
    sd = 0.0015                                  # 15 bp scatter
    df = _panel_on_curve("2011-09-30", mats, betas=[0.02, -0.01, 0.005],
                         noise_sd=sd, seed=42)
    out = noise_from_panel(df)
    rms = out["noise"].iloc[0]
    # RMS residual recovers the injected scatter to within ~40% (finite sample,
    # 3 dof absorbed by the fit).
    assert 0.6 * sd < rms < 1.4 * sd


def test_too_few_bonds_dropped():
    df = _panel_on_curve("2019-01-31", [2, 5, 10], betas=[0.01, 0, 0], noise_sd=0.001)
    out = noise_from_panel(df)                    # 3 bonds < min_bonds -> no row
    assert out.empty


def test_maturity_window_filters():
    # 5 in-window + 3 out-of-window (too short / too long); only in-window counted,
    # and here 5 < default min_bonds(6) so the date is dropped -> empty.
    mats = [0.2, 0.5, 2, 5, 10, 20, 28, 40]
    df = _panel_on_curve("2018-03-30", mats, betas=[0.01, 0, 0], noise_sd=0.0005)
    out = noise_from_panel(df)
    assert out.empty
    # widen min_bonds acceptance by adding more in-window bonds
    mats2 = [2, 3, 4, 5, 7, 10, 15, 20]           # 8 in-window
    df2 = _panel_on_curve("2018-03-30", mats2, betas=[0.01, 0, 0], noise_sd=0.0005)
    out2 = noise_from_panel(df2)
    assert len(out2) == 1 and out2["n_bonds"].iloc[0] == 8


def test_rms_residual_nan_when_too_few():
    assert np.isnan(_rms_residual(np.array([1.0, 2.0]), np.array([0.01, 0.02])))


def test_drop_inflation_linked_removes_dbri_and_il():
    panel = pd.DataFrame({
        "date": pd.Timestamp("2026-08-14"),
        "isin": ["DE0001102408", "DE0001030575", "DE0001030583", "DE000BU25042"],
        "maturity": pd.to_datetime(["2036-08-15", "2046-04-15", "2033-04-15", "2030-04-18"]),
        "yield": [0.024, 0.014, 0.009, 0.028],
        "name": ["DBR 0 08/15/26", "DBRI 0.1 04/15/46 I/L",
                 "DBRI 0.1 04/15/33 I/L", "OBL 2.4 04/18/30 191"],
    })
    out = drop_inflation_linked(panel)
    assert set(out["name"]) == {"DBR 0 08/15/26", "OBL 2.4 04/18/30 191"}
    # no-op when there is no name column (older cache schema)
    bare = panel.drop(columns=["name"])
    assert len(drop_inflation_linked(bare)) == len(bare)


def test_linker_contamination_inflates_noise():
    """A linker whose yield sits far off the nominal curve blows up the RMS; dropping
    it restores a small noise — the exact artefact drop_inflation_linked prevents."""
    d = pd.Timestamp("2026-08-14")
    mats = np.array([2, 3, 5, 7, 10, 15, 20, 30], float)
    ny = _ns_design(mats) @ np.array([0.025, -0.003, 0.004])   # nominal on an NS curve
    rows = [{"date": d, "isin": f"DE{i:012d}", "name": "DBR x",
             "maturity": d + pd.Timedelta(days=int(t * 365.25)), "yield": float(y)}
            for i, (t, y) in enumerate(zip(mats, ny))]
    rows.append({"date": d, "isin": "DE0001030575", "name": "DBRI 0.1 I/L",
                 "maturity": d + pd.Timedelta(days=int(20 * 365.25)), "yield": 0.005})  # real yield
    panel = pd.DataFrame(rows)
    with_linker = noise_from_panel(panel)["noise"].iloc[0]
    without = noise_from_panel(drop_inflation_linked(panel))["noise"].iloc[0]
    assert with_linker > 10 * without       # linker dominates the residual
    assert without < 1e-6                    # nominal-only fit is exact (on-curve)
