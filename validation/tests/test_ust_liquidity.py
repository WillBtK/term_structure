"""Offline tests for the US nominal Treasury 'noise' gauge (no network).

The gauge reads a few basis points of cross-sectional scatter, so the tests build
cross-sections whose true noise is KNOWN — bonds priced exactly off a Svensson curve
(zero noise) and the same bonds with a controlled perturbation — and check the measure
recovers it. That also pins the bond-math wiring end to end: a mistake in accrual or
compounding shows up as noise where there is none.
"""
import numpy as np
import pandas as pd
import pytest

from model.bond_math import disc_from_zero, price_from_curve
from us.fetch_fedinvest import nominal_coupon_bonds, parse_price_table
from us.reconstruct import svensson_yield
from validation.ust_liquidity import noise_from_panel

SETTLE = pd.Timestamp("2026-08-14")
# A plausible upward-sloping curve (GSW units: betas in percent, taus in years).
PARAMS = pd.Series({"BETA0": 4.2, "BETA1": -1.6, "BETA2": -1.0, "BETA3": 0.8,
                    "TAU1": 1.5, "TAU2": 8.0})


def _zero(t):
    return svensson_yield(np.maximum(np.asarray(t, dtype=float), 1e-6),
                          *[PARAMS[k] for k in
                            ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")]) / 100.0


def _synthetic_cross_section(n=24, noise_bp=0.0, seed=0):
    """Bonds spread 1-30y, each priced EXACTLY off the curve, then optionally nudged by
    a known yield error expressed in bp (converted to price via a duration step)."""
    rng = np.random.default_rng(seed)
    disc = disc_from_zero(_zero)
    mats = [SETTLE + pd.Timedelta(days=int(365.25 * t))
            for t in np.linspace(1.0, 30.0, n)]
    coupons = np.round(rng.uniform(0.5, 6.0, n), 3)
    rows = []
    for mat, cpn in zip(mats, coupons):
        p = price_from_curve(SETTLE, mat, float(cpn), disc)
        if noise_bp:
            eps = rng.normal(0.0, noise_bp / 1e4)
            tau = (mat - SETTLE).days / 365.25
            p *= np.exp(-eps * tau)          # first-order price move for a yield shock
        rows.append({"date": SETTLE, "cusip": f"X{len(rows):09d}",
                     "kind": "MARKET BASED NOTE", "coupon": float(cpn),
                     "maturity": mat, "price": p})
    return pd.DataFrame(rows)


def test_bonds_priced_off_the_curve_show_essentially_zero_noise():
    """The decisive wiring check. If accrual, compounding or the discount convention
    were wrong, these bonds — priced off the very curve they are compared to — would
    carry residuals, and the gauge would report illiquidity that is not there."""
    panel = _synthetic_cross_section()
    out = noise_from_panel(panel, gsw_params=pd.DataFrame([PARAMS], index=[SETTLE]))
    assert len(out) == 1
    # vs the true curve there is nothing to fit, so the residual is ~machine-exact
    assert out["noise_gsw"].iloc[0] * 1e4 < 0.2
    # the self-fit still carries its functional-form floor, but the decay grid keeps
    # it under a basis point instead of the 2-9 bp a single fixed decay leaves
    assert out["noise"].iloc[0] * 1e4 < 1.0


def test_injected_scatter_is_recovered_in_basis_points():
    panel = _synthetic_cross_section(noise_bp=8.0, seed=3)
    out = noise_from_panel(panel, gsw_params=pd.DataFrame([PARAMS], index=[SETTLE]))
    assert out["noise_gsw"].iloc[0] * 1e4 == pytest.approx(8.0, rel=0.45)
    # and the fitted measure lands close too, because the dof correction removes the
    # downward bias that plain RMS over six parameters would otherwise carry
    assert out["noise"].iloc[0] * 1e4 == pytest.approx(8.0, rel=0.5)
    # more scatter must read as more noise
    quiet = noise_from_panel(_synthetic_cross_section(noise_bp=2.0, seed=3),
                             gsw_params=pd.DataFrame([PARAMS], index=[SETTLE]))
    assert out["noise"].iloc[0] > quiet["noise"].iloc[0]


def test_thin_cross_sections_are_skipped_not_fitted():
    panel = _synthetic_cross_section(n=6)
    assert len(noise_from_panel(panel)) == 0


def test_maturity_window_excludes_the_very_short_and_very_long_ends():
    panel = _synthetic_cross_section(n=24)
    wide = noise_from_panel(panel)
    narrow = noise_from_panel(panel, min_mat=5.0, max_mat=20.0)
    assert narrow["n_bonds"].iloc[0] < wide["n_bonds"].iloc[0]


def test_only_fixed_coupon_nominal_notes_and_bonds_survive_the_filter():
    """Bills quote on a discount basis, TIPS off an inflation-adjusted principal and
    FRNs have no fixed schedule — each would be a large spurious residual."""
    panel = pd.DataFrame({"kind": ["MARKET BASED BILL", "MARKET BASED NOTE",
                                   "MARKET BASED BOND", "MARKET BASED FRN", "TIPS"]})
    assert list(nominal_coupon_bonds(panel)["kind"]) == ["MARKET BASED NOTE",
                                                         "MARKET BASED BOND"]


# The live table layout, copied from a real response body. RATE carries a PERCENT SIGN
# ("4.125%"), and bills quote "0.000%" rather than blank — an earlier version of this
# fixture wrote the rates bare, so the test passed while the parser coerced every live
# coupon to NaN. Keep the published formatting verbatim; that is the whole point of it.
_LIVE_HTML = """
<table>
<tr><th>CUSIP</th><th>SECURITY TYPE</th><th>RATE</th><th>MATURITY DATE</th>
    <th>CALL DATE</th><th>BUY</th><th>SELL</th><th>END OF DAY</th></tr>
<tr><td>912797UU9</td><td>MARKET BASED BILL</td><td>0.000%</td><td>08/18/2026</td>
    <td></td><td>99.99</td><td>99.99</td><td>99.990028</td></tr>
<tr><td>91282CJK8</td><td>MARKET BASED NOTE</td><td>4.125%</td><td>05/15/2034</td>
    <td></td><td>98.50</td><td>98.55</td><td>98.523438</td></tr>
<tr><td>912810TV0</td><td>MARKET BASED BOND</td><td>4.750%</td><td>11/15/2053</td>
    <td></td><td>99.10</td><td>99.20</td><td>99.156250</td></tr>
</table>
"""


def test_parses_the_live_table_layout():
    df = parse_price_table(_LIVE_HTML, pd.Timestamp("2026-08-14"))
    assert len(df) == 3
    row = df.set_index("cusip").loc["91282CJK8"]
    assert row["coupon"] == pytest.approx(4.125)
    assert row["maturity"] == pd.Timestamp("2034-05-15")
    assert row["price"] == pytest.approx(98.523438)
    assert row["kind"] == "MARKET BASED NOTE"
    assert set(nominal_coupon_bonds(df)["cusip"]) == {"91282CJK8", "912810TV0"}


_CALLABLE_HTML = """
<table>
<tr><th>CUSIP</th><th>SECURITY TYPE</th><th>RATE</th><th>MATURITY DATE</th>
    <th>CALL DATE</th><th>BUY</th><th>SELL</th><th>END OF DAY</th></tr>
<tr><td>912810DN5</td><td>MARKET BASED BOND</td><td>11.750%</td><td>11/15/2014</td>
    <td>11/15/2009</td><td>99.99</td><td>100.01</td><td>100.000000</td></tr>
<tr><td>912810DP0</td><td>MARKET BASED BOND</td><td>11.250%</td><td>02/15/2015</td>
    <td></td><td>144.00</td><td>144.06</td><td>144.031250</td></tr>
</table>
"""


def test_a_callable_bond_priced_to_the_call_is_excluded():
    """A bond nearing its call date prices toward the call, not final maturity — pricing
    it as a plain bullet bond (which this module does) is a real mispricing, not
    illiquidity. Treasury's last callable long bonds ran through ~2009; excluding them
    avoids exactly the artifact a real one (912810DN5, called 2009-11-15) produced."""
    df = parse_price_table(_CALLABLE_HTML, pd.Timestamp("2009-11-13"))
    assert df.set_index("cusip").loc["912810DN5", "call_date"] == pd.Timestamp("2009-11-15")
    assert pd.isna(df.set_index("cusip").loc["912810DP0", "call_date"])
    assert list(nominal_coupon_bonds(df)["cusip"]) == ["912810DP0"]


def test_an_unparseable_coupon_is_dropped_not_carried_into_the_bond_math():
    """A None coupon reaching cashflows() raises a TypeError several frames deep, which
    reads as a bond-math failure rather than the parse failure it is. Drop at the filter."""
    panel = pd.DataFrame({
        "date": [SETTLE] * 2,
        "cusip": ["91282CJK8", "912810TV0"],
        "kind": ["MARKET BASED NOTE", "MARKET BASED BOND"],
        "coupon": [4.125, None],
        "maturity": [pd.Timestamp("2034-05-15"), pd.Timestamp("2053-11-15")],
        "price": [98.5, 99.1]})
    assert list(nominal_coupon_bonds(panel)["cusip"]) == ["91282CJK8"]
