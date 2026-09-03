"""Sanity checks for the six-market export -- runs the real build against the
committed canonical parquet, so this is a correctness gate on the export logic
itself, not a re-validation of any country's term-premium estimate."""
import numpy as np

from exports.multi_market_term_premia_monthly import COUNTRIES, REAL_COUNTRIES, build


def test_all_six_markets_present():
    m = build()
    assert set(m["country"].unique()) == set(COUNTRIES)


def test_real_premium_only_for_us_and_uk():
    m = build()
    has_rtp = set(m.loc[m["rtp"].notna(), "country"].unique())
    assert has_rtp == REAL_COUNTRIES


def test_tenor_grids_match_each_market_config():
    import canada.config as ca_cfg
    import germany.config as de_cfg
    import japan.config as jp_cfg
    import switzerland.config as ch_cfg
    import uk.config as uk_cfg
    import us.config as us_cfg
    m = build()
    expected = {"US": us_cfg.COMMON_GRID, "UK": uk_cfg.COMMON_GRID, "DE": de_cfg.COMMON_GRID,
               "JP": jp_cfg.COMMON_GRID, "CH": ch_cfg.COMMON_GRID, "CA": ca_cfg.COMMON_GRID}
    for c, grid in expected.items():
        got = sorted(m.loc[m["country"] == c, "tenor"].unique())
        assert got == sorted(float(t) for t in grid), c


def test_no_row_below_min_days():
    m = build()
    assert (m["n_days"] >= 5).all()


def test_us_matches_the_single_country_export_at_10y():
    """Cross-check against exports.term_premia_monthly, which this does not reuse."""
    from exports.term_premia_monthly import build as build_wide
    wide = build_wide("US")
    m = build()
    us10 = m[(m["country"] == "US") & (m["tenor"] == 10.0)].set_index("month")["tp"]
    ref = wide["tp10"].dropna()
    common = us10.index.astype("datetime64[ns]").intersection(ref.index)
    assert len(common) > 500
    diff = (us10.reindex(common) - ref.reindex(common)).abs()
    assert diff.max() < 1e-6
