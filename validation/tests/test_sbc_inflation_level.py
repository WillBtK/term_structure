"""Offline tests for the inflation-level crossing point.

No network and no panel build: ``fit`` is a pure function of the frame it is handed,
so it is exercised on synthetic panels with a known answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation import sbc_inflation_level as m


def _panel(slope: float, cross: float, n: int = 40, noise: float = 0.0,
           markets=("US", "UK")) -> pd.DataFrame:
    """A panel whose true fitted line crosses zero at ``cross``."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        for c in markets:
            pi = 0.5 + 8.0 * i / n
            sbc = slope * (pi - cross) + (rng.normal(0, noise) if noise else 0.0)
            rows.append({"country": c, "year": 1980 + i, "pi": pi, "sbc": sbc})
    return pd.DataFrame(rows)


def test_recovers_a_known_crossing():
    out = m.fit(_panel(slope=0.05, cross=3.5), n_boot=200)
    assert out["cross"] == 3.5
    assert out["slope"] == 0.05
    assert out["r2"] > 0.999


def test_crossing_survives_noise_and_the_band_brackets_it():
    out = m.fit(_panel(slope=0.05, cross=3.5, n=60, noise=0.10), n_boot=1500)
    assert abs(out["cross"] - 3.5) < 0.4
    assert out["cross_lo"] < out["cross"] < out["cross_hi"]


def test_bootstrap_is_deterministic():
    p = _panel(slope=0.05, cross=3.5, n=60, noise=0.10)
    a, b = m.fit(p, n_boot=400, seed=7), m.fit(p, n_boot=400, seed=7)
    assert a == b


def test_fitted_values_at_two_and_four_bracket_the_crossing():
    """A crossing at 3.5 must mean bonds hedge at 2% and stop by 4%."""
    out = m.fit(_panel(slope=0.05, cross=3.5), n_boot=100)
    assert out["at2"] < 0 < out["at4"]


def test_per_market_block_is_reported_for_each_market():
    out = m.fit(_panel(slope=0.05, cross=3.5, markets=("US", "UK", "DE")), n_boot=100)
    assert set(out["per_market"]) == {"US", "UK", "DE"}
    assert out["markets"] == 3
    for stat in out["per_market"].values():
        assert stat["n"] > 0


def test_flat_relationship_does_not_produce_a_confident_crossing():
    """A zero slope has no crossing to report; the band must not pretend otherwise."""
    p = _panel(slope=0.0, cross=0.0, n=60, noise=0.20)
    out = m.fit(p, n_boot=1500)
    assert abs(out["slope"]) < 0.02
    # the band is either undefined or far wider than the signal case
    assert not np.isfinite(out["cross_hi"] - out["cross_lo"]) \
        or (out["cross_hi"] - out["cross_lo"]) > 1.0
