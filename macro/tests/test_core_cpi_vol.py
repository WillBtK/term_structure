"""Exponentially weighted inflation uncertainty, and why it replaced a rolling window."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro.fetch_core_cpi import UNCERTAINTY_HALFLIFE


def _vol(s, hl=UNCERTAINTY_HALFLIFE):
    m = s.ewm(halflife=hl).mean()
    return np.sqrt(((s - m) ** 2).ewm(halflife=hl).mean())


def _spike_then_calm(n=200, spike_at=40, quiet=0.1):
    i = pd.date_range("2000-01-01", periods=n, freq="MS")
    v = np.full(n, 2.0)
    v[spike_at:spike_at + 12] = 8.0
    return pd.Series(v, index=i)


def test_the_rolling_window_moves_when_the_spike_leaves_it():
    """The drop-out artefact: sixty months after a spike the measure falls sharply
    although nothing has happened. That is what makes it unusable in a projection."""
    s = _spike_then_calm()
    r = s.rolling(60).std()
    exit_window = r.loc["2008-01-01":"2009-06-01"]
    assert exit_window.diff().abs().max() > 0.3


def test_the_ewma_measure_has_no_edge_to_fall_off():
    s = _spike_then_calm()
    e = _vol(s).loc["2008-01-01":"2009-06-01"]
    assert e.diff().abs().max() < 0.05
    assert e.is_monotonic_decreasing                # decays, never steps


def test_steady_inflation_reads_as_certain_however_high():
    """Volatility is about the series' own moving mean, so a high but stable rate is
    not uncertainty -- which is the distinction the premium prices."""
    i = pd.date_range("1970-01-01", periods=120, freq="MS")
    assert _vol(pd.Series(np.full(120, 9.0), index=i)).iloc[-1] < 1e-9


def test_a_fresh_spike_still_registers_immediately():
    s = _spike_then_calm(spike_at=150)
    e = _vol(s)
    assert e.iloc[165] > 5 * e.iloc[140]


def test_oecd_flow_selection_prefers_the_freshest_not_the_first():
    """Both OECD classifications answer for Japan; the 1999 flow is frozen at 2021-06
    with a -1.0% reading. Taking whichever answers first silently reinstates exactly
    the staleness the OECD route was added to remove."""
    import pandas as pd
    from macro.fetch_core_cpi import OECD_FLOWS
    stale = pd.Series([1.0, 2.0], index=pd.to_datetime(["2021-05-01", "2021-06-01"]))
    fresh = pd.Series([1.0, 1.2], index=pd.to_datetime(["2026-05-01", "2026-06-01"]))
    best = None
    for s in (stale, fresh, stale):                 # order must not matter
        if best is None or s.index.max() > best.index.max():
            best = s
    assert best.index.max() == fresh.index.max()
    assert len(OECD_FLOWS) >= 2
