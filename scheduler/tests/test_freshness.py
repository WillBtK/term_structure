"""Freshness auditing — chiefly that a whole-pipeline freeze is caught.

The bug this exists to prevent: every source is fetched inside a try/except with a
cache fallback, so a source that stops publishing looks exactly like success. The
pre-existing relative check (each market vs the freshest market) is blind to the
worst case — everything freezing together — because then no market is behind any
other. ``test_global_freeze_is_caught`` is that regression.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from scheduler.freshness import (audit, business_days_between, cadence_budget,
                                 report)


def bdays(end: date, n: int, step: int = 1) -> list:
    """``n`` business days ending at ``end``, every ``step``-th one."""
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n * step)
    return sorted({d.date() for d in idx[::step]})


# ------------------------------------------------------------------ business days

def test_business_days_skip_the_weekend():
    # Fri 2026-09-04 -> Mon 2026-09-07 is one business day, not three
    assert business_days_between(date(2026, 9, 4), date(2026, 9, 7)) == 1


# --------------------------------------------------------------------- the budget

def test_a_daily_series_earns_a_small_budget():
    assert cadence_budget(bdays(date(2026, 9, 8), 120)) <= 4


def test_a_weekly_publisher_earns_a_larger_budget_without_being_told():
    """The point of deriving the budget: a genuinely slow source must not be flagged
    just for being slow."""
    weekly = bdays(date(2026, 9, 8), 30, step=5)
    daily = bdays(date(2026, 9, 8), 120)
    assert cadence_budget(weekly) > cadence_budget(daily)


def test_budget_never_drops_below_the_floor():
    assert cadence_budget([date(2026, 9, 7), date(2026, 9, 8)]) >= 3


# ---------------------------------------------------------------------- verdicts

def test_a_current_series_is_not_stale():
    asof = date(2026, 9, 8)
    v = audit({"US:nominal": bdays(asof, 120)}, as_of=asof)[0]
    assert not v.stale and v.status == "ok"


def test_a_holiday_gap_is_not_stale():
    """Sep 7 2026 is Labor Day: the US series has no Sep 7 and that is correct."""
    asof = date(2026, 9, 8)
    dates = [d for d in bdays(asof, 120) if d != date(2026, 9, 7)]
    assert not audit({"US:nominal": dates}, as_of=asof)[0].stale


def test_a_dead_daily_source_is_stale():
    asof = date(2026, 9, 8)
    stopped = bdays(date(2026, 8, 14), 120)          # nothing for ~3.5 weeks
    v = audit({"CH:nominal": stopped}, as_of=asof)[0]
    assert v.stale and v.status == "STALE"


def test_global_freeze_is_caught():
    """THE regression. Every market frozen at the same date: no market is behind any
    other, so the relative check sees nothing. The absolute check must still fire."""
    asof = date(2026, 9, 30)
    frozen = {m: bdays(date(2026, 9, 4), 120) for m in ("US:nominal", "DE:nominal",
                                                        "JP:nominal", "UK:nominal")}
    verdicts = audit(frozen, as_of=asof)
    assert all(v.behind_newest == 0 for v in verdicts)   # relative check is blind...
    assert not any(v.lagging for v in verdicts)
    assert all(v.stale for v in verdicts)                # ...absolute check is not


def test_one_dead_market_among_live_ones_is_flagged_both_ways():
    asof = date(2026, 9, 8)
    series = {"DE:nominal": bdays(asof, 120),
              "CH:nominal": bdays(date(2026, 8, 3), 120)}
    by = {v.key: v for v in audit(series, as_of=asof)}
    assert by["CH:nominal"].stale and by["CH:nominal"].lagging
    assert by["DE:nominal"].status == "ok"


def test_verdicts_are_sorted_worst_first():
    asof = date(2026, 9, 8)
    series = {"ok:x": bdays(asof, 60), "bad:y": bdays(date(2026, 8, 3), 60)}
    assert audit(series, as_of=asof)[0].key == "bad:y"


# ----------------------------------------------------------------------- report

def test_report_renders_every_series_and_its_status():
    asof = date(2026, 9, 8)
    series = {"US:nominal": bdays(asof, 60), "CH:nominal": bdays(date(2026, 8, 3), 60)}
    text = report(audit(series, as_of=asof), annotate_ci=False)
    assert "US:nominal" in text and "CH:nominal" in text and "STALE" in text


def test_report_handles_an_empty_dataset():
    assert "no series" in report([], annotate_ci=False)
