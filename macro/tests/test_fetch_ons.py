"""Offline tests for the ONS website-JSON parser.

Date parsing is the whole risk here: ONS labels periods as '1948 JAN', '1948 Q1' or
'1948', and a mis-parsed quarter would shift the series by up to three months without
raising anything.
"""
from __future__ import annotations

import pandas as pd

from macro.fetch_ons import _parse_date, ons_series


def test_parses_each_published_period_label():
    assert _parse_date({"date": "1948 JAN"}) == pd.Timestamp("1948-01-01")
    assert _parse_date({"date": "2026 JUN"}) == pd.Timestamp("2026-06-01")
    assert _parse_date({"date": "1948 Q1"}) == pd.Timestamp("1948-01-01")
    assert _parse_date({"date": "1975 Q3"}) == pd.Timestamp("1975-07-01")
    assert _parse_date({"date": "2011 Q4"}) == pd.Timestamp("2011-10-01")
    assert _parse_date({"date": "1948"}) == pd.Timestamp("1948-01-01")


def test_rejects_unparseable_labels_instead_of_guessing():
    assert _parse_date({"date": ""}) is None
    assert _parse_date({"date": "1948 XYZ"}) is None
    assert _parse_date({}) is None


def test_reads_the_requested_frequency_and_skips_missing_values(monkeypatch):
    payload = {
        "description": {"title": "TEST SERIES", "cdid": "K222"},
        "months": [{"date": "1948 JAN", "value": "27.2"}],
        "quarters": [
            {"date": "1948 Q1", "value": "27.2"},
            {"date": "1948 Q2", "value": ""},        # ONS blanks a suppressed value
            {"date": "1948 Q3", "value": "28.0"},
        ],
    }
    monkeypatch.setattr("macro.fetch_ons._get", lambda url, retries=4: payload)

    q = ons_series("K222", "diop", freq="quarters")
    assert list(q.index) == [pd.Timestamp("1948-01-01"), pd.Timestamp("1948-07-01")]
    assert q.tolist() == [27.2, 28.0]
    assert q.name == "K222"

    m = ons_series("K222", "diop", freq="months")
    assert len(m) == 1 and m.index[0] == pd.Timestamp("1948-01-01")


def test_unknown_dataset_is_an_error_not_a_bad_url():
    try:
        ons_series("K222", "nosuchdataset")
    except KeyError as exc:
        assert "nosuchdataset" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unmapped dataset")
