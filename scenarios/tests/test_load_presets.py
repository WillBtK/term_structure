"""Offline tests for the scenario-CSV loader.

The loader exists so 288 hand-typed numbers do not become 288 chances to move a
decimal, so its job is to be forgiving about layout and unforgiving about content.
"""
from __future__ import annotations

import json

import pytest

from scenarios import load_presets as lp

YEARS = range(2026, 2032)


def _csv(tmp_path, text: str):
    p = tmp_path / "views.csv"
    p.write_text(text)
    return p


def test_long_layout(tmp_path):
    p = _csv(tmp_path, "scenario,market,row,year,value\n"
                       "Stagflation,US,inflation,2027,4.0\n"
                       "Stagflation,US,inflation,2028,4.2\n")
    out = lp.parse(p, YEARS)
    assert out["Stagflation"]["US"]["inflation"] == {2027: 4.0, 2028: 4.2}


def test_wide_layout_is_detected_and_reshaped(tmp_path):
    p = _csv(tmp_path, "Scenario,Market,Row,2026,2027,2028\n"
                       "Hard Landing,US,Policy rate,3.5,2.0,1.5\n")
    out = lp.parse(p, YEARS)
    assert out["Hard Landing"]["US"]["policy_rate"] == {2026: 3.5, 2027: 2.0, 2028: 1.5}


def test_market_and_row_aliases(tmp_path):
    p = _csv(tmp_path, "scenario,market,row,year,value\n"
                       "Goldilocks,United States,Real GDP growth,2027,2.8\n"
                       "Goldilocks,Germany,CPI,2027,1.9\n"
                       "Goldilocks,Japan,Bank rate,2027,1.0\n"
                       "Goldilocks,Britain,Budget balance,2027,-4.0\n")
    out = lp.parse(p, YEARS)["Goldilocks"]
    assert out["US"]["real_growth"] == {2027: 2.8}
    assert out["DE"]["inflation"] == {2027: 1.9}
    assert out["JP"]["policy_rate"] == {2027: 1.0}
    assert out["UK"]["deficit"] == {2027: -4.0}


def test_scenario_names_tolerate_punctuation_and_case(tmp_path):
    p = _csv(tmp_path, "scenario,market,row,year,value\n"
                       "high growth high inflation,US,inflation,2027,3.5\n"
                       "HARD LANDING,US,inflation,2027,1.0\n")
    out = lp.parse(p, YEARS)
    assert "High Growth High Inflation" in out
    assert "Hard Landing" in out


def test_years_outside_the_grid_are_dropped_not_silently_kept(tmp_path, caplog):
    p = _csv(tmp_path, "scenario,market,row,year,value\n"
                       "Stagflation,US,inflation,2027,4.0\n"
                       "Stagflation,US,inflation,2040,9.9\n")
    with caplog.at_level("WARNING"):
        out = lp.parse(p, YEARS)
    assert out["Stagflation"]["US"]["inflation"] == {2027: 4.0}
    assert "2040" in caplog.text


def test_blank_values_are_skipped(tmp_path):
    """A forecast table with gaps must not become zeroes."""
    p = _csv(tmp_path, "scenario,market,row,year,value\n"
                       "Stagflation,US,inflation,2027,4.0\n"
                       "Stagflation,US,inflation,2028,\n")
    out = lp.parse(p, YEARS)
    assert out["Stagflation"]["US"]["inflation"] == {2027: 4.0}


def test_unknown_names_raise_rather_than_being_dropped(tmp_path):
    for bad, msg in ((("Stagflation,FR,inflation,2027,4.0"), "market"),
                     (("Stagflation,US,unemployment,2027,4.0"), "row"),
                     (("Melt-up,US,inflation,2027,4.0"), "scenario")):
        p = _csv(tmp_path, "scenario,market,row,year,value\n" + bad + "\n")
        with pytest.raises(ValueError, match=msg):
            lp.parse(p, YEARS)


def test_missing_columns_name_what_is_missing(tmp_path):
    p = _csv(tmp_path, "scenario,market,year,value\nStagflation,US,2027,4.0\n")
    with pytest.raises(ValueError, match="row"):
        lp.parse(p, YEARS)


def test_non_numeric_value_names_the_row(tmp_path):
    p = _csv(tmp_path, "scenario,market,row,year,value\n"
                       "Stagflation,US,inflation,2027,about four\n")
    with pytest.raises(ValueError, match="row 2"):
        lp.parse(p, YEARS)


def test_render_is_valid_json_the_sheet_can_read():
    presets = {"Stagflation": {"US": {"inflation": {2027: 4.0}}}}
    line = lp.render(presets)
    assert line.startswith("const PRESETS = ") and line.endswith(";")
    assert json.loads(line[len("const PRESETS = "):-1]) == {
        "Stagflation": {"US": {"inflation": {"2027": 4.0}}}}


def test_splice_replaces_only_the_presets_line(tmp_path):
    sheet = tmp_path / "sheet.html"
    sheet.write_text("before\nconst PRESETS = {};\nafter\n")
    lp.splice({"Stagflation": {"US": {"inflation": {2027: 4.0}}}}, sheet)
    lines = sheet.read_text().split("\n")
    assert lines[0] == "before" and lines[2] == "after"
    assert lines[1].startswith("const PRESETS = {\"Stagflation\"")


def test_splice_refuses_a_sheet_without_the_anchor(tmp_path):
    sheet = tmp_path / "sheet.html"
    sheet.write_text("nothing to replace\n")
    with pytest.raises(ValueError, match="PRESETS"):
        lp.splice({}, sheet)
