"""Turn a scenario CSV into the sheet's ``PRESETS`` block.

The sheet holds five named scenarios and seeds them from a ``PRESETS`` constant. This
writes that constant from a CSV, so pre-populating is one command rather than an edit
by hand -- which matters because the numbers arrive as a table and hand-transcribing
four markets times three rows times six years times four scenarios is 288 chances to
fat-finger a decimal.

    python -m scenarios.load_presets views.csv          # splice into the sheet
    python -m scenarios.load_presets views.csv --check  # parse and report, write nothing

Expected columns, in any order, header row required::

    scenario,market,row,year,value

* ``scenario`` -- one of the five names, matched case- and punctuation-insensitively,
  so "high growth, high inflation" finds "High Growth High Inflation".
* ``market`` -- US, UK, DE, JP (or United States / Germany / Japan / United Kingdom).
* ``row`` -- ``policy_rate``, ``inflation``, ``real_growth`` or ``fiscal_balance``.
  Common aliases are accepted: "policy", "cpi", "growth", "gdp", "balance", "deficit".
* ``year`` -- a calendar year inside the sheet's grid.
* ``value`` -- percent for the three macro rows; percent of GDP for the balance, and
  **deficit negative**, matching what the sheet displays.

Anything you leave out keeps the baseline opening, so a scenario can state a view on
inflation and growth and stay neutral on the policy path. Baseline may be omitted
entirely -- it already opens on the market's own curve and the WEO macro baseline.

**The layout to send.** Six columns, one row per scenario-market-year -- 120 rows for
five scenarios, four markets and six years::

    scenario,market,year,growth,inflation,policy
    Baseline,US,2026,2.32,3.23,3.94
    Baseline,US,2027,2.10,2.14,3.65

Two other shapes are accepted without asking, because forecast tables arrive in them:

* years across the top -- ``scenario,market,row,2026,2027,...``
* one column per variable and scenario, years down the rows --
  ``market,year,GDP_Baseline,CPI_Baseline,policy_Baseline,GDP_Stagflation,...``
  The column name is split into a variable and a scenario however it is written:
  ``GDP_Baseline``, ``cpi baseline``, ``Baseline_policy`` and
  ``growth_HardLanding`` all resolve.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
SHEET = HERE / "sheet" / "index.html"

SCENARIOS = ["Baseline", "Goldilocks", "High Growth High Inflation",
             "Stagflation", "Hard Landing"]
MARKETS = ["US", "UK", "DE", "JP"]
ROWS = ["policy_rate", "inflation", "real_growth", "deficit"]

_MARKET_ALIASES = {
    "us": "US", "usa": "US", "unitedstates": "US", "america": "US",
    "uk": "UK", "gb": "UK", "unitedkingdom": "UK", "britain": "UK",
    "de": "DE", "germany": "DE", "ger": "DE", "bund": "DE",
    "jp": "JP", "japan": "JP", "jpn": "JP", "jgb": "JP"}
_ROW_ALIASES = {
    "policyrate": "policy_rate", "policy": "policy_rate", "rate": "policy_rate",
    "policyrates": "policy_rate", "basrate": "policy_rate", "bankrate": "policy_rate",
    "inflation": "inflation", "cpi": "inflation", "headlineinflation": "inflation",
    "headlinecpi": "inflation", "prices": "inflation",
    "realgrowth": "real_growth", "growth": "real_growth", "gdp": "real_growth",
    "realgdp": "real_growth", "realgdpgrowth": "real_growth", "gdpgrowth": "real_growth",
    "fiscalbalance": "deficit", "balance": "deficit", "deficit": "deficit",
    "fiscal": "deficit", "budgetbalance": "deficit"}


def _key(s: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _match_scenario(raw: object) -> str:
    k = _key(raw)
    for name in SCENARIOS:
        if _key(name) == k:
            return name
    # tolerate a missing comma, extra words, or an abbreviation that is a prefix
    for name in SCENARIOS:
        nk = _key(name)
        if k and (nk.startswith(k) or k.startswith(nk)):
            return name
    raise ValueError(f"unknown scenario {raw!r}; expected one of {SCENARIOS}")


def _match(raw: object, table: dict, what: str) -> str:
    k = _key(raw)
    if k in table:
        return table[k]
    raise ValueError(f"unknown {what} {raw!r}")


def _split_column(col: str) -> tuple[str, str] | None:
    """Split a ``GDP_Baseline``-style header into (row, scenario), either order.

    Tried against every variable and scenario alias rather than by splitting on a
    delimiter, because the scenario names themselves contain spaces and commas --
    ``Hard_Landing`` and ``high growth high inflation`` have no reliable separator.
    """
    k = _key(col)
    for row_alias, row in _ROW_ALIASES.items():
        for scen in SCENARIOS:
            sk = _key(scen)
            if k in (row_alias + sk, sk + row_alias):
                return row, scen
    return None


def _from_variable_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """Reshape ``market,year,GDP_Baseline,...`` -- years down, variable+scenario across."""
    if "year" not in df.columns:
        return None
    hits = {c: _split_column(c) for c in df.columns if c not in ("market", "year")}
    hits = {c: v for c, v in hits.items() if v}
    if not hits:
        return None
    unmatched = [c for c in df.columns
                 if c not in ("market", "year") and c not in hits]
    if unmatched:
        logger.warning("columns not recognised as variable+scenario, ignored: %s",
                       unmatched)
    rows = []
    for _, r in df.iterrows():
        for col, (row, scen) in hits.items():
            rows.append({"scenario": scen, "market": r.get("market", "US"),
                         "row": row, "year": r["year"], "value": r[col]})
    logger.info("variable-per-scenario layout detected (%d columns) -> reshaped to long",
                len(hits))
    return pd.DataFrame(rows)


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape a wide year-per-column table if that is what arrived."""
    year_cols = [c for c in df.columns if re.fullmatch(r"\s*(19|20)\d{2}(\.0)?\s*", str(c))]
    if not year_cols:
        return df
    id_cols = [c for c in df.columns if c not in year_cols]
    out = df.melt(id_vars=id_cols, value_vars=year_cols,
                  var_name="year", value_name="value")
    logger.info("wide layout detected (%d year columns) -> reshaped to long",
                len(year_cols))
    return out


def parse(path: str | Path, years: range | None = None) -> dict:
    """CSV -> ``{scenario: {market: {row: {year: value}}}}``, validated."""
    df = pd.read_csv(path)
    df.columns = [_key(c) for c in df.columns]
    df = _from_variable_columns(df) if _from_variable_columns(df) is not None \
        else _to_long(df)
    # the recommended six-column layout has one column PER ROW rather than a `row`
    # column, so melt those into place
    value_cols = [c for c in df.columns if c in _ROW_ALIASES
                  and c not in ("scenario", "market", "year")]
    if "row" not in df.columns and value_cols:
        df = df.melt(id_vars=[c for c in ("scenario", "market", "year")
                              if c in df.columns],
                     value_vars=value_cols, var_name="row", value_name="value")
        logger.info("six-column layout detected -> reshaped to long")
    missing = {"scenario", "market", "row", "year", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing column(s): {sorted(missing)}. "
                         f"Found: {sorted(df.columns)}")
    df = df.dropna(subset=["value"])

    out: dict = {}
    for i, r in df.iterrows():
        scen = _match_scenario(r["scenario"])
        mkt = _match(r["market"], _MARKET_ALIASES, "market")
        row = _match(r["row"], _ROW_ALIASES, "row")
        try:
            year = int(float(r["year"]))
            value = float(r["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {i + 2}: year/value not numeric "
                             f"({r['year']!r}, {r['value']!r})") from exc
        if years is not None and year not in years:
            logger.warning("row %d: %s is outside the sheet's grid %d-%d, skipped",
                           i + 2, year, years[0], years[-1])
            continue
        out.setdefault(scen, {}).setdefault(mkt, {}).setdefault(row, {})[year] = value
    return out


def _grid_years() -> range:
    """The sheet's own six columns, read from its payload rather than assumed."""
    src = SHEET.read_text(encoding="utf-8")
    m = re.search(r"^const D = (\{.*\});$", src, re.M)
    if not m:
        raise ValueError("no payload line in the sheet")
    y0 = int(json.loads(m.group(1))["year0"])
    return range(y0, y0 + 6)


def render(presets: dict) -> str:
    """The ``PRESETS = {...};`` line the sheet reads."""
    return "const PRESETS = " + json.dumps(presets, indent=None) + ";"


def splice(presets: dict, sheet: Path = SHEET) -> None:
    src = sheet.read_text(encoding="utf-8").split("\n")
    hits = [i for i, l in enumerate(src) if l.startswith("const PRESETS = ")]
    if not hits:
        raise ValueError("no `const PRESETS = ` line in the sheet to replace")
    src[hits[0]] = render(presets)
    sheet.write_text("\n".join(src), encoding="utf-8")


def report(presets: dict) -> str:
    lines = []
    for scen in SCENARIOS:
        blk = presets.get(scen)
        if not blk:
            lines.append(f"  {scen:<30} (not supplied - opens on the baseline)")
            continue
        bits = []
        for mkt in MARKETS:
            rows = blk.get(mkt)
            if rows:
                bits.append(f"{mkt}:" + "/".join(
                    f"{r.split('_')[0]}({len(v)})" for r, v in sorted(rows.items())))
        lines.append(f"  {scen:<30} " + "  ".join(bits))
    return "\n".join(lines)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", help="the scenario CSV")
    ap.add_argument("--check", action="store_true",
                    help="parse and report only; do not touch the sheet")
    a = ap.parse_args()
    years = _grid_years()
    presets = parse(a.csv, years)
    print(f"\ngrid years {years[0]}-{years[-1]}\n")
    print(report(presets))
    if a.check:
        print("\n--check: sheet not modified")
        return
    splice(presets)
    print(f"\nspliced into {SHEET}")


if __name__ == "__main__":
    _main()
