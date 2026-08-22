"""CBO fiscal projections, reached through the GitHub mirrors.

``cbo.gov`` sits behind DataDome and returns 403 to any automated request -- headers,
user-agents and retries do not help, and an earlier pass through this repo wrote the
source off as unreachable for that reason. It is not. CBO publishes the same data to two
public GitHub repositories, and ``raw.githubusercontent.com`` answers normally:

    us-cbo/cbo-data          current data and forward projections, explicitly built for
                             machine consumption; economic projections to 2036, the
                             long-term budget to FY2056
    US-CBO/eval-projections  the vintage archive -- every baseline CBO ever published,
                             back to 1982, which is what makes a REAL-TIME panel possible

The GitHub API is blocked too, so directory listings need a blobless clone rather than
``/repos/.../contents``. Only whole files are fetched here, so that is not needed.

The object built is the **five-year-ahead projected total balance as a percent of GDP,
by vintage** -- the driver in Furceri, Goncalves and Li (2025). It is forward-looking by
construction, which is the point: what should price a bond is the fiscal path expected
over its life, not the deficit that has already been run. Sign is flipped on read so a
larger deficit is a larger positive number.

Two wrinkles the raw data forces:

* CBO reports a **debt** projection on fewer than half its baselines (45 vintages
  against 100 for the deficit), which is why debt is unidentified at this frequency and
  should not be read as a measured zero.
* ``eval-projections`` carries realised GDP only, so vintages whose five-year-ahead year
  has not happened yet have no denominator -- and those are the recent ones that matter
  most. The series stopped in 2021 until CBO's own projected GDP was spliced on from
  ``cbo-data``; with it, the panel reaches 2026-02.
"""
from __future__ import annotations

import io
import logging
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "cbo_balance_5y.csv"

EVAL = "https://raw.githubusercontent.com/US-CBO/eval-projections/main/input_data/"
DATA = "https://raw.githubusercontent.com/us-cbo/cbo-data/main/data/"
GDP_VINTAGE = "2026-02"          # bump when a newer economic_projections vintage lands
HORIZON = 5                      # projected_year_number; 1 is the current fiscal year
_TIMEOUT = 120


def _csv(url: str, retries: int = 4) -> pd.DataFrame:
    last = None
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
                 "-A", "Mozilla/5.0", url], capture_output=True, timeout=_TIMEOUT + 30)
            if out.returncode != 0:
                raise RuntimeError(f"curl {out.returncode}: {out.stderr[:150]!r}")
            return pd.read_csv(io.BytesIO(out.stdout))
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            logger.warning("CBO fetch %s attempt %d: %s", url, attempt + 1, exc)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def gdp_denominator() -> pd.DataFrame:
    """Realised fiscal-year GDP where it exists, CBO's own projection beyond."""
    realised = _csv(EVAL + "actual_GDP.csv").rename(
        columns={"fiscal_year": "fy", "GDP": "gdp"})
    proj = _csv(DATA + f"economic/economic_projections/fiscal_{GDP_VINTAGE}.csv")
    proj = proj[proj["variable"] == "gdp"][["date", "value"]].rename(
        columns={"date": "fy", "value": "gdp"})
    proj["fy"] = proj["fy"].astype(int)
    return pd.concat([realised, proj[~proj["fy"].isin(realised["fy"])]]
                     ).sort_values("fy").reset_index(drop=True)


def projected_balance(component: str = "deficit", horizon: int = HORIZON) -> pd.Series:
    """Projected balance as a percent of GDP, indexed by baseline (vintage) date.

    Returned DEFICIT-positive: CBO reports the balance surplus-positive, and every
    downstream sign convention here has a bigger deficit raising the premium.
    """
    b = _csv(EVAL + "baselines.csv")
    d = b[(b["component"] == component) & (b["category"] == "Total")
          & (b["projected_year_number"] == horizon)].copy()
    d = d.merge(gdp_denominator(), left_on="projected_fiscal_year",
                right_on="fy", how="left")
    d["date"] = pd.to_datetime(d["baseline_date"])
    s = (-d["value"] / d["gdp"] * 100.0)
    s.index = d["date"]
    return s.dropna().sort_index().rename("cbo_deficit5")


def load(refresh: bool = False) -> pd.Series:
    """Cached vintage series; the committed CSV is the fallback if GitHub is down."""
    if not refresh and OUT.is_file():
        df = pd.read_csv(OUT, index_col=0, parse_dates=True)
        return df.iloc[:, 0].rename("cbo_deficit5")
    s = projected_balance()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame("value").to_csv(OUT)
    return s


def monthly(index: pd.DatetimeIndex, refresh: bool = False) -> pd.Series:
    """As-of merged onto a monthly grid: a vintage stands until CBO publishes again."""
    s = load(refresh=refresh)
    return s.reindex(index.union(s.index)).ffill().reindex(index)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = load(refresh=True)
    print(f"CBO {HORIZON}y-ahead projected deficit, % of GDP: {len(s)} vintages, "
          f"{s.index.min().date()}..{s.index.max().date()}")
    print(f"  post-2008-09: {int((s.index >= '2008-09-01').sum())} vintages")
    print(s.tail(6).round(2).to_string())


if __name__ == "__main__":
    _main()
