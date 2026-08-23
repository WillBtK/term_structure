"""Official projected fiscal balances for all four markets, one concept everywhere.

The US fiscal driver is the CBO five-year-ahead projected total balance. Extending that
concept means finding each market's own official projection -- the body whose baseline
the market actually trades on -- rather than one convenient multilateral source:

    US   CBO baseline                        (macro.fetch_cbo; 11 years)
    UK   OBR, public sector net borrowing    (public finances databank; 5 years)
    DE   European Commission AMECO, general government net lending, EDP basis
         (the Commission's horizon is two years -- beyond it the path is extended
          with the IMF WEO shape, level-matched at the last common year, taking
          the horizon to t+5; the two agree to ~0.1-0.2pp where they overlap)
    JP   Cabinet Office medium-to-long-term projections, central+local fiscal
         balance, BASELINE case ("current projection"), to FY2040

All returned DEFICIT-POSITIVE in percent of GDP, keyed by years ahead (0 = the current
year where the source projects it, 1 = the first full year). Committed CSV caches under ``macro/data`` keep everything runnable offline; a
failed fetch keeps the committed file.

Vintage archives -- the real-time panels that would let each market MEASURE its own
fiscal sensitivity the way the US does -- are the recorded upgrade path, not part of
this module: OBR publishes a historical official forecasts database back to 1970, and
AMECO vintages exist twice a year. Until those are built, the spread markets borrow the
US-measured sensitivity, on the same reasoning as the borrowed inflation coefficients:
a relative driver's coefficient is the response to this market's driver moving with the
US held still, and the US sample is the only one with a measured post-break response.
"""
from __future__ import annotations

import io
import logging
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FILES = {"UK": DATA / "obr_balance_uk.csv",
         "DE": DATA / "ameco_balance_de.csv",
         "JP": DATA / "cao_balance_jp.csv"}

AMECO_ZIP = "https://ec.europa.eu/economy_finance/db_indicators/ameco/documents/ameco0.zip"
AMECO_CODE = "DEU.1.0.319.0.UBLG"        # net lending, % of GDP, EDP basis
OBR_DATA_PAGE = "https://obr.uk/data/"
CAO_XLSX = "https://www5.cao.go.jp/keizai2/keizai-syakai/shisan/keisuuhyo.xlsx"
CAO_SHEET = "財政の詳細計数表（現状投影・長期）"     # the baseline ("current projection") case

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
_TIMEOUT = 180


def _get(url: str, referer: str = "", retries: int = 3) -> bytes:
    hdr = ["-H", f"Referer: {referer}"] if referer else []
    last = None
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
                 "-A", _UA, *hdr, url], capture_output=True, timeout=_TIMEOUT + 30)
            if out.returncode != 0:
                raise RuntimeError(f"curl {out.returncode}: {out.stderr[:150]!r}")
            return out.stdout
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            logger.warning("fetch %s attempt %d: %s", url, attempt + 1, exc)
    raise RuntimeError(f"failed to fetch {url}: {last}")


# ------------------------------------------------------------------------- fetchers
# Each returns a Series of the BALANCE in percent of GDP (surplus-positive, as the
# sources publish it), indexed by the calendar year the fiscal year starts in.

def fetch_de() -> pd.Series:
    blob = _get(AMECO_ZIP)
    z = zipfile.ZipFile(io.BytesIO(blob))
    import csv
    for name in z.namelist():
        for line in io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace"):
            if not line.startswith(AMECO_CODE + ";"):
                continue
            row = next(csv.reader([line], delimiter=";"))
            hdr = next(csv.reader([_header(z, name)], delimiter=";"))
            vals = {}
            for h, v in zip(hdr[5:], row[5:]):
                try:
                    vals[int(h)] = float(v)
                except (TypeError, ValueError):
                    continue
            return pd.Series(vals).sort_index()
    raise RuntimeError(f"{AMECO_CODE} not found in AMECO bulk file")


def _header(z: zipfile.ZipFile, name: str) -> str:
    with z.open(name) as f:
        return io.TextIOWrapper(f, encoding="utf-8", errors="replace").readline()


def fetch_uk() -> pd.Series:
    """OBR public finances databank: PSNB, % of GDP, history + the current forecast."""
    import re
    page = _get(OBR_DATA_PAGE).decode("utf-8", "replace")
    m = re.search(r'href="(https://obr\.uk/download/public-finances-databank[^"]+)"',
                  page)
    if not m:
        raise RuntimeError("no public-finances-databank link on obr.uk/data")
    blob = _get(m.group(1), referer=OBR_DATA_PAGE)
    d = pd.read_excel(io.BytesIO(blob), sheet_name="Aggregates (per cent of GDP)",
                      header=None)
    hdr = d.iloc[3].astype(str)
    col = next(j for j in range(d.shape[1])
               if hdr[j].strip().lower() == "public sector net borrowing")
    vals = pd.to_numeric(d.iloc[4:, col], errors="coerce")
    years = d.iloc[4:, 1].astype(str).str.extract(r"^(\d{4})-")[0]
    s = pd.Series(vals.values, index=pd.to_numeric(years, errors="coerce").values)
    s = s.dropna()
    s.index = s.index.astype(int)
    return (-s).sort_index()          # PSNB is borrowing-positive -> balance


def fetch_jp() -> pd.Series:
    """Cabinet Office baseline: central+local fiscal balance, % of nominal GDP."""
    blob = _get(CAO_XLSX)
    d = pd.read_excel(io.BytesIO(blob), sheet_name=CAO_SHEET, header=None)
    # The percent-of-GDP block: a year row, then 財政収支 with a （対名目GDP比） row
    # directly beneath it. Labels move between releases, so locate by text.
    def _label(i: int) -> str:
        return "".join(str(x) for x in d.iloc[i, :3] if pd.notna(x))
    year_rows = [i for i in range(len(d))
                 if "年" in _label(i) and "度" in _label(i)]
    for i in range(len(d) - 1):
        if _label(i).strip() == "財政収支" and "対名目GDP比" in _label(i + 1):
            yr_row = max(r for r in year_rows if r < i)
            years = pd.to_numeric(d.iloc[yr_row], errors="coerce")
            vals = pd.to_numeric(d.iloc[i + 1], errors="coerce")
            s = pd.Series(vals.values, index=years.values).dropna()
            s = s[[isinstance(ix, float) and 1990 < ix < 2100 for ix in s.index]]
            s.index = s.index.astype(int)
            if len(s) > 5:
                return s.sort_index()
    raise RuntimeError("CAO fiscal-balance (% GDP) row not found")


FETCHERS = {"DE": fetch_de, "UK": fetch_uk, "JP": fetch_jp}


# ------------------------------------------------- the sources' own growth baselines
# The endogenous-deficit rule differences a stated growth view against a baseline,
# and the honest baseline is the growth path the DEFICIT projection was conditioned
# on -- each institution's own forecast, not a third party's. Measuring against a
# different source's growth silently converts that source's disagreement with the
# deficit institution into "no view": the Cabinet Office runs 0.4% long-run growth
# against the WEO's 0.6%, so agreeing with the CAO would have registered as growth
# pessimism. WEO (macro.weo) remains the fallback, and extends Germany beyond
# AMECO's horizon exactly as the deficit path is extended.

GROWTH_FILES = {c: DATA / f"growth_{c.lower()}.csv" for c in ("US", "UK", "DE", "JP")}
OBR_LTD_RE = r'href="(https://obr\.uk/download/long-term-economic-determinants[^"]+)"'


def fetch_growth_us() -> pd.Series:
    """CBO economic projections (calendar year), the vintage the deficit is on."""
    from macro.fetch_cbo import DATA as CBO_DATA, GDP_VINTAGE, _csv
    d = _csv(CBO_DATA + f"economic/economic_projections/calendar_{GDP_VINTAGE}.csv")
    d = d[d["variable"] == "real_gdp_pct_change"]
    return pd.Series({int(r["date"]): float(r["value"]) for _, r in d.iterrows()}
                     ).sort_index()


def fetch_growth_uk() -> pd.Series:
    """OBR long-term economic determinants: real GDP growth by fiscal year, keyed by
    the calendar year the fiscal year starts in (matching fetch_uk's convention)."""
    import re
    page = _get(OBR_DATA_PAGE).decode("utf-8", "replace")
    m = re.search(OBR_LTD_RE, page)
    if not m:
        raise RuntimeError("no long-term-economic-determinants link on obr.uk/data")
    blob = _get(m.group(1), referer=OBR_DATA_PAGE)
    d = pd.read_excel(io.BytesIO(blob), sheet_name=0, header=None)
    def _lab(i: int) -> str:      # labels drift between columns across releases
        return " ".join(str(x) for x in d.iloc[i, :3] if pd.notna(x))
    yr_row = next(i for i in range(len(d))
                  if d.iloc[i].astype(str).str.match(r"^\d{4}-\d{2}$").sum() > 5)
    g_row = next(i for i in range(len(d)) if _lab(i).strip().startswith("Real GDP growth"))
    out = {}
    for y, v in zip(d.iloc[yr_row], d.iloc[g_row]):
        my = pd.to_numeric(str(y).split("-")[0], errors="coerce")
        v = pd.to_numeric(v, errors="coerce")
        if pd.notna(my) and pd.notna(v):
            out[int(my)] = float(v)
    if len(out) < 5:
        raise RuntimeError("OBR determinants: real GDP growth row not parsed")
    return pd.Series(out).sort_index()


def fetch_growth_de() -> pd.Series:
    """AMECO real GDP growth (from the OVGD level series), extended with the WEO
    shape beyond the Commission's horizon -- the same splice as the deficit path."""
    import csv
    blob = _get(AMECO_ZIP)
    z = zipfile.ZipFile(io.BytesIO(blob))
    levels = None
    for name in z.namelist():
        for line in io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace"):
            if line.startswith("DEU.1.1.0.0.OVGD;"):
                row = next(csv.reader([line], delimiter=";"))
                hdr = next(csv.reader([_header(z, name)], delimiter=";"))
                levels = {}
                for h, v in zip(hdr[5:], row[5:]):
                    try:
                        levels[int(h)] = float(v)
                    except (TypeError, ValueError):
                        continue
                break
        if levels:
            break
    if not levels:
        raise RuntimeError("DEU.1.1.0.0.OVGD not found in AMECO bulk file")
    g = {y: (levels[y] / levels[y - 1] - 1) * 100.0
         for y in sorted(levels) if y - 1 in levels}
    try:
        from macro import weo
        for y, v in weo.growth("DE").items():
            if y > max(g):
                g[y] = v
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("DE growth WEO extension unavailable (%s)", exc)
    return pd.Series(g).sort_index()


def fetch_growth_jp() -> pd.Series:
    """Cabinet Office baseline real GDP growth, from the same workbook (and the same
    'current projection' case) as the fiscal balance."""
    blob = _get(CAO_XLSX)
    d = pd.read_excel(io.BytesIO(blob), sheet_name="計数表 (長期)", header=None)
    def _lab(i: int) -> str:
        return "".join(str(x) for x in d.iloc[i, :4] if pd.notna(x))
    base_row = next(i for i in range(len(d)) if "現状投影" in _lab(i))
    yr_row = next(i for i in range(base_row, len(d)) if "年" in _lab(i) and "度" in _lab(i))
    g_row = next(i for i in range(base_row, len(d)) if "実質GDP成長率" in _lab(i))
    years = pd.to_numeric(d.iloc[yr_row], errors="coerce")
    vals = pd.to_numeric(d.iloc[g_row], errors="coerce")
    s = pd.Series(vals.values, index=years.values).dropna()
    s = s[[isinstance(ix, float) and 1990 < ix < 2100 for ix in s.index]]
    s.index = s.index.astype(int)
    if len(s) < 5:
        raise RuntimeError("CAO baseline real-growth row not parsed")
    return s.sort_index()


GROWTH_FETCHERS = {"US": fetch_growth_us, "UK": fetch_growth_uk,
                   "DE": fetch_growth_de, "JP": fetch_growth_jp}


def refresh_growth(markets=("US", "UK", "DE", "JP")) -> None:
    for c in markets:
        try:
            s = GROWTH_FETCHERS[c]()
            GROWTH_FILES[c].parent.mkdir(parents=True, exist_ok=True)
            s.to_frame("real_growth_pct").rename_axis("year").to_csv(GROWTH_FILES[c])
            logger.info("%s growth: %d years %d..%d", c, len(s),
                        s.index.min(), s.index.max())
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("%s growth refresh failed (%s) -- keeping committed file",
                           c, exc)


def growth_baseline(country: str) -> dict:
    """{calendar year: real growth, %} -- the deficit source's own projection, current
    year onward. Falls back to WEO when no committed file exists. Rounded to 2dp so
    the sheet's grid default (which displays 2dp) IS the baseline exactly -- an
    untouched growth row must produce a deviation of precisely zero."""
    f = GROWTH_FILES[country]
    now = pd.Timestamp.today().year
    if f.is_file():
        d = pd.read_csv(f)
        out = {int(r["year"]): round(float(r["real_growth_pct"]), 2)
               for _, r in d.iterrows() if int(r["year"]) >= now}
        if out:
            return out
    from macro import weo
    return {y: round(v, 2) for y, v in weo.growth(country).items() if y >= now}


# ------------------------------------------------------------------------- interface

def refresh(markets=("DE", "UK", "JP")) -> None:
    for c in markets:
        try:
            s = FETCHERS[c]()
            FILES[c].parent.mkdir(parents=True, exist_ok=True)
            s.to_frame("balance_pct_gdp").rename_axis("year").to_csv(FILES[c])
            logger.info("%s: %d years %d..%d, latest balance %+.2f%% of GDP",
                        c, len(s), s.index.min(), s.index.max(), s.iloc[-1])
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("%s refresh failed (%s) -- keeping committed file", c, exc)


def path(country: str) -> dict:
    """Projected DEFICIT path, % of GDP, keyed by years ahead (0 = the current year
    where the source projects it, 1 = the first full year).

    Only projection years are returned -- history is not a path. The last projected
    year extends flat downstream, which for Germany means the Commission's two-year
    horizon carries the long run and is the stated weakness of that source.
    """
    if country == "US":
        from macro.fetch_cbo import projected_deficit_path
        return {int(k): float(v) for k, v in projected_deficit_path().items()}
    f = FILES[country]
    if not f.is_file():
        raise FileNotFoundError(f"no committed balance projection for {country}; "
                                "run python -m macro.fiscal_projections")
    d = pd.read_csv(f)
    now = pd.Timestamp.today().year
    # The current year is included when the source projects it (year 0), so a sheet
    # column for this calendar year shows this year's projected deficit rather than
    # falling back to the five-year-ahead statistic and reading as a step. Horizon
    # averages sample whole years from 1 onward, so year 0 never enters them.
    out = {int(r["year"]) - now: -float(r["balance_pct_gdp"])
           for _, r in d.iterrows() if int(r["year"]) >= now}
    if country == "DE" and out:
        # Extend the Commission's two-year horizon with the WEO shape, level-matched
        # at the last AMECO year so the seam introduces no step. The near end stays
        # AMECO's (the source that revises fastest); WEO carries the profile to t+5.
        try:
            from macro import weo
            w = {y - now: v for y, v in weo.deficit("DE").items()}
            last = max(out)
            if last in w:
                offset = out[last] - w[last]
                for k in sorted(w):
                    if k > last:
                        out[k] = w[k] + offset
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("DE WEO extension unavailable (%s); AMECO horizon only",
                           exc)
    return {k: v for k, v in sorted(out.items()) if 0 <= k <= 15}


def five_year_ahead(country: str) -> float:
    """Today's five-year-ahead projected deficit -- the driver's base reading."""
    p = path(country)
    if not p:
        raise RuntimeError(f"{country}: empty projection path")
    ys = sorted(p)
    return float(p[min(5, ys[-1])] if 5 not in p else p[5])


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    refresh()
    refresh_growth()
    for c in ("US", "UK", "DE", "JP"):
        try:
            p = path(c)
            print(f"{c}: 5y-ahead deficit {five_year_ahead(c):+.2f}% of GDP   path "
                  + "  ".join(f"{k}:{v:.1f}" for k, v in sorted(p.items())[:8]))
        except Exception as exc:                                    # noqa: BLE001
            print(f"{c}: {exc}")


if __name__ == "__main__":
    _main()
