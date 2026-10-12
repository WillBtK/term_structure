"""US marketable Treasury ladder and SOMA holdings BEFORE the machine-readable sources
start (MSPD API: 2001-01; NY Fed SOMA by CUSIP: 2003-07), back to 1994.

Outstanding ladder, by month:

* **1998-04 .. 2000-12: the MSPD itself.** TreasuryDirect's archive keeps every monthly
  statement as a text-layer PDF (``/ftp/opd/opdmMMYY.pdf`` in 1998, ``opdmMMYYYY.pdf``
  from 1999). Table III, "Detail of Public Debt Outstanding", is parsed security by
  security (``parse_mspd_text``). The parser is validated against the API's own rows for
  2001, where both exist.
* **1994-01 .. 1998-03: rebuilt from Treasury's auction records.** The statements for
  these months are scanned images (1993-Jan 1997) or missing (Feb 1997 - Mar 1998), so the
  ladder is reconstructed instead: every bill, note, bond and TIPS auctioned since
  1979-11 (Fiscal Data ``auctions_query``, exact CUSIP, coupon, maturity and amount),
  summed by CUSIP over the auctions issued by the month-end and not yet matured. Against
  the statements this matches issued amounts CUSIP by CUSIP to within ~0.1%; before the
  2000-02 buybacks issued == outstanding for every post-1979 security.
  Bonds issued BEFORE the auction record starts (1955-1979 long bonds, still a few percent
  of the long end in 1994) come from ``PRE1979_BONDS`` -- transcribed from the January
  1994 statement (OCR'd, every amount checked against issued - retired = outstanding) and
  carried to their maturity; the few with estate-tax ("flower bond") redemptions decline
  in steps between the January statements.

SOMA holdings, by month: the Federal Reserve's H.4.1 "Maturity distribution of loans and
securities" -- Treasury holdings by remaining maturity (6 buckets), archived on
federalreserve.gov from mid-1996 (``h41_buckets``). See ``supply.us_dv01_hist`` for how
buckets become DV01, and for 1994 - mid-1996.
"""
from __future__ import annotations

import html
import io
import json
import logging
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from supply.fetch_us import DATA

logger = logging.getLogger(__name__)

HIST = DATA / "hist"
MSPD_TEXT_PARQUET = DATA / "us_mspd_hist_text.parquet"
AUCTIONS_PARQUET = DATA / "us_auctions_1979_2004.parquet"
H41_PARQUET = DATA / "us_h41_soma_buckets.parquet"

TD_OPD = "https://www.treasurydirect.gov/ftp/opd/"
AUCTIONS_API = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/"
                "accounting/od/auctions_query")
H41_ARCHIVE = "https://www.federalreserve.gov/releases/h41/"
TEXT_START, TEXT_END = pd.Period("1998-04", "M"), pd.Period("2000-12", "M")
AUCTION_START = pd.Period("1994-01", "M")
_UA = "Mozilla/5.0"


# ------------------------------------------------------------------ http
def _curl(url: str, out: Path | None = None, timeout: int = 90, retries: int = 3):
    last = None
    for attempt in range(retries):
        cmd = ["curl", "-sS", "--http1.1", "-L", "--max-time", str(timeout), "-A", _UA,
               "-w", "%{http_code}"]
        if out is not None:
            cmd += ["-o", str(out)]
        r = subprocess.run(cmd + [url], capture_output=True, timeout=timeout + 30)
        if r.returncode == 0:
            if out is not None:
                return r.stdout.decode().strip()[-3:]
            body = r.stdout[:-3]
            return r.stdout[-3:].decode(), body
        last = r.stderr[:200]
        time.sleep(2 ** attempt)
    raise RuntimeError(f"curl failed for {url}: {last!r}")


# ------------------------------------------------------------------ MSPD text statements
def mspd_pdf_name(p: pd.Period) -> str:
    return f"opdm{p.month:02d}{p.year % 100:02d}.pdf" if p.year == 1998 \
        else f"opdm{p.month:02d}{p.year}.pdf"


def download_mspd_pdfs(start=TEXT_START, end=TEXT_END) -> list:
    """The monthly statements as PDFs (text layer), cached under supply/data/hist/."""
    HIST.mkdir(parents=True, exist_ok=True)
    out = []
    for p in pd.period_range(start, end, freq="M"):
        f = HIST / f"mspd_{p.year}{p.month:02d}.pdf"
        if not (f.exists() and f.stat().st_size > 10_000):
            code = _curl(TD_OPD + mspd_pdf_name(p), out=f)
            if code != "200":
                logger.warning("MSPD %s: HTTP %s", p, code)
                f.unlink(missing_ok=True)
                continue
        out.append((p, f))
    return out


_SECTION = [("Treasury Bills", "bill"), ("Treasury Notes", "nominal"),
            ("Treasury Bonds", "nominal"), ("Treasury Inflation-Indexed Notes", "tips"),
            ("Treasury Inflation-Indexed Bonds", "tips")]
_DATE = r"\d\d/\d\d/\d\d"
_AMT = r"\(?\d{1,3}(?:,\d{3})*\)?"
_BILL = re.compile(rf"^\s*(?:(?P<pre>912\d{{3}})\s*)?(?P<suf>[0-9A-Z]{{3}})\s+\d+\.\d+\s+"
                   rf"(?P<iss>{_DATE})\s+(?P<pay>{_DATE})\s+(?P<mat>{_DATE})\s+(?P<rest>.*)$")
_COUP = re.compile(rf"^\s*(?:(?P<pre>912\d{{3}})\s*)?(?P<suf>[0-9A-Z]{{3}})\s+"
                   rf"(?:[A-Z]{{1,2}}\s+)?(?P<cpn>\d+(?:[- ]\d+/\d+)?|\d+\.\d+)(?:\s+\d{{1,2}})?\s+"
                   rf"(?P<iss>{_DATE})\s+(?P<mat>{_DATE})\s+\d\d/\d\d-\d\d/\d\d\s+(?P<rest>.*)$")


def _coupon(s: str) -> float:
    s = s.replace(" ", "-")
    if "-" in s:
        whole, frac = s.split("-")
        n, d = frac.split("/")
        return float(whole) + float(n) / float(d)
    return float(s)


def _date(s: str) -> pd.Timestamp:
    """mm/dd/yy. %y puts 00-68 in the 2000s, so a 1960 issue date would read 2060: any
    date more than 40 years out (no marketable Treasury runs that long) is a 1900s one."""
    d = pd.to_datetime(s, format="%m/%d/%y")
    return d - pd.DateOffset(years=100) if d.year > 2040 else d


def parse_mspd_text(txt: str, record_date: pd.Timestamp) -> pd.DataFrame:
    """Table III of one statement -> one row per marketable security."""
    # Table III only: stop before the STRIPS/holdings tables that repeat the CUSIPs
    end = txt.find("Total Treasury Inflation-Indexed Bonds")
    end = txt.find("\n", end) if end >= 0 else txt.find("TABLE IV")
    body = txt[txt.find("TABLE III"):end if end > 0 else None]
    cls, pre, rows = None, None, []
    for line in body.split("\n"):
        st = line.strip()
        for label, c in _SECTION:
            if st.startswith(label + ":") or st.startswith(label + " (Maturity Value):"):
                cls = c
        if st.startswith("Total "):
            continue
        if cls is None:
            continue
        m = (_BILL if cls == "bill" else _COUP).match(line)
        if not m:
            continue
        if m.group("pre"):
            pre = m.group("pre")
        if pre is None:
            continue
        amts = re.findall(_AMT, m.group("rest"))
        if not amts:
            continue
        out = float(amts[-1].strip("()").replace(",", ""))
        rows.append({"record_date": record_date, "cusip": pre + m.group("suf"),
                     "asset_class": cls, "issue_date": _date(m.group("iss")),
                     "maturity_date": _date(m.group("mat")),
                     "coupon_pct": 0.0 if cls == "bill" else _coupon(m.group("cpn")),
                     "amount": out})
    return pd.DataFrame(rows)


def mspd_text_ladder(refresh: bool = False) -> pd.DataFrame:
    """All parsed statements, 1998-04 .. 2000-12 (cached)."""
    if not refresh and MSPD_TEXT_PARQUET.exists():
        return pd.read_parquet(MSPD_TEXT_PARQUET)
    frames = []
    for p, f in download_mspd_pdfs():
        txt = subprocess.run(["pdftotext", "-layout", str(f), "-"], capture_output=True,
                             text=True).stdout
        frames.append(parse_mspd_text(txt, p.to_timestamp("M")))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(MSPD_TEXT_PARQUET)
    return df


# ------------------------------------------------------------------ auction reconstruction
def fetch_auctions(refresh: bool = False) -> pd.DataFrame:
    """Every Treasury auction issued 1979-11 .. 2004 (Fiscal Data ``auctions_query``)."""
    if not refresh and AUCTIONS_PARQUET.exists():
        return pd.read_parquet(AUCTIONS_PARQUET)
    fields = ("cusip,security_type,security_term,auction_date,issue_date,maturity_date,"
              "int_rate,total_accepted,reopening,cash_management_bill_cmb,"
              "inflation_index_security,floating_rate,callable,call_date,called_date")
    rows, page = [], 1
    while True:
        url = (f"{AUCTIONS_API}?filter=issue_date:lte:2004-12-31&fields={fields}"
               f"&page%5Bsize%5D=10000&page%5Bnumber%5D={page}")
        code, body = _curl(url)
        if code != "200":
            raise RuntimeError(f"auctions page {page}: HTTP {code}")
        j = json.loads(body)
        rows += j["data"]
        if page >= int(j["meta"]["total-pages"]):
            break
        page += 1
    df = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(AUCTIONS_PARQUET)
    return df


# Bonds issued before the auction record starts (1979-11) and still outstanding in 1994.
# Source: MSPD Table III, January 1994 (scanned; OCR'd and every amount checked as
# issued - retired = outstanding), and the April 1998 statement for those still there.
# ``redeemed`` is the date the bond left the ladder: its maturity, or for a callable bond
# its first call date -- Treasury called every callable bond at its first call (the four
# 1994-96 calls are absent from every statement from 1998-04; the later ones leave the
# MSPD the month before their call). Callables are priced to that date
# (``supply.us_dv01.to_call``). CUSIPs are given where a later statement shows
# them; the rest are labelled by coupon and maturity.
PRE1979_BONDS = [
    # id,              coupon, issued,       maturity,     redeemed,     $mn outstanding
    ("PRE79-9.000-1994",  9.000, "1979-01-11", "1994-02-15", "1994-02-15", 3010),
    ("PRE79-8.500-1999",  8.500, "1974-05-15", "1999-05-15", "1994-05-15", 2378),
    ("PRE79-8.750-1994",  8.750, "1979-07-09", "1994-08-15", "1994-08-15", 1506),
    ("PRE79-10.125-1994", 10.125, "1979-10-18", "1994-11-15", "1994-11-15", 1502),
    ("PRE79-3.000-1995",  3.000, "1955-02-15", "1995-02-15", "1995-02-15", 104),
    ("PRE79-7.875-2000",  7.875, "1975-02-18", "2000-02-15", "1995-02-15", 2749),
    ("PRE79-8.375-2000",  8.375, "1975-08-15", "2000-08-15", "1995-08-15", 4612),
    ("PRE79-8.000-2001",  8.000, "1976-08-16", "2001-08-15", "1996-08-15", 1485),
    ("912810BG2",         3.500, "1960-10-03", "1998-11-15", "1998-11-15", 253),
    ("912810BU1",         8.250, "1975-05-15", "2005-05-15", "2000-05-15", 4224),
    ("912810BX5",         7.625, "1977-02-15", "2007-02-15", "2002-02-15", 4234),
    ("912810BZ0",         7.875, "1977-11-15", "2007-11-15", "2002-11-15", 1495),
    ("912810CC0",         8.375, "1978-08-15", "2008-08-15", "2003-08-15", 2103),
    ("912810CE6",         8.750, "1978-11-15", "2008-11-15", "2003-11-15", 5230),
    ("912810CG1",         9.125, "1979-05-15", "2009-05-15", "2004-05-15", 4606),
]


# The 3-1/2% 1998 was an estate-tax ("flower") bond, redeemed at par for estate taxes in
# small amounts each year: its outstanding from the January statements (OCR) and the first
# text statement, held between readings.
FLOWER_STEPS = {"912810BG2": {"1994-01-31": 253, "1995-01-31": 242, "1996-01-31": 232,
                              "1997-01-31": 229, "1998-04-30": 226}}


def _auction_class(r) -> str:
    if r["security_type"] == "Bill":
        return "bill"
    if str(r.get("inflation_index_security")) == "Yes":
        return "tips"
    if str(r.get("floating_rate")) == "Yes":
        return "frn"
    return "nominal"


def auction_ladder(months, auctions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Month-end ladders rebuilt from auctions + ``PRE1979_BONDS``, ``months`` a
    PeriodIndex. Same columns as the MSPD ladders."""
    a = fetch_auctions() if auctions is None else auctions.copy()
    a["issue_date"] = pd.to_datetime(a["issue_date"])
    a["maturity_date"] = pd.to_datetime(a["maturity_date"])
    a["amount"] = pd.to_numeric(a["total_accepted"], errors="coerce") / 1e6
    a["coupon_pct"] = pd.to_numeric(a["int_rate"], errors="coerce").fillna(0.0)
    a["asset_class"] = a.apply(_auction_class, axis=1)
    a.loc[a["asset_class"] == "bill", "coupon_pct"] = 0.0
    pre = pd.DataFrame(PRE1979_BONDS, columns=["cusip", "coupon_pct", "issue_date",
                                               "maturity_date", "redeemed", "amount"])
    for c in ("issue_date", "maturity_date", "redeemed"):
        pre[c] = pd.to_datetime(pre[c])
    frames = []
    for p in months:
        t = p.to_timestamp("M")
        live = a[(a["issue_date"] <= t) & (a["maturity_date"] > t)]
        g = live.groupby("cusip").agg(asset_class=("asset_class", "first"),
                                      issue_date=("issue_date", "min"),
                                      maturity_date=("maturity_date", "first"),
                                      coupon_pct=("coupon_pct", "first"),
                                      amount=("amount", "sum")).reset_index()
        old = pre[(pre["issue_date"] <= t) & (pre["redeemed"] > t)].assign(asset_class="nominal")
        for cusip, steps in FLOWER_STEPS.items():
            st = pd.Series(steps, dtype=float)
            st.index = pd.to_datetime(st.index)
            known = st[st.index <= t]
            if len(known) and (old["cusip"] == cusip).any():
                old.loc[old["cusip"] == cusip, "amount"] = float(known.iloc[-1])
        g = pd.concat([g, old[g.columns]], ignore_index=True)
        g.insert(0, "record_date", t)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ H.4.1 maturity buckets
H41_KEYS = ["b15", "b90", "b1y", "b5y", "b10y", "b10p"]
# remaining-maturity bucket edges, in years -- the H.4.1 table's own
H41_EDGES = {"b15": (0.0, 15 / 365.25), "b90": (15 / 365.25, 90 / 365.25),
             "b1y": (90 / 365.25, 1.0), "b5y": (1.0, 5.0), "b10y": (5.0, 10.0),
             "b10p": (10.0, 100.0)}
_H41_PATS = [r"Within 15 days?", r"16 (?:days )?to 90(?: days)?", r"91 days to 1 year",
             r"Over 1 (?:year )?to 5 years", r"Over 5 (?:years )?to 10 years", r"Over 10 years"]
_TOK = re.compile(r"[+-]\s?\d+(?:,\d{3})*(?![\d,])|(?<![\d,])\d+(?:,\d{3})*(?![\d,])|-{3,}")


def _h41_text(raw: str) -> str:
    t = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"[ \t]+", " ", t)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _first(t: str, pat: str) -> float:
    m = re.search(pat, t)
    return _num(m.group(1)) if m else float("nan")


def parse_h41(raw: str) -> dict | None:
    """SOMA Treasury holdings by remaining maturity from one H.4.1 release, $mn.

    Three layouts over 1996-2008: (A) loans | govt holdings, change | agency holdings,
    change; (B) the same plus repo and reverse-repo columns; (C) transposed, one
    'Holdings' row per security type. In A and B the government column INCLUDES
    securities held under repurchase agreement (classified within 15 days or 16-90 days);
    they are removed so that only outright holdings remain: the excess of the bucket total
    over bills + notes + bonds bought outright comes out of the shortest buckets first.
    """
    t = _h41_text(raw)
    i = t.upper().find("MATURITY DISTRIBUTION")
    if i < 0:
        return None
    sect = t[i:i + 5000]
    out: dict = {}
    m = re.search(r"U\.S\. Treasury securities[^\n]*\n\s*Holdings\s+([^\n]+)", sect)
    if m:
        vals = [_num(x) for x in re.findall(r"\d{1,3}(?:,\d{3})*", m.group(1))]
        if len(vals) < 7:
            return None
        out.update(dict(zip(H41_KEYS, vals[:6])), layout="C")
    else:
        ncol = 6 if ("Triparty" in sect[:900] or "agreements" in sect[:900].lower()
                     and "Holdings" in sect[:900] and "Reverse" in sect[:900]) else 4
        lines = sect.split("\n")
        for k, pat in zip(H41_KEYS, _H41_PATS):
            line = next((ln for ln in lines if re.search(pat, ln)), None)
            if line is None:
                return None
            last = list(re.finditer(pat, line))[-1]
            toks = _TOK.findall(line[last.end():])
            if len(toks) < ncol:
                return None
            v = toks[len(toks) - ncol]
            if v[0] in "+-":
                return None
            out[k] = _num(v)
        out["layout"] = "B" if ncol == 6 else "A"
    out["bucket_total"] = sum(out[k] for k in H41_KEYS)
    bills = _first(t, r"Bought outright\s*-+\s*Bills\s*(?:\(\d\))?\s+([\d,]+)")
    notes = _first(t, r"\n\s*Notes\s*(?:\(\d\))?\s+([\d,]+)")
    bonds = _first(t, r"\n\s*Bonds\s*(?:\(\d\))?\s+([\d,]+)")
    outright = bills + notes + bonds
    if outright != outright:                 # 1997: only the system-account total
        outright = _first(t, r"Bought outright\s*-+\s*system account\s*(?:\(\d\))?\s+([\d,]+)")
    out["outright"] = outright
    out["tips"] = _first(t, r"Includes \$\s*([\d,]+) million of inflation-indexed")
    if out["layout"] in ("A", "B") and outright == outright:
        excess = out["bucket_total"] - outright
        out["repo_removed"] = excess
        for k in ("b15", "b90"):
            take = min(max(excess, 0.0), out[k])
            out[k] -= take
            excess -= take
    return out


def _h41_month_end(ym: pd.Period, cache: Path):
    """The release for the month's last Wednesday (or the week before, if missing)."""
    last = ym.to_timestamp("M")
    wed = last - pd.Timedelta(days=(last.weekday() - 2) % 7)
    for back in (0, 7):
        w = wed - pd.Timedelta(days=back)
        for off in (1, 2, 0, -1):
            rel = w + pd.Timedelta(days=off)
            f = cache / f"{rel:%Y%m%d}.html"
            if f.exists() and f.stat().st_size > 5000:
                return w, f.read_text(errors="ignore")
            base = f"{H41_ARCHIVE}{rel:%Y%m%d}/"
            code, body = _curl(base, timeout=60)
            if code != "200":
                continue
            body = body.decode(errors="ignore")
            if "aturity" not in body.lower():
                for sub in ("h41.htm", "H41.htm", "h41.txt", "H41.TXT"):
                    c2, b2 = _curl(base + sub, timeout=60)
                    b2 = b2.decode(errors="ignore")
                    if c2 == "200" and "aturity" in b2.lower():
                        body = b2
                        break
            if "aturity" in body.lower():
                f.write_text(body)
                return w, body
    return wed, None


def h41_buckets(start: str = "1996-06", end: str = "2007-12",
                refresh: bool = False) -> pd.DataFrame:
    """Month-end SOMA Treasury holdings by remaining-maturity bucket ($mn, outright),
    from the federalreserve.gov H.4.1 archive (which starts mid-1996)."""
    if not refresh and H41_PARQUET.exists():
        return pd.read_parquet(H41_PARQUET)
    cache = HIST / "h41"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for ym in pd.period_range(start, end, freq="M"):
        wed, body = _h41_month_end(ym, cache)
        r = parse_h41(body) if body else None
        if r is None:
            logger.warning("H.4.1 %s: no parseable release", ym)
            continue
        r["asof"] = wed
        r["month"] = ym.to_timestamp("M")
        rows.append(r)
    df = pd.DataFrame(rows).set_index("month").sort_index()
    df.to_parquet(H41_PARQUET)
    return df


# ------------------------------------------------------------------ Fed Bulletin table 1.19
FRB_URL = "https://fraser.stlouisfed.org/files/docs/publications/FRB/1990s/frb_{m:02d}{y}.pdf"
FRB_PARQUET = DATA / "us_frb_soma_buckets.parquet"
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7,
           "jul": 7, "aug": 8, "sept": 9, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_FRB_ROWS = [("total", r"Total U\.?\s?S\.? Treasury securities"), ("b15", r"Within fifteen days"),
             ("b90", r"Sixteen days to ninety days"), ("b1y", r"Ninety-one days to one year"),
             ("b5y", r"One year to five years"), ("b10y", r"Five years to ten years"),
             ("b10p", r"More than ten years")]


def parse_frb_119(txt: str, issue: pd.Period) -> list:
    """End-of-month columns of Bulletin table 1.19 ("Federal Reserve Banks: maturity
    distribution of loan and security holdings") -> [{month, total, b15..b10p}].

    Each issue's last three columns are the month-ends five, four and three months
    before the issue (Jan 1994: Aug-Oct 1993; May 1994: Dec 1993 - Feb 1994; checked
    against the H.4.1 archive where they overlap). They are dated by that rule, not by
    the OCR'd header, which sometimes spaces a month name out ("F e b . 28"). Holdings
    under repurchase agreement sit in 'within fifteen days' (worth ~0 DV01)."""
    i = txt.find("1.19 FEDERAL RESERVE BANKS")
    if i < 0:
        i = txt.find("Maturity Distribution of Loan and Security")
    if i < 0:
        return []
    block = txt[i:i + 7000]
    j = re.search(r"Total U\.?\s?S\.? Treasury securities", block)
    if not j:
        return []
    lines = block[j.start():].split("\n")
    vals = {}
    for key, pat in _FRB_ROWS:
        line = next((ln for ln in lines if re.search(pat, ln)), None)
        if line is None:
            return []
        tail = line[re.search(pat, line).end():]
        nums = re.findall(r"\d{1,3}(?:,\d{3})+|\b\d{1,3}\b", tail)
        if len(nums) < 3:
            return []
        vals[key] = [_num(x) for x in nums[-3:]]
    rows = []
    for k, lag in enumerate((5, 4, 3)):
        r = {"month": (issue - lag).to_timestamp("M"), "issue": str(issue)}
        r.update({key: v[k] for key, v in vals.items()})
        r["bucket_total"] = sum(r[x] for x in H41_KEYS)
        rows.append(r)
    return rows


def frb_buckets(first_issue: str = "1994-04", last_issue: str = "1996-10",
                refresh: bool = False) -> pd.DataFrame:
    """Month-end SOMA Treasury holdings by bucket from the Federal Reserve Bulletin
    (FRASER scans with an OCR text layer), for the months before the H.4.1 archive.
    Where two issues carry the same month the later one wins (revisions)."""
    if not refresh and FRB_PARQUET.exists():
        return pd.read_parquet(FRB_PARQUET)
    cache = HIST / "frb"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pd.period_range(first_issue, last_issue, freq="M"):
        t = cache / f"frb_{p.year}{p.month:02d}.txt"
        if not t.exists():
            pdf = cache / "tmp.pdf"
            code = _curl(FRB_URL.format(m=p.month, y=p.year), out=pdf, timeout=180)
            if code != "200":
                logger.warning("Bulletin %s: HTTP %s", p, code)
                continue
            txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True,
                                 text=True, errors="ignore").stdout
            t.write_text(txt)
            pdf.unlink(missing_ok=True)
        got = parse_frb_119(t.read_text(errors="ignore"), p)
        if not got:
            logger.warning("Bulletin %s: table 1.19 not parsed", p)
        rows += got
    allrows = pd.DataFrame(rows)
    # every month-end is printed in three issues: keep the latest reading whose buckets
    # add up to the printed total (an OCR'd digit error breaks the sum); failing that,
    # the closest one, flagged
    allrows["gap"] = (allrows["bucket_total"] - allrows["total"]).abs()
    allrows["consistent"] = allrows["gap"] <= 3
    allrows = allrows.sort_values(["month", "consistent", "issue"])
    df = allrows.groupby("month").tail(1).set_index("month").sort_index()
    df.to_parquet(FRB_PARQUET)
    return df
