"""Parse-layer tests for the US benchmark loaders (no network).

Exercises the pure parsers (`parse_acm_xls`, `parse_dkw_csv`) on tiny inline
fixtures: schema, decimal conversion, and the DKW nominal-TP identity
(term_premium = real_term_premium + inflation_risk_premium).
"""
import io

import pandas as pd
import pytest

from validation import benchmarks as b


def _acm_xls_bytes():
    """Build a minimal ACM-layout workbook (sheet 'ACM Daily', DATE + ACM* cols)."""
    cols = {"DATE": ["14-Jun-1961", "15-Jun-1961"]}
    for n in range(1, 11):
        cols[f"ACMY{n:02d}"] = [4.00 + n * 0.1, 4.10 + n * 0.1]      # fitted (percent)
        cols[f"ACMTP{n:02d}"] = [0.10 * n, 0.11 * n]                 # term premium
        cols[f"ACMRNY{n:02d}"] = [3.00 + n * 0.05, 3.05 + n * 0.05]  # risk-neutral
    df = pd.DataFrame(cols)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="ACM Daily", index=False)
    return buf.getvalue()


def test_parse_acm_schema_and_units():
    out = b.parse_acm_xls(_acm_xls_bytes())
    assert list(out.columns) == ["date", "maturity", "term_premium",
                                 "expected_rate", "fitted_yield"]
    assert sorted(out["maturity"].unique()) == [float(n) for n in range(1, 11)]
    row10 = out[(out["maturity"] == 10.0)].sort_values("date").iloc[0]
    # percent -> decimal: ACMTP10 = 1.00% -> 0.01
    assert row10["term_premium"] == pytest.approx(0.01, abs=1e-9)
    assert row10["fitted_yield"] == pytest.approx(0.05, abs=1e-9)     # 5.00% -> 0.05
    assert pd.Timestamp(row10["date"]) == pd.Timestamp("1961-06-14")


_DKW_FIXTURE = (
    '"Some preamble title line."\n'
    '"Source: Federal Reserve Board."\n'
    '\n'
    'date,exp.real.short.rate.5,exp.inflation.5,real.term.prem.5,inflation.risk.prem.5,'
    'tips.liq.prem.5,nominal.yield.raw.5,nominal.yield.fitted.5,ic.raw.5,ic.fitted.5,'
    'exp.real.short.rate.10,exp.inflation.10,real.term.prem.10,inflation.risk.prem.10,'
    'tips.liq.prem.10,nominal.yield.raw.10,nominal.yield.fitted.10,ic.raw.10,ic.fitted.10,'
    'exp.real.short.rate.5f5,exp.inflation.5f5,real.term.prem.5f5,inflation.risk.prem.5f5,'
    'tips.liq.prem.5f5,nominal.yield.raw.5f5,nominal.yield.fitted.5f5,ic.raw.5f5,ic.fitted.5f5\n'
    '2020-01-02,0.5,2.0,0.3,0.2,0.1,3.0,3.0,2.2,2.2,'
    '0.6,2.1,0.8,0.4,0.1,4.0,3.9,2.5,2.5,'
    '0.7,2.2,1.0,0.5,0.1,4.5,4.4,2.7,2.7\n'
)


def test_parse_dkw_schema_units_and_nominal_tp_identity():
    out = b.parse_dkw_csv(_DKW_FIXTURE)
    assert sorted(out["maturity"].unique()) == [5.0, 10.0]           # 5f5 excluded
    for q in ("term_premium", "real_term_premium", "inflation_risk_premium",
              "expected_inflation", "expected_real_short_rate", "nominal_yield"):
        assert q in out.columns
    r10 = out[out["maturity"] == 10.0].iloc[0]
    assert r10["real_term_premium"] == pytest.approx(0.008, abs=1e-9)   # 0.8% -> 0.008
    assert r10["inflation_risk_premium"] == pytest.approx(0.004, abs=1e-9)
    # nominal TP identity = real TP + IRP, in decimal
    assert r10["term_premium"] == pytest.approx(0.008 + 0.004, abs=1e-9)
    assert r10["expected_inflation"] == pytest.approx(0.021, abs=1e-9)
