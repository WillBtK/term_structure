"""The pre-2001 / pre-2003 sources behind the 1994 extension of the US DV01 series."""
import pandas as pd
import pytest

from supply.fetch_us_hist import (_coupon, _date, auction_ladder, parse_h41,
                                  parse_mspd_text)

STATEMENT = """
                     TABLE III - DETAIL OF PUBLIC DEBT OUTSTANDING, JANUARY 31, 1999
    Treasury Bills (Maturity Value):
                BC4                    4.940                      08/27/98           02/25/99           02/25/99          11,299                ...................         22,973
                                       4.450                      11/27/98                                                11,675
    Total Treasury Bills.......................................   ................    ..............     .............   22,973
    Treasury Notes:
    CUSIP:                     Series:      Interest Rate:
        912827 XE7               A            8-7/8 9             02/15/89           02/15/99          02/15-08/15         9,720                ...................          9,720
    912827Y89       Y            6            08/15/96   08/15/99   02/15-08/15     22,708                 ...................         22,708
               6H0     X       6 1/4          07/31/00      07/31/02        01/31-07/31       15,058           ...................        15,058
    Total Treasury Notes......................................   47,486
     Treasury Bonds:
      912810 BG2                         3-1/2 10            10/03/60           11/15/98             05/15-11/15          4,463                (4,237)                  226
    Total Treasury Bonds.......  226
   Treasury Inflation-Indexed Notes:
       912827 3A8         J       3-5/8 8                   07/15/97            07/15/02           01/15-07/15             16,817                   855                 17,672
   Total Treasury Inflation-Indexed Bonds......   0
"""


def test_statement_rows_cover_every_layout_quirk():
    """Joined CUSIPs, spaced and hyphenated fractions, one- and two-digit footnotes,
    retired amounts in brackets, the bill CUSIP prefix carried from the section."""
    df = parse_mspd_text(STATEMENT.replace("BC4 ", "912795 BC4 ", 1),
                         pd.Timestamp("1999-01-31")).set_index("cusip")
    assert df.loc["912795BC4", "amount"] == 22_973          # reopenings summed by MSPD
    assert df.loc["912795BC4", "coupon_pct"] == 0.0
    assert df.loc["912827XE7", "coupon_pct"] == pytest.approx(8.875)
    assert df.loc["912827Y89", "amount"] == 22_708
    assert df.loc["9128276H0", "coupon_pct"] == pytest.approx(6.25)
    assert df.loc["912810BG2", "amount"] == 226             # issued 4,463 less 4,237
    assert df.loc["912810BG2", "issue_date"] == pd.Timestamp("1960-10-03")
    assert df.loc["9128273A8", "asset_class"] == "tips"
    assert set(df["asset_class"]) == {"bill", "nominal", "tips"}


def test_two_digit_years_before_1969_are_twentieth_century():
    assert _date("10/03/60") == pd.Timestamp("1960-10-03")
    assert _date("05/15/09") == pd.Timestamp("2009-05-15")
    assert _coupon("13-3/8") == pytest.approx(13.375) and _coupon("6 1/4") == 6.25


H41_A = """MATURITY DISTRIBUTION OF LOANS AND SECURITIES, DECEMBER 25, 1996
 Holdings changes Holdings changes
 Within 15 days 80 0 Within 15 days 20,574 - 11,651 1,235 - 199
 16 days to 90 days 26 0 16 days to 90 days 93,385 + 3,241 694 0
 Over 90 days 0 0 91 days to 1 year 122,208 + 491 249 0
 Over 1 year to 5 years 95,513 0 520 0
 Over 5 years to 10 years 33,782 0 457 0
 Over 10 years 41,826 0 25 0
 Total 105 0 Total 407,289 - 7,919 3,180 - 199
 U.S. government securities
 Bought outright--Bills 195,120 + 2,794 + 9,061
 Notes 150,922 0 - 91
 Bonds 49,339 0 + 5,270
"""

H41_B = """Maturity Distribution of Loans and Securities, August 25, 2004
 U.S. Treasury Federal agency Reverse
Maturity Loans securities (1) securities Repurchase repurchase
 Weekly Weekly agreements(2)agreements(2)
 Holdings changes Holdings changes
Within 15 days 215 57,989 - 76 0 0 24,500 20,922
16 days to 90 days 21 150,619 - 573 0 0 0 0
91 days to 1 year 0 162,874 + 717 0 0 ------ ------
Over 1 to 5 years ------ 197,636 + 4 0 0 ------ ------
Over 5 to 10 years ------ 49,806 + 3 0 0 ------ ------
Over 10 years ------ 75,916 + 5 0 0 ------ ------
"""

H41_C = """Maturity Distribution of Term Auction Credit, Other Loans, and Securities, October 29, 2008
U.S. Treasury securities (2)
 Holdings 15,320 30,724 61,340 173,435 94,317 101,333 476,469
"""


def test_h41_layout_a_takes_repo_out_of_the_short_buckets():
    r = parse_h41(H41_A)
    assert r["layout"] == "A"
    assert r["b10p"] == 41_826 and r["b5y"] == 95_513
    outright = 195_120 + 150_922 + 49_339
    assert sum(r[k] for k in ("b15", "b90", "b1y", "b5y", "b10y", "b10p")) == pytest.approx(outright)
    buckets = 20_574 + 93_385 + 122_208 + 95_513 + 33_782 + 41_826   # 1 under the printed total
    assert r["b15"] == pytest.approx(20_574 - (buckets - outright))


def test_h41_layout_b_skips_the_loans_column_where_it_is_blank():
    r = parse_h41(H41_B)
    assert r["layout"] == "B"
    assert (r["b15"], r["b90"], r["b5y"], r["b10p"]) == (57_989, 150_619, 197_636, 75_916)


def test_h41_layout_c_reads_the_transposed_holdings_row():
    r = parse_h41(H41_C)
    assert r["layout"] == "C"
    assert (r["b15"], r["b10p"]) == (15_320, 101_333)


def test_auction_ladder_sums_reopenings_and_drops_matured_and_called():
    a = pd.DataFrame({
        "cusip": ["912827AA1", "912827AA1", "912795BB2"],
        "security_type": ["Note", "Note", "Bill"],
        "issue_date": ["1993-02-15", "1993-05-15", "1993-12-09"],
        "maturity_date": ["1996-02-15", "1996-02-15", "1994-01-06"],
        "int_rate": ["5.5", "5.5", "null"], "total_accepted": ["10000000000", "2000000000", "5000000000"],
        "inflation_index_security": ["No", "No", "No"], "floating_rate": ["No", "No", "No"]})
    lad = auction_ladder(pd.period_range("1994-01", "1994-06", freq="M"), auctions=a)
    jan = lad[lad["record_date"] == pd.Timestamp("1994-01-31")].set_index("cusip")
    assert jan.loc["912827AA1", "amount"] == pytest.approx(12_000)
    assert "912795BB2" not in jan.index                       # matured 1994-01-06
    # the pre-1979 callable 8-1/2% 1994-99 leaves at its first call date, 1994-05-15
    may, apr = (lad[lad["record_date"] == pd.Timestamp(d)] for d in ("1994-05-31", "1994-04-30"))
    assert "PRE79-8.500-1999" in set(apr["cusip"]) and "PRE79-8.500-1999" not in set(may["cusip"])


BULLETIN = """             1.19 FEDERAL RESERVE BANKS                          Maturity Distribution of Loan and Security Holding
                                                                        Jan. 26       Feb. 2        Feb. 9      Feb. 16   Feb. 23   Dec. 31       Jan. 31           F e b . 28
              9 Total U.S. Treasury securities                          334,706      332,113       332,508      333,984   338,547   332,015       331,995           333,404
             10 Within fifteen days'                                      11,886       16,814      17,140      19,423     8,532     9,262        12,028         9,168
             11 Sixteen days to ninety days                               77,157       76,422      77,017      79,559    85,486    81,344        79,687        84,699
             12 Ninety-one days to one year                              101,514      100,846     100,589      97,495   100,930   105,184       104,666       106,001
             13 One year to five years                                    75,179       74,942      74,942      74,911    74,911    79,826        79,992        77,654
             14 Five years to ten years                                   22,505       22,505      22,505      21,623    21,623    24,659        23,884        23,818
             15 More than ten years                                       31,111       31,111      31,111      31,111    31,111    31,739        31,739        32,064
"""


def test_bulletin_month_ends_are_dated_by_issue_not_by_the_ocrd_header():
    """The May 1994 issue's month-end columns are Dec 1993 - Feb 1994; the header's
    'F e b . 28' would shift them if they were read from it."""
    from supply.fetch_us_hist import parse_frb_119
    rows = {r["month"]: r for r in parse_frb_119(BULLETIN, pd.Period("1994-05", "M"))}
    assert set(rows) == {pd.Timestamp("1993-12-31"), pd.Timestamp("1994-01-31"),
                         pd.Timestamp("1994-02-28")}
    feb = rows[pd.Timestamp("1994-02-28")]
    assert (feb["total"], feb["b10p"], feb["b5y"]) == (333_404, 32_064, 77_654)
    assert abs(feb["bucket_total"] - feb["total"]) <= 3
