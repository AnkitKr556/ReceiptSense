import pytest

from expense_tracker.parser import (find_amounts, find_date, find_gstin, find_items, find_merchant, find_total,
                                    fix_numbers, parse_lines)
from expense_tracker.synthetic import build_dataset, receipt_lines


@pytest.mark.parametrize("text,expected", [
    ("Total 9.00", [9.0]),
    ("Rs.1042.66", [1042.66]),
    ("1,042.66", [1042.66]),
    ("1,04,250.00", [104250.0]),
    ("Rs. 1,693,00", [1693.0]),      # OCR read the decimal point as a comma
    ("TOP, : RM 73. 00", [73.0]),     # OCR put a space after the point
    ("03.11.26", []),                 # a date, not an amount
    ("CGST @2.5% 24.83", [24.83]),
    ("1,234", []),
    ("Qty 2 x 5", []),
])
def test_find_amounts(text, expected):
    assert find_amounts(text) == expected


def test_fix_numbers_only_touches_number_like_tokens():
    assert fix_numbers("TOTAL 1O4.5O") == "TOTAL 104.50"
    assert fix_numbers("SALE 25/l2/2O18") == "SALE 25/12/2018"
    assert fix_numbers("SOAP") == "SOAP"


@pytest.mark.parametrize("line,iso", [
    ("Date: 03/11/2026", "2026-11-03"),
    ("Dt: 03-11-2026", "2026-11-03"),
    ("Date : 03.11.26", "2026-11-03"),
    ("DATE: 03-Nov-2026", "2026-11-03"),
    ("30 Jan 2026", "2026-01-30"),
    ("2026-01-30", "2026-01-30"),
    ("12/28/2017", "2017-12-28"),     # month-first falls back when day-first is impossible
    ("Mar 5, 2018", "2018-03-05"),
])
def test_find_date_formats(line, iso):
    assert find_date(["SHOP", line]) == iso


def test_find_date_prefers_line_with_date_word():
    lines = ["Plot 3-5-12, Main Road", "Bill No 12   Date: 05/06/2025"]
    assert find_date(lines) == "2025-06-05"


def test_find_date_none():
    assert find_date(["no dates here", "Total 5.00"]) is None


def test_total_prefers_strong_keyword_and_skips_subtotal():
    lines = ["Sub Total 900.00", "CGST 27.00", "TOTAL 954.00", "Grand Total 954.00", "Cash 1000.00", "Change 46.00"]
    assert find_total(lines) == (954.0, 3)


def test_total_ignores_gst_summary_table():
    lines = ["TOTAL 4.90", "CASH 4.90", "CHANGE 0.00", "GST SUMMARY", "TOTAL: 4.60 0.28"]
    assert find_total(lines)[0] == 4.90


def test_total_takes_last_candidate_before_payment():
    lines = ["Total : 9.00", "Rounded Total 9.00", "CASH 10.00", "Total savings 1.00"]
    assert find_total(lines)[0] == 9.00


def test_total_on_next_line():
    assert find_total(["NET AMOUNT", "512.40", "Thank you"])[0] == 512.40


def test_total_integer_amount():
    assert find_total(["Bill Amount 450"])[0] == 450.0


def test_total_fallback_largest_amount():
    lines = ["Rate (Rs/L): 104.65", "Volume (L): 9.56", "Amount (Rs): 1000.00", "Cash 2000.00"]
    assert find_total(lines)[0] == 1000.0


def test_total_none():
    assert find_total(["hello", "world"]) == (None, None)


def test_merchant_skips_headers_and_prefers_business_name():
    lines = ["TAX INVOICE", "TAN WOON YANN", "BOOK TA.K (TAMAN DAYA) SDN BHD", "Tel: 07-123"]
    assert find_merchant(lines)[0] == "BOOK TA.K (TAMAN DAYA) SDN BHD"


def test_merchant_skips_address_lines():
    lines = ["12/5, Lanka, Varanasi - 221005", "Om Kirana Store"]
    assert find_merchant(lines)[0] == "Om Kirana Store"


def test_merchant_none():
    assert find_merchant(["12345", "---"]) == (None, None)


def test_find_items():
    lines = ["Item Qty Rate Amount", "Toor Dal 1kg 1 168.00 168.00", "Butter 500g x2 570.00",
             "CGST 2.5% 9.00", "Sub Total 738.00"]
    assert find_items(lines, 0, len(lines)) == ["Toor Dal 1kg", "Butter 500g"]  # quantity "x2" is stripped


def test_find_gstin():
    assert find_gstin(["GSTIN: 09ABCDE1234F1Z5"]) == "09ABCDE1234F1Z5"
    assert find_gstin(["nothing"]) is None


def test_parse_lines_on_perfect_synthetic_text():
    """With perfect text, the parser should get every synthetic receipt right."""
    ds = build_dataset(seed=3)
    for r in ds.test[:200]:
        f = parse_lines([t for t, _ in receipt_lines(r)])
        assert f.merchant.lower() == r.merchant.name.lower()
        assert f.date == r.date.isoformat()
        assert f.total == pytest.approx(r.total)
