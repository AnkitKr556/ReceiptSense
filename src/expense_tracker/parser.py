"""Pull merchant, date, total and items out of OCR text with regex and layout rules.

The rules follow how printed receipts are laid out:
- the shop name is near the top, usually the first line that is mostly letters
- the date matches a date pattern, best on a line that says "Date"
- the total sits on a line with a word like "Total" or "Net Amount", and is the
  amount at the right end of that line
- item lines are between the header and the total and end in a price

All functions take plain strings, so they can be tested without running OCR.
"""
from __future__ import annotations

import re
from datetime import date

from .models import ReceiptFields

# ---------------------------------------------------------------- numbers

# Characters OCR often confuses with digits. Only fixed inside tokens that are
# already mostly digits, so words like "SALE" are left alone.
_CONFUSABLE = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1",
                             "|": "1", "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})
_SEPARATORS = set(".,/:-")


def _fix_token(tok: str) -> str:
    digits = sum(c.isdigit() for c in tok)
    if digits == 0:
        return tok
    confusable = sum(c in "OoDQlI|iSsBZz" for c in tok)
    seps = sum(c in _SEPARATORS for c in tok)
    other = len(tok) - digits - confusable - seps
    if other == 0 and confusable <= digits:
        return tok.translate(_CONFUSABLE)
    return tok


def fix_numbers(line: str) -> str:
    """Repair digit look-alikes in number-like tokens: '1O4.5O' -> '104.50'."""
    return " ".join(_fix_token(t) for t in line.split())


# An amount has exactly two decimals: 9.00, 1042.66, 1,042.66, 1,04,250.00.
# OCR often reads the decimal point as a comma ("1,093,00"), so either is accepted
# before the last two digits; thousands separators must be commas.
# The look-behinds stop us matching the middle of a longer number, and the
# last look-ahead stops dates like 03.11.26 being read as 3.11.
AMOUNT_RE = re.compile(
    r"(?<!\d)(?<!\d[.,])(\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)[.,] ?(\d{2})(?!\d)(?!\.\d)")
INT_RE = re.compile(r"(?<![\d.,/:-])(\d{1,6})(?![\d.,/:%])")


def find_amounts(line: str) -> list[float]:
    return [float(m.group(1).replace(",", "") + "." + m.group(2)) for m in AMOUNT_RE.finditer(line)]


# ---------------------------------------------------------------- total

_EXCLUDE_TOTAL = re.compile(
    r"sub\s*-?\s*t[o0]ta[l1i]|t[o0]ta[l1i]\s*(qty|quantity|items?|no\b|tax|gst|vat|disc|saving|excl)"
    r"|excl|before\s+tax|t[o0]ta[l1i]\s+sales\s*\(?\s*excl", re.I)
_STRONG_TOTAL = re.compile(
    r"grand\s*t[o0]ta[l1i]|net\s*(amount|amt|t[o0]ta[l1i]|payable)|nett\s*t[o0]ta[l1i]"
    r"|amount\s*(payable|due)|t[o0]ta[l1i]\s*(amount|amt|payable|due|rs|inr|rm|\(rm\)|incl)"
    r"|bill\s*amount|rounded\s*t[o0]ta[l1i]|t[o0]ta[l1i]\s+sales\s*\(?\s*incl|incl\w*\s+(of\s+)?gst|to\s*pay\b", re.I)
_WEAK_TOTAL = re.compile(r"\bt[o0]ta[l1i]\b", re.I)
_NOT_TOTAL_AMOUNT = re.compile(r"cash|change|tender|paid|balance|card|rate|qty|round", re.I)
# Lines where the customer pays. The total is printed just before these.
_PAYMENT = re.compile(r"\bcash\b|change|tender|\bpaid\b|\bcard\b|\bupi\b|visa|master", re.I)
# Tax summary tables below the total ("GST SUMMARY ... TOTAL: 26.00 1.56") repeat
# the word TOTAL with the pre-tax amount, so we stop looking once one starts.
_SUMMARY = re.compile(r"summary", re.I)


def find_total(lines: list[str]) -> tuple[float | None, int | None]:
    """Return (total, line index).

    The highest-priority keyword wins. Among equal ones, take the last one printed
    before the payment lines (cash, change...), or the last one if there are none.
    """
    stop = next((i for i, l in enumerate(lines) if _SUMMARY.search(l)), len(lines))
    candidates = []  # (priority, index, amount)
    for i, line in enumerate(lines[:stop]):
        if _EXCLUDE_TOTAL.search(line):
            continue
        if _STRONG_TOTAL.search(line):
            priority = 2
        elif _WEAK_TOTAL.search(line):
            priority = 1
        else:
            continue
        amounts = find_amounts(line)
        if not amounts and i + 1 < len(lines):
            nxt = lines[i + 1]
            if sum(c.isalpha() for c in nxt) <= 3:
                amounts = find_amounts(nxt)
        if not amounts:
            ints = [int(x) for x in INT_RE.findall(line)]
            amounts = [float(x) for x in ints if x > 0]
        if amounts:
            candidates.append((priority, i, amounts[-1]))
    if candidates:
        top = max(c[0] for c in candidates)
        best_of = [c for c in candidates if c[0] == top]
        first_pay = next((i for i in range(best_of[0][1] + 1, stop) if _PAYMENT.search(lines[i])), None)
        if first_pay is not None:
            before = [c for c in best_of if c[1] < first_pay]
            best_of = before or best_of
        best = best_of[-1]
        return best[2], best[1]
    # Fallback: the largest amount that is not cash handed over, change, a rate...
    pool = [(a, i) for i, line in enumerate(lines[:stop]) if not _NOT_TOTAL_AMOUNT.search(line)
            for a in find_amounts(line)]
    if pool:
        a, i = max(pool)
        return a, i
    return None, None


# ---------------------------------------------------------------- date

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE_PATTERNS = [
    (re.compile(r"(?<!\d)(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?!\d)"), "dmy"),
    (re.compile(r"(?<!\d)(\d{1,2})[\s\-/.]*" + _MON + r"[\s\-/.,']*(\d{4}|\d{2})(?!\d)", re.I), "dMy"),
    (re.compile(_MON + r"[\s\-/.]*(\d{1,2})(?:st|nd|rd|th)?[\s,\-/.]+(\d{4})(?!\d)", re.I), "Mdy"),
]
_DATE_WORD = re.compile(r"\bdate\b|\bdt\b|tarikh", re.I)


def _make_date(d: int, m: int, y: int) -> date | None:
    if y < 100:
        y += 2000
    for dd, mm in ((d, m), (m, d)):  # day-first, then month-first as a fallback
        try:
            out = date(y, mm, dd)
        except ValueError:
            continue
        if 2000 <= out.year <= 2035:
            return out
    return None


def _dates_in(line: str) -> list[tuple[date, bool]]:
    found = []
    for pattern, kind in _DATE_PATTERNS:
        for m in pattern.finditer(line):
            g = m.groups()
            if kind == "ymd":
                d = _make_date(int(g[2]), int(g[1]), int(g[0]))
                four = True
            elif kind == "dmy":
                d = _make_date(int(g[0]), int(g[1]), int(g[2]))
                four = len(g[2]) == 4
            elif kind == "dMy":
                d = _make_date(int(g[0]), _MONTHS[g[1].lower()[:3]], int(g[2]))
                four = len(g[2]) == 4
            else:
                d = _make_date(int(g[1]), _MONTHS[g[0].lower()[:3]], int(g[2]))
                four = True
            if d:
                found.append((d, four))
    return found


def find_date(lines: list[str]) -> str | None:
    best = None  # (score, -index, date)
    for i, line in enumerate(lines):
        for d, four_digit_year in _dates_in(line):
            score = 2 * bool(_DATE_WORD.search(line)) + four_digit_year + (d.year >= 2015)
            key = (score, -i)
            if best is None or key > best[0]:
                best = (key, d)
    return best[1].isoformat() if best else None


# ---------------------------------------------------------------- merchant

_NOT_MERCHANT = re.compile(
    r"tax\s*invoice|invoice|receipt|cash\s*memo|\bbill\s*(no|of|:|#|date)|gst|\btel\b|phone|\bph\b|fax|www|http|@"
    r"|welcome|original|duplicate|copy|\bdate\b|\btime\b|cashier|table|counter|\bno\s*[:.]", re.I)
_BUSINESS_WORDS = re.compile(
    r"\bsdn\b|\bbhd\b|pvt|ltd|llp|enterprise|trading|store|stores|mart|supermarket|restaurant|cafe|hotel"
    r"|bakery|pharmacy|medical|chemist|petroleum|fuel|filling|station|electronic|traders|sons"
    r"|bhandar|kirana|dhaba|bhavan|sweets|foods|kitchen|garments|fashion|cinema|telecom|agency"
    r"|centre|center|point|house|world|zone|gallery|services|clinic|collection|emporium", re.I)


def _plausible_merchant(line: str) -> bool:
    letters = sum(c.isalpha() for c in line)
    digits = sum(c.isdigit() for c in line)
    if letters < 3 or letters < 0.5 * len(line.replace(" ", "")):
        return False
    if digits > 0.3 * max(1, letters):
        return False
    return not _NOT_MERCHANT.search(line)


def find_merchant(lines: list[str], window: int = 6) -> tuple[str | None, int | None]:
    top = [(i, l) for i, l in enumerate(lines[:window]) if _plausible_merchant(l)]
    if not top:
        return None, None
    for i, line in top:  # prefer a line that names a kind of business
        if _BUSINESS_WORDS.search(line):
            return _clean(line), i
    return _clean(top[0][1]), top[0][0]


def _clean(text: str) -> str:
    text = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9).]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- items and GSTIN

_SKIP_ITEM = re.compile(
    r"t[o0]ta[l1i]|tax|gst|vat|discount|round|cash|change|tender|paid|balance|card|upi|\bdate\b"
    r"|\btime\b|bill\s*no|invoice|phone|\btel\b|\bqty\b|\brate\b|amount|thank|visit|cashier|table"
    r"|saving|receipt|nozzle|volume", re.I)
_PRICE_TOKEN = re.compile(r"^(rs\.?|inr|rm|[@x×*]|[\d.,/@x×*%-]+)$", re.I)
GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b")


def find_items(lines: list[str], start: int, end: int) -> list[str]:
    items = []
    for line in lines[start:end]:
        if _SKIP_ITEM.search(line) or not find_amounts(line):
            continue
        if sum(c.isalpha() for c in line) < 3:
            continue
        tokens = line.split()
        while tokens and _PRICE_TOKEN.match(tokens[-1]):
            tokens.pop()
        name = _clean(" ".join(tokens))
        if sum(c.isalpha() for c in name) >= 3:
            items.append(name)
    return items


def find_gstin(lines: list[str]) -> str | None:
    for line in lines:
        m = GSTIN_RE.search(line.replace(" ", "").upper())
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------- all together

def parse_lines(raw_lines: list[str]) -> ReceiptFields:
    lines = [fix_numbers(l) for l in raw_lines if l.strip()]
    merchant, m_idx = find_merchant(lines)
    total, t_idx = find_total(lines)
    start = (m_idx + 1) if m_idx is not None else 0
    end = t_idx if t_idx is not None and t_idx > start else len(lines)
    return ReceiptFields(
        merchant=merchant,
        date=find_date(lines),
        total=total,
        items=find_items(lines, start, end),
        gstin=find_gstin(lines),
    )
