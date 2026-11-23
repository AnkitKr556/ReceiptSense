"""Metrics for comparing extracted fields with ground truth."""
from __future__ import annotations

import difflib
import re

from .parser import _dates_in


def norm_text(s: str | None) -> str:
    """Uppercase letters and digits only, so spacing and punctuation don't count."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def similarity(a: str | None, b: str | None) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def label_date_iso(label: str) -> str | None:
    """Turn a dataset's raw date string ('25/12/2018') into ISO, if it parses."""
    found = _dates_in(label or "")
    return found[0][0].isoformat() if found else None


def total_matches(pred: float | None, truth: float | str | None) -> bool:
    if pred is None or truth in (None, ""):
        return False
    try:
        t = float(str(truth).replace(",", "").replace("RM", "").replace("$", "").strip())
    except ValueError:
        return False
    return abs(pred - t) < 0.005


def item_recall(pred_items: list[str], true_items: list[str], cutoff: float = 0.8) -> float:
    """Share of true item names that have a close match among the extracted items."""
    if not true_items:
        return 1.0
    hits = sum(any(similarity(t, p) >= cutoff for p in pred_items) for t in true_items)
    return hits / len(true_items)


def field_scores(fields, truth: dict, fuzzy: float = 0.9) -> dict:
    """Per-receipt correctness flags for merchant, date and total."""
    true_date = truth.get("date_iso") or label_date_iso(truth.get("date", "")) or truth.get("date")
    sim = similarity(fields.merchant, truth.get("merchant"))
    out = {
        "merchant_exact": norm_text(fields.merchant) == norm_text(truth.get("merchant")) and bool(fields.merchant),
        "merchant_fuzzy": sim >= fuzzy,
        "merchant_similarity": round(sim, 3),
        "date": fields.date is not None and fields.date == true_date,
        "total": total_matches(fields.total, truth.get("total")),
    }
    out["all_three"] = out["merchant_fuzzy"] and out["date"] and out["total"]
    if "items" in truth:
        out["item_recall"] = item_recall(fields.items, truth["items"])
    return out
