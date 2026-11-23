"""Plain data classes passed between the pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OcrLine:
    """One line of text found by the OCR engine."""

    text: str
    conf: float  # mean word confidence, 0-100
    bbox: tuple[int, int, int, int]  # x, y, width, height in pixels


@dataclass
class ReceiptFields:
    """What the parser pulls out of the OCR text."""

    merchant: str | None = None
    date: str | None = None  # ISO format, yyyy-mm-dd
    total: float | None = None
    items: list[str] = field(default_factory=list)
    gstin: str | None = None


@dataclass
class Expense:
    """A labelled expense: the input to the categorizer plus its true category."""

    merchant: str
    items: list[str]
    category: str


@dataclass
class Prediction:
    """A categorizer's answer for one expense."""

    category: str
    confidence: float  # 0-1
    source: str  # "rule", "keyword", "ml" or "none"
    needs_review: bool = False
