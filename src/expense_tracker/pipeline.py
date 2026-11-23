"""The full receipt pipeline: image -> preprocess -> OCR -> parse -> categorize."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .categorizer import Categorizer
from .models import OcrLine, Prediction, ReceiptFields
from .ocr import run_ocr
from .parser import parse_lines
from .preprocess import PreprocessConfig, preprocess

BLUR_WARNING_THRESHOLD = 60.0  # variance of Laplacian; tuned by eye on the dev images


class InvalidImageError(ValueError):
    """The file is missing, not an image, or empty."""


@dataclass
class PipelineResult:
    fields: ReceiptFields
    prediction: Prediction | None
    ocr_lines: list[OcrLine]
    timings: dict[str, float]  # seconds per stage
    info: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "merchant": self.fields.merchant,
            "date": self.fields.date,
            "total": self.fields.total,
            "items": self.fields.items,
            "gstin": self.fields.gstin,
            "category": self.prediction.category if self.prediction else None,
            "confidence": round(self.prediction.confidence, 3) if self.prediction else None,
            "source": self.prediction.source if self.prediction else None,
            "needs_review": self.prediction.needs_review if self.prediction else True,
            "warnings": self.warnings,
            "timings_ms": {k: round(v * 1000, 1) for k, v in self.timings.items()},
        }


class ReceiptPipeline:
    def __init__(self, categorizer: Categorizer | None = None,
                 config: PreprocessConfig = PreprocessConfig(), psm: int = 4):
        self.categorizer = categorizer
        self.config = config
        self.psm = psm

    def process_file(self, path: str | Path) -> PipelineResult:
        path = Path(path)
        if not path.is_file():
            raise InvalidImageError(f"file not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError(f"not a readable image: {path}")
        return self.process_image(img)

    def process_image(self, img: np.ndarray) -> PipelineResult:
        if img is None or img.size == 0 or min(img.shape[:2]) < 50:
            raise InvalidImageError("image is empty or too small")
        timings: dict[str, float] = {}

        t = time.perf_counter()
        clean, info = preprocess(img, self.config)
        timings["preprocess"] = time.perf_counter() - t

        t = time.perf_counter()
        lines = run_ocr(clean, psm=self.psm)
        timings["ocr"] = time.perf_counter() - t

        t = time.perf_counter()
        fields = parse_lines([l.text for l in lines])
        timings["parse"] = time.perf_counter() - t

        prediction = None
        t = time.perf_counter()
        if self.categorizer is not None and fields.merchant:
            prediction = self.categorizer.predict(fields.merchant, fields.items)
        timings["categorize"] = time.perf_counter() - t
        timings["total"] = sum(timings.values())

        warnings = []
        if info["blur_score"] < BLUR_WARNING_THRESHOLD:
            warnings.append("image looks blurry; please check the extracted fields")
        if not lines:
            warnings.append("no text found in the image")
        for name in ("merchant", "date", "total"):
            if getattr(fields, name) is None:
                warnings.append(f"could not find the {name}")
        return PipelineResult(fields, prediction, lines, timings, info, warnings)
