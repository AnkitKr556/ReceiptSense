"""Thin wrapper around Tesseract that returns text lines with positions."""
from __future__ import annotations

import numpy as np
import pytesseract
from pytesseract import Output

from .models import OcrLine


def run_ocr(img: np.ndarray, psm: int = 4, lang: str = "eng") -> list[OcrLine]:
    """OCR an image and group the words into lines.

    psm 4 tells Tesseract to expect a single column of text of varying sizes,
    which is what a receipt is.
    """
    data = pytesseract.image_to_data(img, lang=lang, config=f"--oem 1 --psm {psm}",
                                     output_type=Output.DICT)
    groups: dict[tuple[int, int, int], list[tuple]] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append(
            (data["left"][i], data["top"][i], data["width"][i], data["height"][i], word, conf))

    lines = []
    for words in groups.values():
        words.sort(key=lambda w: w[0])
        x0 = min(w[0] for w in words)
        y0 = min(w[1] for w in words)
        x1 = max(w[0] + w[2] for w in words)
        y1 = max(w[1] + w[3] for w in words)
        text = " ".join(w[4] for w in words)
        conf = sum(w[5] for w in words) / len(words)
        lines.append(OcrLine(text, conf, (x0, y0, x1 - x0, y1 - y0)))
    return merge_rows(lines)


def merge_rows(lines: list[OcrLine]) -> list[OcrLine]:
    """Join lines that sit on the same visual row.

    Tesseract sometimes puts a label ("TOTAL") and its value ("9.00") in separate
    blocks when they are far apart. Joining them keeps the label next to its amount.
    """
    lines = sorted(lines, key=lambda l: l.bbox[1] + l.bbox[3] / 2)
    merged: list[list[OcrLine]] = []
    for line in lines:
        cy = line.bbox[1] + line.bbox[3] / 2
        if merged:
            row = merged[-1]
            ref = row[0]
            ref_cy = ref.bbox[1] + ref.bbox[3] / 2
            tol = 0.5 * min(ref.bbox[3], line.bbox[3])
            overlaps = any(_x_overlap(line, other) for other in row)
            if abs(cy - ref_cy) <= tol and not overlaps:
                row.append(line)
                continue
        merged.append([line])

    out = []
    for row in merged:
        row.sort(key=lambda l: l.bbox[0])
        if len(row) == 1:
            out.append(row[0])
            continue
        x0 = min(l.bbox[0] for l in row)
        y0 = min(l.bbox[1] for l in row)
        x1 = max(l.bbox[0] + l.bbox[2] for l in row)
        y1 = max(l.bbox[1] + l.bbox[3] for l in row)
        text = " ".join(l.text for l in row)
        conf = sum(l.conf for l in row) / len(row)
        out.append(OcrLine(text, conf, (x0, y0, x1 - x0, y1 - y0)))
    return out


def _x_overlap(a: OcrLine, b: OcrLine) -> bool:
    return a.bbox[0] < b.bbox[0] + b.bbox[2] and b.bbox[0] < a.bbox[0] + a.bbox[2]
