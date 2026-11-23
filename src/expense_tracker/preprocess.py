"""Image clean-up before OCR.

Phone photos of receipts are small, tilted, noisy and unevenly lit. Tesseract does
much better on a large, straight, clean image, so we fix those things first:

1. grayscale
2. upscale, so the text is tall enough for Tesseract (about 30 px per line)
3. flatten the lighting, by dividing out a blurred estimate of the paper background
4. deskew, by finding the rotation that makes the text rows sharpest
5. denoise, with a small median filter, but only on soft (not sharp) images:
   on sharp images the filter erodes fine detail such as decimal points
6. binarize (optional), Otsu or adaptive thresholding

Every step can be switched off through PreprocessConfig, which is how the
ablation experiment measures what each step is worth.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    upscale: bool = True
    min_width: int = 1000
    flatten_light: bool = True
    deskew: bool = True
    max_skew: float = 6.0  # degrees searched on each side
    denoise: str = "auto"  # "auto" (only when blur_score < sharp_threshold), "always" or "never"
    sharp_threshold: float = 1500.0  # chosen on the dev sets
    binarize: str = "none"  # "none", "otsu" or "adaptive"


RAW = PreprocessConfig(upscale=False, flatten_light=False, deskew=False, denoise="never", binarize="none")


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale(gray: np.ndarray, min_width: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    if w >= min_width:
        return gray, 1.0
    f = min_width / w
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), f


def flatten_light(gray: np.ndarray) -> np.ndarray:
    """Remove shadows and uneven light.

    A morphological closing with a kernel wider than the text strokes erases the
    dark text and leaves the paper; dividing by that background makes the paper
    evenly white everywhere.
    """
    k = max(15, (gray.shape[1] // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), k / 2)
    return cv2.divide(gray, np.maximum(background, 1), scale=255)


def rotate(gray: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    """Rotate counter-clockwise by `angle` degrees, growing the canvas so no corner is cut."""
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _profile_score(bw: np.ndarray, angle: float) -> float:
    rotated = rotate(bw, angle, border=0)
    profile = rotated.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(profile) ** 2))


def estimate_skew(gray: np.ndarray, max_angle: float = 6.0) -> float:
    """Return the rotation (degrees) that straightens the text.

    Projection-profile method: when text rows are horizontal, the row sums of the
    ink pixels alternate sharply between text and gaps, so the squared differences
    of the profile peak. Coarse search in 0.5 degree steps, then 0.1 around the best.
    """
    h, w = gray.shape
    scale = 800 / w if w > 800 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.5)
    best = max(coarse, key=lambda a: _profile_score(bw, a))
    fine = np.arange(best - 0.4, best + 0.4 + 1e-9, 0.1)
    best = max(fine, key=lambda a: _profile_score(bw, a))
    return round(float(best), 2)


def binarize(gray: np.ndarray, method: str) -> np.ndarray:
    if method == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
    if method == "none":
        return gray
    raise ValueError(f"unknown binarize method: {method}")


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Low values mean a blurry image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess(img: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> tuple[np.ndarray, dict]:
    """Run the enabled steps and return the cleaned image plus what was done."""
    gray = to_gray(img)
    info = {"scale": 1.0, "skew": 0.0, "blur_score": blur_score(gray), "denoised": False}
    if cfg.upscale:
        gray, info["scale"] = upscale(gray, cfg.min_width)
    if cfg.flatten_light:
        gray = flatten_light(gray)
    if cfg.deskew:
        angle = estimate_skew(gray, cfg.max_skew)
        info["skew"] = angle
        if abs(angle) >= 0.2:
            gray = rotate(gray, angle, border=int(np.median(gray)))
    if cfg.denoise not in ("auto", "always", "never"):
        raise ValueError(f"unknown denoise mode: {cfg.denoise}")
    if cfg.denoise == "always" or (cfg.denoise == "auto" and info["blur_score"] < cfg.sharp_threshold):
        gray = cv2.medianBlur(gray, 3)
        info["denoised"] = True
    gray = binarize(gray, cfg.binarize)
    return gray, info
