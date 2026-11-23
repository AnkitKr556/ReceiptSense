import shutil

import cv2
import numpy as np
import pytest

from expense_tracker.models import OcrLine
from expense_tracker.ocr import merge_rows, run_ocr
from expense_tracker.preprocess import (RAW, PreprocessConfig, binarize, blur_score, estimate_skew,
                                        flatten_light, preprocess, rotate, to_gray, upscale)
from expense_tracker.synthetic import build_dataset, render_receipt

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


@pytest.fixture(scope="module")
def receipt_image():
    r = build_dataset(seed=2, n_train=10, n_val=10, n_test=10).test[0]
    return render_receipt(r, np.random.default_rng(0), "clean"), r


@pytest.mark.parametrize("angle", [-4.0, -1.5, 2.0, 3.5])
def test_estimate_skew_recovers_rotation(receipt_image, angle):
    gray = to_gray(receipt_image[0])
    tilted = rotate(gray, angle)
    assert estimate_skew(tilted) == pytest.approx(-angle, abs=0.3)


def test_upscale_only_when_small():
    small = np.full((100, 400), 255, np.uint8)
    big, f = upscale(small, 1000)
    assert big.shape[1] == 1000 and f == pytest.approx(2.5)
    same, f = upscale(np.full((100, 1200), 255, np.uint8), 1000)
    assert same.shape[1] == 1200 and f == 1.0


def test_flatten_light_evens_out_background():
    h, w = 400, 600
    gradient = np.tile(np.linspace(250, 120, w), (h, 1)).astype(np.uint8)
    flat = flatten_light(gradient)
    assert gradient[:, :50].mean() - gradient[:, -50:].mean() > 100
    assert abs(float(flat[:, :50].mean()) - float(flat[:, -50:].mean())) < 15


def test_binarize_methods():
    gray = np.random.default_rng(0).integers(0, 255, (60, 60)).astype(np.uint8)
    for method in ("otsu", "adaptive"):
        out = binarize(gray, method)
        assert set(np.unique(out)) <= {0, 255}
    assert binarize(gray, "none") is gray
    with pytest.raises(ValueError):
        binarize(gray, "magic")


def test_to_gray_handles_channels():
    assert to_gray(np.zeros((5, 5), np.uint8)).shape == (5, 5)
    assert to_gray(np.zeros((5, 5, 3), np.uint8)).shape == (5, 5)
    assert to_gray(np.zeros((5, 5, 4), np.uint8)).shape == (5, 5)


def test_preprocess_reports_steps(receipt_image):
    out, info = preprocess(receipt_image[0], PreprocessConfig())
    assert out.ndim == 2 and out.shape[1] >= 1000
    assert set(info) == {"scale", "skew", "blur_score", "denoised"}
    raw, info_raw = preprocess(receipt_image[0], RAW)
    assert raw.shape == receipt_image[0].shape[:2] and info_raw["skew"] == 0.0 and not info_raw["denoised"]


def test_denoise_modes(receipt_image):
    img = receipt_image[0]
    assert not preprocess(img)[1]["denoised"]  # a clean render is sharp, so "auto" leaves it alone
    assert preprocess(img, PreprocessConfig(sharp_threshold=1e9))[1]["denoised"]
    assert preprocess(img, PreprocessConfig(denoise="always"))[1]["denoised"]
    with pytest.raises(ValueError):
        preprocess(img, PreprocessConfig(denoise="sometimes"))


def test_blur_score_drops_when_blurred(receipt_image):
    gray = to_gray(receipt_image[0])
    assert blur_score(cv2.GaussianBlur(gray, (0, 0), 3)) < blur_score(gray) / 3


def test_merge_rows_joins_label_and_value():
    lines = [OcrLine("TOTAL", 90, (10, 100, 80, 20)), OcrLine("9.00", 80, (400, 102, 50, 20)),
             OcrLine("CASH", 90, (10, 140, 60, 20))]
    merged = merge_rows(lines)
    assert [l.text for l in merged] == ["TOTAL 9.00", "CASH"]
    assert merged[0].conf == pytest.approx(85)


@needs_tesseract
def test_run_ocr_reads_rendered_receipt(receipt_image):
    img, r = receipt_image
    lines = run_ocr(preprocess(img)[0])
    text = " ".join(l.text for l in lines).upper()
    assert r.merchant.name.upper().split()[0] in text
    assert all(0 <= l.conf <= 100 for l in lines)
