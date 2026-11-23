import json
import shutil

import numpy as np
import pytest

from expense_tracker.categorizer import HybridCategorizer
from expense_tracker.cli import main
from expense_tracker.evaluate import field_scores, item_recall, label_date_iso, similarity, total_matches
from expense_tracker.pipeline import InvalidImageError, ReceiptPipeline
from expense_tracker.synthetic import (CATEGORIES, HELD_OUT, build_dataset, print_name, receipt_lines,
                                       render_receipt)

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


@pytest.fixture(scope="module")
def small_ds():
    return build_dataset(seed=4, n_train=300, n_val=50, n_test=60)


# ---------------------------------------------------------------- synthetic data

def test_dataset_splits(small_ds):
    assert len(small_ds.train) == 300 and len(small_ds.test) == 60
    known = {m.name for m in small_ds.merchants["known"]}
    assert all(r.merchant.name in known for r in small_ds.train)
    new = [r for r, k in zip(small_ds.test, small_ds.test_is_known) if not k]
    assert all(r.merchant.name not in known for r in new)
    assert {m.category for m in small_ds.merchants["known"]} == set(CATEGORIES)


def test_held_out_items_never_in_training(small_ds):
    styles = {"title", "upper", "trunc", "vowel"}
    printed_held_out = {print_name(n, s) for n in HELD_OUT for s in styles}
    for r in small_ds.train:
        assert not set(r.item_names) & printed_held_out


def test_receipt_amounts_add_up(small_ds):
    for r in small_ds.test:
        if r.fuel:
            continue
        assert r.subtotal == pytest.approx(sum(i[3] for i in r.items))
        assert r.total == pytest.approx(r.subtotal + sum(t[1] for t in r.taxes) + r.round_off, abs=0.011)


def test_print_name_styles():
    assert print_name("Paneer Butter Masala", "vowel") == "PNR BTTR MSL"
    assert print_name("Toor Dal 1kg", "trunc") == "TOOR DAL 1KG"
    assert print_name("Toor Dal", "upper") == "TOOR DAL"
    assert print_name("Toor Dal", "title") == "Toor Dal"


def test_render_levels(small_ds):
    r = small_ds.test[0]
    sizes = [render_receipt(r, np.random.default_rng(0), lv).shape for lv in ("clean", "medium", "hard")]
    assert all(len(s) == 3 for s in sizes)
    assert sizes[2][1] < sizes[0][1]  # hard photos are smaller


def test_receipt_lines_has_total(small_ds):
    r = small_ds.test[1]
    text = "\n".join(t for t, _ in receipt_lines(r))
    assert f"{r.total:.2f}" in text.replace(",", "")


# ---------------------------------------------------------------- metrics

def test_metrics():
    assert similarity("Book Ta.K Sdn Bhd", "BOOK TA K SDN BHD") == 1.0
    assert similarity(None, "x") == 0.0
    assert label_date_iso("25/12/2018") == "2018-12-25"
    assert label_date_iso("garbage") is None
    assert total_matches(9.0, "9.00") and total_matches(8.2, "$8.20")
    assert not total_matches(None, "1.00") and not total_matches(1.0, "abc") and not total_matches(1.0, "")
    assert item_recall(["TOOR DAL 1KG"], ["Toor Dal 1kg", "Sugar 1kg"]) == 0.5
    assert item_recall([], []) == 1.0


def test_field_scores():
    class F:
        merchant, date, total, items = "OM STORE", "2025-01-02", 10.0, ["Milk"]
    s = field_scores(F, {"merchant": "Om Store", "date": "02/01/2025", "total": "10.00", "items": ["Milk"]})
    assert s["merchant_exact"] and s["date"] and s["total"] and s["all_three"] and s["item_recall"] == 1.0


# ---------------------------------------------------------------- pipeline

def test_invalid_inputs(tmp_path):
    pipe = ReceiptPipeline()
    with pytest.raises(InvalidImageError):
        pipe.process_file(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    with pytest.raises(InvalidImageError):
        pipe.process_file(bad)
    with pytest.raises(InvalidImageError):
        pipe.process_image(np.zeros((10, 10, 3), np.uint8))


@needs_tesseract
def test_blank_image_gives_warnings_not_crash():
    res = ReceiptPipeline().process_image(np.full((600, 400, 3), 245, np.uint8))
    assert res.fields.total is None
    assert any("total" in w for w in res.warnings)
    assert res.to_dict()["needs_review"] is True


@needs_tesseract
def test_end_to_end_clean_receipt(small_ds):
    train = [r.to_expense() for r in small_ds.train]
    model = HybridCategorizer().fit(train)
    r = next(r for r, k in zip(small_ds.test, small_ds.test_is_known) if k and not r.fuel)
    img = render_receipt(r, np.random.default_rng(1), "clean")
    res = ReceiptPipeline(categorizer=model).process_image(img)
    assert res.fields.date == r.date.isoformat()
    assert res.fields.total == pytest.approx(r.total)
    assert similarity(res.fields.merchant, r.merchant.name) > 0.9
    assert res.prediction.category == r.merchant.category
    d = res.to_dict()
    assert set(d["timings_ms"]) == {"preprocess", "ocr", "parse", "categorize", "total"}


# ---------------------------------------------------------------- CLI

@needs_tesseract
def test_cli_train_sample_process(tmp_path, capsys):
    csv_path = tmp_path / "train.csv"
    rows = ["merchant,items,category"] + [
        f"{e.merchant},{';'.join(e.items)},{e.category}"
        for e in (r.to_expense() for r in build_dataset(seed=5, n_train=200, n_val=10, n_test=10).train)]
    csv_path.write_text("\n".join(rows))
    model = tmp_path / "m" / "model.pkl"
    assert main(["train", "--csv", str(csv_path), "--out", str(model)]) == 0
    img = tmp_path / "r.jpg"
    assert main(["sample", str(img), "--level", "clean"]) == 0
    capsys.readouterr()
    assert main(["process", str(img), "--model", str(model), "--show-text"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {"merchant", "date", "total", "category", "ocr_text"} <= set(out)
    assert main(["process", str(tmp_path / "nope.jpg")]) == 2


def test_cli_rejects_bad_csv(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("shop,cat\nA,B\n")
    with pytest.raises(SystemExit):
        main(["train", "--csv", str(bad), "--out", str(tmp_path / "m.pkl")])
