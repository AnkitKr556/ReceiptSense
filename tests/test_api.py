"""Tests for the web API and the SQLite store."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from expense_tracker.api import create_app  # noqa: E402
from expense_tracker.categorizer import HybridCategorizer  # noqa: E402
from expense_tracker.models import Expense  # noqa: E402
from expense_tracker.storage import ExpenseStore  # noqa: E402
from expense_tracker.synthetic import build_dataset, render_receipt  # noqa: E402

SMALL_TRAIN = [
    Expense("Sharma General Store", ["toor dal", "basmati rice"], "Groceries"),
    Expense("Gupta Kirana", ["atta 5kg", "sugar"], "Groceries"),
    Expense("Hotel Annapurna", ["masala dosa", "filter coffee"], "Dining"),
    Expense("Biryani House", ["chicken biryani", "raita"], "Dining"),
    Expense("Apollo Pharmacy", ["paracetamol", "cough syrup"], "Health & Pharmacy"),
    Expense("City Medicals", ["vitamin c", "bandage"], "Health & Pharmacy"),
    Expense("HP Filling Station", ["petrol"], "Fuel & Transport"),
    Expense("Indian Oil Petrol Pump", ["diesel"], "Fuel & Transport"),
]


@pytest.fixture
def client(tmp_path):
    model_path = tmp_path / "base.pkl"
    HybridCategorizer(tau=0.4, retrain_every=3).fit(SMALL_TRAIN).save(model_path)
    app = create_app(data_dir=tmp_path / "app", model_path=model_path)
    with TestClient(app) as c:
        yield c


def _jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_and_categories(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    cats = client.get("/api/categories").json()
    assert "Groceries" in cats and len(cats) == 8


def test_dashboard_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ReceiptSense" in r.text
    assert client.get("/app.js").status_code == 200


def test_upload_receipt_end_to_end(client):
    receipt = build_dataset(seed=0).test[0]
    img = render_receipt(receipt, np.random.default_rng(0), "clean")
    r = client.post("/api/receipts", files={"file": ("r.jpg", _jpeg(img), "image/jpeg")})
    assert r.status_code == 201, r.text
    e = r.json()
    assert e["total"] == pytest.approx(receipt.total)
    assert e["date"] == receipt.date.isoformat()
    assert e["has_image"] and e["ocr_text"]
    assert client.get(f"/api/expenses/{e['id']}/image").headers["content-type"] == "image/jpeg"
    assert len(client.get("/api/expenses").json()) == 1


def test_upload_rejects_non_images(client):
    r = client.post("/api/receipts", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    tiny = np.full((20, 20, 3), 255, np.uint8)
    r = client.post("/api/receipts", files={"file": ("t.jpg", _jpeg(tiny), "image/jpeg")})
    assert r.status_code == 400


def test_manual_expense_predicts_category(client):
    r = client.post("/api/expenses", json={"merchant": "Sharma General Store", "date": "2026-11-02",
                                           "total": 250, "items": ["toor dal"]})
    assert r.status_code == 201
    e = r.json()
    assert e["category"] == "Groceries" and e["source"] == "rule" and not e["needs_review"]


def test_manual_expense_with_category_teaches_model(client):
    client.post("/api/expenses", json={"merchant": "PVR Cinemas", "total": 600, "category": "Entertainment"})
    e = client.post("/api/expenses", json={"merchant": "PVR Cinemas", "total": 400}).json()
    assert e["category"] == "Entertainment" and e["source"] == "rule"


def test_correction_learns_and_retrains(client):
    ids = [client.post("/api/expenses", json={"merchant": f"Zeta Shop {i}", "total": 10}).json()["id"]
           for i in range(3)]
    before = client.get("/api/model").json()
    results = [client.patch(f"/api/expenses/{i}", json={"category": "Electronics"}).json() for i in ids]
    assert all(r["learned"] and r["category"] == "Electronics" and r["source"] == "user" for r in results)
    assert results[-1]["retrained"]  # retrain_every=3 in the fixture
    after = client.get("/api/model").json()
    assert after["known_shops"] == before["known_shops"] + 3
    assert after["retrain_count"] == before["retrain_count"] + 1
    new = client.post("/api/expenses", json={"merchant": "Zeta Shop 1", "total": 5}).json()
    assert new["category"] == "Electronics"


def test_patch_validation_and_404(client):
    e = client.post("/api/expenses", json={"merchant": "Gupta Kirana", "total": 99}).json()
    assert client.patch(f"/api/expenses/{e['id']}", json={"category": "Snacks"}).status_code == 422
    assert client.patch(f"/api/expenses/{e['id']}", json={"date": "02/11/2026"}).status_code == 422
    r = client.patch(f"/api/expenses/{e['id']}", json={"total": 120.5, "items": ["atta"]}).json()
    assert r["total"] == 120.5 and r["items"] == ["atta"] and not r["learned"]
    assert client.patch("/api/expenses/999", json={"total": 1}).status_code == 404
    assert client.get("/api/expenses/999").status_code == 404


def test_delete_removes_row_and_image(client, tmp_path):
    img = render_receipt(build_dataset(seed=0).test[1], np.random.default_rng(1), "clean")
    e = client.post("/api/receipts", files={"file": ("r.jpg", _jpeg(img), "image/jpeg")}).json()
    assert len(list((tmp_path / "app" / "uploads").iterdir())) == 1
    assert client.delete(f"/api/expenses/{e['id']}").status_code == 204
    assert list((tmp_path / "app" / "uploads").iterdir()) == []
    assert client.delete(f"/api/expenses/{e['id']}").status_code == 404


def test_summary_and_filters(client):
    for m, d, t in [("Sharma General Store", "2026-10-05", 100), ("Hotel Annapurna", "2026-10-09", 300),
                    ("Gupta Kirana", "2026-11-01", 50)]:
        client.post("/api/expenses", json={"merchant": m, "date": d, "total": t})
    s = client.get("/api/summary?month=2026-10").json()
    assert s["total"] == 400 and s["count"] == 2
    assert s["by_category"][0] == {"category": "Dining", "count": 1, "total": 300}
    assert [m["month"] for m in s["by_month"]] == ["2026-10", "2026-11"]
    assert "known_shops" in s["model"]
    assert len(client.get("/api/expenses?category=Groceries").json()) == 2
    assert len(client.get("/api/expenses?q=annapurna").json()) == 1
    assert len(client.get("/api/expenses?month=2026-11").json()) == 1
    assert client.get("/api/expenses?month=2026-1").status_code == 422


def test_sample_endpoint(client):
    r = client.get("/api/sample?level=clean&seed=3")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert "total" in r.headers["x-truth"]
    assert client.get("/api/sample?level=awful").status_code == 422


def test_model_persists_between_restarts(tmp_path):
    base = tmp_path / "base.pkl"
    HybridCategorizer(tau=0.4).fit(SMALL_TRAIN).save(base)
    with TestClient(create_app(tmp_path / "app", base)) as c:
        c.post("/api/expenses", json={"merchant": "Croma Electronics", "category": "Electronics"})
    with TestClient(create_app(tmp_path / "app", base)) as c:
        assert len(c.get("/api/expenses").json()) == 1
        e = c.post("/api/expenses", json={"merchant": "Croma Electronics"}).json()
        assert e["category"] == "Electronics" and e["source"] == "rule"


def test_store_in_memory():
    s = ExpenseStore(":memory:")
    e = s.add(merchant="A", items=["x"], total=1.0, needs_review=True)
    assert e["items"] == ["x"] and e["needs_review"] and not e["has_image"]
    assert s.update(e["id"], needs_review=False, bogus=1)["needs_review"] is False
    assert s.list(needs_review=True) == []
    with pytest.raises(KeyError):
        s.delete(42)
    s.close()
