"""FastAPI backend: upload receipts, review categories, browse spending.

    python -m expense_tracker serve            # then open http://127.0.0.1:8000

Everything stays on this machine: images, the SQLite database and the
user's copy of the categorizer all live in the data folder
(default `data/app`, or set EXPENSE_TRACKER_DATA).
"""
from __future__ import annotations

import json
import os
import random
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .categorizer import HybridCategorizer
from .pipeline import InvalidImageError, ReceiptPipeline
from .storage import ExpenseStore
from .synthetic import CATEGORIES

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DEFAULT_MODEL = Path("models/categorizer.pkl")


class ExpenseIn(BaseModel):
    merchant: str = Field(min_length=1, max_length=200)
    date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    total: Optional[float] = Field(default=None, ge=0)
    items: list[str] = []
    category: Optional[str] = None


class ExpensePatch(BaseModel):
    merchant: Optional[str] = Field(default=None, max_length=200)
    date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    total: Optional[float] = Field(default=None, ge=0)
    items: Optional[list[str]] = None
    category: Optional[str] = None


class ModelService:
    """Holds the user's categorizer, applies feedback, and saves it after every change."""

    def __init__(self, user_path: Path, base_path: Path | None):
        self.user_path = user_path
        self.lock = threading.Lock()
        if user_path.is_file():
            self.model = HybridCategorizer.load(user_path)
        elif base_path and Path(base_path).is_file():
            self.model = HybridCategorizer.load(base_path)
        else:  # no trained model yet: train on the synthetic training set
            from .synthetic import build_dataset
            data = [r.to_expense() for r in build_dataset(seed=0).train]
            self.model = HybridCategorizer(tau=0.4).fit(data)
            self.model.save(user_path)

    def predict(self, merchant: str, items: list[str]):
        with self.lock:
            return self.model.predict(merchant, items)

    def feedback(self, merchant: str, items: list[str], category: str) -> bool:
        with self.lock:
            retrained = self.model.feedback(merchant, items, category)
            self.model.save(self.user_path)
        return retrained

    def stats(self) -> dict:
        m = self.model
        return {
            "known_shops": len(m.rules.table),
            "training_examples": len(m.train_data),
            "corrections_until_retrain": max(m.retrain_every - m.pending, 0),
            "retrain_count": m.retrain_count,
            "tau": m.tau,
        }


@lru_cache(maxsize=1)
def _sample_receipts():
    from .synthetic import build_dataset
    return build_dataset(seed=0).test


def create_app(data_dir: str | Path | None = None, model_path: str | Path | None = DEFAULT_MODEL) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("EXPENSE_TRACKER_DATA", "data/app"))
    uploads = data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    store = ExpenseStore(data_dir / "expenses.db")
    models = ModelService(data_dir / "categorizer.pkl", Path(model_path) if model_path else None)
    pipeline = ReceiptPipeline(categorizer=None)

    app = FastAPI(title="ReceiptSense API", version="1.0.0",
                  description="OCR + hybrid categorization of receipts, running locally.\n\n"
                              "**This page is the developer API reference. "
                              "[Open the dashboard →](/)**")
    app.state.store = store
    app.state.models = models

    def _get_or_404(expense_id: int) -> dict:
        e = store.get(expense_id)
        if e is None:
            raise HTTPException(404, "expense not found")
        return e

    def _check_category(cat: str | None) -> None:
        if cat is not None and cat not in CATEGORIES:
            raise HTTPException(422, f"unknown category; use one of: {', '.join(CATEGORIES)}")

    # ------------------------------------------------------------ meta

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/categories")
    def categories():
        return CATEGORIES

    @app.get("/api/model")
    def model_stats():
        return models.stats()

    # ------------------------------------------------------------ receipts

    @app.post("/api/receipts", status_code=201)
    def upload_receipt(file: UploadFile = File(...)):
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image is larger than 15 MB")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is None:
            raise HTTPException(400, "not a readable image (use JPG, PNG, BMP, TIFF or WebP)")
        try:
            result = pipeline.process_image(img)
        except InvalidImageError as e:
            raise HTTPException(400, str(e))

        fields = result.fields
        pred = models.predict(fields.merchant, fields.items) if fields.merchant else None
        name = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(uploads / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        expense = store.add(
            merchant=fields.merchant, date=fields.date, total=fields.total, items=fields.items,
            gstin=fields.gstin,
            category=pred.category if pred else None,
            confidence=round(pred.confidence, 3) if pred else None,
            source=pred.source if pred else None,
            needs_review=pred.needs_review if pred else True,
            image=name, ocr_text=[l.text for l in result.ocr_lines], warnings=result.warnings,
        )
        expense["timings_ms"] = {k: round(v * 1000, 1) for k, v in result.timings.items()}
        return expense

    @app.get("/api/expenses/{expense_id}/image")
    def receipt_image(expense_id: int):
        name = store.image_name(expense_id)
        if not name or not (uploads / name).is_file():
            raise HTTPException(404, "no image for this expense")
        return FileResponse(uploads / name, media_type="image/jpeg")

    @app.get("/api/sample", response_class=Response)
    def sample_receipt(level: str = Query("medium", pattern="^(clean|medium|hard)$"),
                       seed: Optional[int] = None):
        """A synthetic receipt photo to try the app with."""
        from .synthetic import render_receipt
        seed = random.randrange(10_000) if seed is None else seed
        receipts = _sample_receipts()
        receipt = receipts[seed % len(receipts)]
        img = render_receipt(receipt, np.random.default_rng(seed), level)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(buf.tobytes(), media_type="image/jpeg",
                        headers={"X-Truth": json.dumps(receipt.truth(), ensure_ascii=True, default=str)})

    # ------------------------------------------------------------ expenses

    @app.get("/api/expenses")
    def list_expenses(month: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$"),
                      category: Optional[str] = None, needs_review: Optional[bool] = None,
                      q: Optional[str] = None):
        return store.list(month=month, category=category, needs_review=needs_review, q=q)

    @app.post("/api/expenses", status_code=201)
    def add_expense(body: ExpenseIn):
        _check_category(body.category)
        if body.category:
            models.feedback(body.merchant, body.items, body.category)
            cat, conf, src, review = body.category, 1.0, "manual", False
        else:
            p = models.predict(body.merchant, body.items)
            cat, conf, src, review = p.category, round(p.confidence, 3), p.source, p.needs_review
        return store.add(merchant=body.merchant, date=body.date, total=body.total, items=body.items,
                         category=cat, confidence=conf, source=src, needs_review=review)

    @app.get("/api/expenses/{expense_id}")
    def get_expense(expense_id: int):
        return _get_or_404(expense_id)

    @app.patch("/api/expenses/{expense_id}")
    def update_expense(expense_id: int, body: ExpensePatch):
        current = _get_or_404(expense_id)
        changes = body.model_dump(exclude_unset=True)
        _check_category(changes.get("category"))
        learned = retrained = False
        if changes.get("category"):
            merchant = changes.get("merchant") or current["merchant"]
            items = changes["items"] if changes.get("items") is not None else current["items"]
            if merchant:  # the user confirmed or corrected the category: learn from it
                retrained = models.feedback(merchant, items, changes["category"])
                learned = True
            changes.update(source="user", confidence=1.0, needs_review=False)
        expense = store.update(expense_id, **changes)
        return {**expense, "learned": learned, "retrained": retrained}

    @app.delete("/api/expenses/{expense_id}", status_code=204)
    def delete_expense(expense_id: int):
        try:
            image = store.delete(expense_id)
        except KeyError:
            raise HTTPException(404, "expense not found")
        if image:
            (uploads / image).unlink(missing_ok=True)
        return Response(status_code=204)

    @app.get("/api/summary")
    def summary(month: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$")):
        return {**store.summary(month), "model": models.stats()}

    # ------------------------------------------------------------ dashboard

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
