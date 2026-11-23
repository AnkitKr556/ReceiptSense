"""SQLite storage for saved expenses (Repository pattern).

The API talks to `ExpenseStore` only, so the database could be swapped for
another one without touching the endpoints.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant     TEXT,
    date         TEXT,              -- yyyy-mm-dd
    total        REAL,
    items        TEXT NOT NULL,     -- JSON list of item names
    gstin        TEXT,
    category     TEXT,
    confidence   REAL,
    source       TEXT,              -- rule / ml / user / manual
    needs_review INTEGER NOT NULL DEFAULT 0,
    image        TEXT,              -- file name inside the uploads folder
    ocr_text     TEXT,              -- JSON list of OCR lines
    warnings     TEXT,              -- JSON list
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);
"""

EDITABLE = ("merchant", "date", "total", "items", "gstin", "category", "confidence",
            "source", "needs_review")
_JSON_COLS = ("items", "ocr_text", "warnings")


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for col in _JSON_COLS:
        d[col] = json.loads(d[col]) if d.get(col) else []
    d["needs_review"] = bool(d["needs_review"])
    d["has_image"] = bool(d.get("image"))
    d.pop("image", None)
    return d


class ExpenseStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------ writes

    def add(self, **fields) -> dict:
        record = {
            "merchant": fields.get("merchant"),
            "date": fields.get("date"),
            "total": fields.get("total"),
            "items": json.dumps(fields.get("items") or []),
            "gstin": fields.get("gstin"),
            "category": fields.get("category"),
            "confidence": fields.get("confidence"),
            "source": fields.get("source"),
            "needs_review": int(bool(fields.get("needs_review"))),
            "image": fields.get("image"),
            "ocr_text": json.dumps(fields.get("ocr_text") or []),
            "warnings": json.dumps(fields.get("warnings") or []),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        cols = ", ".join(record)
        marks = ", ".join("?" for _ in record)
        with self._lock:
            cur = self._conn.execute(f"INSERT INTO expenses ({cols}) VALUES ({marks})",
                                     list(record.values()))
            self._conn.commit()
            new_id = cur.lastrowid
        return self.get(new_id)

    def update(self, expense_id: int, **changes) -> dict | None:
        changes = {k: v for k, v in changes.items() if k in EDITABLE}
        if "items" in changes:
            changes["items"] = json.dumps(changes["items"] or [])
        if "needs_review" in changes:
            changes["needs_review"] = int(bool(changes["needs_review"]))
        if changes:
            sets = ", ".join(f"{k} = ?" for k in changes)
            with self._lock:
                self._conn.execute(f"UPDATE expenses SET {sets} WHERE id = ?",
                                   [*changes.values(), expense_id])
                self._conn.commit()
        return self.get(expense_id)

    def delete(self, expense_id: int) -> str | None:
        """Deletes the row. Returns the image file name (if any) so the caller can remove it."""
        image = self.image_name(expense_id)
        with self._lock:
            cur = self._conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
            self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(expense_id)
        return image

    # ------------------------------------------------------------ reads

    def get(self, expense_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM expenses WHERE id = ?", (expense_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def image_name(self, expense_id: int) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT image FROM expenses WHERE id = ?", (expense_id,)).fetchone()
        return row["image"] if row else None

    def list(self, month: str | None = None, category: str | None = None,
             needs_review: bool | None = None, q: str | None = None) -> list[dict]:
        where, args = [], []
        if month:
            where.append("date LIKE ?")
            args.append(f"{month}%")
        if category:
            where.append("category = ?")
            args.append(category)
        if needs_review is not None:
            where.append("needs_review = ?")
            args.append(int(needs_review))
        if q:
            where.append("(merchant LIKE ? OR items LIKE ?)")
            args += [f"%{q}%", f"%{q}%"]
        sql = "SELECT * FROM expenses"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY COALESCE(date, substr(created_at, 1, 10)) DESC, id DESC"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_dict(r) for r in rows]

    def summary(self, month: str | None = None) -> dict:
        where, args = "", []
        if month:
            where, args = "WHERE date LIKE ?", [f"{month}%"]
        with self._lock:
            by_cat = self._conn.execute(
                f"SELECT COALESCE(category, 'Uncategorized') AS category, COUNT(*) AS count, "
                f"COALESCE(SUM(total), 0) AS total FROM expenses {where} "
                f"GROUP BY 1 ORDER BY total DESC", args).fetchall()
            by_month = self._conn.execute(
                "SELECT substr(date, 1, 7) AS month, COUNT(*) AS count, COALESCE(SUM(total), 0) AS total "
                "FROM expenses WHERE date IS NOT NULL GROUP BY 1 ORDER BY 1 DESC LIMIT 12").fetchall()
            review = self._conn.execute(
                "SELECT COUNT(*) FROM expenses WHERE needs_review = 1").fetchone()[0]
        cats = [dict(r) for r in by_cat]
        return {
            "month": month,
            "total": round(sum(c["total"] for c in cats), 2),
            "count": sum(c["count"] for c in cats),
            "by_category": cats,
            "by_month": [dict(r) for r in reversed(by_month)],
            "needs_review": review,
        }

    def close(self) -> None:
        self._conn.close()
