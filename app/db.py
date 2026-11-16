"""SQLite persistence layer for CosList AI.

Deliberately dependency-free (``sqlite3`` from the stdlib). A fresh
connection is opened per operation and WAL mode is enabled, which keeps the
code simple and safe under FastAPI's threadpool.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional

from .config import settings
from .schemas import (
    Category,
    ConditionGrade,
    Confidence,
    Listing,
    ListingCreate,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title_th TEXT NOT NULL,
    description_th TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'other',
    character_guess TEXT,
    character_confidence TEXT,
    condition_grade TEXT NOT NULL DEFAULT 'B',
    price_min INTEGER,
    price_max INTEGER,
    tags TEXT NOT NULL DEFAULT '[]',
    colors TEXT NOT NULL DEFAULT '[]',
    materials TEXT NOT NULL DEFAULT '[]',
    missing_info TEXT NOT NULL DEFAULT '[]',
    warnings TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    size TEXT NOT NULL DEFAULT '',
    brand TEXT NOT NULL DEFAULT '',
    purchase_price_thb INTEGER,
    source TEXT NOT NULL DEFAULT 'ai',
    images TEXT NOT NULL DEFAULT '[]',
    thumb TEXT,
    ai_draft TEXT,
    final TEXT,
    accepted_as_is INTEGER,
    edited_fields TEXT NOT NULL DEFAULT '[]',
    latency_ms INTEGER,
    provider TEXT,
    model TEXT
);
CREATE INDEX IF NOT EXISTS idx_listings_created ON listings(created_at DESC);
"""

_LIST_COLUMNS = {"tags", "colors", "materials", "missing_info", "warnings", "images", "edited_fields"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def _loads(raw: Any, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class Database:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or settings.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- helpers -------------------------------------------------------
    def _row_to_listing(self, row: sqlite3.Row) -> Listing:
        images = _loads(row["images"], [])
        return Listing(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            title_th=row["title_th"],
            description_th=row["description_th"] or "",
            category=Category(row["category"] or "other"),
            character_guess=row["character_guess"],
            character_confidence=(
                Confidence(row["character_confidence"])
                if row["character_confidence"]
                else None
            ),
            condition_grade=ConditionGrade(row["condition_grade"] or "B"),
            price_min=row["price_min"],
            price_max=row["price_max"],
            tags=_loads(row["tags"], []),
            colors=_loads(row["colors"], []),
            materials=_loads(row["materials"], []),
            missing_info=_loads(row["missing_info"], []),
            warnings=_loads(row["warnings"], []),
            notes=row["notes"] or "",
            size=row["size"] or "",
            brand=row["brand"] or "",
            purchase_price_thb=row["purchase_price_thb"],
            source=row["source"] or "ai",
            thumb_url=f"/media/thumbs/{row['thumb']}" if row["thumb"] else None,
            image_urls=[f"/media/images/{name}" for name in images],
            ai_draft=_loads(row["ai_draft"], None),
            accepted_as_is=(
                None if row["accepted_as_is"] is None else bool(row["accepted_as_is"])
            ),
            edited_fields=_loads(row["edited_fields"], []),
            latency_ms=row["latency_ms"],
            provider=row["provider"],
            model=row["model"],
        )

    # -- CRUD ----------------------------------------------------------
    def create_listing(
        self,
        payload: ListingCreate,
        images: Optional[List[str]] = None,
        thumb: Optional[str] = None,
        latency_ms: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Listing:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO listings (
                    created_at, updated_at, title_th, description_th, category,
                    character_guess, character_confidence, condition_grade,
                    price_min, price_max, tags, colors, materials, missing_info,
                    warnings, notes, size, brand, purchase_price_thb, source,
                    images, thumb, ai_draft, accepted_as_is, edited_fields,
                    latency_ms, provider, model
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now,
                    now,
                    payload.title_th,
                    payload.description_th,
                    payload.category.value,
                    payload.character_guess,
                    payload.character_confidence.value
                    if payload.character_confidence
                    else None,
                    payload.condition_grade.value,
                    payload.price_min,
                    payload.price_max,
                    _dumps(payload.tags),
                    _dumps(payload.colors),
                    _dumps(payload.materials),
                    _dumps(payload.missing_info),
                    _dumps(payload.warnings),
                    payload.notes,
                    payload.size,
                    payload.brand,
                    payload.purchase_price_thb,
                    payload.source,
                    _dumps(images or []),
                    thumb,
                    _dumps(payload.ai_draft) if payload.ai_draft else None,
                    None if payload.accepted_as_is is None else int(payload.accepted_as_is),
                    _dumps(payload.edited_fields),
                    latency_ms,
                    provider,
                    model,
                ),
            )
            new_id = int(cur.lastrowid)
        return self.get_listing(new_id)  # type: ignore[return-value]

    def get_listing(self, listing_id: int) -> Optional[Listing]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM listings WHERE id = ?", (listing_id,)
            ).fetchone()
        return self._row_to_listing(row) if row else None

    def list_listings(self, search: Optional[str] = None) -> List[Listing]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM listings ORDER BY created_at DESC"
            ).fetchall()
        items = [self._row_to_listing(r) for r in rows]
        if search:
            needle = search.strip().lower()
            items = [
                it
                for it in items
                if needle in it.title_th.lower()
                or any(needle in t.lower() for t in it.tags)
                or needle in (it.description_th or "").lower()
            ]
        return items

    def update_listing(self, listing_id: int, changes: dict) -> Optional[Listing]:
        if not changes:
            return self.get_listing(listing_id)
        columns = []
        values: List[Any] = []
        for key, value in changes.items():
            if key in _LIST_COLUMNS:
                value = _dumps(value)
            elif hasattr(value, "value"):
                value = value.value
            columns.append(f"{key} = ?")
            values.append(value)
        columns.append("updated_at = ?")
        values.append(_now())
        values.append(listing_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE listings SET {', '.join(columns)} WHERE id = ?", values
            )
        return self.get_listing(listing_id)

    def delete_listing(self, listing_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM listings WHERE id = ?", (listing_id,))
        return cur.rowcount > 0

    def set_feedback(
        self,
        listing_id: int,
        accepted_as_is: bool,
        edited_fields: List[str],
        final: Optional[dict],
    ) -> Optional[Listing]:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE listings
                SET accepted_as_is = ?, edited_fields = ?, final = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    int(accepted_as_is),
                    _dumps(edited_fields),
                    _dumps(final) if final is not None else None,
                    _now(),
                    listing_id,
                ),
            )
        return self.get_listing(listing_id)

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM listings").fetchone()["c"]
            avg_latency = conn.execute(
                "SELECT AVG(latency_ms) AS a FROM listings WHERE latency_ms IS NOT NULL"
            ).fetchone()["a"]
            with_feedback = conn.execute(
                "SELECT COUNT(*) AS c FROM listings WHERE accepted_as_is IS NOT NULL"
            ).fetchone()["c"]
            accepted = conn.execute(
                "SELECT COUNT(*) AS c FROM listings WHERE accepted_as_is = 1"
            ).fetchone()["c"]
            grade_rows = conn.execute(
                "SELECT condition_grade AS g, COUNT(*) AS c FROM listings GROUP BY condition_grade"
            ).fetchall()
            edit_rows = conn.execute(
                "SELECT edited_fields FROM listings WHERE accepted_as_is IS NOT NULL"
            ).fetchall()

        edit_counts = [_loads(r["edited_fields"], []) for r in edit_rows]
        avg_edit_rate = None
        if edit_counts:
            avg_edit_rate = sum(len(e) for e in edit_counts) / len(edit_counts)

        return {
            "total_listings": total,
            "avg_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
            "drafts_with_feedback": with_feedback,
            "drafts_accepted_without_edits": accepted,
            "accepted_rate": round(accepted / with_feedback, 4) if with_feedback else None,
            "avg_edit_rate": round(avg_edit_rate, 2) if avg_edit_rate is not None else None,
            "by_grade": {r["g"]: r["c"] for r in grade_rows},
        }


_db: Optional[Database] = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
