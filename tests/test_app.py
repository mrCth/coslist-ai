"""Tests for the CosList AI service.

Run:  .venv\\Scripts\\python -m pytest -q
Live provider test:  set RUN_LIVE=1  (requires a running Ollama with a vision model)
"""

from __future__ import annotations

import os

import pytest
from starlette.testclient import TestClient

from app.config import settings
from app.main import app

from .conftest import make_image_bytes, valid_draft_json


# --------------------------------------------------------------------------- #
# System endpoints
# --------------------------------------------------------------------------- #
def test_health(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"ok", "degraded"}
    assert "provider" in body and "model" in body


def test_config(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["max_images"] == settings.max_images
    assert "cosplay_costume" in body["categories"]


def test_index_served(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "คอสลิสต์" in res.text


# --------------------------------------------------------------------------- #
# /api/analyze
# --------------------------------------------------------------------------- #
def test_analyze_valid(client: TestClient, mock_provider):
    provider = mock_provider([valid_draft_json()])
    res = client.post(
        "/api/analyze",
        files=[("images", ("a.jpg", make_image_bytes(), "image/jpeg"))],
        data={"notes": "ชุดสีม่วง", "size": "M"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["title_th"]
    assert body["category"] == "cosplay_costume"
    assert 5 <= len(body["tags"]) <= 10
    assert body["meta"]["provider"] == "mock"
    assert body["meta"]["latency_ms"] >= 0
    assert len(provider.calls) == 1


def test_analyze_invalid_file_type(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    res = client.post(
        "/api/analyze",
        files=[("images", ("a.txt", b"not an image", "image/jpeg"))],
    )
    assert res.status_code == 415


def test_analyze_unsupported_content_type(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    res = client.post(
        "/api/analyze",
        files=[("images", ("a.txt", b"hello", "text/plain"))],
    )
    assert res.status_code == 415


def test_analyze_oversized(client: TestClient, mock_provider, monkeypatch):
    mock_provider([valid_draft_json()])
    monkeypatch.setattr(settings, "max_image_bytes", 1000)
    res = client.post(
        "/api/analyze",
        files=[("images", ("big.jpg", make_image_bytes(size=(1200, 1200)), "image/jpeg"))],
    )
    assert res.status_code == 413


def test_analyze_too_many_images(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    files = [
        ("images", (f"p{i}.jpg", make_image_bytes(), "image/jpeg"))
        for i in range(settings.max_images + 1)
    ]
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 422


def test_analyze_no_images(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    res = client.post("/api/analyze", data={"notes": "hello"})
    assert res.status_code == 422


def test_analyze_invalid_json_then_502(client: TestClient, mock_provider):
    provider = mock_provider(["not json at all", "still not json"])
    res = client.post(
        "/api/analyze",
        files=[("images", ("a.jpg", make_image_bytes(), "image/jpeg"))],
    )
    assert res.status_code == 502
    assert len(provider.calls) == 2  # retried once


def test_analyze_repair_succeeds(client: TestClient, mock_provider):
    provider = mock_provider(["oops not json", valid_draft_json()])
    res = client.post(
        "/api/analyze",
        files=[("images", ("a.jpg", make_image_bytes(), "image/jpeg"))],
    )
    assert res.status_code == 200
    assert res.json()["meta"]["retries"] == 1
    assert len(provider.calls) == 2


def test_analyze_missing_tags_triggers_repair(client: TestClient, mock_provider):
    bad = valid_draft_json(tags=["only-one"])
    provider = mock_provider([bad, valid_draft_json()])
    res = client.post(
        "/api/analyze",
        files=[("images", ("a.jpg", make_image_bytes(), "image/jpeg"))],
    )
    assert res.status_code == 200
    assert len(provider.calls) == 2


def test_analyze_rate_limit(client: TestClient, mock_provider):
    provider = mock_provider([valid_draft_json()])
    # shrink the limiter to force a 429
    from app import ai_service

    ai_service._service.rate_limiter.limit = 1
    first = client.post(
        "/api/analyze",
        files=[("images", ("a.jpg", make_image_bytes(), "image/jpeg"))],
    )
    second = client.post(
        "/api/analyze",
        files=[("images", ("a.jpg", make_image_bytes(), "image/jpeg"))],
    )
    assert first.status_code == 200
    assert second.status_code == 429


# --------------------------------------------------------------------------- #
# Listings CRUD
# --------------------------------------------------------------------------- #
def _create_listing(client: TestClient) -> dict:
    payload = {
        "title_th": "ชุดทดสอบ",
        "description_th": "คำอธิบายทดสอบยาวพอสมควรสำหรับการทดสอบ",
        "category": "wig",
        "condition_grade": "A",
        "price_min": 300,
        "price_max": 600,
        "tags": ["wig", "ทดสอบ"],
        "ai_draft": {"meta": {"latency_ms": 1234, "provider": "mock", "model": "mock-vision"}},
    }
    res = client.post(
        "/api/listings",
        data={"payload": __import__("json").dumps(payload, ensure_ascii=False)},
        files=[("images", ("a.jpg", make_image_bytes(), "image/jpeg"))],
    )
    assert res.status_code == 201, res.text
    return res.json()


def test_listing_crud(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    created = _create_listing(client)
    assert created["id"] > 0
    assert created["thumb_url"]
    assert created["image_urls"]
    assert created["latency_ms"] == 1234

    lid = created["id"]
    got = client.get(f"/api/listings/{lid}")
    assert got.status_code == 200
    assert got.json()["title_th"] == "ชุดทดสอบ"

    listed = client.get("/api/listings")
    assert listed.status_code == 200
    assert any(item["id"] == lid for item in listed.json())

    search = client.get("/api/listings", params={"q": "ทดสอบ"})
    assert any(item["id"] == lid for item in search.json())

    updated = client.put(
        f"/api/listings/{lid}",
        json={"title_th": "ชุดทดสอบแก้ไข", "price_min": 350},
    )
    assert updated.status_code == 200
    assert updated.json()["title_th"] == "ชุดทดสอบแก้ไข"
    assert updated.json()["price_min"] == 350

    deleted = client.delete(f"/api/listings/{lid}")
    assert deleted.status_code == 204
    assert client.get(f"/api/listings/{lid}").status_code == 404


def test_listing_404s(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    assert client.get("/api/listings/99999").status_code == 404
    assert client.put("/api/listings/99999", json={"title_th": "x"}).status_code == 404
    assert client.delete("/api/listings/99999").status_code == 404
    assert client.post("/api/listings/99999/feedback", json={"accepted_as_is": True}).status_code == 404


def test_feedback_and_stats(client: TestClient, mock_provider):
    mock_provider([valid_draft_json()])
    a = _create_listing(client)
    b = _create_listing(client)

    fb = client.post(
        f"/api/listings/{a['id']}/feedback",
        json={"accepted_as_is": True, "edited_fields": []},
    )
    assert fb.status_code == 200
    assert fb.json()["accepted_as_is"] is True

    fb2 = client.post(
        f"/api/listings/{b['id']}/feedback",
        json={
            "accepted_as_is": False,
            "edited_fields": ["title_th", "price_min"],
            "final": {"title_th": "แก้ไขแล้ว"},
        },
    )
    assert fb2.status_code == 200
    assert fb2.json()["saved_final"] is True

    stats = client.get("/api/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["total_listings"] == 2
    assert body["drafts_with_feedback"] == 2
    assert body["drafts_accepted_without_edits"] == 1
    assert body["accepted_rate"] == 0.5


# --------------------------------------------------------------------------- #
# Optional live test
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real provider")
def test_live_provider_analyze():
    from app.ai_service import get_ai_service, process_image

    service = get_ai_service()
    img = process_image(make_image_bytes())
    import asyncio

    result = asyncio.run(service.analyze([img], notes="ชุดคอสเพลย์สีม่วง ใช้แล้วหนึ่งครั้ง"))
    assert result.draft.title_th
    assert 5 <= len(result.draft.tags) <= 10
    assert result.draft.meta.provider
