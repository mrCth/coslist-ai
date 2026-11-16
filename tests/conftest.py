"""Shared pytest fixtures for the CosList AI test-suite."""

from __future__ import annotations

import io
import json
from typing import List

import pytest
from PIL import Image
from starlette.testclient import TestClient

import app.db as db_module
from app import ai_service
from app.ai_service import AIService, BaseProvider, ProviderInfo
from app.db import Database
from app.main import app


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_image_bytes(color=(120, 60, 200), size=(640, 480), fmt="JPEG") -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def valid_draft_json(**overrides) -> str:
    draft = {
        "title_th": "ชุดคอสเพลย์มือสอง สภาพดี",
        "description_th": "ชุดคอสเพลย์สีม่วง สภาพดีมาก ใส่เพียงครั้งเดียว เหมาะกับงานอีเวนต์ ราคาคุ้มค่า",
        "category": "cosplay_costume",
        "character_or_series_guess": {
            "name": "ตัวละครทดสอบ",
            "confidence": "low",
            "disclaimer_th": "เป็นการเดาจากภาพเท่านั้น",
        },
        "colors": ["ม่วง", "ดำ"],
        "materials_guess": ["ผ้าโพลีเอสเตอร์"],
        "tags": ["cosplay", "ชุดคอสเพลย์", "มือสอง", "ม่วง", "purple"],
        "condition_grade": "B",
        "condition_reasons": ["มีรอยใช้งานเล็กน้อยที่ชายกระโปรง"],
        "suggested_price_thb": {
            "min": 500,
            "max": 900,
            "reasoning": "สภาพดีแต่เป็นมือสอง",
            "disclaimer_th": "ราคาเป็นเพียงการประมาณการ",
        },
        "missing_info": ["ไซซ์", "รอบอก/รอบเอว"],
        "warnings": ["ภาพอาจไม่ชัดเจน"],
    }
    draft.update(overrides)
    return json.dumps(draft, ensure_ascii=False)


class MockProvider(BaseProvider):
    """Deterministic provider used by tests."""

    name = "mock"
    mode = "vision"

    def __init__(self, responses: List[str]) -> None:
        super().__init__("mock-vision")
        self.responses = responses
        self.calls: List[str] = []

    async def generate(self, system_prompt, user_prompt, images) -> str:
        self.calls.append(user_prompt)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]

    async def probe(self) -> ProviderInfo:
        return ProviderInfo(self.name, self.model, self.mode, True, "mock ready")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def database(tmp_path) -> Database:
    db = Database(tmp_path / "test.db")
    db_module._db = db
    yield db
    db_module._db = None


@pytest.fixture
def client(database) -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_provider():
    def _make(responses: List[str]) -> MockProvider:
        provider = MockProvider(responses)
        service = AIService(provider=provider)
        service.rate_limiter.limit = 10_000
        ai_service._service = service
        return provider

    yield _make
    ai_service._service = None
