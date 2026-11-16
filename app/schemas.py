"""Pydantic v2 schemas for CosList AI.

Two groups of models live here:

* The **AI contract** (``ListingDraft`` and friends) - what the generative
  service must return for ``POST /api/analyze``. Every field is validated so a
  hallucinated / malformed model response fails loudly and can be repaired.
* The **API contract** (``ListingCreate``/``ListingUpdate``/``Listing``/
  ``FeedbackRequest``/``Stats``) used by the CRUD + feedback endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Category(str, Enum):
    cosplay_costume = "cosplay_costume"
    wig = "wig"
    accessory = "accessory"
    prop = "prop"
    shoes = "shoes"
    top = "top"
    bottom = "bottom"
    dress = "dress"
    outerwear = "outerwear"
    other = "other"


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ConditionGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


# --------------------------------------------------------------------------- #
# AI contract
# --------------------------------------------------------------------------- #
def _clean_str_list(value: Any, *, max_items: int = 20) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out[:max_items]


class CharacterGuess(BaseModel):
    """A *guess* about the character/series - never stated as fact."""

    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = Field(
        default=None,
        description="Guessed character or series name, or null when unknown.",
    )
    confidence: Confidence = Confidence.low
    disclaimer_th: str = Field(
        default="เป็นการเดาจากภาพเท่านั้น อาจไม่ถูกต้อง",
        description="Thai disclaimer reminding the seller this is only a guess.",
    )

    @field_validator("name")
    @classmethod
    def _blank_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @model_validator(mode="after")
    def _force_disclaimer(self) -> "CharacterGuess":
        if self.name is not None and "เดา" not in self.disclaimer_th:
            self.disclaimer_th = (
                self.disclaimer_th.strip() + " (เป็นการเดาจากภาพเท่านั้น)"
            ).strip()
        return self


class PriceEstimate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min: int = Field(ge=0, le=1_000_000)
    max: int = Field(ge=0, le=1_000_000)
    currency: str = "THB"
    reasoning: str = Field(default="")
    disclaimer_th: str = Field(
        default="ราคานี้เป็นเพียงการประมาณการจากภาพ ไม่ใช่ราคาตลาดจริง",
    )

    @model_validator(mode="after")
    def _order(self) -> "PriceEstimate":
        if self.max < self.min:
            self.min, self.max = self.max, self.min
        return self


class ListingMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    prompt_version: str = ""
    retries: int = 0
    mode: str = "vision"


class ListingDraft(BaseModel):
    """The validated result returned by ``POST /api/analyze``."""

    model_config = ConfigDict(extra="ignore")

    title_th: str = Field(default="")
    description_th: str = Field(default="")
    category: Category = Category.other
    character_or_series_guess: Optional[CharacterGuess] = None
    colors: List[str] = Field(default_factory=list)
    materials_guess: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    condition_grade: ConditionGrade = ConditionGrade.B
    condition_reasons: List[str] = Field(default_factory=list)
    suggested_price_thb: PriceEstimate = Field(
        default_factory=lambda: PriceEstimate(min=0, max=0, reasoning="")
    )
    missing_info: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    meta: ListingMeta = Field(default_factory=ListingMeta)

    # -- normalisation -------------------------------------------------
    @field_validator("title_th")
    @classmethod
    def _title(cls, v: str) -> str:
        v = (v or "").strip()
        return v[:80]

    @field_validator("description_th")
    @classmethod
    def _desc(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("colors", "materials_guess", "condition_reasons")
    @classmethod
    def _list(cls, v: Any) -> List[str]:
        return _clean_str_list(v, max_items=12)

    @field_validator("missing_info", "warnings")
    @classmethod
    def _checklist(cls, v: Any) -> List[str]:
        return _clean_str_list(v, max_items=12)

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: Any) -> List[str]:
        tags = _clean_str_list(v, max_items=10)
        if not (5 <= len(tags) <= 10):
            raise ValueError(
                f"tags must contain between 5 and 10 items, got {len(tags)}"
            )
        return tags

    @model_validator(mode="after")
    def _require_title_and_desc(self) -> "ListingDraft":
        if not self.title_th:
            raise ValueError("title_th must not be empty")
        if len(self.description_th) < 20:
            raise ValueError("description_th is too short (min 20 chars)")
        return self


# --------------------------------------------------------------------------- #
# API contract
# --------------------------------------------------------------------------- #
class ListingCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title_th: str = Field(min_length=1, max_length=200)
    description_th: str = ""
    category: Category = Category.other
    character_guess: Optional[str] = None
    character_confidence: Optional[Confidence] = None
    condition_grade: ConditionGrade = ConditionGrade.B
    price_min: Optional[int] = Field(default=None, ge=0)
    price_max: Optional[int] = Field(default=None, ge=0)
    tags: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    notes: str = ""
    size: str = ""
    brand: str = ""
    purchase_price_thb: Optional[int] = Field(default=None, ge=0)
    source: str = "ai"
    ai_draft: Optional[dict] = None
    accepted_as_is: Optional[bool] = None
    edited_fields: List[str] = Field(default_factory=list)

    @field_validator("tags", "colors", "materials", "missing_info", "warnings")
    @classmethod
    def _lists(cls, v: Any) -> List[str]:
        return _clean_str_list(v, max_items=30)


class ListingUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title_th: Optional[str] = Field(default=None, max_length=200)
    description_th: Optional[str] = None
    category: Optional[Category] = None
    character_guess: Optional[str] = None
    character_confidence: Optional[Confidence] = None
    condition_grade: Optional[ConditionGrade] = None
    price_min: Optional[int] = Field(default=None, ge=0)
    price_max: Optional[int] = Field(default=None, ge=0)
    tags: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    materials: Optional[List[str]] = None
    missing_info: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    notes: Optional[str] = None
    size: Optional[str] = None
    brand: Optional[str] = None
    purchase_price_thb: Optional[int] = Field(default=None, ge=0)
    accepted_as_is: Optional[bool] = None
    edited_fields: Optional[List[str]] = None

    @field_validator("tags", "colors", "materials", "missing_info", "warnings")
    @classmethod
    def _lists(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        return _clean_str_list(v, max_items=30)


class Listing(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    created_at: str
    updated_at: str
    title_th: str
    description_th: str = ""
    category: Category = Category.other
    character_guess: Optional[str] = None
    character_confidence: Optional[Confidence] = None
    condition_grade: ConditionGrade = ConditionGrade.B
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    notes: str = ""
    size: str = ""
    brand: str = ""
    purchase_price_thb: Optional[int] = None
    source: str = "ai"
    thumb_url: Optional[str] = None
    image_urls: List[str] = Field(default_factory=list)
    ai_draft: Optional[dict] = None
    accepted_as_is: Optional[bool] = None
    edited_fields: List[str] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accepted_as_is: bool
    edited_fields: List[str] = Field(default_factory=list)
    final: Optional[dict] = Field(
        default=None,
        description="Optional final version of the listing after the seller edited it.",
    )
    note: str = ""

    @field_validator("edited_fields")
    @classmethod
    def _fields(cls, v: Any) -> List[str]:
        return _clean_str_list(v, max_items=50)


class FeedbackResponse(BaseModel):
    listing_id: int
    accepted_as_is: bool
    edited_fields: List[str]
    saved_final: bool


class Stats(BaseModel):
    total_listings: int
    avg_latency_ms: Optional[float] = None
    drafts_accepted_without_edits: int
    drafts_with_feedback: int
    accepted_rate: Optional[float] = None
    avg_edit_rate: Optional[float] = None
    by_grade: dict[str, int] = Field(default_factory=dict)
    provider: str
    model: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    ai_ready: bool
    ai_detail: str
    mode: str
    version: str


class ConfigResponse(BaseModel):
    app_name: str
    app_version: str
    provider: str
    model: str
    mode: str
    prompt_version: str
    max_images: int
    max_image_bytes: int
    max_long_side: int
    rate_limit_per_min: int
    categories: List[str]
    condition_grades: List[str]


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
