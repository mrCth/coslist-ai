"""CosList AI - FastAPI application.

Exposes the generative AI *service* (``/api/analyze``) plus a small CRUD API
for saved listings, feedback capture and stats, and serves the single-file
Thai web UI from ``/``.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .ai_service import (
    AIAnalysisError,
    ImageValidationError,
    get_ai_service,
    make_thumbnail,
    process_image,
)
from .config import settings
from .db import Database, get_db
from .schemas import (
    Category,
    ConditionGrade,
    ConfigResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    Listing,
    ListingCreate,
    ListingDraft,
    ListingUpdate,
    Stats,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("coslist.api")

TAGS_METADATA = [
    {"name": "analyze", "description": "ส่งรูปสินค้าให้ AI วิเคราะห์และร่างประกาศขาย"},
    {"name": "listings", "description": "บันทึก/อ่าน/แก้ไข/ลบประกาศที่ผู้ขายบันทึกไว้"},
    {"name": "feedback", "description": "บันทึกว่าผู้ขายใช้ร่าง AI ตามเดิมหรือแก้ไข (คำนวณ edit rate)"},
    {"name": "system", "description": "สถานะระบบ, การตั้งค่า, สถิติ"},
]

app = FastAPI(
    title="CosList AI Service",
    version=settings.app_version,
    description=(
        "**CosList AI** เป็น Generative AI Service สำหรับช่วยผู้ขายชุดคอสเพลย์และ"
        "เสื้อผ้ามือสองในประเทศไทย ร่างประกาศขายจากรูปภาพ\n\n"
        "* `POST /api/analyze` รับรูป 1-4 รูป + ข้อมูลเพิ่มเติม แล้วคืนร่างประกาศ JSON\n"
        "* ผลลัพธ์ทุกฟิลด์ผ่านการตรวจสอบด้วย Pydantic\n"
        "* ราคาเป็นเพียงการประมาณการ ไม่ใช่ราคาตลาดจริง และไม่รับรองความแท้ของสินค้า"
    ),
    openapi_tags=TAGS_METADATA,
    contact={"name": "DIT323 AI as a Service Project"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.image_dir.exists():
    app.mount("/media/images", StaticFiles(directory=str(settings.image_dir)), name="images")
if settings.thumb_dir.exists():
    app.mount("/media/thumbs", StaticFiles(directory=str(settings.thumb_dir)), name="thumbs")
if settings.static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")


# --------------------------------------------------------------------------- #
# Exception handlers
# --------------------------------------------------------------------------- #
@app.exception_handler(ImageValidationError)
async def _image_error_handler(request: Request, exc: ImageValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message_th, "code": exc.code},
    )


@app.exception_handler(AIAnalysisError)
async def _ai_error_handler(request: Request, exc: AIAnalysisError) -> JSONResponse:
    logger.error("AI analysis error code=%s: %s", exc.code, exc.message_th)
    return JSONResponse(
        status_code=502,
        content={"detail": exc.message_th, "code": exc.code},
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _read_uploads(
    files: List[UploadFile], *, enforce_count: bool = True
) -> List:
    files = [f for f in (files or []) if f is not None]
    if enforce_count:
        if len(files) > settings.max_images:
            raise HTTPException(
                status_code=422,
                detail=f"อัปโหลดได้สูงสุด {settings.max_images} รูป (ได้รับ {len(files)} รูป)",
            )
    processed = []
    for f in files:
        if f.content_type and f.content_type not in {
            "image/jpeg",
            "image/png",
            "image/webp",
            "application/octet-stream",
        }:
            raise HTTPException(
                status_code=415,
                detail="รองรับเฉพาะไฟล์รูปภาพ JPG, PNG หรือ WEBP เท่านั้น",
            )
        raw = await f.read()
        processed.append(process_image(raw))
    return processed


def _save_images(processed) -> tuple[List[str], Optional[str]]:
    names: List[str] = []
    for img in processed:
        name = f"{uuid.uuid4().hex}.jpg"
        (settings.image_dir / name).write_bytes(img.data)
        names.append(name)
    thumb: Optional[str] = None
    if processed:
        thumb = f"{uuid.uuid4().hex}.jpg"
        (settings.thumb_dir / thumb).write_bytes(make_thumbnail(processed[0].data))
    return names, thumb


# --------------------------------------------------------------------------- #
# System endpoints
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="สถานะระบบและโมเดล AI")
async def health() -> HealthResponse:
    service = get_ai_service()
    info = await service.provider_info(refresh=True)
    return HealthResponse(
        status="ok" if info.ready else "degraded",
        provider=info.name,
        model=info.model,
        ai_ready=info.ready,
        ai_detail=info.detail,
        mode=info.mode,
        version=settings.app_version,
    )


@app.get("/api/config", response_model=ConfigResponse, tags=["system"], summary="การตั้งค่าและข้อจำกัด")
async def api_config() -> ConfigResponse:
    service = get_ai_service()
    info = await service.provider_info()
    return ConfigResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        provider=info.name,
        model=info.model,
        mode=info.mode,
        prompt_version=settings.prompt_version,
        max_images=settings.max_images,
        max_image_bytes=settings.max_image_bytes,
        max_long_side=settings.max_long_side,
        rate_limit_per_min=settings.rate_limit_per_min,
        categories=[c.value for c in Category],
        condition_grades=[g.value for g in ConditionGrade],
    )


@app.get("/api/stats", response_model=Stats, tags=["system"], summary="สถิติการใช้งาน")
async def api_stats(db: Database = Depends(get_db)) -> Stats:
    service = get_ai_service()
    info = await service.provider_info()
    data = db.stats()
    return Stats(provider=info.name, model=info.model, **data)


# --------------------------------------------------------------------------- #
# AI service endpoint
# --------------------------------------------------------------------------- #
@app.post(
    "/api/analyze",
    response_model=ListingDraft,
    tags=["analyze"],
    summary="วิเคราะห์รูปและร่างประกาศขายด้วย AI",
    responses={
        413: {"model": ErrorResponse, "description": "รูปใหญ่เกินไป"},
        415: {"model": ErrorResponse, "description": "ชนิดไฟล์ไม่รองรับ"},
        422: {"model": ErrorResponse, "description": "จำนวนรูปไม่ถูกต้อง"},
        429: {"model": ErrorResponse, "description": "เรียกใช้บ่อยเกินไป"},
        502: {"model": ErrorResponse, "description": "AI ไม่สามารถสร้างผลลัพธ์ที่ถูกต้องได้"},
    },
)
async def analyze(
    request: Request,
    images: List[UploadFile] = File(default=[], description="รูปสินค้า 1-4 รูป (jpg/png/webp, สูงสุด 5 MB/รูป)"),
    notes: str = Form("", description="บันทึกเพิ่มเติมจากผู้ขาย"),
    size: str = Form("", description="ไซซ์ เช่น M, L, ฟรีไซซ์"),
    brand: str = Form("", description="แบรนด์ (ถ้าทราบ)"),
    purchase_price_thb: Optional[int] = Form(None, description="ราคาที่ซื้อมา (บาท)"),
) -> ListingDraft:
    # request-size guard
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > settings.max_request_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"ขนาดคำขอใหญ่เกินไป (สูงสุด {settings.max_request_bytes // (1024 * 1024)} MB)",
            )

    service = get_ai_service()
    info = await service.provider_info()

    if not service.rate_limiter.check(client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail="เรียกใช้บ่อยเกินไป กรุณารอสักครู่แล้วลองใหม่",
        )

    processed = await _read_uploads(images)

    if not processed:
        if info.mode == "text" and notes.strip():
            pass
        elif info.mode == "text":
            raise HTTPException(
                status_code=422,
                detail="โหมดข้อความ (ไม่มีโมเดลวิเคราะห์ภาพ) ต้องกรอกคำอธิบายสินค้า",
            )
        else:
            raise HTTPException(
                status_code=422, detail="กรุณาอัปโหลดรูปสินค้าอย่างน้อย 1 รูป"
            )

    result = await service.analyze(
        images=processed,
        notes=notes,
        size=size,
        brand=brand,
        purchase_price=purchase_price_thb,
    )
    return result.draft


# --------------------------------------------------------------------------- #
# Listings CRUD
# --------------------------------------------------------------------------- #
@app.post("/api/listings", response_model=Listing, status_code=201, tags=["listings"], summary="บันทึกประกาศ")
async def create_listing(
    payload: str = Form(..., description="JSON ของ ListingCreate"),
    images: List[UploadFile] = File(default=[]),
    db: Database = Depends(get_db),
) -> Listing:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"payload ไม่ใช่ JSON: {exc}")
    try:
        create = ListingCreate.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    processed = await _read_uploads(images, enforce_count=False)
    names, thumb = _save_images(processed)
    latency = None
    provider = None
    model = None
    if isinstance(create.ai_draft, dict):
        meta = create.ai_draft.get("meta") or {}
        latency = meta.get("latency_ms")
        provider = meta.get("provider")
        model = meta.get("model")

    listing = db.create_listing(
        create,
        images=names,
        thumb=thumb,
        latency_ms=latency,
        provider=provider,
        model=model,
    )
    return listing  # type: ignore[return-value]


@app.get("/api/listings", response_model=List[Listing], tags=["listings"], summary="รายการประกาศทั้งหมด")
async def list_listings(
    q: Optional[str] = Query(None, description="ค้นหาจากหัวข้อ/แท็ก/คำอธิบาย"),
    db: Database = Depends(get_db),
) -> List[Listing]:
    return db.list_listings(q)


@app.get("/api/listings/{listing_id}", response_model=Listing, tags=["listings"], summary="อ่านประกาศเดียว")
async def get_listing(listing_id: int, db: Database = Depends(get_db)) -> Listing:
    listing = db.get_listing(listing_id)
    if listing is None:
        raise HTTPException(status_code=404, detail="ไม่พบประกาศนี้")
    return listing


@app.put("/api/listings/{listing_id}", response_model=Listing, tags=["listings"], summary="แก้ไขประกาศ")
async def update_listing(
    listing_id: int,
    update: ListingUpdate,
    db: Database = Depends(get_db),
) -> Listing:
    if db.get_listing(listing_id) is None:
        raise HTTPException(status_code=404, detail="ไม่พบประกาศนี้")
    changes = update.model_dump(exclude_unset=True, exclude_none=True)
    listing = db.update_listing(listing_id, changes)
    return listing  # type: ignore[return-value]


@app.delete(
    "/api/listings/{listing_id}",
    status_code=204,
    response_model=None,
    tags=["listings"],
    summary="ลบประกาศ",
)
async def delete_listing(listing_id: int, db: Database = Depends(get_db)) -> None:
    if not db.delete_listing(listing_id):
        raise HTTPException(status_code=404, detail="ไม่พบประกาศนี้")


@app.post(
    "/api/listings/{listing_id}/feedback",
    response_model=FeedbackResponse,
    tags=["feedback"],
    summary="บันทึกผลตอบรับ (ใช้ร่างเดิมหรือแก้ไข)",
)
async def listing_feedback(
    listing_id: int,
    feedback: FeedbackRequest,
    db: Database = Depends(get_db),
) -> FeedbackResponse:
    listing = db.get_listing(listing_id)
    if listing is None:
        raise HTTPException(status_code=404, detail="ไม่พบประกาศนี้")
    db.set_feedback(
        listing_id,
        feedback.accepted_as_is,
        feedback.edited_fields,
        feedback.final,
    )
    return FeedbackResponse(
        listing_id=listing_id,
        accepted_as_is=feedback.accepted_as_is,
        edited_fields=feedback.edited_fields,
        saved_final=feedback.final is not None,
    )
