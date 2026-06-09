import json
import logging
import os
import re
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from retrieve import retrieve

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
TOP_K = int(os.getenv("TOP_K", "5"))
CV_SYSTEM_URL = os.getenv("CV_SYSTEM_URL", "http://localhost:8001/inspection/inspect")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ProductionContext(BaseModel):
    """Historical and production data from PostgreSQL — enriches LLM decision making."""
    # Factory & station
    factory_id: str | None = None
    factory_name: str | None = None
    production_line_id: str | None = None
    station_id: str | None = None
    shift: str | None = None

    # Product
    product_model: str | None = None
    safety_criticality: str | None = None
    functional_zone: bool | None = None
    board_zone: str | None = None

    # Defect history
    same_defect_count_30d: int | None = None
    same_defect_count_90d: int | None = None
    trend: str | None = None
    most_common_shift: str | None = None
    recurring_location: str | None = None
    affected_lines: str | None = None
    affected_factories: str | None = None

    # Repair history
    common_repair_action: str | None = None
    recurrence_after_repair: bool | None = None
    last_repair_result: str | None = None
    previous_dispositions: str | None = None

    # Maintenance
    maintenance_note: str | None = None
    cleaning_status: str | None = None

    # Supplier
    supplier_id: str | None = None
    supplier_quality_status: str | None = None
    supplier_lot: str | None = None

    # Batch
    batch_id: str | None = None
    quarantine_status: str | None = None

    # Calibration
    calibration_event: str | None = None
    detection_rate_change: str | None = None


class DefectInput(BaseModel):
    defect_type: str = Field(..., examples=["short"])
    location: str = Field(..., examples=["top-left"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.96])
    severity: str = Field(..., examples=["high"])
    production_context: ProductionContext | None = None


class CVDetection(BaseModel):
    detection_id: str | None = None
    defect_code: str | None = None
    defect_name: str | None = None
    defect_group: str | None = None
    confidence: float | None = None
    severity_hint: str | None = None
    location_description: str | None = None
    possible_impact: str | None = None
    default_action_hint: str | None = None


class CVVisionResult(BaseModel):
    defect_detected: bool = False
    primary_detection: CVDetection | None = None
    detections: list[CVDetection] = []


class CVPayload(BaseModel):
    vision_result: CVVisionResult
    production_context: ProductionContext | None = None
    model_config = {"extra": "allow"}


class DefectOutput(BaseModel):
    defect_explanation: str
    severity_assessment: str
    recommended_action: str   # pass | rework | reject | escalate | clean_station | quarantine_lot | manual_review
    justification: str
    sop_references: list[str]
    confidence: float
    inspection_timestamp: str


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(defect: DefectInput, chunks: list[dict]) -> str:
    sop_context = "\n\n".join(
        f"[{c['metadata'].get('source_file', 'doc')}]\n{c['content'][:400]}"
        for c in chunks[:2]
    )

    # Build production context block only if data is present
    ctx = defect.production_context
    prod_lines = []
    if ctx:
        if ctx.factory_name:
            prod_lines.append(f"Factory: {ctx.factory_name} | Line: {ctx.production_line_id} | Station: {ctx.station_id}")
        if ctx.product_model:
            prod_lines.append(f"Product: {ctx.product_model} | Safety criticality: {ctx.safety_criticality}")
        if ctx.same_defect_count_30d is not None:
            prod_lines.append(f"Same defect last 30 days: {ctx.same_defect_count_30d} | 90 days: {ctx.same_defect_count_90d} | Trend: {ctx.trend}")
        if ctx.most_common_shift:
            prod_lines.append(f"Most common shift: {ctx.most_common_shift} | Recurring location: {ctx.recurring_location}")
        if ctx.recurrence_after_repair is not None:
            prod_lines.append(f"Previous repair: {ctx.common_repair_action} | Recurred after repair: {ctx.recurrence_after_repair}")
        if ctx.maintenance_note:
            prod_lines.append(f"Maintenance note: {ctx.maintenance_note} | Cleaning status: {ctx.cleaning_status}")
        if ctx.supplier_quality_status:
            prod_lines.append(f"Supplier: {ctx.supplier_id} | Status: {ctx.supplier_quality_status} | Lot: {ctx.supplier_lot}")
        if ctx.affected_lines:
            prod_lines.append(f"Affected lines: {ctx.affected_lines} | Affected factories: {ctx.affected_factories}")
        if ctx.functional_zone is not None:
            prod_lines.append(f"Board zone: {ctx.board_zone} | Functional zone: {ctx.functional_zone}")
        if ctx.calibration_event:
            prod_lines.append(f"Calibration event: {ctx.calibration_event} | Detection rate change: {ctx.detection_rate_change}")
        if ctx.previous_dispositions:
            prod_lines.append(f"Previous dispositions: {ctx.previous_dispositions}")

    prod_block = "\n".join(prod_lines) if prod_lines else "No historical context available."

    return f"""/no_think
PCB defect detected: {defect.defect_type.replace("_", " ")}, location: {defect.location}, confidence: {defect.confidence:.0%}, severity: {defect.severity}.

PRODUCTION HISTORY:
{prod_block}

KNOWLEDGE BASE:
{sop_context}

Based on the defect, production history, and knowledge base, reply with only this JSON:
{{"defect_explanation": "...", "severity_assessment": "low|medium|high|critical", "recommended_action": "pass|rework|reject|escalate|clean_station|quarantine_lot|manual_review", "justification": "...", "sop_references": ["..."], "confidence": {defect.confidence}}}"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sovra RAG API",
    description="RAG-powered PCB defect analysis. Accepts CV output + production context, returns structured disposition.",
    version="2.0.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health_check() -> dict:
    from pathlib import Path
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "connected" if resp.status_code == 200 else "error"
    except Exception:
        ollama_status = "disconnected"

    index_path = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index"))
    return {
        "status": "ok",
        "ollama": ollama_status,
        "faiss_index": "ready" if (index_path / "index.faiss").exists() else "not_built",
    }


@app.post("/explain-defect", response_model=DefectOutput)
async def explain_defect(defect: DefectInput) -> DefectOutput:
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(f"defect={defect.defect_type!r} confidence={defect.confidence:.2f} severity={defect.severity!r}")

    try:
        chunks = retrieve(defect.defect_type, top_k=TOP_K)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not chunks:
        raise HTTPException(status_code=404, detail=f"No KB entries for: {defect.defect_type}")

    logger.info(f"Retrieved {len(chunks)} chunks")

    prompt = build_prompt(defect, chunks)

    ollama_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512, "num_ctx": 4096},
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=ollama_payload)
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama not running. Start with: ollama serve")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Ollama timed out.")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.text[:200]}")

    raw_text: str = response.json().get("response", "")

    try:
        result: dict = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {raw_text[:200]}")

    result["confidence"] = defect.confidence
    result["inspection_timestamp"] = timestamp
    result.setdefault("sop_references", [c["metadata"].get("source_file", "unknown") for c in chunks[:3]])
    result["recommended_action"] = result.get("recommended_action", "reject").lower()
    result["severity_assessment"] = result.get("severity_assessment", defect.severity).lower()

    logger.info(f"action={result['recommended_action']!r} severity={result['severity_assessment']!r} ts={timestamp}")

    return DefectOutput(**result)


# ---------------------------------------------------------------------------
# CV bridge
# ---------------------------------------------------------------------------

DEFECT_NAME_MAP = {
    "short": "short", "sh": "short",
    "open": "open", "op": "open",
    "mouse bite": "mouse_bite", "mouse_bite": "mouse_bite", "mb": "mouse_bite",
    "hole breakout": "hole_breakout", "hole_breakout": "hole_breakout", "hb": "hole_breakout",
    "spur": "spur", "sp": "spur",
    "spurious copper": "spurious_copper", "spurious_copper": "spurious_copper", "sc": "spurious_copper",
    "conductor scratch": "conductor_scratch", "conductor_scratch": "conductor_scratch", "cs": "conductor_scratch",
    "conductor foreign object": "conductor_foreign_object", "conductor_foreign_object": "conductor_foreign_object", "cfo": "conductor_foreign_object",
    "base material foreign object": "base_material_foreign_object", "base_material_foreign_object": "base_material_foreign_object", "bmfo": "base_material_foreign_object",
}


@app.post("/inspect", response_model=DefectOutput)
async def inspect(payload: CVPayload) -> DefectOutput:
    """Accepts Sovra Vision /inspection/inspect payload + optional production_context."""
    vision = payload.vision_result

    if not vision.defect_detected:
        return DefectOutput(
            defect_explanation="No defect detected.",
            severity_assessment="low",
            recommended_action="pass",
            justification="CV system found no defect above confidence threshold.",
            sop_references=[],
            confidence=1.0,
            inspection_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    detection = vision.primary_detection
    if not detection:
        raise HTTPException(status_code=422, detail="defect_detected=true but primary_detection is missing.")

    raw_name = (detection.defect_name or detection.defect_code or "").strip().lower()
    defect_type = DEFECT_NAME_MAP.get(raw_name, raw_name.replace(" ", "_"))

    logger.info(f"/inspect → '{raw_name}' → '{defect_type}'")

    return await explain_defect(DefectInput(
        defect_type=defect_type,
        location=detection.location_description or "unknown",
        confidence=detection.confidence or 0.0,
        severity=(detection.severity_hint or "medium").lower(),
        production_context=payload.production_context,
    ))


@app.post("/analyze-image", response_model=DefectOutput)
async def analyze_image(request: Request) -> DefectOutput:
    """Send raw image → CV system → RAG → decision."""
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            cv_response = await client.post(CV_SYSTEM_URL, files={"image": ("image.jpg", body, "image/jpeg")})
            cv_response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"CV system not reachable at {CV_SYSTEM_URL}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"CV system error: {exc.response.text[:200]}")

    return await inspect(CVPayload(**cv_response.json()))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")), reload=True)
