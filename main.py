"""
main.py — PCB Defect Inspection RAG API

FastAPI application exposing a POST /explain-defect endpoint.
Retrieves relevant SOP/knowledge chunks via FAISS, then calls Ollama
(qwen3:4b by default) with the context to generate a structured JSON response.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Environment variables (see .env.example):
    OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K, FAISS_INDEX_PATH, EMBEDDING_MODEL
"""

import json
import logging
import os
import re

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from retrieve import retrieve

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DefectInput(BaseModel):
    defect_type: str = Field(..., examples=["short"])
    location: str = Field(..., examples=["top-left"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.96])
    severity: str = Field(..., examples=["high"])


# CV system payload models (from Sovra-Vision-Repo /inspection/inspect)
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
    model_config = {"extra": "allow"}  # accept full payload without strict schema


class DefectOutput(BaseModel):
    defect_explanation: str
    severity_assessment: str
    recommended_action: str
    justification: str
    sop_references: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# JSON schema sent to Ollama's `format` parameter to enforce structured output
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "defect_explanation": {
            "type": "string",
            "description": "Detailed explanation of the defect, its nature and implications",
        },
        "severity_assessment": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Severity classification of the detected defect",
        },
        "recommended_action": {
            "type": "string",
            "enum": ["pass", "rework", "reject"],
            "description": "Recommended disposition action per quality SOPs",
        },
        "justification": {
            "type": "string",
            "description": "Reasoning for the recommended action, citing SOPs and quality criteria",
        },
        "sop_references": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of SOP or knowledge base document references used",
        },
        "confidence": {
            "type": "number",
            "description": "Detection confidence score from the inspection system (0.0 – 1.0)",
        },
    },
    "required": [
        "defect_explanation",
        "severity_assessment",
        "recommended_action",
        "justification",
        "sop_references",
        "confidence",
    ],
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(defect: DefectInput, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("source_file", "Unknown")
        category = meta.get("folder_category", "Unknown")
        context_blocks.append(
            f"[Context {i} | {source} ({category})]\n{chunk['content']}"
        )

    context_text = "\n\n".join(context_blocks)

    return f"""You are a senior PCB quality control engineer analyzing a defect flagged by an automated inspection system.

DEFECT DETECTION REPORT:
  Defect Type : {defect.defect_type.replace("_", " ").title()}
  Location    : {defect.location}
  Confidence  : {defect.confidence:.1%}
  Severity    : {defect.severity.upper()}

KNOWLEDGE BASE CONTEXT (SOPs, defect definitions, quality criteria):
{context_text}

TASK:
Using the defect report and knowledge base context above, produce a structured quality assessment.
Your response MUST be a single JSON object with these fields:
  - defect_explanation   : what this defect is and its technical implications
  - severity_assessment  : one of "low" | "medium" | "high" | "critical"
  - recommended_action   : one of "pass" | "rework" | "reject"
  - justification        : clear reasoning citing specific SOPs or quality thresholds
  - sop_references       : list of source documents or SOP codes referenced
  - confidence           : {defect.confidence} (preserve the input confidence value)

Output only the JSON object. No preamble, no markdown code fences."""


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PCB Defect Inspection RAG API",
    description=(
        "RAG-powered API for industrial PCB defect explanation and disposition. "
        "Retrieves relevant SOPs from a local FAISS knowledge base, then calls "
        "Ollama (qwen3:4b) for structured JSON analysis."
    ),
    version="1.0.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", summary="Health check")
async def health_check() -> dict:
    """Returns API status, Ollama connectivity, and FAISS index readiness."""
    import os
    from pathlib import Path

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "connected" if resp.status_code == 200 else "error"
            models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        ollama_status = "disconnected"
        models = []

    index_path = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index"))
    index_ready = index_path.exists() and (index_path / "index.faiss").exists()

    return {
        "status": "ok",
        "ollama": {
            "status": ollama_status,
            "url": OLLAMA_BASE_URL,
            "configured_model": OLLAMA_MODEL,
            "available_models": models,
        },
        "faiss_index": {
            "status": "ready" if index_ready else "not_built",
            "path": str(index_path),
        },
    }


@app.post(
    "/explain-defect",
    response_model=DefectOutput,
    summary="Explain a PCB defect with RAG-enhanced LLM analysis",
)
async def explain_defect(defect: DefectInput) -> DefectOutput:
    """
    Accepts a defect detection event, retrieves relevant SOPs and quality
    criteria from the local knowledge base, and returns a structured
    disposition analysis powered by a local Ollama LLM.
    """
    logger.info(
        f"Received: defect_type={defect.defect_type!r} "
        f"location={defect.location!r} "
        f"confidence={defect.confidence:.2f} "
        f"severity={defect.severity!r}"
    )

    # --- Step 1: Retrieve relevant knowledge base chunks ---
    try:
        chunks = retrieve(defect.defect_type, top_k=TOP_K)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No knowledge base entries found for defect type: "
                f"'{defect.defect_type}'. Rebuild index or add relevant documents."
            ),
        )

    logger.info(f"Retrieved {len(chunks)} chunks (top score: {chunks[0]['score']:.4f})")

    # --- Step 2: Build prompt ---
    prompt = build_prompt(defect, chunks)

    # --- Step 3: Call Ollama ---
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 1024,
            "num_ctx": 8192,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Ensure Ollama is running: ollama serve"
            ),
        )
    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Ollama did not respond within {OLLAMA_TIMEOUT}s. "
                "Try increasing OLLAMA_TIMEOUT or using a smaller model."
            ),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama API error {exc.response.status_code}: {exc.response.text[:300]}",
        )

    # --- Step 4: Parse LLM response ---
    raw_text: str = response.json().get("response", "")
    logger.debug(f"Ollama raw response: {raw_text[:500]}")

    try:
        result: dict = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: extract first {...} block if model added extra text
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM response is not valid JSON: {raw_text[:300]}",
                )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"LLM response is not valid JSON: {raw_text[:300]}",
            )

    # --- Step 5: Normalise and return ---
    # Always preserve the input confidence value
    result["confidence"] = defect.confidence

    # Provide fallback SOP references from retrieved chunk filenames
    if not result.get("sop_references"):
        result["sop_references"] = [
            chunk["metadata"].get("source_file", "unknown")
            for chunk in chunks[:3]
        ]

    # Normalise action / severity to lower-case
    result["recommended_action"] = result.get("recommended_action", "reject").lower()
    result["severity_assessment"] = result.get("severity_assessment", defect.severity).lower()

    logger.info(
        f"Result: action={result['recommended_action']!r} "
        f"severity={result['severity_assessment']!r}"
    )

    return DefectOutput(**result)


# Defect name normalisation — maps CV defect names/codes to knowledge base filenames
DEFECT_NAME_MAP = {
    "short": "short",
    "sh": "short",
    "open": "open",
    "op": "open",
    "mouse bite": "mouse_bite",
    "mouse_bite": "mouse_bite",
    "mb": "mouse_bite",
    "hole breakout": "hole_breakout",
    "hole_breakout": "hole_breakout",
    "hb": "hole_breakout",
    "spur": "spur",
    "sp": "spur",
    "spurious copper": "spurious_copper",
    "spurious_copper": "spurious_copper",
    "sc": "spurious_copper",
    "conductor scratch": "conductor_scratch",
    "conductor_scratch": "conductor_scratch",
    "cs": "conductor_scratch",
    "conductor foreign object": "conductor_foreign_object",
    "conductor_foreign_object": "conductor_foreign_object",
    "cfo": "conductor_foreign_object",
    "base material foreign object": "base_material_foreign_object",
    "base_material_foreign_object": "base_material_foreign_object",
    "bmfo": "base_material_foreign_object",
}


@app.post(
    "/inspect",
    response_model=DefectOutput,
    summary="Accept Sovra Vision payload and return RAG analysis",
)
async def inspect(payload: CVPayload) -> DefectOutput:
    """
    Bridge endpoint for the Sovra CV system.
    Accepts the full /inspection/inspect payload from Sovra-Vision-Repo
    and returns the same structured analysis as /explain-defect.
    """
    vision = payload.vision_result

    if not vision.defect_detected:
        return DefectOutput(
            defect_explanation="No defect detected by the vision system.",
            severity_assessment="low",
            recommended_action="pass",
            justification="Vision system reported no defect detected.",
            sop_references=[],
            confidence=1.0,
        )

    detection = vision.primary_detection
    if not detection:
        raise HTTPException(status_code=422, detail="defect_detected is true but primary_detection is missing.")

    raw_name = (detection.defect_name or detection.defect_code or "").strip().lower()
    defect_type = DEFECT_NAME_MAP.get(raw_name, raw_name.replace(" ", "_"))
    location = detection.location_description or "unknown"
    confidence = detection.confidence or 0.0
    severity = (detection.severity_hint or "medium").lower()

    logger.info(f"/inspect → mapped '{raw_name}' to '{defect_type}'")

    return await explain_defect(
        DefectInput(
            defect_type=defect_type,
            location=location,
            confidence=confidence,
            severity=severity,
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
