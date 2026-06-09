# sovrarag — PCB Defect Knowledge Base

Local RAG pipeline for PCB inspection. No cloud, no subscriptions.

---

## What it does

Takes a defect detection event from the CV system, pulls the most relevant SOPs and quality criteria from a local knowledge base, and returns a structured decision — defect explanation, severity, and recommended action.

Supported actions: `pass` `rework` `reject` `escalate` `clean_station` `quarantine_lot` `manual_review`

---

## Stack

- **FAISS** — local vector index
- **nomic-embed-text** via Ollama — embeddings
- **LangChain** — document loading and chunking
- **FastAPI** — API layer
- **Ollama** `qwen3:4b` — local LLM inference

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

---

## Setup

```bash
git clone https://github.com/ALMOWAFI/sovraragai.git
cd sovraragai

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
cp .env.example .env
```

Pull models (one-time):
```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

Build the FAISS index:
```bash
python build_index.py
```

Start the API:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Endpoints

### `POST /inspect`
Main endpoint — accepts Sovra Vision output directly.

```json
{
  "vision_result": {
    "defect_detected": true,
    "primary_detection": {
      "defect_code": "SH",
      "defect_name": "Short",
      "confidence": 0.97,
      "severity_hint": "critical",
      "location_description": "bottom-left"
    }
  },
  "production_context": {
    "factory_name": "Berlin Electronics Plant",
    "station_id": "AOI-04",
    "same_defect_count_30d": 9,
    "trend": "increasing",
    "most_common_shift": "Afternoon",
    "recurrence_after_repair": true,
    "maintenance_note": "dust near board loading area"
  }
}
```

`production_context` is optional — if not provided, decision is based on defect + SOPs only.

Response:
```json
{
  "defect_explanation": "...",
  "severity_assessment": "critical",
  "recommended_action": "clean_station",
  "justification": "...",
  "sop_references": ["short.txt", "reject_thresholds.txt"],
  "confidence": 0.97,
  "inspection_timestamp": "2026-06-09T18:32:05Z"
}
```

### `POST /analyze-image`
Send a raw image — this endpoint calls the CV system, gets detection, runs RAG, returns decision.

### `POST /explain-defect`
Manual input without CV system.

### `GET /health`
Returns Ollama and FAISS index status.

---

## Test cases

10 hardcoded test cases from the Sovra LLM test document:

```bash
python test_cases.py          # run all 10
python test_cases.py TC-001   # run single case
```

---

## Knowledge base

```
knowledge_base/
  defect_definitions/   ← short, open, mouse_bite, hole_breakout, spur,
                           spurious_copper, conductor_scratch,
                           conductor_foreign_object, base_material_foreign_object
  sops/                 ← inspection and rework procedures
  quality_criteria/     ← pass / rework / reject thresholds
```

Add `.txt` or `.pdf` files to any subfolder, then re-run `python build_index.py`.

---

## Config (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `KNOWLEDGE_BASE_DIR` | `knowledge_base` | |
| `FAISS_INDEX_PATH` | `faiss_index` | rebuilt by `build_index.py` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | |
| `CHUNK_SIZE` | `400` | tokens |
| `CHUNK_OVERLAP` | `50` | tokens |
| `TOP_K` | `5` | chunks per query |
| `OLLAMA_MODEL` | `qwen3:4b` | |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_TIMEOUT` | `300` | seconds |
| `CV_SYSTEM_URL` | `http://localhost:8001/inspection/inspect` | CV system endpoint |
| `API_HOST` | `0.0.0.0` | |
| `API_PORT` | `8000` | |
