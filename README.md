# sovrarag — PCB Defect Knowledge Base

Local RAG pipeline for PCB inspection. No cloud, no subscriptions. Point it at your defect docs, build the index, ask it questions via API.

---

## What it does

You send a defect detection event (type, location, confidence, severity) to the API. It pulls the most relevant SOPs and quality criteria from a local knowledge base, feeds them to a local LLM (Ollama), and returns a structured decision: explain the defect, assess severity, recommend pass / rework / reject.

---

## Stack

- **FAISS** — vector index, runs locally
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **LangChain** — document loading and chunking
- **FastAPI** — API layer
- **Ollama** (`qwen3:4b`) — local LLM inference

---

## Setup

```bash
git clone https://github.com/ALMOWAFI/sovrarag.git
cd sovrarag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

Pull the model (one-time, ~2.5 GB):
```bash
ollama pull qwen3:4b
```

Build the index:
```bash
python build_index.py
```

Start the API:
```bash
uvicorn main:app --reload
```

---

## Knowledge base structure

```
knowledge_base/
  defect_definitions/   ← what each defect is, causes, impact
  sops/                 ← inspection and rework procedures
  quality_criteria/     ← pass / rework / reject thresholds
```

Add or edit `.txt` / `.pdf` files in any subfolder, then re-run `build_index.py`.

---

## API

### `POST /explain-defect`

```json
{
  "defect_type": "surface_crack",
  "location": "top-left",
  "confidence": 0.96,
  "severity": "high"
}
```

Response:
```json
{
  "defect_explanation": "...",
  "severity_assessment": "high",
  "recommended_action": "reject",
  "justification": "...",
  "sop_references": ["reject_thresholds.txt", "surface_crack.txt"],
  "confidence": 0.96
}
```

### `GET /health`

Returns Ollama connectivity status and whether the index is built.

---

## CLI retriever

```bash
python retrieve.py surface_crack
python retrieve.py solder_bridge --top-k 3 --json
```

---

## Config (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `KNOWLEDGE_BASE_DIR` | `knowledge_base` | |
| `FAISS_INDEX_PATH` | `faiss_index` | rebuilt by `build_index.py` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | |
| `CHUNK_SIZE` | `400` | tokens |
| `CHUNK_OVERLAP` | `50` | tokens |
| `TOP_K` | `5` | chunks returned per query |
| `OLLAMA_MODEL` | `qwen3:4b` | |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_TIMEOUT` | `120` | seconds |
