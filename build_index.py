"""
build_index.py — PCB Defect RAG Knowledge Base Index Builder

Loads all .txt and .pdf files from knowledge_base subfolders, chunks them,
embeds them with sentence-transformers, and saves a FAISS vector index locally.

Run:
    python build_index.py

Environment variables (see .env.example):
    KNOWLEDGE_BASE_DIR, FAISS_INDEX_PATH, EMBEDDING_MODEL,
    CHUNK_SIZE (tokens), CHUNK_OVERLAP (tokens)
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# all-MiniLM-L6-v2 averages ~4 chars/token for English technical text
CHARS_PER_TOKEN = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KNOWN_DEFECT_TYPES = [
    "surface_crack",
    "solder_bridge",
    "missing_component",
    "tombstoning",
    "cold_solder_joint",
    "lifted_pad",
    "open_circuit",
    "short_circuit",
    "void",
    "delamination",
    "oxidation",
    "misalignment",
    "wrong_component",
    "insufficient_solder",
    "excess_solder",
]


def detect_defect_type(stem: str) -> str | None:
    """Infer defect type from filename stem (underscored, lowercased)."""
    normalized = stem.lower().replace(" ", "_").replace("-", "_")
    for defect in KNOWN_DEFECT_TYPES:
        if defect in normalized:
            return defect
    return None


def load_documents() -> list:
    """Walk knowledge_base subfolders and load all .txt / .pdf files."""
    documents = []
    kb_path = Path(KNOWLEDGE_BASE_DIR)

    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: '{KNOWLEDGE_BASE_DIR}'"
        )

    for folder in sorted(kb_path.iterdir()):
        if not folder.is_dir():
            continue

        folder_category = folder.name
        files = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in {".txt", ".pdf"}
        )

        if not files:
            logger.warning(f"No .txt/.pdf files found in: {folder}")
            continue

        for file_path in files:
            try:
                if file_path.suffix.lower() == ".txt":
                    loader = TextLoader(str(file_path), encoding="utf-8")
                else:
                    loader = PyPDFLoader(str(file_path))

                docs = loader.load()
                defect_type = detect_defect_type(file_path.stem)

                for doc in docs:
                    doc.metadata.update(
                        {
                            "source_file": file_path.name,
                            "folder_category": folder_category,
                            "defect_type": defect_type or "general",
                            "file_path": str(file_path),
                        }
                    )

                documents.extend(docs)
                logger.info(
                    f"  Loaded '{file_path.name}' → {len(docs)} doc(s) "
                    f"[category={folder_category}, defect={defect_type or 'general'}]"
                )

            except Exception as exc:
                logger.error(f"  Failed to load '{file_path}': {exc}")

    return documents


def chunk_documents(documents: list) -> list:
    """
    Split documents into chunks of ~CHUNK_SIZE tokens with CHUNK_OVERLAP token overlap.
    Uses character-count approximation (CHARS_PER_TOKEN chars ≈ 1 token).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * CHARS_PER_TOKEN,
        chunk_overlap=CHUNK_OVERLAP * CHARS_PER_TOKEN,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
    return chunks


def build_index() -> None:
    """Main entry point: load → chunk → embed → save FAISS index."""
    logger.info(f"Knowledge base directory : {KNOWLEDGE_BASE_DIR}")
    logger.info(f"FAISS index output path  : {FAISS_INDEX_PATH}")
    logger.info(f"Embedding model          : {EMBEDDING_MODEL}")
    logger.info(f"Chunk size               : {CHUNK_SIZE} tokens (~{CHUNK_SIZE * CHARS_PER_TOKEN} chars)")
    logger.info(f"Chunk overlap            : {CHUNK_OVERLAP} tokens (~{CHUNK_OVERLAP * CHARS_PER_TOKEN} chars)")
    logger.info("-" * 60)

    logger.info("Step 1/4 — Loading documents...")
    documents = load_documents()

    if not documents:
        logger.error("No documents loaded. Add .txt or .pdf files to knowledge_base/ subfolders.")
        return

    logger.info(f"Step 2/4 — Chunking {len(documents)} document(s)...")
    chunks = chunk_documents(documents)

    logger.info(f"Step 3/4 — Loading embedding model '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"Step 4/4 — Building FAISS index from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    out_path = Path(FAISS_INDEX_PATH)
    out_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_path))
    logger.info(f"FAISS index saved → {out_path}/")

    # Write stats alongside the index for traceability
    stats = {
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "knowledge_base_dir": KNOWLEDGE_BASE_DIR,
        "categories": sorted(
            {doc.metadata.get("folder_category", "unknown") for doc in documents}
        ),
        "defect_types_detected": sorted(
            {
                doc.metadata.get("defect_type", "general")
                for doc in documents
                if doc.metadata.get("defect_type") not in (None, "general")
            }
        ),
    }
    stats_path = out_path / "index_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    logger.info("-" * 60)
    logger.info("Index build complete.")
    logger.info(f"  Documents : {stats['total_documents']}")
    logger.info(f"  Chunks    : {stats['total_chunks']}")
    logger.info(f"  Categories: {stats['categories']}")
    logger.info(f"  Stats file: {stats_path}")


if __name__ == "__main__":
    build_index()
