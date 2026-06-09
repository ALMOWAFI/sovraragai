"""
retrieve.py — PCB Defect RAG Retriever

Loads the pre-built FAISS index and returns the top-K most relevant
knowledge base chunks for a given defect type query.

Usage (CLI):
    python retrieve.py surface_crack
    python retrieve.py solder_bridge --top-k 3

Usage (module):
    from retrieve import retrieve
    chunks = retrieve("tombstoning", top_k=5)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Module-level singletons — loaded once, reused across calls
_embeddings: OllamaEmbeddings | None = None
_vectorstore: FAISS | None = None


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    return _embeddings


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        index_path = Path(FAISS_INDEX_PATH)
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
                "Run  python build_index.py  first."
            )
        logger.info(f"Loading FAISS index from: {index_path}")
        _vectorstore = FAISS.load_local(
            str(index_path),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve(defect_type: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """
    Retrieve the top-K most semantically relevant knowledge base chunks
    for a given PCB defect type.

    Args:
        defect_type: Defect name, e.g. "surface_crack", "solder_bridge".
        top_k:       Number of results to return (default: TOP_K env var or 5).

    Returns:
        List of dicts, each with keys:
            "content"  — chunk text
            "metadata" — source_file, folder_category, defect_type, file_path
            "score"    — L2 distance (lower = more similar)
    """
    query = (
        f"PCB defect type: {defect_type.replace('_', ' ')}. "
        f"Inspection criteria, SOP procedure, severity assessment, "
        f"accept reject rework thresholds."
    )

    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in results
    ]


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Query the PCB defect RAG knowledge base."
    )
    parser.add_argument(
        "defect_type",
        help="Defect type to query, e.g. surface_crack, solder_bridge",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of results to return (default: {TOP_K})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as raw JSON",
    )
    args = parser.parse_args()

    results = retrieve(args.defect_type, top_k=args.top_k)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"\nTop {len(results)} chunks for defect: '{args.defect_type}'\n")
    print("=" * 70)
    for i, chunk in enumerate(results, 1):
        meta = chunk["metadata"]
        print(f"\n[{i}] Score: {chunk['score']:.4f}")
        print(f"    Source   : {meta.get('source_file', 'N/A')}")
        print(f"    Category : {meta.get('folder_category', 'N/A')}")
        print(f"    Defect   : {meta.get('defect_type', 'N/A')}")
        print(f"    Content  :\n")
        # Print first 400 chars with indent
        preview = chunk["content"][:400].replace("\n", "\n    ")
        print(f"    {preview}")
        if len(chunk["content"]) > 400:
            print(f"    ... [{len(chunk['content']) - 400} more chars]")
        print("-" * 70)


if __name__ == "__main__":
    _cli()
