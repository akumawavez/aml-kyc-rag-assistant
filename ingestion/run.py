"""
Run the full ingestion pipeline: load data -> chunk -> embed (OpenRouter) -> Qdrant.
Usage:
  python -m ingestion.run
  python -m ingestion.run --data-dir ./data --recreate
Requires: OPENROUTER_API_KEY in env (and Qdrant running, e.g. docker compose up -d).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ingestion.loader import load_all
from ingestion.chunker import chunk_documents
from ingestion.embedder import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion: load -> chunk -> embed -> Qdrant")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Data directory (contains processed/, regulatory/)",
    )
    parser.add_argument(
        "--qdrant-host",
        default=os.environ.get("QDRANT_HOST", "localhost"),
        help="Qdrant host (default: QDRANT_HOST or localhost)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.environ.get("QDRANT_PORT", "6333")),
        help="Qdrant port (default: QDRANT_PORT or 6333)",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "aml_rag"),
        help="Qdrant collection name (default: QDRANT_COLLECTION or aml_rag)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collection before upserting",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size in characters (~512 tokens)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters",
    )
    args = parser.parse_args()

    print("Loading documents...")
    docs = load_all(args.data_dir)
    print(f"  Loaded {len(docs)} documents")

    print("Chunking...")
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"  {len(chunks)} chunks")

    print("Embedding and writing to Qdrant...")
    n = run_ingestion(
        chunks,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
        recreate_collection=args.recreate,
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    )
    print(f"  Upserted {n} points to collection {args.collection!r}")


if __name__ == "__main__":
    main()
