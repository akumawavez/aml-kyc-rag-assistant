"""
Run the full ingestion pipeline: load data -> chunk -> embed (OpenRouter) -> Qdrant.
Usage:
  python -m ingestion.run
  python -m ingestion.run --data-dir ./data --recreate
Requires: OPENROUTER_API_KEY in env (and Qdrant running, e.g. docker compose up -d qdrant).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ingestion.loader import load_all
from ingestion.chunker import chunk_documents
from ingestion.embedder import run_ingestion


def _check_setup(data_dir: Path, qdrant_host: str, qdrant_port: int) -> None:
    """Verify env and connectivity; exit with message if something is missing."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        print("  Add it to your .env or set it in the shell (see .env.example).", file=sys.stderr)
        sys.exit(1)

    cfpb_path = data_dir / "processed" / "cfpb_filtered.csv"
    if not cfpb_path.exists():
        print(f"Warning: {cfpb_path} not found. Run scripts/download_and_filter_cfpb.py first.", file=sys.stderr)
        print("  Ingestion will only load PDFs from data/regulatory/ if any.", file=sys.stderr)

    try:
        from qdrant_client import QdrantClient
        QdrantClient(host=qdrant_host, port=qdrant_port)
    except Exception as e:
        print(f"Error: Cannot connect to Qdrant at {qdrant_host}:{qdrant_port}.", file=sys.stderr)
        print("  Start it with: docker compose up -d qdrant", file=sys.stderr)
        print(f"  Details: {e}", file=sys.stderr)
        sys.exit(1)


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

    _check_setup(args.data_dir, args.qdrant_host, args.qdrant_port)

    print("Loading documents...")
    docs = load_all(args.data_dir)
    print(f"  Loaded {len(docs)} documents")
    if len(docs) == 0:
        print("Error: No documents to ingest. Ensure data/processed/cfpb_filtered.csv exists and/or data/regulatory/ has PDFs.", file=sys.stderr)
        sys.exit(1)

    print("Chunking...")
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"  {len(chunks)} chunks")

    total_chunks = len(chunks)
    batch_size_embed = 32  # must match embedder.BATCH_SIZE
    _last_pct = [-1.0, -1.0]  # [embed, upsert] — throttle: only print when pct advances by ≥1%

    def _progress(current: int, total: int, phase: str) -> None:
        if total == 0:
            return
        pct = current / total * 100
        idx = 0 if phase == "embed" else 1
        if pct - _last_pct[idx] < 1.0 and current < total:
            return  # skip update to avoid extra I/O
        _last_pct[idx] = pct
        if phase == "embed":
            chunks_done = min(current * batch_size_embed, total_chunks)
            print(f"\r  Embedding: {pct:.1f}% ({chunks_done}/{total_chunks} chunks)", end="", flush=True)
        else:
            print(f"\r  Upserting: {pct:.1f}% ({current}/{total} points)", end="", flush=True)

    print("Embedding and writing to Qdrant...")
    n = run_ingestion(
        chunks,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
        recreate_collection=args.recreate,
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        progress_callback=_progress,
    )
    print()  # newline after progress
    print(f"  Upserted {n} points to collection {args.collection!r}")

    # Verify collection actually has points (fail if ingestion wrote 0)
    from qdrant_client import QdrantClient
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    count = 0
    try:
        info = client.get_collection(args.collection)
        count = getattr(info, "points_count", 0)
    except Exception:
        pass
    if not count and hasattr(client, "count"):
        try:
            r = client.count(collection_name=args.collection)
            count = getattr(r, "count", 0)
        except Exception:
            pass
    if count == 0:
        print("Error: Verification failed — collection has 0 points. Ingestion did not persist data.", file=sys.stderr)
        sys.exit(1)
    print(f"  Verified: collection {args.collection!r} has {count} points.")


if __name__ == "__main__":
    main()
