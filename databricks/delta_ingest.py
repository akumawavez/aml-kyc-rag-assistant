"""
Delta Lake ingest for Databricks: load CFPB + regulatory data, chunk, embed, write to Delta.
Run on a Databricks cluster (Spark required). API key from env or dbutils.secrets.

Usage (in Databricks notebook or job):
  - Set INPUT_CFPB_PATH, INPUT_REGULATORY_DIR (optional), DELTA_TABLE_NAME, OPENROUTER_API_KEY (or use secrets).
  - Or: python delta_ingest.py --cfpb-path /dbfs/FileStore/aml_kyc/cfpb_filtered.csv --delta-table main.default.aml_kyc_chunks
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

# Add project root for imports when run as job
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField
except ImportError:
    SparkSession = None  # type: ignore[misc, assignment]

OPENROUTER_EMBED_API = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
BATCH_SIZE = 32
RATE_DELAY = 0.5


def _get_secret(key: str) -> str:
    """Get secret from env or, on Databricks, from dbutils."""
    v = os.environ.get(key, "").strip()
    if v:
        return v
    try:
        import dbutils
        scope = os.environ.get("DATABRICKS_SECRET_SCOPE", "aml-kyc")
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return ""


def _embed_batch(texts: list[str], api_key: str, model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    import requests
    resp = requests.post(
        OPENROUTER_EMBED_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aml-kyc-rag-assistant",
        },
        json={"model": model, "input": texts if len(texts) > 1 else texts[0]},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("data", [])
    if isinstance(raw, dict):
        raw = [raw]
    return [item["embedding"] for item in raw]


def load_and_chunk(cfpb_path: str, regulatory_dir: str | None) -> list[dict]:
    """Load documents and chunk. Uses project ingestion modules."""
    from ingestion.loader import load_cfpb, load_regulatory_pdfs
    from ingestion.chunker import chunk_documents

    docs: list[dict] = []
    if cfpb_path and os.path.exists(cfpb_path):
        docs.extend(load_cfpb(cfpb_path))
    if regulatory_dir and os.path.isdir(regulatory_dir):
        try:
            docs.extend(load_regulatory_pdfs(regulatory_dir))
        except ImportError:
            pass
    if not docs:
        return []
    return chunk_documents(docs, chunk_size=2000, overlap=200)


def embed_chunks(chunks: list[dict], api_key: str, model: str = DEFAULT_EMBED_MODEL) -> list[tuple[list[float], dict]]:
    """Embed all chunks; returns list of (vector, metadata_dict)."""
    results = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = _embed_batch(texts, api_key=api_key, model=model)
        for vec, chunk in zip(vectors, batch):
            meta = dict(chunk.get("metadata", {}))
            meta["text"] = (chunk.get("text") or "")[:65535]
            results.append((vec, meta))
        if i + BATCH_SIZE < len(chunks):
            time.sleep(RATE_DELAY)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CFPB + regulatory data to Delta on Databricks")
    parser.add_argument("--cfpb-path", default=os.environ.get("INPUT_CFPB_PATH", ""), help="Path to CFPB CSV (local or DBFS)")
    parser.add_argument("--regulatory-dir", default=os.environ.get("INPUT_REGULATORY_DIR", ""), help="Path to regulatory PDFs dir")
    parser.add_argument("--delta-table", default=os.environ.get("DELTA_TABLE_NAME", "main.default.aml_kyc_chunks"), help="Delta table name (catalog.schema.table)")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="OpenRouter embedding model")
    args = parser.parse_args()

    api_key = _get_secret("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set and not in secrets.", file=sys.stderr)
        sys.exit(1)

    cfpb_path = args.cfpb_path or os.path.join("/dbfs", "FileStore", "aml_kyc", "cfpb_filtered.csv")
    print(f"Loading and chunking from {cfpb_path} and {args.regulatory_dir or 'N/A'}...")
    chunks = load_and_chunk(cfpb_path, args.regulatory_dir or None)
    if not chunks:
        print("Error: No documents to ingest.", file=sys.stderr)
        sys.exit(1)
    print(f"  Chunks: {len(chunks)}")

    print("Embedding...")
    embedded = embed_chunks(chunks, api_key, model=args.embed_model)

    if not SparkSession:
        print("Error: PySpark not available. Run this script on a Databricks cluster.", file=sys.stderr)
        sys.exit(1)

    spark = SparkSession.builder.getOrCreate()
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("text", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), False),
        StructField("product", StringType(), True),
        StructField("complaint_id", StringType(), True),
        StructField("source", StringType(), True),
        StructField("issue", StringType(), True),
        StructField("company", StringType(), True),
    ])
    rows = []
    for vec, meta in embedded:
        rows.append((
            str(uuid.uuid4()),
            meta.get("text", ""),
            vec,
            meta.get("product"),
            meta.get("complaint_id"),
            meta.get("source", "cfpb"),
            meta.get("issue"),
            meta.get("company"),
        ))
    df = spark.createDataFrame(rows, schema)
    print(f"Writing to Delta table {args.delta_table}...")
    df.write.format("delta").mode("overwrite").saveAsTable(args.delta_table)
    print(f"Done. Table {args.delta_table} has {df.count()} rows.")


if __name__ == "__main__":
    main()
