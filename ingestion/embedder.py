"""
Embed chunks via OpenRouter (nvidia/llama-nemotron-embed-vl-1b-v2:free) and persist to Qdrant.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

OPENROUTER_EMBED_API = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
BATCH_SIZE = 32
RATE_DELAY = 0.5  # seconds between batches to avoid rate limits
UPSERT_BATCH_SIZE = 500  # points per upsert batch for progress reporting


def _embed_batch(
    texts: list[str],
    api_key: str,
    model: str = DEFAULT_EMBED_MODEL,
) -> list[list[float]]:
    """Call OpenRouter embeddings API. Input can be a list of strings."""
    import requests

    resp = requests.post(
        OPENROUTER_EMBED_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aml-kyc-rag-assistant",
        },
        json={
            "model": model,
            "input": texts if len(texts) > 1 else texts[0],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("data", [])
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raw = [raw]
    return [item["embedding"] for item in raw]


def embed_chunks(
    chunks: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_EMBED_MODEL,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[tuple[list[float], dict[str, Any]]]:
    """Embed all chunks; returns list of (vector, metadata). progress_callback(batch_done, total_batches, 'embed')."""
    results = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, i in enumerate(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        try:
            vectors = _embed_batch(texts, api_key=api_key, model=model)
        except Exception as e:
            raise RuntimeError(f"Embedding batch at index {i}: {e}") from e
        for j, (vec, chunk) in enumerate(zip(vectors, batch)):
            meta = chunk.get("metadata", {})
            payload = {k: (v if v is None or isinstance(v, (str, int, float, bool)) else str(v)) for k, v in meta.items()}
            payload["text"] = chunk["text"][:65535]
            results.append((vec, payload))
        if progress_callback:
            progress_callback(batch_idx + 1, total_batches, "embed")
        if i + BATCH_SIZE < len(chunks):
            time.sleep(RATE_DELAY)
    return results


def get_embedding_dimension(api_key: str, model: str = DEFAULT_EMBED_MODEL) -> int:
    """Get vector size by embedding one short string."""
    vecs = _embed_batch(["test"], api_key=api_key, model=model)
    return len(vecs[0])


def run_ingestion(
    chunks: list[dict[str, Any]],
    *,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "aml_rag",
    openrouter_api_key: str | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    recreate_collection: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """
    Embed chunks and upsert into Qdrant. Returns number of points upserted.
    progress_callback(current, total, phase) with phase in ('embed', 'upsert').
    """
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required for embeddings")

    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    if recreate_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    dim = get_embedding_dimension(api_key, model=embed_model)
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    embedded = embed_chunks(
        chunks,
        api_key=api_key,
        model=embed_model,
        progress_callback=progress_callback,
    )
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
        for vec, payload in embedded
    ]
    total_points = len(points)
    # Upsert in batches for progress reporting
    for i in range(0, total_points, UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name, points=batch)
        if progress_callback:
            progress_callback(min(i + len(batch), total_points), total_points, "upsert")
    return total_points
