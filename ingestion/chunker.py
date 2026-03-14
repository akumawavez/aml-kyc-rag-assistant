"""
Chunk documents with overlap and metadata preservation.
Uses character-based splitting (~512 tokens ≈ 2000 chars, 50 tokens overlap ≈ 200 chars).
"""
from __future__ import annotations

from typing import Any


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks (character-based)."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at a word boundary
        if end < len(text) and chunk:
            last_space = chunk.rfind(" ")
            if last_space > chunk_size // 2:
                chunk = chunk[: last_space + 1]
                end = start + len(chunk)
        chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text):
            break
    return [c for c in chunks if c]


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """
    Split each document's text into chunks; preserve metadata on every chunk.
    chunk_size/overlap in characters (~512 tokens / 50 token overlap).
    """
    result = []
    for doc in documents:
        text = doc.get("text", "")
        meta = dict(doc.get("metadata", {}))
        for piece in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            result.append({
                "text": piece,
                "metadata": {**meta},
            })
    return result
