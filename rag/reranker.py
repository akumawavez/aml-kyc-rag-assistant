"""
Rerank retrieved documents via Cohere reranker service (Docker).
"""
from __future__ import annotations

import os
from typing import List

import requests
from langchain_core.documents import Document

RERANKER_URL_DEFAULT = "http://localhost:8000"


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5,
    reranker_url: str | None = None,
) -> List[Document]:
    """Call Cohere reranker service and return top_n documents in order."""
    if not documents:
        return []
    url = (reranker_url or os.environ.get("RERANKER_URL", RERANKER_URL_DEFAULT)).rstrip("/") + "/rerank"
    texts = [doc.page_content for doc in documents]
    try:
        resp = requests.post(
            url,
            json={"query": query, "documents": texts, "top_n": min(top_n, len(texts))},
            timeout=30,
        )
        resp.raise_for_status()
    except (requests.RequestException, Exception):
        # Service down or no COHERE_API_KEY: return original order
        return documents[:top_n]
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return documents[:top_n]
    # results: [{"index": 0, "relevance_score": 0.99}, ...]
    ordered = [documents[r["index"]] for r in results]
    return ordered
