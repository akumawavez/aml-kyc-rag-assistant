"""
OpenRouter embeddings for LangChain (same model as ingestion: nvidia/llama-nemotron-embed-vl-1b-v2:free).
"""
from __future__ import annotations

import os
from typing import List

from langchain_core.embeddings import Embeddings

OPENROUTER_EMBED_API = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


def _embed_batch(texts: List[str], api_key: str, model: str) -> List[List[float]]:
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


class OpenRouterEmbeddings(Embeddings):
    """LangChain Embeddings that call OpenRouter embeddings API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required for OpenRouterEmbeddings")
        out = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out.extend(_embed_batch(batch, self.api_key, self.model))
        return out

    def embed_query(self, text: str) -> List[float]:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required for OpenRouterEmbeddings")
        return _embed_batch([text], self.api_key, self.model)[0]
