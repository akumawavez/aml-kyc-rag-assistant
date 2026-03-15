"""
Build LangChain retriever over Qdrant using OpenRouter embeddings (same model as ingestion).
"""
from __future__ import annotations

import os
from typing import Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from qdrant_client import QdrantClient

from rag.embeddings_openrouter import OpenRouterEmbeddings, DEFAULT_MODEL


def get_qdrant_retriever(
    collection_name: str = "aml_rag",
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    openrouter_api_key: str | None = None,
    embed_model: str = DEFAULT_MODEL,
    k: int = 10,
) -> BaseRetriever:
    """Build a retriever that uses Qdrant + OpenRouter embeddings."""
    try:
        from langchain_qdrant import QdrantVectorStore
    except ImportError:
        try:
            from langchain_qdrant import Qdrant as QdrantVectorStore
        except ImportError:
            from langchain_community.vectorstores import Qdrant as QdrantVectorStore

    host = qdrant_host or os.environ.get("QDRANT_HOST", "localhost")
    port = qdrant_port or int(os.environ.get("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port)
    embeddings = OpenRouterEmbeddings(
        api_key=openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"),
        model=embed_model,
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})
