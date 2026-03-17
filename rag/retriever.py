"""
Build LangChain retriever over Qdrant or Databricks Vector Search (Phase 3) using OpenRouter embeddings.
"""
from __future__ import annotations

import os

from langchain_core.retrievers import BaseRetriever

from qdrant_client import QdrantClient

from rag.embeddings_openrouter import OpenRouterEmbeddings, DEFAULT_MODEL


def get_retriever(
    *,
    backend: str | None = None,
    **kwargs,
) -> BaseRetriever:
    """Return Qdrant or Databricks retriever based on VECTOR_BACKEND env or backend= argument."""
    backend = backend or os.environ.get("VECTOR_BACKEND", "qdrant")
    if backend == "databricks":
        from rag.retriever_databricks import get_databricks_retriever
        db_kwargs = {k: v for k, v in kwargs.items() if k in ("index_name", "workspace_url", "personal_access_token", "endpoint_name", "openrouter_api_key", "embed_model", "k")}
        return get_databricks_retriever(**db_kwargs)
    return get_qdrant_retriever(**kwargs)


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
