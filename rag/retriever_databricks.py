"""
LangChain retriever over Databricks Mosaic AI Vector Search (Phase 3).
Uses OpenRouter to embed the query, then queries the Vector Search index.
"""
from __future__ import annotations

import os
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.embeddings_openrouter import OpenRouterEmbeddings, DEFAULT_MODEL


class DatabricksVectorSearchRetriever(BaseRetriever):
    """Retriever that queries a Databricks Vector Search index using query embedding from OpenRouter."""

    index_name: str
    host: str
    token: str
    embeddings: OpenRouterEmbeddings
    k: int = 10
    columns_to_return: List[str] = ["id", "text", "product", "complaint_id", "source", "issue", "company"]

    def _get_collection(
        self,
    ):
        try:
            from databricks.vector_search.client import VectorSearchClient
        except ImportError as e:
            raise ImportError("Install databricks-vectorsearch: pip install databricks-vectorsearch") from e
        client = VectorSearchClient(host=self.host, token=self.token)
        return client.get_index(self.index_name)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        index = self._get_collection()
        results = index.similarity_search(
            columns=self.columns_to_return,
            query_vector=query_vector,
            num_results=self.k,
        )
        # Response may be dict with result.rows or a list of hits
        rows = []
        if isinstance(results, dict):
            rows = results.get("result", {}).get("rows", results.get("rows", []))
        elif isinstance(results, list):
            rows = results
        docs = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = row.get("text") or row.get("content") or ""
            meta = {k: v for k, v in row.items() if k not in ("text", "content") and v is not None}
            docs.append(Document(page_content=text, metadata=meta))
        return docs


def get_databricks_retriever(
    index_name: str | None = None,
    host: str | None = None,
    token: str | None = None,
    openrouter_api_key: str | None = None,
    embed_model: str = DEFAULT_MODEL,
    k: int = 10,
) -> BaseRetriever:
    """Build a retriever that uses Databricks Vector Search + OpenRouter for query embedding."""
    index_name = index_name or os.environ.get("DATABRICKS_VECTOR_SEARCH_INDEX_NAME", "")
    host = host or os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = token or os.environ.get("DATABRICKS_TOKEN", "")
    if not index_name or not host or not token:
        raise ValueError(
            "For Databricks retriever set DATABRICKS_VECTOR_SEARCH_INDEX_NAME, DATABRICKS_HOST, DATABRICKS_TOKEN"
        )
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required for query embedding")
    embeddings = OpenRouterEmbeddings(api_key=api_key, model=embed_model)
    return DatabricksVectorSearchRetriever(
        index_name=index_name,
        host=host,
        token=token,
        embeddings=embeddings,
        k=k,
    )
