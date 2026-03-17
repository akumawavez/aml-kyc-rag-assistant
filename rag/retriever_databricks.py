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
    workspace_url: str
    personal_access_token: str
    endpoint_name: str | None
    embeddings: OpenRouterEmbeddings
    k: int = 10
    columns_to_return: List[str] = ["id", "text", "product", "complaint_id", "source", "issue", "company"]

    def _get_index(self):
        try:
            from databricks.vector_search.client import VectorSearchClient
        except ImportError as e:
            raise ImportError("Install databricks-vectorsearch: pip install databricks-vectorsearch") from e
        client = VectorSearchClient(
            workspace_url=self.workspace_url,
            personal_access_token=self.personal_access_token,
        )
        return client.get_index(endpoint_name=self.endpoint_name, index_name=self.index_name)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        index = self._get_index()
        results = index.similarity_search(
            columns=self.columns_to_return,
            query_vector=query_vector,
            num_results=self.k,
        )
        # API returns result["result"]["data_array"] = list of rows (each row = list of values in column order)
        data_array = results.get("result", {}).get("data_array", [])
        cols = self.columns_to_return
        docs = []
        for row in data_array:
            if not isinstance(row, (list, tuple)) or len(row) != len(cols):
                continue
            row_dict = dict(zip(cols, row))
            text = row_dict.get("text") or row_dict.get("content") or ""
            meta = {k: v for k, v in row_dict.items() if k not in ("text", "content") and v is not None}
            docs.append(Document(page_content=text, metadata=meta))
        return docs


def get_databricks_retriever(
    index_name: str | None = None,
    workspace_url: str | None = None,
    personal_access_token: str | None = None,
    endpoint_name: str | None = None,
    openrouter_api_key: str | None = None,
    embed_model: str = DEFAULT_MODEL,
    k: int = 10,
) -> BaseRetriever:
    """Build a retriever that uses Databricks Vector Search + OpenRouter for query embedding."""
    index_name = index_name or os.environ.get("DATABRICKS_VECTOR_SEARCH_INDEX_NAME", "")
    workspace_url = workspace_url or os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    personal_access_token = personal_access_token or os.environ.get("DATABRICKS_TOKEN", "")
    endpoint_name = endpoint_name or os.environ.get("DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME")
    if not index_name or not workspace_url or not personal_access_token:
        raise ValueError(
            "For Databricks retriever set DATABRICKS_VECTOR_SEARCH_INDEX_NAME, DATABRICKS_HOST, DATABRICKS_TOKEN"
        )
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required for query embedding")
    embeddings = OpenRouterEmbeddings(api_key=api_key, model=embed_model)
    return DatabricksVectorSearchRetriever(
        index_name=index_name,
        workspace_url=workspace_url,
        personal_access_token=personal_access_token,
        endpoint_name=endpoint_name,
        embeddings=embeddings,
        k=k,
    )
