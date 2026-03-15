"""RAG: retriever (Qdrant + OpenRouter), reranker, chain (LangChain)."""
from rag.chain import build_rag_chain, ask
from rag.retriever import get_qdrant_retriever
from rag.prompts import RAG_PROMPT
from rag.reranker import rerank_documents
from rag.embeddings_openrouter import OpenRouterEmbeddings

__all__ = [
    "build_rag_chain",
    "ask",
    "get_qdrant_retriever",
    "RAG_PROMPT",
    "rerank_documents",
    "OpenRouterEmbeddings",
]
