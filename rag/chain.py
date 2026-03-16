"""
RAG chain: retriever (Qdrant) -> optional rerank -> format context -> LLM (OpenRouter).
"""
from __future__ import annotations

import os
from typing import Any, List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag.retriever import get_qdrant_retriever
from rag.reranker import rerank_documents
from rag.prompts import RAG_PROMPT

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    *,
    collection_name: str = "aml_rag",
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    openrouter_api_key: str | None = None,
    embed_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    llm_model: str = "google/gemma-2-9b-it:free",
    retriever_k: int = 10,
    rerank_top_n: int = 5,
    use_reranker: bool = True,
):
    """Build RAG chain: retrieve -> rerank (optional) -> LLM."""
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required")

    retriever = get_qdrant_retriever(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        openrouter_api_key=api_key,
        embed_model=embed_model,
        k=retriever_k,
    )

    llm = ChatOpenAI(
        model=llm_model,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE,
        temperature=0,
    )

    def retrieve_and_rerank(query: str) -> str:
        docs = retriever.invoke(query)
        if use_reranker and docs:
            docs = rerank_documents(query, docs, top_n=rerank_top_n)
        return _format_docs(docs)

    chain = (
        RunnablePassthrough.assign(context=lambda x: retrieve_and_rerank(x["question"]))
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, **kwargs: Any) -> str:
    """One-shot RAG query. Uses build_rag_chain with default or kwargs."""
    chain = build_rag_chain(**kwargs)
    return chain.invoke({"question": question})


def _get_langfuse_handler():
    """Return Langfuse callback handler if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set."""
    pk = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    sk = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    if not pk or not sk:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception:
        return None


def ask_with_sources(
    question: str,
    *,
    collection_name: str = "aml_rag",
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    openrouter_api_key: str | None = None,
    embed_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    llm_model: str = "google/gemma-2-9b-it:free",
    retriever_k: int = 10,
    rerank_top_n: int = 5,
    use_reranker: bool = True,
    **kwargs: Any,
) -> Tuple[str, List[Document], str | None]:
    """Run RAG and return (answer, source documents, trace_url or None) for citations and observability."""
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required")

    retriever = get_qdrant_retriever(
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        openrouter_api_key=api_key,
        embed_model=embed_model,
        k=retriever_k,
    )

    docs = retriever.invoke(question)
    if use_reranker and docs:
        docs = rerank_documents(question, docs, top_n=rerank_top_n)

    if not docs:
        return (
            "I couldn't find any relevant context to answer that. Try rephrasing or ask about complaints or regulatory content.",
            [],
            None,
        )

    llm = ChatOpenAI(
        model=llm_model,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE,
        temperature=0,
    )
    context_str = _format_docs(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()

    langfuse_handler = _get_langfuse_handler()
    invoke_kwargs: dict = {"context": context_str, "question": question}
    config: dict = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    answer = chain.invoke(invoke_kwargs, config=config or None)

    trace_url = None
    if langfuse_handler:
        try:
            from langfuse import get_client
            get_client().flush()
            if hasattr(langfuse_handler, "get_trace_url"):
                trace_url = langfuse_handler.get_trace_url()
            elif hasattr(langfuse_handler, "trace_id") and langfuse_handler.trace_id:
                base = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
                trace_url = f"{base}/trace/{langfuse_handler.trace_id}"
        except Exception:
            pass

    return answer, docs, trace_url
