"""Tests for RAG: prompts, reranker helper, chain with mocks (no OpenRouter/Qdrant)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.prompts import RAG_PROMPT
from rag.reranker import rerank_documents
from rag.chain import _format_docs, build_rag_chain
from rag.embeddings_openrouter import OpenRouterEmbeddings


def test_prompt_has_placeholders():
    # RAG_PROMPT expects "context" and "question"
    msg = RAG_PROMPT.invoke({"context": "Some context.", "question": "What?"})
    assert len(msg.messages) >= 1
    assert "Some context" in str(msg)
    assert "What?" in str(msg)


def test_format_docs():
    docs = [
        Document(page_content="First.", metadata={}),
        Document(page_content="Second.", metadata={}),
    ]
    s = _format_docs(docs)
    assert "First" in s and "Second" in s
    assert "---" in s


def test_rerank_documents_empty():
    assert rerank_documents("query", [], top_n=5) == []


def test_rerank_documents_service_down_returns_sublist():
    docs = [
        Document(page_content="A", metadata={}),
        Document(page_content="B", metadata={}),
    ]
    with patch("rag.reranker.requests.post") as mock_post:
        mock_post.side_effect = Exception("connection refused")
        out = rerank_documents("q", docs, top_n=1)
    assert len(out) == 1
    assert out[0].page_content == "A"


def test_openrouter_embeddings_requires_key():
    emb = OpenRouterEmbeddings(api_key="")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        emb.embed_query("test")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        emb.embed_documents(["test"])


def test_openrouter_embeddings_query_calls_api():
    with patch("rag.embeddings_openrouter._embed_batch") as mock_embed:
        mock_embed.return_value = [[0.1] * 256]  # fake vector
        emb = OpenRouterEmbeddings(api_key="sk-fake")
        vec = emb.embed_query("hello")
        assert vec == [0.1] * 256
        mock_embed.assert_called_once()
        assert mock_embed.call_args[0][0] == ["hello"]


def test_build_rag_chain_requires_api_key():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        with patch.dict("os.environ", {}, clear=True):
            build_rag_chain()


def test_build_rag_chain_invoke_mocked(monkeypatch):
    # Mock retriever to return fixed docs, LLM to return fixed answer
    from langchain_core.messages import AIMessage
    from rag import chain as chain_mod

    fake_docs = [Document(page_content="Context about debt collection.", metadata={})]
    with patch.object(chain_mod, "get_retriever") as mock_retriever_cls:
        mock_ret = MagicMock()
        mock_ret.invoke.return_value = fake_docs
        mock_retriever_cls.return_value = mock_ret
        with patch.object(chain_mod, "ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            msg = AIMessage(content="Based on the context, this is about debt.")
            mock_llm.invoke.return_value = msg
            mock_llm.return_value = msg  # in case chain calls llm(input) not llm.invoke(input)
            mock_llm_cls.return_value = mock_llm
            monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
            chain = build_rag_chain(use_reranker=False)
            out = chain.invoke({"question": "What is this about?"})
            assert "debt" in out.lower() or "context" in out.lower()
            mock_ret.invoke.assert_called_once_with("What is this about?")
