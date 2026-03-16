"""
AML/KYC RAG Assistant – Streamlit chat with citations.
Run: streamlit run app/main.py
"""
from __future__ import annotations

import os
from typing import Any

import streamlit as st

from rag.chain import ask_with_sources

# Page config
st.set_page_config(page_title="AML/KYC RAG Assistant", layout="wide")

# Sidebar: provider/model when OpenRouter key is set
openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
if openrouter_key:
    with st.sidebar:
        st.subheader("Model")
        llm_model = st.selectbox(
            "OpenRouter model",
            options=[
                "google/gemma-2-9b-it:free",
                "meta-llama/llama-3.2-3b-instruct:free",
                "mistralai/mistral-7b-instruct:free",
            ],
            index=0,
            key="llm_model",
        )
else:
    llm_model = "google/gemma-2-9b-it:free"

# Session state for chat history and observability
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace_url" not in st.session_state:
    st.session_state.last_trace_url = None

# Tabs: Chat and Observability
tab_chat, tab_observability = st.tabs(["Chat", "Observability"])

langfuse_configured = bool(
    (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    and (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
)
langfuse_host = (os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")

def _source_label(meta: dict) -> str:
    label_parts = []
    if meta.get("product"):
        label_parts.append(f"Product: {meta['product']}")
    if meta.get("complaint_id"):
        label_parts.append(f"Complaint ID: {meta['complaint_id']}")
    if meta.get("source") and meta["source"] != "cfpb":
        label_parts.append(f"Source: {meta['source']}")
    return " | ".join(label_parts) if label_parts else "Source"


def _render_sources(sources: list[dict[str, Any]]) -> None:
    for i, doc in enumerate(sources, 1):
        meta = doc.get("metadata") or {}
        label = _source_label(meta)
        if label == "Source":
            label = f"Source {i}"
        st.markdown(f"**{label}**")
        content = doc.get("page_content", "")
        st.text(content[:500] + ("..." if len(content) > 500 else ""))


with tab_chat:
    st.title("AML/KYC RAG Assistant")
    st.caption("Ask about CFPB complaints and regulatory content. Answers are grounded in retrieved context with citations.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources / citations", expanded=False):
                    _render_sources(msg["sources"])

    if prompt := st.chat_input("Ask a question about complaints or regulations..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    answer, sources, trace_url = ask_with_sources(
                        prompt,
                        llm_model=llm_model,
                        use_reranker=bool((os.environ.get("COHERE_API_KEY") or "").strip()),
                    )
                    if trace_url:
                        st.session_state.last_trace_url = trace_url
                except ValueError as e:
                    answer = f"Configuration error: {e}. Set OPENROUTER_API_KEY in .env."
                    sources = []
                except Exception as e:
                    answer = f"Error: {e}. Check Qdrant is running (docker compose up -d qdrant) and ingestion has been run."
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander("Sources / citations", expanded=True):
                    _render_sources([{"page_content": d.page_content, "metadata": d.metadata} for d in sources])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [{"page_content": d.page_content, "metadata": d.metadata} for d in sources],
            })

with tab_observability:
    st.subheader("Observability")
    if langfuse_configured:
        st.markdown("**Langfuse** is configured. Token consumption and cost for each RAG request are recorded in Langfuse.")
        st.markdown(f"- **Dashboard:** [{langfuse_host}]({langfuse_host})")
        if st.session_state.last_trace_url:
            st.markdown(f"- **Last trace:** [{st.session_state.last_trace_url}]({st.session_state.last_trace_url})")
        st.markdown("Open your Langfuse project to view traces, token usage, and cost estimates.")
    else:
        st.markdown("To enable observability, set **LANGFUSE_PUBLIC_KEY** and **LANGFUSE_SECRET_KEY** in your `.env` (see `.env.example`).")
        st.markdown(f"Then use [Langfuse]({langfuse_host}) to view traces, token usage, and costs.")
