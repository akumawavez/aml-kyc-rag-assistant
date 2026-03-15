"""Tests for ingestion: loader, chunker (no external API)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ingestion.loader import load_cfpb, load_regulatory_pdfs, load_all, _build_cfpb_text
from ingestion.chunker import chunk_text, chunk_documents
import pandas as pd


# --- Loader ---


def test_build_cfpb_text_empty_row():
    row = pd.Series({})
    assert _build_cfpb_text(row) == ""


def test_build_cfpb_text_narrative_only():
    row = pd.Series({"consumer_complaint_narrative": "I was charged twice."})
    assert "I was charged twice" in _build_cfpb_text(row)


def test_build_cfpb_text_structured_only():
    row = pd.Series({
        "product": "Debt collection",
        "issue": "Attempts to collect debt not owed",
        "company": "ACME Corp",
    })
    t = _build_cfpb_text(row)
    assert "Debt collection" in t and "ACME Corp" in t


def test_load_cfpb_missing_file():
    assert load_cfpb(Path("/nonexistent/file.csv")) == []


def test_load_cfpb_small_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date_received,product,sub_product,issue,sub_issue,consumer_complaint_narrative,company,state,complaint_id\n")
        f.write("2020-01-01,Debt collection,Other,Attempts to collect,Debt not yours,,ACME,FL,1\n")
        f.write("2020-01-02,Debt collection,Other,Harassment,,Narrative here.,Beta Inc,CA,2\n")
        path = f.name
    try:
        docs = load_cfpb(path)
        assert len(docs) == 2
        assert docs[0]["metadata"]["product"] == "Debt collection"
        assert docs[0]["metadata"]["complaint_id"] == "1"
        assert "ACME" in docs[0]["text"]
        assert "Narrative here" in docs[1]["text"]
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_regulatory_pdfs_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        assert load_regulatory_pdfs(d) == []


def test_load_all_empty_data_dir():
    with tempfile.TemporaryDirectory() as d:
        # No processed CSV and no PDFs
        docs = load_all(d)
        assert docs == []


# --- Chunker ---


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_text_splits_with_overlap():
    text = " ".join("word" for _ in range(500))  # long string
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) <= 100 + 20 for c in chunks)


def test_chunk_documents_preserves_metadata():
    docs = [
        {"text": "Short.", "metadata": {"source": "cfpb", "product": "Debt collection"}},
    ]
    out = chunk_documents(docs, chunk_size=2000, overlap=200)
    assert len(out) == 1
    assert out[0]["metadata"]["source"] == "cfpb"
    assert out[0]["metadata"]["product"] == "Debt collection"
    assert out[0]["text"] == "Short."


def test_chunk_documents_splits_long():
    docs = [
        {"text": "x " * 500, "metadata": {"source": "test"}},
    ]
    out = chunk_documents(docs, chunk_size=100, overlap=10)
    assert len(out) >= 2
    for o in out:
        assert o["metadata"]["source"] == "test"
