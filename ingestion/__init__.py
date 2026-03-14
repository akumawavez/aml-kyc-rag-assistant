"""Ingestion pipeline: load CFPB + PDFs, chunk, embed via OpenRouter, store in Qdrant."""
from ingestion.loader import load_all, load_cfpb, load_regulatory_pdfs
from ingestion.chunker import chunk_documents
from ingestion.embedder import run_ingestion

__all__ = [
    "load_all",
    "load_cfpb",
    "load_regulatory_pdfs",
    "chunk_documents",
    "run_ingestion",
]
