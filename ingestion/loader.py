"""
Load CFPB complaint CSV and regulatory PDFs into a unified document list with text + metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# PDF loading
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore[misc, assignment]


def _build_cfpb_text(row: pd.Series) -> str:
    """Build searchable text from a CFPB row (narrative + structured fields)."""
    parts = []
    if pd.notna(row.get("consumer_complaint_narrative")) and str(row["consumer_complaint_narrative"]).strip():
        parts.append(str(row["consumer_complaint_narrative"]).strip())
    # Always add structured context so we have something to embed when narrative is empty
    structured = []
    for col in ["product", "sub_product", "issue", "sub_issue", "company", "company_response_to_consumer"]:
        if col in row.index and pd.notna(row.get(col)) and str(row[col]).strip():
            structured.append(str(row[col]).strip())
    if structured:
        parts.append(" | ".join(structured))
    return "\n\n".join(parts) if parts else ""


def load_cfpb(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load CFPB filtered CSV into documents with text and metadata."""
    path = Path(csv_path)
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False, on_bad_lines="warn")
    docs = []
    for _, row in df.iterrows():
        text = _build_cfpb_text(row)
        if not text.strip():
            continue
        docs.append({
            "text": text,
            "metadata": {
                "source": "cfpb",
                "product": str(row.get("product", "")).strip() or None,
                "issue": str(row.get("issue", "")).strip() or None,
                "sub_issue": str(row.get("sub_issue", "")).strip() or None,
                "date": str(row.get("date_received", "")).strip() or None,
                "company": str(row.get("company", "")).strip() or None,
                "state": str(row.get("state", "")).strip() or None,
                "complaint_id": str(row.get("complaint_id", "")).strip() or None,
                "company_response_to_consumer": str(row.get("company_response_to_consumer", "")).strip() or None,
            },
        })
    return docs


def load_regulatory_pdfs(dir_path: str | Path) -> list[dict[str, Any]]:
    """Load all PDFs from a directory into documents (one doc per page or whole doc)."""
    if PdfReader is None:
        raise ImportError("Install pypdf: pip install pypdf")
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []
    docs = []
    for pdf_path in sorted(dir_path.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not (text and text.strip()):
                    continue
                docs.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": "regulatory",
                        "file": pdf_path.name,
                        "page": i + 1,
                    },
                })
        except Exception as e:
            # Log and skip broken PDFs
            import warnings
            warnings.warn(f"Skip {pdf_path}: {e}", UserWarning)
    return docs


def load_all(
    data_dir: str | Path,
    cfpb_rel: str = "processed/cfpb_filtered.csv",
    regulatory_rel: str = "regulatory",
) -> list[dict[str, Any]]:
    """Load CFPB and regulatory docs from data_dir. Returns list of {text, metadata}."""
    data_dir = Path(data_dir)
    out: list[dict[str, Any]] = []
    cfpb_path = data_dir / cfpb_rel
    out.extend(load_cfpb(cfpb_path))
    reg_path = data_dir / regulatory_rel
    out.extend(load_regulatory_pdfs(reg_path))
    return out
