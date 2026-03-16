"""
Golden set for RAG evaluation: question / ground_truth pairs.
Used by evaluation/ragas_eval.py to run RAG and compute RAGAS metrics.
"""
from __future__ import annotations

# Each item: question (str), ground_truth (str). Contexts and answer are filled by the RAG pipeline.
GOLDEN_SET = [
    {
        "question": "What are common issues in debt collection complaints?",
        "ground_truth": "Common issues include attempts to collect debt not owed, harassment, and communication tactics. Complaints often mention repeated calls, disputes about debt ownership, and lack of validation.",
    },
    {
        "question": "How do consumers describe company responses to debt collection complaints?",
        "ground_truth": "Company responses vary; they may close with explanation, close with monetary relief, or close with non-monetary relief. Some consumers report no response or dispute resolution.",
    },
    {
        "question": "What product is most associated with debt collection complaints in the CFPB data?",
        "ground_truth": "Debt collection is the product category. Complaints are often categorized under sub-products and specific issues like attempts to collect debt not owed.",
    },
]
