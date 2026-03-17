"""
Run RAGAS evaluation on the golden set.
Usage: python -m evaluation.ragas_eval
Requires: OPENROUTER_API_KEY, Qdrant running, ingestion done. Optional: ragas, datasets, mlflow.
Logs metrics (and optional params) to MLflow experiment "aml-kyc-rag-eval" when mlflow is available.
"""
from __future__ import annotations

import os
import sys

# Optional deps for RAGAS
try:
    from datasets import Dataset
except ImportError:
    print("Install datasets: pip install datasets", file=sys.stderr)
    sys.exit(1)

try:
    from ragas import evaluate
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
except ImportError:
    print("Install ragas: pip install ragas", file=sys.stderr)
    sys.exit(1)

from evaluation.golden_set import GOLDEN_SET
from rag.chain import ask_with_sources


def main() -> None:
    if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print("Running RAG on golden set...")
    for i, row in enumerate(GOLDEN_SET):
        q = row["question"]
        gt = row["ground_truth"]
        try:
            answer, docs, _ = ask_with_sources(q, use_reranker=bool((os.environ.get("COHERE_API_KEY") or "").strip()))
            ctx_list = [d.page_content for d in docs]
        except Exception as e:
            print(f"  Row {i+1} failed: {e}", file=sys.stderr)
            answer = ""
            ctx_list = []

        questions.append(q)
        ground_truths.append(gt)
        answers.append(answer)
        contexts.append(ctx_list)

    if not any(contexts):
        print("Error: No contexts retrieved. Run ingestion and ensure Qdrant is up.", file=sys.stderr)
        sys.exit(1)

    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts,
    })

    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        show_progress=True,
    )

    print("\nRAGAS scores:")
    for name, score in result.items():
        print(f"  {name}: {score}")

    # Log to MLflow (experiment aml-kyc-rag-eval) when available
    _log_ragas_to_mlflow(
        result=result,
        params={
            "vector_backend": os.environ.get("VECTOR_BACKEND", "qdrant"),
            "use_reranker": bool((os.environ.get("COHERE_API_KEY") or "").strip()),
            "num_questions": len(GOLDEN_SET),
        },
    )
    print()


def _log_ragas_to_mlflow(result: dict, params: dict | None = None) -> None:
    """Log RAGAS metrics (and optional params) to MLflow experiment aml-kyc-rag-eval."""
    try:
        import mlflow
        exp_name = "aml-kyc-rag-eval"
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
        with mlflow.start_run():
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            metrics = {k: float(v) for k, v in result.items() if v is not None and str(v) != "nan"}
            if metrics:
                mlflow.log_metrics(metrics)
            print("Logged metrics and params to MLflow.")
    except Exception:
        # MLflow optional: no run, not installed, or tracking server down
        pass


if __name__ == "__main__":
    main()
