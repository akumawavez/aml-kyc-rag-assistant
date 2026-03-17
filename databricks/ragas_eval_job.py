"""
RAGAS evaluation job for Databricks.
Run as a Databricks job with VECTOR_BACKEND=databricks and Databricks env vars set.
Optionally logs metrics to MLflow.

Usage:
  - In job config: set env VECTOR_BACKEND=databricks, DATABRICKS_HOST, DATABRICKS_TOKEN,
    DATABRICKS_VECTOR_SEARCH_INDEX_NAME, OPENROUTER_API_KEY (or from secrets).
  - Run this script as the job entry point, or call main() from a notebook.
"""
from __future__ import annotations

import os
import sys

# Project root for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _get_secret(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if v:
        return v
    try:
        import dbutils
        scope = os.environ.get("DATABRICKS_SECRET_SCOPE", "aml-kyc")
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return ""


def main() -> None:
    # Ensure Databricks retriever is used and API key is available
    os.environ["VECTOR_BACKEND"] = "databricks"
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        api_key = _get_secret("OPENROUTER_API_KEY")
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
    if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        print("Error: OPENROUTER_API_KEY not set and not in secrets.", file=sys.stderr)
        sys.exit(1)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError as e:
        print(f"Install ragas and datasets: {e}", file=sys.stderr)
        sys.exit(1)

    from evaluation.golden_set import GOLDEN_SET
    from rag.chain import ask_with_sources

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print("Running RAG on golden set (Databricks Vector Search)...")
    for i, row in enumerate(GOLDEN_SET):
        q = row["question"]
        gt = row["ground_truth"]
        try:
            answer, docs, _ = ask_with_sources(q, use_reranker=False)
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
        print("Error: No contexts retrieved. Check Vector Search index and env.", file=sys.stderr)
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

    # Log to MLflow (experiment aml-kyc-rag-eval); start run if not already in one (e.g. job UI)
    _log_ragas_to_mlflow(
        result=result,
        params={
            "vector_backend": "databricks",
            "use_reranker": False,
            "num_questions": len(GOLDEN_SET),
        },
    )
    print()


def _log_ragas_to_mlflow(result: dict, params: dict | None = None) -> None:
    """Log RAGAS metrics and params to MLflow experiment aml-kyc-rag-eval."""
    try:
        import mlflow
        exp_name = "aml-kyc-rag-eval"
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                mlflow.create_experiment(exp_name)
        except Exception:
            pass
        mlflow.set_experiment(exp_name)
        started_run = mlflow.active_run() is None
        if started_run:
            mlflow.start_run()
        try:
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            metrics = {k: float(v) for k, v in result.items() if v is not None and str(v) != "nan"}
            if metrics:
                mlflow.log_metrics(metrics)
            print("Logged metrics and params to MLflow.")
        finally:
            if started_run and mlflow.active_run() is not None:
                mlflow.end_run()
    except Exception:
        pass


if __name__ == "__main__":
    main()
