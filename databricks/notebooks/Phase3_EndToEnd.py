# Databricks notebook source
# MAGIC %md
# MAGIC # AML/KYC RAG – Phase 3 End-to-End (Databricks)
# MAGIC
# MAGIC This notebook runs the full Phase 3 pipeline on Databricks:
# MAGIC 1. **Delta ingest** – Load CFPB + regulatory data, chunk, embed, write to Delta.
# MAGIC 2. **Vector Search index** – Create endpoint and Delta Sync index; wait until Ready.
# MAGIC 3. **Sample RAG query** – Query the index via the RAG chain (Databricks retrieval).
# MAGIC 4. **RAGAS evaluation** – Run golden set through RAG; compute and log metrics to MLflow.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Cluster with PySpark, or attach to one.
# MAGIC - CFPB CSV at `dbfs:/FileStore/aml_kyc/cfpb_filtered.csv` (or set `INPUT_CFPB_PATH`).
# MAGIC - Secret scope `aml-kyc` with `OPENROUTER_API_KEY`, or set `OPENROUTER_API_KEY` in cluster env.
# MAGIC - Unity Catalog enabled; create catalog/schema if needed (e.g. `main.default`).

# COMMAND ----------

# MAGIC %pip install pandas pypdf requests langchain-core langchain langchain-openai langchain-community qdrant-client databricks-vectorsearch ragas datasets mlflow
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Set project root and env for repo layout (if code is in Repos or uploaded)
import os
import sys

# Project root: parent of 'databricks' folder. In Repos: /Workspace/Repos/<user>/<repo>.
# In notebook path .../databricks/notebooks/Phase3_EndToEnd.py, go up two levels.
try:
    _notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(_notebook_path), "..", ".."))
except Exception:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Optional: load OPENROUTER_API_KEY from secrets (scope aml-kyc)
try:
    os.environ["OPENROUTER_API_KEY"] = dbutils.secrets.get(scope="aml-kyc", key="OPENROUTER_API_KEY")
except Exception:
    pass  # use cluster env if already set

# Config (override via widget or env)
os.environ.setdefault("DELTA_TABLE_NAME", "main.default.aml_kyc_chunks")
os.environ.setdefault("DATABRICKS_VECTOR_SEARCH_INDEX_NAME", "main.default.aml_kyc_chunks_index")
os.environ.setdefault("DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME", "aml-kyc-vs-endpoint")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 – Delta ingest
# MAGIC Load CFPB CSV (and optional regulatory PDFs), chunk, embed with OpenRouter, write to Delta.

# COMMAND ----------

# Run delta_ingest: expects Spark, writes to DELTA_TABLE_NAME
import sys
sys.path.insert(0, _repo_root)

# So delta_ingest.main() gets correct args from argparse (it reads sys.argv)
_cfpb = os.environ.get("INPUT_CFPB_PATH", "/dbfs/FileStore/aml_kyc/cfpb_filtered.csv")
_delta = os.environ.get("DELTA_TABLE_NAME", "main.default.aml_kyc_chunks")
sys.argv = ["delta_ingest", "--cfpb-path", _cfpb, "--delta-table", _delta]

from databricks.delta_ingest import main as delta_ingest_main
delta_ingest_main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 – Enable Change Data Feed (required for Delta Sync index)
# MAGIC Run once per table.

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE IF EXISTS main.default.aml_kyc_chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 – Create Vector Search endpoint and index
# MAGIC Creates the endpoint (if missing) and a Delta Sync index; waits until the index is Ready.

# COMMAND ----------

import os
import sys
sys.path.insert(0, _repo_root)

from databricks.create_vector_index import EMBEDDING_DIMENSION

from databricks.vector_search.client import VectorSearchClient

delta_table = os.environ.get("DELTA_TABLE_NAME", "main.default.aml_kyc_chunks")
index_name = os.environ.get("DATABRICKS_VECTOR_SEARCH_INDEX_NAME", f"{delta_table}_index")
endpoint_name = os.environ.get("DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME", "aml-kyc-vs-endpoint")

client = VectorSearchClient()

if not client.endpoint_exists(name=endpoint_name):
    print(f"Creating endpoint {endpoint_name}...")
    client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    client.wait_for_endpoint(name=endpoint_name, verbose=True)
else:
    print(f"Endpoint {endpoint_name} already exists.")

if not client.index_exists(index_name=index_name):
    print(f"Creating Delta Sync index {index_name}...")
    client.create_delta_sync_index_and_wait(
        endpoint_name=endpoint_name,
        index_name=index_name,
        primary_key="id",
        source_table_name=delta_table,
        pipeline_type="TRIGGERED",
        embedding_dimension=EMBEDDING_DIMENSION,
        embedding_vector_column="embedding",
        columns_to_sync=["id", "text", "product", "complaint_id", "source", "issue", "company"],
        verbose=True,
    )
    print("Index is Ready.")
else:
    print(f"Index {index_name} already exists. Sync if needed via UI or index.sync().")

# COMMAND ----------

# Set Databricks workspace URL and token for RAG (when running in notebook; jobs set these in task env)
try:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    if not os.environ.get("DATABRICKS_HOST"):
        os.environ["DATABRICKS_HOST"] = ctx.apiUrl().get()
    if not os.environ.get("DATABRICKS_TOKEN"):
        os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()
except Exception:
    pass
os.environ.setdefault("DATABRICKS_VECTOR_SEARCH_INDEX_NAME", index_name)
os.environ.setdefault("DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME", endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 – Sample RAG query (Databricks retrieval)
# MAGIC Use the app's RAG chain with `VECTOR_BACKEND=databricks` to run one question.

# COMMAND ----------

import os
import sys
sys.path.insert(0, _repo_root)

os.environ["VECTOR_BACKEND"] = "databricks"

from rag.chain import ask_with_sources

question = "What are common issues in debt collection complaints?"
answer, sources, trace_url = ask_with_sources(question, use_reranker=False)
print("Question:", question)
print("Answer:", answer)
print("Sources count:", len(sources))
if trace_url:
    print("Trace:", trace_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 – RAGAS evaluation and MLflow logging
# MAGIC Run the golden set through RAG (Databricks Vector Search), compute RAGAS metrics, log to MLflow experiment `aml-kyc-rag-eval`.

# COMMAND ----------

import sys
sys.path.insert(0, _repo_root)

from databricks.ragas_eval_job import main as ragas_main

ragas_main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 – MLflow experiment summary
# MAGIC Open the experiment **`aml-kyc-rag-eval`** in MLflow to compare runs (params: vector_backend, num_questions; metrics: faithfulness, answer_relevancy, context_precision, context_recall).

# COMMAND ----------

import mlflow

exp = mlflow.get_experiment_by_name("aml-kyc-rag-eval")
if exp:
    print("MLflow experiment:", exp.name, "| ID:", exp.experiment_id)
    print("View runs in the MLflow UI (Experiments -> aml-kyc-rag-eval).")
else:
    print("Experiment aml-kyc-rag-eval not found; RAGAS job may not have logged yet.")
