# Phase 3 – Azure Databricks

This folder contains scripts and notebooks for running the AML/KYC RAG pipeline on **Azure Databricks**: Delta Lake storage, Mosaic AI Vector Search, and optional RAGAS evaluation as a job.

**Prerequisites:** Databricks workspace (Unity Catalog, serverless compute), Azure Key Vault (optional, for secrets).

---

## Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `delta_ingest.py` | Ingest CFPB + regulatory data → chunk → embed → write to Delta table |
| 2 | `create_vector_index.py` | Create Vector Search endpoint and Delta Sync index from the Delta table |
| 3 | App / RAG | Set `VECTOR_BACKEND=databricks` and Databricks env vars; app uses Databricks retrieval |
| 4 | `ragas_eval_job.py` | Run RAGAS evaluation on Databricks; optional MLflow logging |

---

## Secrets (Databricks + Key Vault)

Store API keys in Databricks secrets so jobs and notebooks can read them without hardcoding.

1. **Create a secret scope** (e.g. backed by Azure Key Vault):
   - In Databricks: **Settings → Secret scopes → Create**.
   - Or CLI: `databricks secrets create-scope --scope aml-kyc --scope-backend-type AZURE_KEYVAULT --resource-id <key-vault-resource-id> --dns-name <vault-name>.vault.azure.net`

2. **Store secrets** in the scope (or in Databricks-backed scope):
   - `OPENROUTER_API_KEY` – for embeddings and LLM (or use Azure OpenAI secrets below).
   - `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` – optional, if using Azure OpenAI for embeddings/LLM.

3. **In jobs/notebooks**, read via `dbutils.secrets.get(scope="aml-kyc", key="OPENROUTER_API_KEY")`.

---

## Environment variables (for app when using Databricks)

When running the Streamlit app or RAG locally but querying Databricks Vector Search, set:

- `VECTOR_BACKEND=databricks`
- `DATABRICKS_HOST=https://<workspace>.azuredatabricks.net`
- `DATABRICKS_TOKEN=<personal-access-token>` (or use Azure CLI / service principal)
- `DATABRICKS_VECTOR_SEARCH_INDEX_NAME=<catalog>.<schema>.<index_name>`

Optional: `DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME` if different from default.

---

## Running the ingest job

1. Upload or clone this repo to the Databricks workspace (e.g. Repos).
2. Upload CFPB CSV to DBFS or a path accessible to the cluster (e.g. `dbfs:/FileStore/aml_kyc/cfpb_filtered.csv`).
3. In a **notebook** or **job**:
   - Install dependencies: `%pip install pandas pypdf requests`
   - Run `delta_ingest.py` (or copy its logic into a notebook). Configure:
     - `INPUT_CFPB_PATH`, `INPUT_REGULATORY_PATH` (or use defaults).
     - `DELTA_TABLE_NAME` (e.g. `main.default.aml_kyc_chunks`).
     - Embedding: set `OPENROUTER_API_KEY` from secrets, or use Azure OpenAI.
4. After the Delta table is populated, run `create_vector_index.py` to create the Vector Search endpoint and index.

---

## Creating the Vector Search index

- Run `create_vector_index.py` on a cluster with `databricks-vectorsearch` installed.
- It creates (or reuses) an endpoint and a **Delta Sync** index on the table written by `delta_ingest.py`.
- The index must be in **Ready** state before the app or RAGAS job can query it.

---

## RAGAS evaluation job

- Run `ragas_eval_job.py` as a Databricks job (or in a notebook).
- It uses the same golden set as `evaluation/golden_set.py`, runs retrieval via Databricks Vector Search and your LLM, then computes RAGAS metrics.
- Optional: log scores to MLflow with `mlflow.log_metrics(result)`.
