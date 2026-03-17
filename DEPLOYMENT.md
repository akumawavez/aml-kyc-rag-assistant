# Deployment guide – AML/KYC RAG Assistant

This document describes how to run the app locally and deploy it to common platforms.

---

## Local run

### 1. Environment

- Copy `.env.example` to `.env` and set at least:
  - **`OPENROUTER_API_KEY`** (required) – [openrouter.ai/keys](https://openrouter.ai/keys)
- Optional: `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION` (defaults: localhost, 6333, aml_rag); `COHERE_API_KEY` for reranker.

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Services

Start Qdrant (required for ingestion and RAG):

```bash
docker compose up -d qdrant
```

Optional: Cohere reranker service (see [DOCKER.md](DOCKER.md)).

### 4. Data and ingestion

- Ensure `data/processed/cfpb_filtered.csv` exists (see [README](README.md) – run `scripts/download_and_filter_cfpb.py` if needed).
- Optional: add PDFs under `data/regulatory/`.

Run ingestion:

```bash
python -m ingestion.run
```

Use `python -m ingestion.run --recreate` to wipe and refill the vector collection.

### 5. App

From the project root:

```bash
streamlit run app/main.py
```

Open the URL shown in the terminal (e.g. http://localhost:8501).

---

## Environment variables summary

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for embeddings and LLM |
| `QDRANT_HOST` | No | Default: localhost |
| `QDRANT_PORT` | No | Default: 6333 |
| `QDRANT_COLLECTION` | No | Default: aml_rag |
| `COHERE_API_KEY` | No | Enables reranker when set (and reranker service running) |

See [.env.example](.env.example) for optional vars (Langfuse, Neo4j, etc.).

---

## Platform notes

### Streamlit Community Cloud

1. Connect the repo and set the run command to: `streamlit run app/main.py`.
2. Add **Secrets** in the app settings:
   - `OPENROUTER_API_KEY`
   - Optionally `QDRANT_HOST`, `QDRANT_PORT` if using a hosted Qdrant.
3. For a hosted Qdrant, ensure the URL is reachable from Streamlit’s runners. For a local Qdrant, use a tunnel (e.g. ngrok) and set `QDRANT_HOST`/`QDRANT_PORT` accordingly, or run the app locally only.

### Hugging Face Spaces

1. Create a new Space, choose **Streamlit**.
2. In “Repository contents” include this repo (or copy `app/`, `rag/`, `ingestion/`, `requirements.txt`, etc.).
3. Set `OPENROUTER_API_KEY` (and optional Qdrant/Cohere vars) in **Settings → Variables and secrets**.
4. In “App file” use `app/main.py` and run command: `streamlit run app/main.py --server.port 7860`.
5. As with Streamlit Cloud, Qdrant must be reachable (hosted or tunnel).

### Render

1. New **Web Service**; connect repo.
2. Build: `pip install -r requirements.txt` (or use a Dockerfile).
3. Start command: `streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0`.
4. Add env vars in Render dashboard: `OPENROUTER_API_KEY`, and Qdrant/Cohere if used.
5. Ensure Qdrant is reachable from Render (hosted DB or tunnel).

---

## Phase 3 – Databricks

### End-to-end proof (one notebook)

Run the **Phase 3 end-to-end notebook** to execute the full pipeline on Databricks and satisfy “build a complete end-to-end solution using Databricks”:

1. In Databricks, open **`databricks/notebooks/Phase3_EndToEnd.py`** (from Repos or after uploading the repo).
2. Ensure CFPB CSV is at `dbfs:/FileStore/aml_kyc/cfpb_filtered.csv` (or set `INPUT_CFPB_PATH`), and set `OPENROUTER_API_KEY` (cluster env or secret scope `aml-kyc`).
3. Run all cells in order. The notebook will:
   - Ingest CFPB data into a Delta table
   - Create the Vector Search endpoint and Delta Sync index (and wait until Ready)
   - Run a sample RAG query using Databricks retrieval
   - Run RAGAS evaluation and log metrics to MLflow experiment **`aml-kyc-rag-eval`**
4. In **MLflow → Experiments → aml-kyc-rag-eval**, confirm runs and metrics (faithfulness, answer_relevancy, context_precision, context_recall).

### Using Databricks retrieval from the app

To use **Databricks Mosaic AI Vector Search** instead of Qdrant from the Streamlit app:

1. **On Databricks:** Run the ingest job (`databricks/delta_ingest.py`), then create the Vector Search index (`databricks/create_vector_index.py`). See [databricks/README.md](databricks/README.md) for secrets (e.g. Azure Key Vault), Delta table name, and index creation.
2. **For the app:** Set in `.env`:
   - `VECTOR_BACKEND=databricks`
   - `DATABRICKS_HOST=https://<workspace>.azuredatabricks.net`
   - `DATABRICKS_TOKEN=<personal-access-token>`
   - `DATABRICKS_VECTOR_SEARCH_INDEX_NAME=<catalog>.<schema>.<index_name>`
3. Run the app as usual; retrieval will query the Databricks index. RAGAS evaluation can run as a Databricks job (`databricks/ragas_eval_job.py`) with optional MLflow logging.

---

## MLflow (RAGAS evaluation)

When you run RAGAS evaluation (locally with `python -m evaluation.ragas_eval` or on Databricks with `databricks/ragas_eval_job.py`), metrics and run parameters are logged to MLflow when the library is available:

- **Experiment name:** `aml-kyc-rag-eval`
- **Logged metrics:** RAGAS scores (faithfulness, answer_relevancy, context_precision, context_recall)
- **Logged params:** e.g. `vector_backend`, `use_reranker`, `num_questions`

Use the MLflow UI (local or Databricks) to compare runs. If MLflow is not installed or no tracking server is configured, logging is skipped and the eval still completes.

---

## CI/CD

The repo includes `.github/workflows/ci.yml`: on push/PR to main/master it runs `ruff check .` and `pytest tests/ -v`. No secrets are required for CI; tests use mocks and do not call OpenRouter or Qdrant.
