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

## CI/CD

The repo includes `.github/workflows/ci.yml`: on push/PR to main/master it runs `ruff check .` and `pytest tests/ -v`. No secrets are required for CI; tests use mocks and do not call OpenRouter or Qdrant.
