# AML/KYC RAG Assistant – Phase Plan and Completion Tests

This document defines the **implementation phases** and **concrete tests** (with commands and success criteria) so each phase is shippable and verifiable. **Knowledge graph (Neo4j/Graphiti) is out of scope for now**; the project focuses on RAG over CFPB + regulatory data with citations, hybrid retrieval, and evaluation.

---

## Overview

| Phase | Name | Summary |
|-------|------|---------|
| **1** | Foundation (MVP) | Data + ingestion, RAG chain (vector + optional BM25, reranker), Streamlit chat with citations, tests, CI, deployment docs. |
| **2** | Observability & evaluation | Langfuse tracing, RAGAS golden set and eval script, Prometheus/Grafana (optional), responsible-AI notes. |
| **3** | Optional: Azure Databricks | Delta Lake, Mosaic AI Vector Search, same app/eval approach; no KG. |

### Quick test reference

| Phase | Key tests / proof |
|-------|-------------------|
| **1** | `python -m ingestion.run` → verification prints "Verified: collection has N points". `streamlit run app/main.py`: chat, citations, hybrid retrieval. `pytest` and CI pass. `DEPLOYMENT.md` and `.env.example` exist. |
| **2** | Observability tab (tokens/cost, Langfuse link). RAGAS golden set + script; optional Prometheus `/metrics` and Grafana. |
| **3** | Delta ingest, Vector Search index, app or job using Databricks retrieval. |

---

## Phase 1 – Foundation (MVP)

**Outcome:** Ingestion pipeline, RAG with citations, single Streamlit app, tests, CI, and deployment documentation. No knowledge graph.

### Scope

- **Data:** CFPB filtered CSV + regulatory PDFs; loader, chunker, embedder (OpenRouter free embedding model); Qdrant as vector store.
- **RAG:** Hybrid retrieval (Qdrant vector + optional BM25), optional Cohere reranker, LLM via OpenRouter; prompts that constrain answers to retrieved context.
- **Single app entry point:** Streamlit app with chat, **citations** (sources per turn, expandable or visible).
- **Provider/model in sidebar (optional):** OpenRouter (and model dropdown) from `.env`; only show providers with keys set.
- **Tests:** Unit tests for ingestion (loader, chunker), RAG (prompts, reranker fallback, chain with mocks); ingestion verification (script exits 1 if 0 points).
- **CI/CD:** `.github/workflows/ci.yml` – lint and pytest.
- **Deployment:** `DEPLOYMENT.md`, `.env.example` with all required vars; notes for Streamlit Cloud, Hugging Face Spaces, or Render.

### Deliverables (evidence)

| Deliverable | Location / Evidence |
|-------------|---------------------|
| Ingestion pipeline | `ingestion/loader.py`, `chunker.py`, `embedder.py`, `run.py`; CFPB + PDFs → chunks → Qdrant |
| Ingestion verification | `python -m ingestion.run` ends with "Verified: collection 'aml_rag' has N points" or exits 1 if 0 |
| RAG chain | `rag/retriever.py`, `reranker.py`, `chain.py`, `prompts.py`; OpenRouter embeddings + LLM |
| Streamlit app | `app/main.py` – chat, citations (sources per turn), optional role selector |
| Unit tests | `tests/test_ingestion.py`, `tests/test_rag.py`; pytest passes |
| CI | `.github/workflows/ci.yml` – lint + pytest |
| Deployment docs | `DEPLOYMENT.md`; `.env.example` with OPENROUTER_API_KEY, Qdrant, optional Cohere |

### Prerequisites for Phase 1 tests

- Python env, `requirements.txt`, `.env` (OPENROUTER_API_KEY, QDRANT_HOST, QDRANT_PORT).
- Qdrant running (e.g. `docker compose up -d qdrant`).
- Data: `data/processed/cfpb_filtered.csv` and/or PDFs in `data/regulatory/`.

### Phase 1 completion tests

Run from **project root**.

#### Test 1.1 – Ingestion and verification

**Command:** `python -m ingestion.run` (optionally `--recreate`)

**Success criteria:**

- Exit 0; output shows "Loaded N documents", "X chunks", "Upserted M points", "Verified: collection 'aml_rag' has M points" with M > 0.
- If collection has 0 points after run, script exits 1 with error message.

**Proves:** Ingestion pipeline and Qdrant persistence are working.

---

#### Test 1.2 – Single app and chat with citations

**Command:** `streamlit run app/main.py` (or `app.py` if single file)

**Success criteria:**

- App starts without import/config error.
- Chat is the main interface; after a question, a response is shown.
- **Citations/sources** are visible (expandable source panel or inline) for the current or selected turn.

**Proves:** Single entry point and interactive citations.

---

#### Test 1.3 – RAG uses retrieval (hybrid + optional reranker)

**Command:** Use the app; ask a question that should use CFPB or regulatory content.

**Success criteria:**

- Response is grounded in retrieved context (citations point to complaint or regulatory chunks).
- Retrieval uses Qdrant (vector); optional BM25 and Cohere reranker when configured.

**Proves:** RAG chain and retrieval path.

---

#### Test 1.4 – Unit tests (pytest)

**Command:** `pytest tests/ -v`

**Success criteria:**

- Exit code 0.
- Tests exist for ingestion (loader, chunker) and RAG (prompts, reranker fallback, chain with mocks); optional ingestion integration test (Qdrant point count when running).

**Proves:** Test coverage (section 7).

---

#### Test 1.5 – CI/CD

**Command:** Run lint + test as in CI (e.g. `ruff check .`, `pytest`), or push and verify workflow.

**Success criteria:**

- `.github/workflows/ci.yml` exists; runs lint and pytest; passes on clean tree.

**Proves:** CI/CD.

---

#### Test 1.6 – Deployment documentation

**Evidence:**

- `DEPLOYMENT.md` exists: how to run the app, env vars, platform notes (e.g. Streamlit Cloud, HF Spaces, Render).
- `.env.example` lists all required variables (OPENROUTER_API_KEY, Qdrant, optional Cohere); no secrets.

**Proves:** Deployment (section 6).

---

### Phase 1 sign-off checklist

- [ ] Test 1.1 – Ingestion run and verification (N points > 0).
- [ ] Test 1.2 – Streamlit app runs; chat and citations visible.
- [ ] Test 1.3 – RAG retrieval and citations.
- [ ] Test 1.4 – pytest passes (ingestion + RAG).
- [ ] Test 1.5 – CI workflow (lint + test).
- [ ] Test 1.6 – DEPLOYMENT.md and .env.example.

**Phase 1 is complete when all items above are checked.**

---

## Phase 2 – Observability & evaluation

**Outcome:** Observability tab (tokens/cost, Langfuse link), RAGAS golden set and eval script, optional Prometheus/Grafana; responsible-AI notes.

### Scope

- **Observability tab:** Langfuse callback in RAG chain; token consumption and cost estimation; link(s) to Langfuse.
- **RAG evaluation:** Golden dataset (question/answer/context triples); RAGAS script (faithfulness, answer relevance, context precision/recall); scores logged or reported.
- **Optional:** Prometheus `/metrics` (latency, retrieval count, errors); Grafana dashboard for RAG; document setup.
- **Optional:** Short responsible-AI doc (traceability, transparency, limitations).

### Deliverables (evidence)

| Deliverable | Location / Evidence |
|-------------|---------------------|
| Observability tab | Token/cost KPIs; Langfuse link |
| Golden set | `evaluation/test_set.py` or similar – Q/A/context triples |
| RAGAS script | `evaluation/ragas_eval.py` – runs RAGAS, outputs scores |
| Optional /metrics | Prometheus-compatible endpoint; Grafana doc |
| Optional RESPONSIBLE_AI | `docs/RESPONSIBLE_AI.md` or section in README |

### Phase 2 completion tests

#### Test 2.1 – Observability tab

**Command:** Open Observability tab in the app.

**Success criteria:** Tab shows token consumption and cost (or link to Langfuse where these are visible). Langfuse callback is wired to the RAG chain.

**Proves:** Observability tab.

---

#### Test 2.2 – RAGAS evaluation

**Command:** `python -m evaluation.ragas_eval` (or `scripts/run_ragas.py`)

**Success criteria:** Golden set exists; script runs and produces RAGAS scores (e.g. faithfulness, answer relevance); results logged or printed.

**Proves:** RAG evals pipeline.

---

#### Test 2.3 – Prometheus / Grafana (optional)

**Evidence:** If implemented: `/metrics` returns Prometheus format; Grafana dashboard or doc for RAG (latency, retrieval, errors).

**Proves:** Optional monitoring.

---

### Phase 2 sign-off checklist

- [ ] Test 2.1 – Observability tab (tokens, cost, Langfuse).
- [ ] Test 2.2 – RAGAS golden set and script.
- [ ] Test 2.3 – Optional /metrics and Grafana.

**Phase 2 is complete when all implemented items are checked.**

---

## Phase 3 – Optional: Azure Databricks

**Outcome:** Data and retrieval can run on Azure Databricks (Delta Lake, Mosaic AI Vector Search); same app and evaluation approach. No knowledge graph.

### Scope

- **Data:** Ingest CFPB (and regulatory) into Delta Lake; optional Change Data Feed for vector index sync.
- **Embeddings:** Compute embeddings in Databricks (Azure OpenAI or Foundation Model APIs); write to Delta with vector column.
- **Vector Search:** Mosaic AI Vector Search index from Delta; hybrid (vector + keyword) where supported.
- **App:** Use same Streamlit + RAG logic; retriever backend switches to Databricks Vector Search via config; secrets in Databricks + Key Vault.
- **Eval:** Same RAGAS flow run as job or notebook; log to MLflow if desired.

### Deliverables (evidence)

| Deliverable | Location / Evidence |
|-------------|---------------------|
| Delta ingest | Job or notebook: CSV/PDFs → Delta tables |
| Vector index | Mosaic AI Vector Search index from Delta (with embeddings) |
| Retriever backend | Config or adapter to query Databricks Vector Search instead of Qdrant |
| Secrets | Databricks secrets (Key Vault) for API keys |
| Eval on Databricks | RAGAS run as job; optional MLflow logging |

### Phase 3 sign-off checklist

- [ ] Delta tables populated; Vector Search index created and queryable.
- [ ] App (or Databricks-hosted Streamlit) uses Databricks retrieval.
- [ ] RAGAS (or equivalent) runs on Databricks; results visible.

**Phase 3 is optional and complete when the above are done.**

---

## Run steps (after Phase 1)

1. **Environment:** Copy `.env.example` to `.env`; set `OPENROUTER_API_KEY`, optionally `QDRANT_HOST`, `QDRANT_PORT`, `COHERE_API_KEY`.
2. **Install:** `pip install -r requirements.txt`
3. **Services:** `docker compose up -d qdrant` (and optional reranker if using Cohere).
4. **Data:** Ensure `data/processed/cfpb_filtered.csv` and/or `data/regulatory/*.pdf` exist.
5. **Ingest:** `python -m ingestion.run` (or `--recreate` to reset). Expect "Verified: collection 'aml_rag' has N points."
6. **App:** `streamlit run app/main.py`
7. **Tests:** `pytest tests/ -v`
8. **CI:** Push and confirm `.github/workflows/ci.yml` passes.

---

## Document history

- Phases aligned to foundation (RAG + app + tests + CI + deployment), observability & evaluation, optional Databricks.
- **Knowledge graph (Neo4j/Graphiti) is not part of the current phases;** can be added later as a separate phase if needed.
- Each phase has concrete tests and success criteria for sign-off.
