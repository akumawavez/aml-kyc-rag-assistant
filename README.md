# AML/KYC RAG Assistant

**Repo:** `aml-kyc-rag-assistant`

A **retrieval-augmented GenAI assistant** for AML/KYC analysts: explore and summarize financial crime risk signals from consumer complaint narratives and regulatory text. Ask case-style, pattern, and policy questions and get **cited answers** from CFPB complaints and AML/regulatory documents. (Knowledge graph is out of scope for the current phases; see [PHASES.md](PHASES.md).)

---

## What it does

- **Case-style questions** — Summarize complaint history, red flags, company responses; generate executive summaries for case files.
- **Pattern and trend questions** — Top issues by product/company, emerging themes (e.g. crypto, fraud), under-remediation risk.
- **Policy and regulatory questions** — Answers grounded in FATF, CFPB, FinCEN-style guidance (when regulatory PDFs are loaded).

The system uses the **CFPB Consumer Complaint Database** (filtered by product), optional **AML/regulatory PDFs**, **hybrid retrieval** (vector + optional keyword), **reranking** (e.g. Cohere), and an **LLM** (OpenRouter/OpenAI) with citations.

---

## Phases (summary)

| Phase | Focus | Proof |
|-------|--------|--------|
| **1** | Foundation | Ingestion → Qdrant; RAG chain; Streamlit chat + citations; tests; CI; DEPLOYMENT.md |
| **2** | Observability & evaluation | Langfuse; RAGAS golden set + script; optional Prometheus/Grafana |
| **3** | Optional: Databricks | Delta Lake; Mosaic AI Vector Search; same app/eval |

**Full phase plan, deliverables, and completion tests:** [PHASES.md](PHASES.md).

---

## Tech stack

| Layer | Choices |
|-------|--------|
| **Data** | CFPB CSV, regulatory PDFs; Delta Lake (Phase 3 on Databricks) |
| **Vector DB** | Qdrant (local Docker); Mosaic AI Vector Search (Phase 3) |
| **RAG** | Hybrid retrieval (Qdrant + optional BM25), Cohere reranker, LLM (OpenRouter / OpenAI) |
| **Embeddings** | OpenRouter free model (`nvidia/llama-nemotron-embed-vl-1b-v2:free`) |
| **Observability** | Langfuse (Phase 2), RAGAS (Phase 2), Prometheus + Grafana (optional) |
| **App** | Streamlit; Docker Compose for Qdrant, reranker, etc. |

---

## Project structure

```
├── data/                 # Raw + processed CFPB; regulatory PDFs
│   ├── raw/              # complaints.csv (full CFPB)
│   ├── processed/        # cfpb_filtered.csv (one product, capped)
│   ├── regulatory/       # AML/regulatory PDFs
│   ├── DATA.md           # Data description, sources, schema
│   └── README.md         # Data folder overview
├── scripts/              # download_and_filter_cfpb.py, etc.
├── ingestion/            # loader, chunker, embedder, run
├── rag/                  # retriever, reranker, chain, prompts (LangChain)
├── evaluation/           # RAGAS test set and eval (Phase 2)
├── app/                  # Streamlit UI
├── services/reranker/    # Cohere reranker proxy (Docker)
├── prometheus/           # Prometheus config
├── tests/                # pytest: ingestion, RAG
├── docker-compose.yml    # Qdrant, Prometheus, Grafana, reranker, n8n, Neo4j (optional)
├── PHASES.md             # Phase plan and completion tests
├── .env.example          # Env template (copy to .env)
└── DOCKER.md             # Docker stack and ports
```

---

## Quick start

### 1. Clone and env

```bash
git clone <your-repo-url>
cd aml-kyc-rag-assistant
cp .env.example .env
```

Edit `.env`:
- **For ingestion and RAG:** set **`OPENROUTER_API_KEY`** (required; get it at [openrouter.ai/keys](https://openrouter.ai/keys)).
- Optional: `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION` (defaults: localhost, 6333, aml_rag); for reranker: `COHERE_API_KEY`. See [.env.example](.env.example) and [DOCKER.md](DOCKER.md) for full stack.

### 2. Data

- **CFPB:** ensure `data/processed/cfpb_filtered.csv` exists (Debt collection, ~98k rows). To (re)create:
  ```bash
  python scripts/download_and_filter_cfpb.py --product "Debt collection" --max-rows 50000
  ```
- **Regulatory PDFs:** optional; place in `data/regulatory/` (see [data/DATA.md](data/DATA.md) for links).

### 3. Start Qdrant (for ingestion and RAG)

```bash
docker compose up -d qdrant
```

Or start the full stack: `docker compose up -d`. Details: [DOCKER.md](DOCKER.md).

### 4. Run ingestion

From the project root (with `.env` loaded, e.g. by your IDE or `set -a; source .env; set +a` on Unix):

```bash
pip install -r requirements.txt
python -m ingestion.run
```

The script checks: **OPENROUTER_API_KEY** set, Qdrant reachable, and at least one document (CFPB or PDFs). To wipe and refill the collection: `python -m ingestion.run --recreate`.

---

## Data

- **Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/); filtered to one product (e.g. Debt collection), capped rows.
- **Schema and sources:** [data/DATA.md](data/DATA.md) — columns, filtering, suggested regulatory PDF links.
- **Folder overview:** [data/README.md](data/README.md).

---

## Next steps

1. **Phase 1:** Finish Streamlit app (`app/main.py`) with chat and citations; add `DEPLOYMENT.md` and CI workflow. Run and pass tests in [PHASES.md](PHASES.md) (Test 1.1–1.6).
2. **Phase 2:** Add observability tab (Langfuse), RAGAS golden set and `evaluation/ragas_eval.py`; optional Prometheus/Grafana.
3. **Phase 3 (optional):** Azure Databricks path (Delta, Vector Search) when you need scale.

---

## License

Use and adapt as needed for your context. CFPB data is public; respect regulatory document terms and your own compliance policies.
