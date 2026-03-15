# AML/KYC RAG Assistant

**Repo:** `aml-kyc-rag-assistant`

A **retrieval-augmented GenAI assistant** for AML/KYC analysts: explore and summarize financial crime risk signals from consumer complaint narratives and regulatory text. Ask case-style questions, get cited answers, and inspect relationship graphs across customers, institutions, products, and issues.

---

## What it does

- **Case-style questions** — Summarize complaint history, red flags, company responses; generate executive summaries for case files.
- **Pattern and trend questions** — Top issues by product/company, emerging themes (e.g. crypto, fraud), under-remediation risk.
- **Policy and regulatory questions** — Answers grounded in FATF, CFPB, FinCEN-style guidance (when regulatory PDFs are loaded).
- **Knowledge-graph questions** — Entities and relationships from complaints (via Graphiti MCP + Neo4j); multi-hop and network-style queries.

The system uses the **CFPB Consumer Complaint Database** (filtered by product), optional **AML/regulatory PDFs**, and a **knowledge graph** (Graphiti over Neo4j) with **hybrid retrieval** (vector + keyword) and **reranking** (e.g. Cohere).

---

## Tech stack

| Layer | Choices |
|-------|--------|
| **Data** | CFPB CSV, regulatory PDFs; Delta Lake (Phase 2 on Databricks) |
| **Vector DB** | Qdrant (local Docker); Mosaic AI Vector Search (Phase 2) |
| **Graph** | Neo4j + Graphiti MCP |
| **RAG** | Hybrid retrieval (Qdrant + BM25), Cohere reranker, LLM (OpenAI / OpenRouter / Azure OpenAI) |
| **Observability** | Langfuse (tracing), RAGAS (evaluation), Prometheus + Grafana (optional) |
| **App** | Streamlit (planned); Docker Compose for local stack |

---

## Project structure

```
├── data/                 # Raw + processed CFPB; regulatory PDFs
│   ├── raw/              # complaints.csv (full CFPB)
│   ├── processed/       # cfpb_filtered.csv (one product, capped)
│   ├── regulatory/      # AML/regulatory PDFs
│   ├── DATA.md          # Data description, sources, schema
│   └── README.md        # Data folder overview
├── scripts/              # One-off scripts (e.g. download & filter CFPB)
├── ingestion/            # Loader, chunker, embedder (to be added)
├── graph/                # Graphiti/Neo4j builder (to be added)
├── rag/                  # Retriever, reranker, chain, prompts (to be added)
├── evaluation/           # RAGAS test set and eval (to be added)
├── app/                  # Streamlit UI (to be added)
├── services/
│   └── reranker/        # Cohere reranker proxy (Docker)
├── prometheus/           # Prometheus config
├── docker-compose.yml    # n8n, Qdrant, Prometheus, Grafana, Cohere reranker, Neo4j, Graphiti MCP
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
- **For ingestion:** set **`OPENROUTER_API_KEY`** (required; get it at [openrouter.ai/keys](https://openrouter.ai/keys)).
- Optional: `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION` (defaults: localhost, 6333, aml_rag).
- For Docker stack: `NEO4J_PASSWORD`; for reranker: `COHERE_API_KEY`; for Graphiti MCP: `OPENAI_API_KEY`. See [.env.example](.env.example).

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

## Roadmap

- **Phase 1 (local):** Ingestion pipeline → Qdrant + Graphiti/Neo4j → RAG chain (hybrid + reranker) → Streamlit UI → RAGAS/Langfuse evaluation.
- **Phase 2 (optional):** Azure Databricks — Delta Lake, Mosaic AI Vector Search, MLflow; same app and evaluation approach.

---

## License

Use and adapt as needed for your context. CFPB data is public; respect regulatory document terms and your own compliance policies.
