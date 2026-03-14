# Dataset for AML/KYC GraphRAG Copilot

This folder holds **raw** and **processed** CFPB Consumer Complaint data plus **regulatory** documents used for RAG.

**Full data description:** See **[DATA.md](DATA.md)** for sources, schema, what the filtered data contains, and usage details.

## Folder structure

| Folder        | Purpose |
|---------------|--------|
| `raw/`        | Original CFPB CSV (and any other raw inputs). Filled by `scripts/download_and_filter_cfpb.py` or manual download. |
| `processed/`  | Filtered/subsets ready for ingestion: e.g. `cfpb_filtered.csv` (one product, ~50k rows). |
| `regulatory/` | AML/regulatory PDFs (e.g. FATF guidelines, CFPB guidance). Add files here for the RAG pipeline. |

## CFPB Consumer Complaint Database

- **Source:** [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **Direct CSV (zipped):** `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`
- **API:** [Technical documentation](https://www.consumerfinance.gov/complaintdatabase/technical-documentation/)
- **Update:** Data is updated daily. Complaints are published after company response or after 15 days.

### Key columns (for filtering and RAG)

- **Product** — e.g. Mortgage, Debt collection, Credit card, Student loan
- **Sub-product**, **Issue**, **Sub-issue** — for tagging and metadata
- **Consumer complaint narrative** — free text (main RAG content)
- **Company**, **State**, **ZIP code** — for graph and analytics
- **Date received** — for temporal filtering
- **Tags** — e.g. "Older American", "Servicemember" (optional filters)

### Filtering strategy (this project)

- **Product:** One category per run (e.g. **Mortgage** or **Debt collection**) to mirror AML case volume and keep local dev manageable.
- **Size:** Cap at ~50k complaints (configurable in the download script).
- **Output:** Filtered CSV written to `processed/cfpb_filtered.csv` with same schema; ingestion layer adds derived fields (e.g. `risk_level`) later.

## Regulatory documents

Place PDFs in `regulatory/` for the RAG pipeline, e.g.:

- FATF AML/CFT recommendations or guidance
- CFPB consumer protection / complaint-handling guidance
- Any other public AML/KYC or consumer-protection docs you want to query

The ingestion pipeline will chunk and embed these together with CFPB complaint text.

## Quick start

1. **Download and filter CFPB data:**
   ```bash
   python scripts/download_and_filter_cfpb.py
   ```
   Options: `--product "Debt collection"`, `--max-rows 50000`, `--out data/processed/cfpb_filtered.csv`. Raw CSV is saved under `data/raw/`.

2. **Add regulatory PDFs** into `data/regulatory/` as needed.

3. **Ingestion** (later): `ingestion/loader.py` will read from `data/processed/` and `data/regulatory/`.
