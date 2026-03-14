# Data Description — AML/KYC GraphRAG Copilot

This document describes all data used in this project: sources, contents, filtering, and usage.

---

## 1. Overview

The project uses two main data types:

1. **CFPB Consumer Complaint Database** — Structured complaint records (CSV) with optional free-text narratives. Used for RAG, knowledge-graph construction, and AML-style case/pattern analysis.
2. **Regulatory documents** — PDFs (e.g. FATF AML guidance, CFPB materials) placed in `regulatory/` for policy and compliance-style questions.

All paths below are relative to the **`data/`** folder unless otherwise stated.

---

## 2. Folder Structure

| Folder       | Purpose |
|--------------|--------|
| **`raw/`**   | Original, unfiltered data. Contains the full CFPB CSV after download (`complaints.csv`). |
| **`processed/`** | Filtered subsets ready for ingestion. Primary file: `cfpb_filtered.csv` (one product, capped rows). |
| **`regulatory/`** | AML/regulatory PDFs. Add files here; the ingestion pipeline will chunk and embed them. |

---

## 3. Data Sources

### 3.1 CFPB Consumer Complaint Database

- **Owner:** Consumer Financial Protection Bureau (CFPB), U.S.
- **What it is:** Public database of consumer complaints about financial products and services, sent to companies for response.
- **Links:**
  - Main portal: [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
  - Direct CSV (zipped): `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`
  - API: [Technical documentation](https://www.consumerfinance.gov/complaintdatabase/technical-documentation/)
- **Update frequency:** Updated daily. Complaints are published after the company responds or after 15 days, whichever is first.
- **Scope:** Complaints referred to other regulators (e.g. certain depository institutions) are not published; only complaints in the CFPB’s published set are included.
- **Use in this project:** Mirrors the kind of narrative and structured data AML/KYC analysts see (complaints, companies, products, issues, dates). Used for case summarization, trend analysis, and graph (entities/relationships).

### 3.2 Regulatory Documents

- **Source:** You supply these. Typical examples:
  - FATF AML/CFT recommendations or guidance
  - CFPB consumer protection or complaint-handling guidance
  - Other public AML/KYC or consumer-protection documents
- **Format:** PDF (or other formats supported by the ingestion loader).
- **Location:** Place files in **`data/regulatory/`**. The ingestion pipeline will load, chunk, and embed them for RAG.

---

## 4. Raw Data (`raw/`)

### 4.1 What It Is

- **File:** `raw/complaints.csv` (after running the download script or extracting the CFPB zip manually).
- **Origin:** Full CFPB Consumer Complaint Database export (zipped at the URL above).
- **How it gets here:** Run `python scripts/download_and_filter_cfpb.py` from the project root; the script downloads the zip and extracts `complaints.csv` into `data/raw/`. If `complaints.csv` already exists, the script skips the download.

### 4.2 Raw Data Schema (CFPB)

The CFPB CSV uses human-readable column names. The project’s download script normalizes them to **snake_case** when producing the filtered file. Raw CFPB columns (conceptually) include:

| Column (typical) | Description |
|------------------|-------------|
| Date received | When the CFPB received the complaint |
| Product | High-level product (e.g. Mortgage, Debt collection, Credit card) |
| Sub-product | Finer product category |
| Issue | Main issue category |
| Sub-issue | Finer issue category |
| Consumer complaint narrative | Free-text narrative (optional; many rows are empty) |
| Company public response | Optional public response from the company |
| Company | Company name |
| State | U.S. state (abbreviation) |
| ZIP code | Consumer ZIP code |
| Tags | E.g. "Older American", "Servicemember" |
| Consumer consent provided | Whether narrative can be published |
| Submitted via | Channel (Web, Phone, etc.) |
| Date sent to company | When the complaint was sent to the company |
| Company response to consumer | E.g. "Closed with explanation", "Closed with monetary relief" |
| Timely response | Whether the company responded on time |
| Consumer disputed | Whether the consumer disputed the company’s response |
| Complaint ID | Unique identifier |

---

## 5. Filtered Data (`processed/cfpb_filtered.csv`)

### 5.1 What It Contains

- **Purpose:** A single-product subset of the CFPB data, capped in size, for ingestion (RAG, graph, analytics).
- **Default product:** **Debt collection** (configurable via the download script).
- **Default max rows:** **50,000** (configurable). If the product has more rows, the first 50,000 are kept (order as in the raw CSV).
- **Schema:** Same columns as the raw CSV, with names **normalized to snake_case** (e.g. `date_received`, `product`, `sub_product`, `issue`, `consumer_complaint_narrative`, `company`, `state`, `zip_code`, `complaint_id`).

### 5.2 How It Is Produced

1. Load `data/raw/complaints.csv`.
2. Normalize column names to snake_case.
3. Filter rows where **`product`** equals the chosen product (e.g. `"Debt collection"`), case-insensitive.
4. Take the first **`--max-rows`** rows (default 50,000).
5. Write to **`data/processed/cfpb_filtered.csv`** (or the path given by `--out`).

### 5.3 Filtered File Columns (snake_case)

After normalization, the filtered CSV includes (among others):

- `date_received`, `product`, `sub_product`, `issue`, `sub_issue`
- `consumer_complaint_narrative` — main free text for RAG (often empty)
- `company_public_response`, `company`, `state`, `zip_code`
- `tags`, `consumer_consent_provided`, `submitted_via`, `date_sent_to_company`
- `company_response_to_consumer`, `timely_response`, `consumer_disputed`
- `complaint_id`

All rows in the filtered file have the same **`product`** value. No other columns are removed; only rows are filtered and truncated.

### 5.4 Example Rows

Typical row (Debt collection):

- `date_received`: 2020-03-28  
- `product`: Debt collection  
- `sub_product`: Other debt  
- `issue`: Attempts to collect debt not owed  
- `sub_issue`: Debt is not yours  
- `consumer_complaint_narrative`: (optional; may be empty)  
- `company`: ENCORE CAPITAL GROUP INC.  
- `state`: FL  
- `zip_code`: 33175  
- `complaint_id`: 3583440  

### 5.5 Customizing Filtered Data

From the project root:

```bash
# Default: Debt collection, 50k rows → data/processed/cfpb_filtered.csv
python scripts/download_and_filter_cfpb.py

# Different product and size
python scripts/download_and_filter_cfpb.py --product "Mortgage" --max-rows 40000

# Custom output path
python scripts/download_and_filter_cfpb.py --out data/processed/cfpb_mortgage.csv
```

---

## 6. Regulatory Data (`regulatory/`)

- **Content:** Any AML/KYC or consumer-protection documents you want to query (e.g. FATF, CFPB PDFs).
- **Format:** PDF (or whatever the ingestion loader supports).
- **Usage:** The ingestion pipeline will chunk and embed these and add them to the vector store and/or graph so the copilot can answer policy and regulatory questions with citations.

### 6.1 Suggested regulatory documents to download

Place downloaded PDFs in **`data/regulatory/`** for the RAG pipeline. All links below are to official or widely used public sources.

**FATF (AML/CFT international standards)**

- **FATF Recommendations (40 Recommendations)** — Core AML/CFT framework (risk-based approach, customer due diligence, etc.).  
  - Page (with PDF link): [FATF Recommendations](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html)  
  - Direct PDF (FATF Standards): [FATF Standards - 40 Recommendations](https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/FATF%20Standards%20-%2040%20Recommendations%20rc.pdf) (check [FATF Publications](https://www.fatf-gafi.org/en/publications.html) for the latest version).
- **FATF Guidance** — Risk-based approach, customer due diligence, virtual assets, etc.: [FATF Guidance](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Guidance.html).

**CFPB (consumer protection and complaints)**

- **Consumer complaint process** — How complaints are handled and published: [Learn how the complaint process works](https://www.consumerfinance.gov/complaint/process/).
- **CFPB guidance and reports** — Guidance compendium and annual reports (PDFs): [CFPB Publications / files.consumerfinance.gov](https://www.consumerfinance.gov/data-research/research-reports/). For a guidance compendium PDF, search “guidance compendium” on consumerfinance.gov or use: [CFPB Guidance Compendium (example)](https://files.consumerfinance.gov/f/documents/cfpb_guidance-compendium_2025-01.pdf) if still available.

**FinCEN (U.S. BSA/AML)**

- **Guidance index** — All FinCEN guidance (BSA, SAR, CDD, etc.): [FinCEN Guidance](https://www.fincen.gov/resources/statutes-regulations/guidance).
- **Customer Due Diligence (CDD) requirements** — FAQ on CDD: [FinCEN CDD Final Rule FAQ](https://www.fincen.gov/resources/statutes-regulations/guidance/frequently-asked-questions-regarding-customer-due-diligence).
- **AML/CFT National Priorities** — National priorities for AML/CFT: [FinCEN AML/CFT Priorities](https://www.fincen.gov/resources/statutes-regulations/guidance).

**FFIEC (U.S. BSA/AML examination)**

- **BSA/AML Examination Manual** — Examiner guidance (BSA, AML, OFAC). Sections available as PDFs: [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual).

**Optional (identity theft / fraud)**

- **FTC – Identity Theft** — Consumer and business guidance: [FTC Identity Theft](https://consumer.ftc.gov/identity-theft-and-identity-fraud).
- **OIG / Fed – CFPB complaint database controls** — Example of oversight of complaint data: search “OIG CFPB consumer complaint database” for PDF reports if relevant.

Download the PDFs you need, then save them into **`data/regulatory/`** (e.g. `FATF-40-Recommendations.pdf`, `FinCEN-CDD-FAQ.pdf`). The ingestion pipeline will pick them up from that folder.

---

## 7. Downstream Use

- **Ingestion:** `ingestion/loader.py` (or equivalent) reads from `processed/` and `regulatory/` to build chunks and metadata (e.g. `source`, `product`, `issue`, `date`, `risk_level`).
- **Vector store:** Embeddings from chunks are stored in Qdrant (local Docker).
- **Graph:** Complaint narratives and metadata can be sent to Graphiti MCP / Neo4j for entity and relationship extraction.
- **RAG:** Retrieval uses both vector search and (optionally) graph; the Cohere reranker and LLM produce answers with cited sources.

---

## 8. Other Information

### 8.1 Data Freshness

- Re-run the download script periodically to refresh raw and filtered CFPB data.
- If `raw/complaints.csv` already exists, the script does not re-download; delete the file (or zip) to force a fresh download.

### 8.2 Privacy and Compliance

- The CFPB database is public and intended for research and analysis. Consumer complaint narratives are published only when consent is given (see CFPB documentation).
- Use regulatory PDFs in line with their terms and your internal policies.

### 8.3 Limitations

- **Narratives:** Many complaint rows have an empty `consumer_complaint_narrative`; RAG and graph will be richer for rows that include narrative text.
- **Product filter:** Filtered data contains only one product per file; for multiple products, run the script multiple times with different `--product` and `--out` and combine or process separately as needed.
- **Order:** The 50k cap uses the order of rows in the raw CSV (no date or randomness); for time-based subsets, extend the script or filter after loading.

---

## 9. Quick Reference

| Item | Location / Value |
|------|-------------------|
| Raw CFPB CSV | `data/raw/complaints.csv` |
| Filtered CFPB (default) | `data/processed/cfpb_filtered.csv` |
| Default product | Debt collection |
| Default max rows | 50,000 |
| Regulatory PDFs | `data/regulatory/` |
| Download/filter script | `scripts/download_and_filter_cfpb.py` |
| CFPB zip URL | https://files.consumerfinance.gov/ccdb/complaints.csv.zip |

For a short folder overview and quick start, see **`data/README.md`**.
