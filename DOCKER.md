# Local Docker Stack

Run n8n, Qdrant, Prometheus, Grafana, Cohere reranker, Neo4j, and Graphiti MCP locally.

## Prerequisites

- Docker and Docker Compose
- **Reranker:** [Cohere](https://cohere.com) API key (optional; returns 503 if not set)
- **Graphiti MCP:** [OpenAI](https://platform.openai.com) API key (for entity extraction and graph operations over Neo4j)

## Setup

1. Copy env example and set secrets:
   ```bash
   copy .env.example .env
   ```
   Edit `.env` and set:
   - `COHERE_API_KEY` — for the reranker service (optional)
   - `NEO4J_PASSWORD` — Neo4j password (user is `neo4j`); default `localdev`
   - `OPENAI_API_KEY` — for Graphiti MCP (knowledge graph over Neo4j)

2. Start the stack:
   ```bash
   docker compose up -d
   ```

## Services and ports

| Service      | URL (local)            | Notes |
|-------------|-------------------------|--------|
| n8n         | http://localhost:5678   | Workflow automation; data in volume `n8n_data` |
| Qdrant      | http://localhost:6333   | Vector DB; storage in `qdrant_storage` |
| Prometheus  | http://localhost:9090   | Metrics; config in `prometheus/prometheus.yml` |
| Grafana     | http://localhost:3000   | Login: `admin` / `admin` (change in UI) |
| Reranker    | http://localhost:8000   | Cohere proxy: `POST /rerank`, `GET /health` |
| Neo4j       | http://localhost:7474   | Browser UI; Bolt: `bolt://localhost:7687` |
| Graphiti MCP| http://localhost:8001/mcp/ | MCP server for knowledge graph (Neo4j backend); use in Cursor/Claude MCP config |

## Reranker API

- **POST /rerank** — Body: `{ "query": "...", "documents": ["doc1", "doc2", ...], "top_n": 3 }`  
  Returns: `{ "results": [ { "index": 0, "relevance_score": 0.99 }, ... ] }`
- **GET /health** — Check if Cohere is configured.

## Grafana + Prometheus

In Grafana, add a data source: type **Prometheus**, URL `http://prometheus:9090` (from inside Docker network) or `http://host.docker.internal:9090` on Windows/Mac if Grafana runs in Docker and you need to reach host.

## Graphiti MCP (knowledge graph)

Graphiti MCP runs on port **8001** and uses your **Neo4j** instance. MCP endpoint: `http://localhost:8001/mcp/`.

- **Cursor:** In MCP settings, add a server with URL `http://localhost:8001/mcp/` (HTTP transport).
- **Claude Desktop:** Configure the MCP server to point to the same URL.
- Requires `OPENAI_API_KEY` in `.env` for entity extraction and graph updates.

## Stop

```bash
docker compose down
```

To remove volumes as well: `docker compose down -v`.
