"""
Create Mosaic AI Vector Search endpoint and Delta Sync index from the Delta table written by delta_ingest.py.
Run on a Databricks cluster with databricks-vectorsearch installed.

Usage:
  - Set DELTA_TABLE_NAME, INDEX_NAME, ENDPOINT_NAME (or use defaults).
  - Enable Change Data Feed on the Delta table if using a standard endpoint:
      ALTER TABLE main.default.aml_kyc_chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""
from __future__ import annotations

import os
import sys
import time

# Embedding dimension for nvidia/llama-nemotron-embed-vl-1b-v2 (OpenRouter)
EMBEDDING_DIMENSION = 256


def main() -> None:
    try:
        from databricks.vector_search.client import VectorSearchClient
    except ImportError:
        print("Install databricks-vectorsearch: pip install databricks-vectorsearch", file=sys.stderr)
        sys.exit(1)

    delta_table = os.environ.get("DELTA_TABLE_NAME", "main.default.aml_kyc_chunks")
    index_name = os.environ.get("DATABRICKS_VECTOR_SEARCH_INDEX_NAME", f"{delta_table}_index")
    endpoint_name = os.environ.get("DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME", "aml-kyc-vs-endpoint")

    client = VectorSearchClient()

    # Create endpoint if it does not exist
    try:
        client.get_endpoint(name=endpoint_name)
        print(f"Endpoint {endpoint_name} already exists.")
    except Exception:
        print(f"Creating endpoint {endpoint_name}...")
        client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print("Waiting for endpoint to be ready...")
        time.sleep(30)

    # Create Delta Sync index (source table must have Change Data Feed enabled for standard endpoints)
    print(f"Creating Delta Sync index {index_name} from {delta_table}...")
    client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        source_table_name=delta_table,
        index_name=index_name,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_dimension=EMBEDDING_DIMENSION,
        embedding_vector_column="embedding",
        columns_to_sync=["id", "text", "product", "complaint_id", "source", "issue", "company"],
    )
    print(f"Index creation started. Index: {index_name}")
    print("Wait for index to reach Ready state in the Databricks UI (Vector Search).")
    print("Then set DATABRICKS_VECTOR_SEARCH_INDEX_NAME=", index_name, " when using the app with VECTOR_BACKEND=databricks.")


if __name__ == "__main__":
    main()
