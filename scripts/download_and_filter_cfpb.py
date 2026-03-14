"""
Download CFPB Consumer Complaint Database and filter to one product and a max row count.
Output: data/raw/complaints.csv (full), data/processed/cfpb_filtered.csv (filtered).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

CFPB_CSV_ZIP = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
DEFAULT_PRODUCT = "Debt collection"
DEFAULT_MAX_ROWS = 50_000


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case for consistency."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("?", "")
    )
    return df


def download_raw(raw_dir: Path) -> Path:
    """Download complaints.csv.zip and extract to raw_dir. Returns path to complaints.csv."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "complaints.csv.zip"
    csv_path = raw_dir / "complaints.csv"

    if csv_path.exists():
        print(f"Using existing {csv_path}")
        return csv_path

    print(f"Downloading {CFPB_CSV_ZIP} ...")
    urlretrieve(CFPB_CSV_ZIP, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    # CFPB zip often contains a single file; ensure we have complaints.csv
    extracted = list(raw_dir.glob("*.csv"))
    if extracted and extracted[0] != csv_path:
        extracted[0].rename(csv_path)
    return csv_path


def filter_and_save(
    csv_path: Path,
    processed_path: Path,
    product: str,
    max_rows: int,
) -> None:
    """Load CSV, filter by product, limit rows, save to processed_path."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="warn")
    df = normalize_columns(df)

    # CFPB column is "Product" (may be "product" after normalize)
    product_col = "product"
    if product_col not in df.columns:
        raise KeyError(
            f"Column '{product_col}' not found. Available: {list(df.columns)}"
        )

    subset = df[df[product_col].astype(str).str.strip().str.lower() == product.lower().strip()]
    total = len(subset)
    if total == 0:
        raise ValueError(
            f"No rows for product '{product}'. Sample values: {df[product_col].dropna().unique()[:10].tolist()}"
        )

    if total > max_rows:
        subset = subset.head(max_rows)
        print(f"Filtered to product '{product}': {len(subset)} rows (capped from {total})")
    else:
        print(f"Filtered to product '{product}': {len(subset)} rows")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(processed_path, index=False)
    print(f"Saved to {processed_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CFPB complaints and filter to one product and max rows."
    )
    parser.add_argument(
        "--product",
        default=DEFAULT_PRODUCT,
        help=f"Product to keep (default: {DEFAULT_PRODUCT})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"Max rows in filtered output (default: {DEFAULT_MAX_ROWS})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Directory for raw complaints.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "processed" / "cfpb_filtered.csv",
        help="Output path for filtered CSV",
    )
    args = parser.parse_args()

    csv_path = download_raw(args.raw_dir)
    filter_and_save(csv_path, args.out, args.product, args.max_rows)


if __name__ == "__main__":
    main()
