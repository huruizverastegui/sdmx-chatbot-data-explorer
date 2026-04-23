"""
Build country-partitioned Parquet files from the full data dictionary.

Source (in order of preference):
  data/data_dictionary.parquet   <- preferred; convert from CSV once with --convert
  data/mapping_sdmx_*.csv        <- fallback / source for --convert

Output:
  data/partitions/{country_slug}.parquet
  data/partitions/_index.json

Usage:
  # First time: convert CSV → parquet, then partition
  python scripts/partition_by_country.py --convert

  # Subsequent runs: partition from existing parquet
  python scripts/partition_by_country.py
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent          # repo root
DATA = ROOT / "data"
PARQUET_SRC = DATA / "data_dictionary.parquet"
CSV_SRC = DATA / "mapping_sdmx_2025_03_20_country_category_cleaned_no_index.csv"
OUT = DATA / "partitions"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def convert_csv_to_parquet() -> None:
    if not CSV_SRC.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_SRC}")
    print(f"Reading {CSV_SRC} ...")
    df = pd.read_csv(CSV_SRC, low_memory=False)
    # Drop unnamed index column if present (subset CSV artefact)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    print(f"  {len(df):,} rows, {df['country'].nunique()} countries")
    df.to_parquet(PARQUET_SRC, index=False)
    size_mb = PARQUET_SRC.stat().st_size / 1e6
    print(f"  Saved → {PARQUET_SRC}  ({size_mb:.1f} MB)")


def partition() -> None:
    if not PARQUET_SRC.exists():
        raise FileNotFoundError(
            f"Parquet source not found: {PARQUET_SRC}\n"
            "Run with --convert first to build it from the CSV."
        )
    print(f"Reading {PARQUET_SRC} ...")
    df = pd.read_parquet(PARQUET_SRC)
    print(f"  {len(df):,} rows, {df['country'].nunique()} countries")

    OUT.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}

    for country, group in df.groupby("country", sort=True):
        slug = slugify(str(country))
        fname = f"{slug}.parquet"
        group.reset_index(drop=True).to_parquet(OUT / fname, index=False)
        index[str(country)] = fname

    index_path = OUT / "_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"  Written {len(index)} partition files → {OUT}/")
    print(f"  Index  → {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert the source CSV to data_dictionary.parquet before partitioning.",
    )
    args = parser.parse_args()

    if args.convert:
        convert_csv_to_parquet()

    partition()
