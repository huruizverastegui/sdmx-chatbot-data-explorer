"""
Split mapping_sdmx_*.csv into one Parquet file per country.

Output: data/partitions/{country_slug}.parquet
        data/partitions/_index.json   <- country → filename map

Run once:
    conda run -n genai_vector python data/partition_by_country.py
"""

import json
import re
import pandas as pd
from pathlib import Path

SRC = Path(__file__).parent / "mapping_sdmx_2025_03_20_country_category_cleaned_no_index.csv"
OUT = Path(__file__).parent / "partitions"
OUT.mkdir(exist_ok=True)

def slugify(name: str) -> str:
    """'South Africa' → 'south_africa'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

print(f"Reading {SRC} ...")
df = pd.read_csv(SRC, low_memory=False)
print(f"  {len(df):,} rows, {df['country'].nunique()} unique countries")

index: dict[str, str] = {}   # country_name → filename

for country, group in df.groupby("country", sort=True):
    slug = slugify(str(country))
    fname = f"{slug}.parquet"
    group.reset_index(drop=True).to_parquet(OUT / fname, index=False)
    index[str(country)] = fname

index_path = OUT / "_index.json"
with open(index_path, "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)

print(f"  Written {len(index)} partition files to {OUT}/")
print(f"  Index saved to {index_path}")
