"""
Rebuild the SDMX data dictionary from scratch.

Pipeline:
  1. Fetch all dataflows from the SDMX API
  2. Download each dataflow as CSV and standardise column names
  3. Classify unique indicators by category using Azure OpenAI
  4. Resolve geographies to a canonical country name
  5. Write  data/data_dictionary.parquet
  6. Re-run  scripts/embed_indicators.py

Usage:
  conda run -n genai_vector python scripts/build_data_dictionary.py

Flags:
  --skip-download   Re-use data/build_cache/raw_mapping.parquet from a previous run
  --skip-classify   Re-use data/build_cache/classified.parquet   from a previous run
  --skip-embed      Do not re-run embed_indicators.py after writing the dictionary
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ROOT  = Path(__file__).parent.parent
DATA  = ROOT / "data"
CACHE = DATA / "build_cache"
OUT   = DATA / "data_dictionary.parquet"

SDMX_BASE = "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest"

# ---------------------------------------------------------------------------
# Column name mappings (Colab notebook logic, cleaned up)
# ---------------------------------------------------------------------------

_INDICATOR_ID_COLS   = ["INDICATOR", "CDCOVERAGEINDICATORS", "CDDDEMINDICS",
                         "CDDRIVERINDICATORS", "CDT2INDICS", "UNICEF_INDICATOR",
                         "SITREP_INDICATOR"]
_INDICATOR_NAME_COLS = ["Indicator", "Coverage indicators", "Demographic indicators",
                         "Driver indicators", "Tier 2 indicators",
                         "Situation Report Indicator"]
_GEO_ID_COLS         = ["REF_AREA", "RefRegion", "COUNTRY", "Areas", "Subregion"]
_GEO_NAME_COLS       = ["Geographic area", "Reference Areas", "Reference Area",
                         "Country", "REGION", "SUBREGION", "CDAREAS"]
_DEFINITION_COLS     = ["DEFINITION"]

_CATEGORIES = (
    "Health, Nutrition, Education, WASH, Social policy, Climate, "
    "Gender Inequalities, Child protection, Population and demographics, "
    "Humanitarian and Emergencies, Others"
)


# ---------------------------------------------------------------------------
# Step 1 — fetch dataflow list
# ---------------------------------------------------------------------------

def fetch_dataflows() -> dict:
    """Return {dataflow_id: (agency, name)} for every published dataflow."""
    url = (
        f"{SDMX_BASE}/dataflow/all/all/latest/"
        "?format=sdmx-json&detail=full&references=none"
    )
    print("Fetching dataflow list ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dataflows = r.json().get("data", {}).get("dataflows", [])
    result = {
        d["id"]: (d.get("agencyID", ""), d.get("name", ""))
        for d in dataflows
    }
    print(f"  Found {len(result)} dataflows.")
    return result


# ---------------------------------------------------------------------------
# Step 2 — download + standardise each dataflow
# ---------------------------------------------------------------------------

def _first_col(df: pd.DataFrame, candidates: list, new_name: str) -> pd.DataFrame:
    for col in candidates:
        if col in df.columns:
            return df.rename(columns={col: new_name})
    return df


def _standardise(df: pd.DataFrame, agency: str, dataflow_id: str, dataflow_name: str) -> pd.DataFrame:
    df = _first_col(df, _INDICATOR_ID_COLS,   "indicator_id")
    df = _first_col(df, _INDICATOR_NAME_COLS, "indicator")
    df = _first_col(df, _GEO_ID_COLS,         "geography_id")
    df = _first_col(df, _GEO_NAME_COLS,        "geography")
    df = _first_col(df, _DEFINITION_COLS,      "definition")

    # Concatenate all source-like columns
    src_cols = [c for c in df.columns if "source" in c.lower()]
    df["concatenated_sources"] = (
        df[src_cols].astype(str).apply(lambda r: "-".join(r), axis=1)
        if src_cols else ""
    )

    for col in ("indicator_id", "indicator", "geography_id", "geography",
                "definition", "concatenated_sources"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(str)

    df["agency"]        = agency
    df["dataflow_id"]   = dataflow_id
    df["dataflow_name"] = dataflow_name

    # Compute year_min / year_max per (indicator, geography) from TIME_PERIOD
    group_cols = ["indicator_id", "geography_id", "agency", "dataflow_id", "dataflow_name"]
    if "TIME_PERIOD" in df.columns:
        years = pd.to_numeric(
            df["TIME_PERIOD"].astype(str).str[:4], errors="coerce"
        )
        df["_year"] = years
        year_stats = (
            df.groupby(group_cols)["_year"]
            .agg(year_min="min", year_max="max")
            .reset_index()
        )
        df = df.drop(columns=["_year"])
    else:
        year_stats = None

    keep = ["indicator_id", "indicator", "geography_id", "geography",
            "definition", "concatenated_sources", "agency", "dataflow_id", "dataflow_name"]
    df = df[keep].drop_duplicates()

    if year_stats is not None:
        df = df.merge(year_stats, on=group_cols, how="left")
        df["year_min"] = df["year_min"].astype("Int64").where(df["year_min"].between(1900, 2100), pd.NA)
        df["year_max"] = df["year_max"].astype("Int64").where(df["year_max"].between(1900, 2100), pd.NA)
    else:
        df["year_min"] = pd.NA
        df["year_max"] = pd.NA

    return df


def download_all_dataflows(dataflows: dict) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    total = len(dataflows)
    for i, (df_id, (agency, name)) in enumerate(dataflows.items(), 1):
        url = (
            f"{SDMX_BASE}/data/{agency},{df_id},1.0/all"
            "?format=csv&labels=both"
        )
        tmp = CACHE / f"raw_{agency}_{df_id}.csv"
        if not tmp.exists():
            try:
                urlretrieve(url, tmp)
            except Exception as e:
                print(f"  [{i}/{total}] SKIP {df_id}: {e}")
                continue
        try:
            df = pd.read_csv(tmp, low_memory=False)
            df = _standardise(df, agency, df_id, name)
            frames.append(df)
            print(f"  [{i}/{total}] OK   {df_id} ({len(df):,} rows)")
        except Exception as e:
            print(f"  [{i}/{total}] ERR  {df_id}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(CACHE / "raw_mapping.parquet", index=False)
    print(f"\nRaw mapping: {len(combined):,} rows saved to cache.")
    return combined


# ---------------------------------------------------------------------------
# Step 3 — classify indicators with Azure OpenAI
# ---------------------------------------------------------------------------

def _classify_batch(client: AzureOpenAI, deployment: str, batch: pd.DataFrame) -> list:
    prompt = (
        f"Classify the following indicators into one of these categories: {_CATEGORIES}.\n"
        "For each indicator output a JSON object with keys 'indicator_id' and 'category'. "
        "Indicators may be in English, Portuguese, Spanish, or French. "
        "If unsure, use 'Others'. "
        "Output a JSON array only — no other text.\n\nIndicators:\n"
    )
    for _, row in batch.iterrows():
        prompt += f"ID: {row['indicator_id']}; Name: {row['indicator']}\n"

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"    Classification error: {e}")
        return []


def classify_indicators(df: pd.DataFrame) -> pd.DataFrame:
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_LLM_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]

    unique = (
        df[["indicator_id", "indicator"]]
        .dropna(subset=["indicator_id"])
        .drop_duplicates("indicator_id")
        .reset_index(drop=True)
    )
    print(f"Classifying {len(unique):,} unique indicators in batches of 25 ...")

    results = []
    batch_size = 25
    for start in range(0, len(unique), batch_size):
        batch = unique.iloc[start : start + batch_size]
        batch_results = _classify_batch(client, deployment, batch)
        results.extend(batch_results)
        print(f"  {min(start + batch_size, len(unique))}/{len(unique)}", end="\r")
        time.sleep(0.05)

    print(f"\n  Done — {len(results)} classifications returned.")

    cat_df = pd.DataFrame(results)
    if cat_df.empty or "indicator_id" not in cat_df.columns:
        df["category"] = None
        return df

    df = df.merge(cat_df[["indicator_id", "category"]], on="indicator_id", how="left")
    classified = CACHE / "classified.parquet"
    df.to_parquet(classified, index=False)
    print(f"  Classified mapping saved to cache.")
    return df


# ---------------------------------------------------------------------------
# Step 4 — resolve geographies to a canonical country name
# ---------------------------------------------------------------------------

def _build_country_lists(df: pd.DataFrame) -> tuple[list, list]:
    """Extract (country_names, iso_codes) from GLOBAL_DATAFLOW rows."""
    gdf = df[df["dataflow_id"] == "GLOBAL_DATAFLOW"][["geography", "geography_id"]].drop_duplicates()
    # Keep only ISO-style codes (no digits, no underscores)
    gdf = gdf[
        gdf["geography_id"].astype(str).str.match(r"^[A-Za-z]{2,4}$")
    ].copy()
    gdf["geography"] = gdf["geography"].str.lower()
    countries = gdf["geography"].dropna().tolist()
    codes     = gdf["geography_id"].tolist()
    return countries, codes


def resolve_geographies(df: pd.DataFrame) -> pd.DataFrame:
    countries, codes = _build_country_lists(df)
    countries_lower = [str(c).lower() for c in countries]
    codes_lower     = [str(c).lower() for c in codes]

    geo_id  = df["geography_id"].astype(str).str.lower()
    geo_nm  = df["geography"].astype(str).str.lower()

    n1 = geo_id.isin(codes_lower).astype(int)
    n2 = geo_nm.isin(countries_lower).astype(int)
    n3 = geo_id.isin(countries_lower).astype(int)
    n4 = geo_nm.isin(codes_lower).astype(int)
    df["national"] = pd.concat([n1, n2, n3, n4], axis=1).max(axis=1)

    # Rows with digits in geography_id are subnational
    has_digit = df["geography_id"].astype(str).str.contains(r"\d")
    df.loc[has_digit, "national"] = 0

    # national_country: longest of (geography, geography_id) for national rows
    def _national_country(row):
        if row["national"] == 1:
            g, gi = str(row["geography"]), str(row["geography_id"])
            return g if len(g) >= len(gi) else gi
        return np.nan

    df["national_country"] = df.apply(_national_country, axis=1)

    # subnational: match country code / name inside dataflow_id or dataflow_name
    def _sub_code(row):
        if row["national"] == 0:
            haystack = [str(row["dataflow_id"]), str(row["dataflow_name"])]
            for code in codes:
                if any(f"_{code}" in h or f"{code}_" in h for h in haystack):
                    return code
        return np.nan

    def _sub_name(row):
        if row["national"] == 0:
            haystack = [str(row["dataflow_id"]).lower(), str(row["dataflow_name"]).lower()]
            for cname in countries_lower:
                if len(cname) > 3 and any(cname in h for h in haystack):
                    return cname
        return np.nan

    df["subnational_code_retrieved"]    = df.apply(_sub_code, axis=1)
    df["subnational_country_retrieved"] = df.apply(_sub_name, axis=1)

    # Reverse-map code → name using the country list
    code_to_name = dict(zip([c.lower() for c in codes], countries_lower))
    df["subnational_country_retrieved_from_code"] = (
        df["subnational_code_retrieved"]
        .fillna("")
        .astype(str)
        .str.lower()
        .map(code_to_name)
    )

    # Final country column: coalesce in priority order
    country = (
        df["subnational_country_retrieved"]
        .fillna(df["national_country"])
        .fillna(df["subnational_country_retrieved_from_code"])
    )
    df["country"] = country.where(country.isna(), country.astype(str).str.lower().str.strip())

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download",  action="store_true",
                        help="Re-use cached raw_mapping.parquet")
    parser.add_argument("--skip-classify",  action="store_true",
                        help="Re-use cached classified.parquet")
    parser.add_argument("--skip-embed",     action="store_true",
                        help="Do not re-run embed_indicators.py")
    args = parser.parse_args()

    # --- Step 1 + 2: download ---
    if args.skip_download:
        raw_cache = CACHE / "raw_mapping.parquet"
        if not raw_cache.exists():
            sys.exit(f"Cache file not found: {raw_cache}")
        print(f"Loading raw mapping from cache: {raw_cache}")
        df = pd.read_parquet(raw_cache)
    else:
        dataflows = fetch_dataflows()
        df = download_all_dataflows(dataflows)

    # --- Step 3: classify ---
    # If skip_classify and a classified cache exists, merge categories from it
    # without re-calling the API (preserves year columns from the fresh download).
    cat_cache = CACHE / "classified.parquet"
    if args.skip_classify:
        if not cat_cache.exists():
            sys.exit(f"Classified cache not found: {cat_cache}")
        print("Merging categories from cache (no API calls) ...")
        cat_df = (
            pd.read_parquet(cat_cache, columns=["indicator_id", "category"])
            .drop_duplicates("indicator_id")
        )
        df = df.merge(cat_df, on="indicator_id", how="left")
    elif "category" not in df.columns:
        df = classify_indicators(df)
    else:
        print("'category' column already present — skipping classification.")

    # --- Step 4: geography resolution ---
    print("Resolving geographies ...")
    df = resolve_geographies(df)
    print(f"  {df['country'].notna().sum():,} rows with a resolved country "
          f"(out of {len(df):,} total).")

    # --- Step 5: write output ---
    # Enforce column order consistent with existing data_dictionary.parquet
    FINAL_COLS = [
        "indicator_id", "indicator", "geography_id", "geography",
        "definition", "concatenated_sources",
        "agency", "dataflow_id", "dataflow_name", "category",
        "national", "national_country",
        "subnational_code_retrieved", "subnational_country_retrieved",
        "subnational_country_retrieved_from_code", "country",
        "year_min", "year_max",
    ]
    for col in FINAL_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[FINAL_COLS]

    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nData dictionary written → {OUT}  ({size_mb:.1f} MB)")
    print(f"  {len(df):,} rows  |  {df['country'].nunique()} countries  "
          f"|  {df['indicator_id'].nunique()} unique indicators")

    # --- Step 6: re-embed ---
    if not args.skip_embed:
        print("\nRunning embed_indicators.py ...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "embed_indicators.py")],
            check=False,
        )
        if result.returncode != 0:
            print("  embed_indicators.py failed — run it manually.")
    else:
        print("\nSkipped embed step. Run when ready:")
        print("  python scripts/embed_indicators.py")


if __name__ == "__main__":
    main()
