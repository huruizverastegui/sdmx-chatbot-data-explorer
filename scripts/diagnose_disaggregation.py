"""
Diagnose disaggregation patterns across 50 SDMX indicator samples.

For each sampled (dataflow, indicator), reports:
  - How many disaggregation dimensions the data has
  - Which dimensions have real sub-category data vs Total-only
  - Which dimensions have CODES but null OBS_VALUE (the false-positive bug we fixed)
  - Whether a naive plot (no filters) would produce duplicate rows per time point
  - Cardinality of each dimension

Usage:
    conda run -n genai_vector python scripts/diagnose_disaggregation.py
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

ROOT  = Path(__file__).parent.parent
CACHE = ROOT / "data" / "build_cache"

# ── Column classification constants (mirrors sdmx_agent.py) ──────────────────

_NON_DISAGG_COLS = {
    "TIME_PERIOD", "Year", "YEAR",
    "OBS_VALUE", "LOWER_BOUND", "UPPER_BOUND", "OBS_STATUS", "Observation Status",
    "REF_AREA", "Geographic area", "Reference Areas",
    "INDICATOR", "Indicator",
    "UNIT_MEASURE", "Unit of measure",
    "DATA_SOURCE", "SHORT_SOURCE", "CDDATASOURCE", "PROVIDER",
    "DEFINITION", "MDG_REGION", "REF_PERIOD", "COVERAGE_TIME",
    "OBS_VALUE_CHARACTER", "OBS_CONF", "OBS_FOOTNOTE",
    "COUNTRY_NOTES", "SERIES_FOOTNOTE", "SOURCE_LINK", "DATAFLOW",
}

_TOTAL_CODES = {
    "_T", "_ALL", "_Z", "_NA", "TOTAL", "Total", "All",
    "Not applicable", "Not Applicable", "N/A", "NA", "Unknown", "",
}


def classify_columns(df: pd.DataFrame) -> dict:
    """
    For every potential disaggregation column, return a classification dict:
      - "breakable"       : non-total codes present AND those rows have OBS_VALUE
      - "codes_no_data"   : non-total codes present BUT those rows are all NaN
      - "total_only"      : only aggregate/total value
      - "skipped"         : in _NON_DISAGG_COLS or starts with _
    """
    results = {}
    obs_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else None

    for col in df.columns:
        if col in _NON_DISAGG_COLS or col.startswith("_"):
            results[col] = {"status": "skipped"}
            continue

        unique_vals = [str(v) for v in df[col].dropna().unique()]
        if not unique_vals:
            results[col] = {"status": "skipped"}
            continue

        non_total = [v for v in unique_vals if v not in _TOTAL_CODES]
        n_non_total = len(non_total)
        n_total = len(unique_vals) - n_non_total

        if n_non_total <= 1:
            results[col] = {"status": "total_only", "unique": len(unique_vals)}
            continue

        # Has multiple non-total values — check if they have data
        if obs_col:
            mask = df[col].astype(str).isin(non_total)
            obs_vals = pd.to_numeric(df.loc[mask, obs_col], errors="coerce")
            n_with_data = int(obs_vals.notna().sum())
        else:
            n_with_data = -1  # unknown

        if n_with_data == 0:
            results[col] = {
                "status": "codes_no_data",
                "n_non_total": n_non_total,
                "sample": sorted(non_total)[:5],
            }
        else:
            results[col] = {
                "status": "breakable",
                "n_non_total": n_non_total,
                "n_with_data": n_with_data,
                "sample": sorted(non_total)[:5],
            }

    return results


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """
    Simulate a naive 'total trend' plot: group by TIME_PERIOD + geography,
    count how many OBS_VALUE rows there are. If >1 per group without filtering,
    a chart would incorrectly aggregate them.
    """
    time_col = next((c for c in ("TIME_PERIOD", "Year", "YEAR") if c in df.columns), None)
    geo_col  = next((c for c in ("Geographic area", "REF_AREA") if c in df.columns), None)

    if not time_col or "OBS_VALUE" not in df.columns:
        return {"duplicate_risk": False, "reason": "no time or obs column"}

    group_cols = [time_col] + ([geo_col] if geo_col else [])
    obs = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df2 = df.copy()
    df2["_obs"] = obs
    df2 = df2[df2["_obs"].notna()]

    if df2.empty:
        return {"duplicate_risk": False, "reason": "no non-null OBS_VALUE"}

    counts = df2.groupby(group_cols)["_obs"].count()
    max_per_group = int(counts.max())
    pct_duplicated = (counts > 1).mean()

    return {
        "duplicate_risk": max_per_group > 1,
        "max_rows_per_group": max_per_group,
        "pct_groups_duplicated": round(float(pct_duplicated), 3),
    }


def sample_indicators(df: pd.DataFrame, n: int = 3) -> list:
    """Return up to n unique indicator_id values from a dataframe."""
    ids = df["indicator_id"].dropna().unique().tolist() if "indicator_id" in df.columns else []
    if not ids:
        # Try to find an indicator-like column
        for col in df.columns:
            if "indicator" in col.lower() or col == "INDICATOR":
                ids = df[col].dropna().unique().tolist()
                break
    return ids[:n]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    csv_files = sorted(CACHE.glob("raw_*.csv"))
    print(f"Found {len(csv_files)} raw CSV files in build_cache.\n")

    records = []
    target = 50
    seen_indicators = set()

    for csv_path in csv_files:
        if len(records) >= target:
            break
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  SKIP {csv_path.name}: {e}")
            continue

        dataflow = csv_path.stem.replace("raw_", "")
        indicators = sample_indicators(df, n=2)

        for ind_id in indicators:
            if len(records) >= target:
                break
            key = (dataflow, str(ind_id))
            if key in seen_indicators:
                continue
            seen_indicators.add(key)

            ind_col = next(
                (c for c in ("indicator_id", "INDICATOR") if c in df.columns), None
            )
            subset = df[df[ind_col] == ind_id].copy() if ind_col else df.copy()
            if subset.empty or len(subset) < 2:
                continue

            col_classes = classify_columns(subset)
            dup_check   = check_duplicate_rows(subset)

            n_breakable    = sum(1 for v in col_classes.values() if v["status"] == "breakable")
            n_codes_no_data= sum(1 for v in col_classes.values() if v["status"] == "codes_no_data")
            n_total_only   = sum(1 for v in col_classes.values() if v["status"] == "total_only")

            breakable_cols = {
                col: v for col, v in col_classes.items() if v["status"] == "breakable"
            }
            codes_no_data_cols = {
                col: v for col, v in col_classes.items() if v["status"] == "codes_no_data"
            }

            records.append({
                "dataflow":          dataflow,
                "indicator_id":      str(ind_id),
                "n_rows":            len(subset),
                "n_cols":            len(df.columns),
                "n_breakable":       n_breakable,
                "n_codes_no_data":   n_codes_no_data,
                "n_total_only":      n_total_only,
                "breakable_cols":    breakable_cols,
                "codes_no_data_cols":codes_no_data_cols,
                "dup_check":         dup_check,
            })

    print(f"Analysed {len(records)} (dataflow, indicator) pairs.\n")

    # ── Summary stats ─────────────────────────────────────────────────────────
    df_rec = pd.DataFrame([{
        "dataflow":         r["dataflow"],
        "indicator_id":     r["indicator_id"],
        "n_rows":           r["n_rows"],
        "n_breakable":      r["n_breakable"],
        "n_codes_no_data":  r["n_codes_no_data"],
        "n_total_only":     r["n_total_only"],
        "dup_risk":         r["dup_check"]["duplicate_risk"],
        "max_per_group":    r["dup_check"].get("max_rows_per_group", 1),
        "pct_dup":          r["dup_check"].get("pct_groups_duplicated", 0),
    } for r in records])

    print("═" * 60)
    print("DISAGGREGATION PATTERN SUMMARY")
    print("═" * 60)
    print(f"{'Indicators analysed:':<35} {len(records)}")
    print(f"{'With ≥1 breakable dimension:':<35} {(df_rec.n_breakable >= 1).sum()} ({(df_rec.n_breakable >= 1).mean():.0%})")
    print(f"{'With ≥3 breakable dimensions:':<35} {(df_rec.n_breakable >= 3).sum()} ({(df_rec.n_breakable >= 3).mean():.0%})")
    print(f"{'With false-positive dims (codes/no data):':<35} {(df_rec.n_codes_no_data >= 1).sum()} ({(df_rec.n_codes_no_data >= 1).mean():.0%})")
    print(f"{'With duplicate-row risk (no filter):':<35} {df_rec.dup_risk.sum()} ({df_rec.dup_risk.mean():.0%})")
    print()
    print(f"Avg breakable dims per indicator:    {df_rec.n_breakable.mean():.1f}")
    print(f"Avg codes-no-data dims:              {df_rec.n_codes_no_data.mean():.1f}")
    print(f"Avg total-only dims:                 {df_rec.n_total_only.mean():.1f}")
    print(f"Avg max rows per (time, geo) group:  {df_rec.max_per_group.mean():.1f}")
    print()

    # ── Breakable dimension name frequency ───────────────────────────────────
    dim_counter: Counter = Counter()
    for r in records:
        for col in r["breakable_cols"]:
            dim_counter[col] += 1

    print("Most common BREAKABLE dimensions (across all indicators):")
    for dim, cnt in dim_counter.most_common(15):
        print(f"  {dim:<35} {cnt:>3} indicators")
    print()

    # ── False-positive dimension frequency ───────────────────────────────────
    fp_counter: Counter = Counter()
    for r in records:
        for col in r["codes_no_data_cols"]:
            fp_counter[col] += 1

    if fp_counter:
        print("Most common FALSE-POSITIVE dimensions (codes present, no data):")
        for dim, cnt in fp_counter.most_common(10):
            print(f"  {dim:<35} {cnt:>3} indicators")
        print()

    # ── Duplicate-row risk cases ──────────────────────────────────────────────
    risky = df_rec[df_rec.dup_risk].sort_values("max_per_group", ascending=False)
    print(f"Duplicate-row risk: {len(risky)} indicators")
    print(f"  (a naive plot without filters would average/sum multiple rows per time point)\n")
    print("  Worst cases:")
    for _, row in risky.head(10).iterrows():
        rec = next(r for r in records
                   if r["dataflow"] == row["dataflow"] and r["indicator_id"] == row["indicator_id"])
        dims = list(rec["breakable_cols"].keys())
        print(f"  {row['dataflow']:<35} {row['indicator_id']:<20} "
              f"max={row['max_per_group']:.0f}x  dims={dims}")
    print()

    # ── High-cardinality breakdown cases ─────────────────────────────────────
    print("High-cardinality breakdowns (>8 unique values — would produce messy charts):")
    found = False
    for r in records:
        for col, info in r["breakable_cols"].items():
            if info["n_non_total"] > 8:
                print(f"  {r['dataflow']:<35} col={col:<25} "
                      f"n={info['n_non_total']}  sample={info['sample'][:3]}")
                found = True
    if not found:
        print("  None found.")
    print()

    # ── Per-indicator detail for worst duplicate cases ────────────────────────
    print("═" * 60)
    print("DETAILED VIEW — top 5 duplicate-risk indicators")
    print("═" * 60)
    for _, row in risky.head(5).iterrows():
        rec = next(r for r in records
                   if r["dataflow"] == row["dataflow"] and r["indicator_id"] == row["indicator_id"])
        print(f"\n{rec['dataflow']} / {rec['indicator_id']}  ({rec['n_rows']} rows)")
        print(f"  Breakable dims ({rec['n_breakable']}):")
        for col, info in rec["breakable_cols"].items():
            print(f"    {col:<30} n_cats={info['n_non_total']}  "
                  f"rows_with_data={info['n_with_data']}  sample={info['sample'][:4]}")
        print(f"  Dup check: max {row['max_per_group']:.0f} rows per (time,geo) group, "
              f"{row['pct_dup']:.0%} of groups have duplicates")


if __name__ == "__main__":
    main()
