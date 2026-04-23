"""
SDMX agent powered by Azure OpenAI.

Pipeline:
  1. search_data_dictionary  — keyword search within country partition file(s)
  2. query_sdmx_api          — fetch data from the UNICEF SDMX REST endpoint
  3. run_quality_checks      — validate completeness, coverage, and plausibility
"""

import json  # used for tool call argument parsing
import os
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Data – loaded once at first use
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DICTIONARY_FILE = ROOT / "data" / "data_dictionary.parquet"
EMBEDDINGS_FILE = ROOT / "data" / "embeddings" / "indicator_embeddings.parquet"
SDMX_BASE_URL = "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest"

_DICTIONARY_DF: Optional[pd.DataFrame] = None
_COUNTRY_KEYS: Optional[List[str]] = None   # unique lowercase country values
_EMBEDDINGS_DF: Optional[pd.DataFrame] = None
_EMBEDDINGS_MATRIX: Optional[np.ndarray] = None
# Accumulates every DataFrame fetched during one agent run
_fetched_dfs: List[pd.DataFrame] = []
_source_urls: List[Dict] = []   # {geography_name, indicator_id, url} per fetch
_viz_spec: Optional[Dict] = None
_pending_reset: bool = False  # lazy reset — cleared on first API call of a new question

# Cache populated by search_data_dictionary: (indicator_id, geography_id) → [(agency, dataflow_id, geo_id, actual_indicator_id)]
_dataflow_cache: Dict[Tuple[str, str], List[Tuple]] = {}
# UNICEF fallback flows per geography, built across all concepts during search.
# Used as last resort when all primary cache candidates fail.
# geo_id → [(agency, dataflow_id, actual_indicator_id)]
_unicef_fallbacks: Dict[str, List[Tuple[str, str, str]]] = {}
# Circuit breaker: counts consecutive failures per "agency/dataflow_id".
# Dataflows that reach CIRCUIT_OPEN_THRESHOLD are skipped for the session.
# Resets on session reset so a dataflow that recovers is retried next session.
_dataflow_failures: Dict[str, int] = {}
CIRCUIT_OPEN_THRESHOLD = 3


def get_fetched_dfs() -> List[pd.DataFrame]:
    return _fetched_dfs


def get_source_urls() -> List[Dict]:
    return _source_urls


def get_viz_spec() -> Optional[Dict]:
    return _viz_spec


def reset_session_data() -> None:
    global _fetched_dfs, _source_urls, _viz_spec, _pending_reset, _unicef_fallbacks, _dataflow_failures
    _fetched_dfs = []
    _source_urls = []
    _viz_spec = None
    _pending_reset = False
    _unicef_fallbacks = {}
    _dataflow_failures = {}


def mark_new_question() -> None:
    """Signal that a new question was submitted without clearing data yet.
    Data is cleared lazily on the first query_sdmx_api call so that follow-up
    questions that don't fetch new data leave the existing chart visible.
    """
    global _pending_reset
    _pending_reset = True


def _load_dictionary() -> pd.DataFrame:
    global _DICTIONARY_DF, _COUNTRY_KEYS
    if _DICTIONARY_DF is None:
        _DICTIONARY_DF = pd.read_parquet(DICTIONARY_FILE)
        _COUNTRY_KEYS = sorted(_DICTIONARY_DF["country"].dropna().unique().tolist())
    return _DICTIONARY_DF


def _resolve_country(name: str) -> Optional[str]:
    """Return the canonical country key for a user-supplied name, or None."""
    _load_dictionary()
    keys = _COUNTRY_KEYS
    normalised = name.lower().strip()

    # 1. Exact match
    if normalised in keys:
        return normalised

    # 2. Substring match — "cambodia" in "cambodia", "drc" in "democratic republic..."
    for key in keys:
        if normalised in key or key in normalised:
            return key

    # 3. Token-prefix match — "Laos" → "lao people's democratic republic"
    query_tokens = [t for t in normalised.split() if len(t) >= 3]
    for key in keys:
        key_tokens = [t for t in key.split() if len(t) >= 3]
        for qt in query_tokens:
            for kt in key_tokens:
                if qt.startswith(kt) or kt.startswith(qt):
                    return key

    return None


def _load_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:
    global _EMBEDDINGS_DF, _EMBEDDINGS_MATRIX
    if _EMBEDDINGS_DF is None:
        df = pd.read_parquet(EMBEDDINGS_FILE)
        matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        _EMBEDDINGS_MATRIX = matrix / norms
        _EMBEDDINGS_DF = df.reset_index(drop=True)
    return _EMBEDDINGS_DF, _EMBEDDINGS_MATRIX


def _rank_dataflow(agency: str, dataflow_id: str) -> int:
    """Higher score = more likely to return live API data."""
    score = 0
    if agency.upper() == "UNICEF":
        score += 10
    if "CONSOLIDATED" not in dataflow_id.upper():
        score += 5
    if "ALL" not in dataflow_id.upper():
        score += 3
    return score


def _iso_geo_id(row: pd.Series) -> str:
    """
    Return whichever of geography_id / geography looks like an ISO country code
    (2–4 uppercase letters/digits). Falls back to geography_id.
    Different dataflows store the ISO code in different columns.
    """
    for col in ("geography_id", "geography"):
        val = str(row.get(col, "") or "")
        if val and len(val) <= 4 and val == val.upper() and val.isalpha():
            return val
    return str(row.get("geography_id", ""))


def _embed_query(query: str) -> np.ndarray:
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    response = client.embeddings.create(model=deployment, input=query)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# Tool 1 – search data dictionary (semantic, embedding-based)
# ---------------------------------------------------------------------------

def search_data_dictionary(query: str, countries: List[str], top_k: int = 8) -> str:
    """Search the data dictionary for indicators matching the query using semantic search."""
    top_k = min(top_k, 20)

    full_df = _load_dictionary()

    found: List[str] = []
    missing: List[str] = []
    frames: List[pd.DataFrame] = []

    for name in countries:
        key = _resolve_country(name)
        if key is None:
            missing.append(name)
            continue
        found.append(key)
        subset = full_df[full_df["country"] == key].copy()
        subset["_country_key"] = key
        frames.append(subset)

    lines: List[str] = []
    if missing:
        lines.append(f"Countries not found in dictionary: {missing}")
        suggestions = [k for k in _COUNTRY_KEYS if any(m.lower()[:4] in k for m in missing)][:5]
        if suggestions:
            lines.append(f"  Did you mean one of: {suggestions}?")

    if not frames:
        lines.append("No data loaded – check country names and try again.")
        return "\n".join(lines)

    combined = pd.concat(frames, ignore_index=True)
    lines.append(f"Loaded {len(combined):,} rows for: {found}. Searching for '{query}'...\n")

    # Semantic search: embed query, score against pre-computed indicator embeddings
    emb_df, emb_matrix = _load_embeddings()
    query_vec = _embed_query(query)
    similarities = emb_matrix @ query_vec  # cosine similarity (vectors are pre-normalised)

    # Get top-k indicator_ids by similarity that exist in the loaded partitions.
    # Search 3× top_k candidates so deduplication doesn't starve the final list.
    MIN_SIMILARITY = 0.40
    available_ids = set(combined["indicator_id"].unique())
    ranked = sorted(
        ((float(similarities[i]), str(emb_df.at[i, "indicator_id"]))
         for i in range(len(emb_df))
         if str(emb_df.at[i, "indicator_id"]) in available_ids
         and float(similarities[i]) >= MIN_SIMILARITY),
        reverse=True,
    )

    # Build lookups from the embeddings (canonical names/text used at index time).
    # Using embedding text for deduplication avoids partitions where the same indicator_id
    # was mapped to a different indicator name across countries (data inconsistency).
    emb_text_lookup: dict = {
        str(emb_df.at[i, "indicator_id"]): str(emb_df.at[i, "text"])
        for i in range(len(emb_df))
    }
    sim_lookup: dict = {
        str(emb_df.at[i, "indicator_id"]): float(similarities[i])
        for i in range(len(emb_df))
    }

    # Group ranked candidates by concept name. Same-concept variants (e.g. B12, COB12,
    # IM_BCG all named "Immunization - BCG") are merged: their country coverage is unioned
    # and UNICEF fallback flows are shared across countries so the API is tried even when
    # the local partition is incomplete.
    def _norm_emb_name(iid: str) -> str:
        text = emb_text_lookup.get(iid, iid)
        name = text.split(".")[0].lower()
        return re.sub(r"[^a-z0-9]", "", name)

    def _display_name(country_key: str) -> str:
        return re.sub(r"'([A-Z])", lambda m: "'" + m.group(1).lower(), country_key.title())

    # Build concept groups: primary indicator_id (highest sim) + list of secondary ids
    seen_names: set = set()
    top_ids: List[str] = []
    concept_secondaries: Dict[str, List[str]] = {}  # primary_id -> [secondary_ids]

    for _, iid in ranked:
        norm = _norm_emb_name(iid)
        if norm not in seen_names:
            seen_names.add(norm)
            if len(top_ids) < top_k:
                top_ids.append(iid)
                concept_secondaries[iid] = []
        else:
            primary = next((p for p in top_ids if _norm_emb_name(p) == norm), None)
            if primary:
                concept_secondaries[primary].append(iid)

    # Load rows for all variant indicator_ids (primaries + secondaries)
    all_variant_ids = set(top_ids)
    for secs in concept_secondaries.values():
        all_variant_ids.update(secs)
    hits = combined[combined["indicator_id"].isin(all_variant_ids)].copy()

    # Build one record per concept, merging country coverage across all variants.
    records: List[dict] = []
    for primary_id in top_ids:
        secondary_ids = concept_secondaries[primary_id]
        variant_ids = [primary_id] + secondary_ids

        # country_key → {iso_id, display_name, flows}
        # flows are 4-tuples: (agency, dataflow_id, geo_id, actual_indicator_id)
        country_map: Dict[str, dict] = {}
        unicef_flows: List[Tuple] = []  # (agency, dataflow_id, indicator_id) to borrow

        for var_id in variant_ids:
            var_hits = hits[hits["indicator_id"] == var_id]
            for country_key, grp in var_hits.groupby("_country_key", sort=False):
                grp = grp.copy()
                grp["_df_score"] = grp.apply(
                    lambda r: _rank_dataflow(r["agency"], r["dataflow_id"]), axis=1
                )
                if grp.empty:
                    continue
                grp = grp.sort_values("_df_score", ascending=False)
                best_row = grp.iloc[0]
                iso_id = _iso_geo_id(best_row)
                flows = [
                    (str(r["agency"]), str(r["dataflow_id"]), iso_id, str(var_id))
                    for _, r in grp.drop_duplicates(subset=["agency", "dataflow_id"]).iterrows()
                ]
                # Collect UNICEF flows for cross-country fallback borrowing
                for ag, df_id, _, ind in flows:
                    if ag.upper() == "UNICEF":
                        key3 = (ag, df_id, ind)
                        if key3 not in unicef_flows:
                            unicef_flows.append(key3)
                if country_key not in country_map:
                    country_map[country_key] = {
                        "iso_id": iso_id,
                        "display_name": _display_name(country_key),
                        "flows": flows,
                    }

        # Borrow UNICEF flows into countries that lack them (covers partition gaps)
        if unicef_flows:
            for country_key, data in country_map.items():
                existing = {(ag, df_id) for ag, df_id, *_ in data["flows"]}
                for ag, df_id, ind in unicef_flows:
                    if (ag, df_id) not in existing:
                        data["flows"].append((ag, df_id, data["iso_id"], ind))

            # Also add countries missing entirely from partition but in requested list
            for country_key in found:
                if country_key not in country_map:
                    part = _load_partition(country_key)
                    sample = part.head(1)
                    if sample.empty:
                        continue
                    iso_id = _iso_geo_id(sample.iloc[0])
                    flows = [(ag, df_id, iso_id, ind) for ag, df_id, ind in unicef_flows]
                    country_map[country_key] = {
                        "iso_id": iso_id,
                        "display_name": _display_name(country_key),
                        "flows": flows,
                    }

        # Register UNICEF flows globally (after country_map is fully built) so
        # query_sdmx_api can use them as last resort for any geography.
        if unicef_flows:
            for data in country_map.values():
                iso_id = data["iso_id"]
                if iso_id not in _unicef_fallbacks:
                    _unicef_fallbacks[iso_id] = []
                seen = {(ag, df_id) for ag, df_id, _ in _unicef_fallbacks[iso_id]}
                for ag, df_id, ind in unicef_flows:
                    if (ag, df_id) not in seen:
                        _unicef_fallbacks[iso_id].append((ag, df_id, ind))
                        seen.add((ag, df_id))

        # Populate the dataflow cache keyed by (primary_id, iso_id)
        for data in country_map.values():
            _dataflow_cache[(primary_id, data["iso_id"])] = data["flows"]

        available_countries = [
            {"country_key": ck, "geography_id": d["iso_id"], "display_name": d["display_name"]}
            for ck, d in country_map.items()
        ]
        missing_countries = [k for k in found if k not in country_map]

        meta_rows = hits[hits["indicator_id"] == primary_id]
        meta = meta_rows.iloc[0] if not meta_rows.empty else hits[hits["indicator_id"].isin(variant_ids)].iloc[0]
        emb_text = emb_text_lookup.get(primary_id, "")
        canonical_name = emb_text.split(".")[0].strip() if emb_text else str(meta["indicator"])
        records.append({
            "indicator_id":        primary_id,
            "indicator":           canonical_name,
            "definition":          meta.get("definition"),
            "category":            str(meta["category"]),
            "available_countries": available_countries,
            "missing_countries":   missing_countries,
        })

    lines.append(f"Top {len(records)} indicator matches:\n")
    for rec in records:
        sim = sim_lookup.get(rec["indicator_id"], 0.0)
        defn = str(rec.get("definition") or "N/A")[:200]
        avail_str = ", ".join(
            f"{ac['display_name']} ({ac['geography_id']})"
            for ac in rec["available_countries"]
        ) or "none"
        entry = (
            f"indicator_id={rec['indicator_id']}\n"
            f"   Name: {rec['indicator']}\n"
            f"   Definition: {defn}\n"
            f"   Available for: {avail_str}\n"
        )
        if rec["missing_countries"]:
            entry += f"   NOT available for: {', '.join(rec['missing_countries'])}\n"
        entry += f"   Similarity: {sim:.3f}\n"
        lines.append(entry)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 – ask user to select an indicator
# ---------------------------------------------------------------------------

def ask_user_to_select_indicator(question: str) -> str:
    """Present indicator options to the user and return their selection."""
    print(f"\n{question}")
    user_input = input("\nYour choice: ").strip()
    return f"User selected: {user_input}"


# ---------------------------------------------------------------------------
# Tool 3 – query SDMX API
# ---------------------------------------------------------------------------

def _dataflow_quality_score(df: pd.DataFrame) -> float:
    """Score a dataflow result on completeness, recency and time span."""
    score = 0.0
    if "OBS_VALUE" not in df.columns:
        return score

    obs = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    total = len(obs)
    if total == 0:
        return score

    # Completeness (weight 0.4)
    completeness = obs.notna().sum() / total
    score += completeness * 0.4

    time_col = next((c for c in ("TIME_PERIOD", "Year", "YEAR") if c in df.columns), None)
    if time_col is not None:
        years = pd.to_numeric(df[time_col], errors="coerce").dropna()
        if not years.empty:
            # Recency: normalise against a reference of 2025 (weight 0.35)
            recency = min((years.max() - 2000) / 25, 1.0)
            score += recency * 0.35
            # Span: cap at 15 years (weight 0.25)
            span = min(years.nunique() / 15, 1.0)
            score += span * 0.25

    return score


def _fetch_dataflow(
    agency: str,
    dataflow_id: str,
    indicator_id: str,
    geography_id: str,
    geography_name: str,
    qs: str,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Try three URL patterns for one (agency, dataflow_id) pair. Returns (df, url) or None."""
    flow_ref = f"{agency},{dataflow_id},1.0"
    urls = [
        f"{SDMX_BASE_URL}/data/{flow_ref}/{geography_id}.{indicator_id}{qs}",
        f"{SDMX_BASE_URL}/data/{flow_ref}/.{geography_id}..{indicator_id}{qs}",
        f"{SDMX_BASE_URL}/data/{flow_ref}/{indicator_id}..{geography_name}.{qs}",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=20)
        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None
        if resp.status_code == 200:
            try:
                df = pd.read_csv(StringIO(resp.text))
                if not df.empty:
                    return df, url
            except Exception:
                pass
    return None


def query_sdmx_api(
    agency: str,
    dataflow_id: str,
    indicator_id: str,
    geography_id: str,
    geography_name: str,
    start_period: str = "2015",
    end_period: str = "2023",
) -> str:
    """Try all cached dataflows for this indicator, rank by quality, return the best."""
    qs = f"?format=csv&labels=both&startPeriod={start_period}&endPeriod={end_period}"

    # Use all known dataflows from the search cache; fall back to what the model passed.
    # Cache values are 4-tuples: (agency, dataflow_id, geo_id, actual_indicator_id).
    # actual_indicator_id may differ from the queried indicator_id when variants of the
    # same concept use different IDs in different dataflow families (e.g. B12 vs IM_BCG).
    cache_key = (str(indicator_id), str(geography_id))
    raw_candidates = _dataflow_cache.get(cache_key, [(agency, dataflow_id, geography_id, indicator_id)])

    def _normalise(c: tuple) -> tuple:
        if len(c) == 4:
            return c
        if len(c) == 3:
            return (c[0], c[1], c[2], indicator_id)
        return (c[0], c[1], geography_id, indicator_id)

    candidates = [_normalise(c) for c in raw_candidates]

    results: List[Tuple[float, pd.DataFrame, str, str, str]] = []  # score, df, url, ag, df_id

    for ag, df_id, geo, actual_ind_id in candidates:
        flow_key = f"{ag}/{df_id}"
        if _dataflow_failures.get(flow_key, 0) >= CIRCUIT_OPEN_THRESHOLD:
            continue  # circuit open — skip until next session reset
        outcome = _fetch_dataflow(ag, df_id, actual_ind_id, geo, geography_name, qs)
        if outcome is not None:
            _dataflow_failures[flow_key] = 0  # success — reset counter
            df, url = outcome
            score = _dataflow_quality_score(df)
            results.append((score, df, url, ag, df_id))
        else:
            _dataflow_failures[flow_key] = _dataflow_failures.get(flow_key, 0) + 1

    if not results:
        # Last resort: try UNICEF fallback flows registered during search for this geography
        fallbacks = _unicef_fallbacks.get(geography_id, [])
        primary_keys = {(ag, df_id) for ag, df_id, *_ in candidates}
        for ag, df_id, actual_ind_id in fallbacks:
            if (ag, df_id) in primary_keys:
                continue  # already tried
            flow_key = f"{ag}/{df_id}"
            if _dataflow_failures.get(flow_key, 0) >= CIRCUIT_OPEN_THRESHOLD:
                continue
            outcome = _fetch_dataflow(ag, df_id, actual_ind_id, geography_id, geography_name, qs)
            if outcome is not None:
                _dataflow_failures[flow_key] = 0
                df, url = outcome
                score = _dataflow_quality_score(df)
                results.append((score, df, url, ag, df_id))
            else:
                _dataflow_failures[flow_key] = _dataflow_failures.get(flow_key, 0) + 1

    if not results:
        tried = ", ".join(f"{ag}/{df_id}(geo={geo},ind={ind})" for ag, df_id, geo, ind in candidates)
        return (
            f"No data found for indicator={indicator_id}, geography={geography_id}. "
            f"Tried {len(candidates)} dataflow(s): {tried}."
        )

    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_df, best_url, best_ag, best_df_id = results[0]

    summary_lines = [
        f"Found data in {len(results)} dataflow(s). "
        f"Selected: {best_ag}/{best_df_id} (quality score: {best_score:.2f})",
    ]
    if len(results) > 1:
        others = ", ".join(
            f"{ag}/{df_id} ({s:.2f})" for s, _, _, ag, df_id in results[1:]
        )
        summary_lines.append(f"Others tested: {others}")

    return "\n".join(summary_lines) + "\n\n" + _format_csv_response(
        best_df, indicator_id, geography_id, best_url
    )


def _format_csv_response(df: pd.DataFrame, indicator_id: str, geography_id: str, url: str) -> str:
    global _fetched_dfs, _source_urls, _viz_spec, _pending_reset
    if _pending_reset:
        _fetched_dfs = []
        _source_urls = []
        _viz_spec = None
        _pending_reset = False

    # Replace existing entry for this geography/indicator rather than appending
    # (prevents duplicates when the agent retries or calls with different dataflows)
    existing_idx = next(
        (i for i, s in enumerate(_source_urls)
         if s["geography_id"] == geography_id and s["indicator_id"] == indicator_id),
        None,
    )
    if existing_idx is not None:
        _fetched_dfs[existing_idx] = df
        _source_urls[existing_idx]["url"] = url
    else:
        _fetched_dfs.append(df)
    # Extract data source label — column name varies by dataflow
    _DATA_SOURCE_COLS = ["DATA_SOURCE", "Short_Source", "SHORT_SOURCE", "CDDATASOURCE",
                         "PROVIDER", "Data Source", "Source"]
    data_source = None
    for col in _DATA_SOURCE_COLS:
        if col in df.columns:
            val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if val:
                data_source = str(val)
                break
    if existing_idx is None:
        _source_urls.append({
            "geography_id":  geography_id,
            "indicator_id":  indicator_id,
            "url":           url,
            "data_source":   data_source,
        })
    else:
        _source_urls[existing_idx]["data_source"] = data_source

    if df.empty:
        return f"API returned an empty CSV from {url}."

    lines = [
        f"Source URL: {url}",
        f"Indicator: {indicator_id}  |  Geography: {geography_id}",
        f"Rows returned: {len(df)}",
        f"Columns: {', '.join(df.columns.tolist())}",
    ]

    if "OBS_VALUE" in df.columns:
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        obs_vals = df["OBS_VALUE"].dropna()
        total, nonnull = len(df), len(obs_vals)
        lines += [f"Total observations: {total}", f"Non-null observations: {nonnull} / {total}"]
        if not obs_vals.empty:
            lines += [f"Value range: {obs_vals.min():.4g} – {obs_vals.max():.4g}", f"Mean value: {obs_vals.mean():.4g}"]

    for time_col in ("TIME_PERIOD", "Year", "YEAR"):
        if time_col in df.columns:
            periods = sorted(df[time_col].dropna().unique().tolist())
            lines.append(f"Time periods covered: {periods[0]} – {periods[-1]}  ({len(periods)} periods)")
            break

    for geo_col in ("Geographic area", "REF_AREA", "GEOGRAPHIC_AREA"):
        if geo_col in df.columns:
            lines.append(f"Geographies in response: {df[geo_col].unique().tolist()}")
            break

    lines.append(f"\nFirst 5 rows:\n{df.head(5).to_string(index=False)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4 – quality checks
# ---------------------------------------------------------------------------

def run_quality_checks(
    data_summary: str,
    indicator_id: str,
    indicator_name: str,
    geography_id: str,
) -> str:
    """Validate retrieved SDMX data and return a PASS / PARTIAL / FAIL report."""
    issues: List[str] = []
    passed: List[str] = []
    score = 1.0

    failed_phrases = ["No data found", "404", "HTTP 4", "HTTP 5", "timed out", "Network error"]
    if any(p in data_summary for p in failed_phrases):
        issues.append("CRITICAL: Data retrieval failed or returned no usable data.")
        score = 0.0
    else:
        passed.append("Data retrieved successfully from the API.")

    if score > 0:
        m = re.search(r"Total observations: (\d+)", data_summary)
        if m:
            n_obs = int(m.group(1))
            if n_obs == 0:
                issues.append("WARNING: Zero observations in the dataset.")
                score -= 0.3
            elif n_obs < 5:
                issues.append(f"WARNING: Very few observations ({n_obs}).")
                score -= 0.1
            else:
                passed.append(f"Observation count is adequate ({n_obs}).")

        m = re.search(r"Non-null observations: (\d+) / (\d+)", data_summary)
        if m:
            nonnull, total = int(m.group(1)), int(m.group(2))
            null_rate = (total - nonnull) / total if total > 0 else 0
            if null_rate > 0.5:
                issues.append(f"WARNING: High null rate ({null_rate:.0%}).")
                score -= 0.2
            else:
                passed.append(f"Completeness is good ({1 - null_rate:.0%} non-null).")

        m = re.search(r"Value range: ([\d.e+-]+) . ([\d.e+-]+)", data_summary)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            rate_kw = {"rate", "ratio", "percent", "proportion", "share", "coverage"}
            is_rate = bool(rate_kw & set(indicator_name.lower().split()))
            if lo < 0:
                issues.append(f"{'CRITICAL' if is_rate else 'WARNING'}: Negative minimum value ({lo:.4g}).")
                score -= 0.3 if is_rate else 0.1
            if is_rate and hi > 1000:
                issues.append(f"WARNING: Max value ({hi:.4g}) unusually high for a rate.")
                score -= 0.1
            else:
                passed.append(f"Values in plausible range ({lo:.4g} – {hi:.4g}).")

    score = max(0.0, min(1.0, score))
    verdict = "PASS" if score >= 0.7 else ("PARTIAL" if score >= 0.4 else "FAIL")

    lines = [
        "=== Quality Check Report ===",
        f"Indicator : {indicator_name} ({indicator_id})",
        f"Geography : {geography_id}",
        f"Score     : {score:.0%}  |  Verdict: {verdict}",
        "",
        f"Checks passed ({len(passed)}):",
        *[f"  ✓ {c}" for c in passed],
    ]
    if issues:
        lines += [f"\nIssues ({len(issues)}):", *[f"  ✗ {i}" for i in issues]]
    else:
        lines.append("\nNo issues detected.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5 – create visualization
# ---------------------------------------------------------------------------

def create_visualization(
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str,
    color_by: Optional[str] = None,
    filters: Optional[Dict] = None,
) -> str:
    """Store a visualization spec; the UI will render it from all accumulated data."""
    global _viz_spec
    _viz_spec = {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "color_by": color_by,
        "title": title,
        "filters": filters or {},
    }
    return f"Visualization spec stored: {chart_type} chart — x={x_column}, y={y_column}, color={color_by}."


# ---------------------------------------------------------------------------
# Tool schemas for Azure OpenAI function calling
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_data_dictionary",
            "description": (
                "Search the UNICEF SDMX data dictionary for indicators relevant to the query. "
                "Loads only the partition file(s) for the requested countries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of the data you are looking for."},
                    "countries": {"type": "array", "items": {"type": "string"}, "description": "List of country names mentioned in the question, e.g. [\"Cambodia\", \"Brazil\"]."},
                    "top_k": {"type": "integer", "description": "Number of top results to return (default 8, max 20)."},
                },
                "required": ["query", "countries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_to_select_indicator",
            "description": (
                "Present the top indicator matches to the user and ask them to choose one. "
                "Call this after search_data_dictionary, before querying the API. "
                "Format the options clearly — number them and include indicator name, category, and dataflow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The full message to show the user, including a numbered list of indicator options "
                            "and a clear prompt asking which one they want to explore."
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_sdmx_api",
            "description": (
                "Query the UNICEF SDMX REST API to retrieve actual data. "
                "Tries three URL patterns automatically and returns a CSV-based summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agency":         {"type": "string", "description": "Agency code from the search results, e.g. 'UNICEF'."},
                    "dataflow_id":    {"type": "string", "description": "Dataflow ID from the search results, e.g. 'CME'."},
                    "indicator_id":   {"type": "string", "description": "Indicator ID, e.g. 'CME_MRY0T4'."},
                    "geography_id":   {"type": "string", "description": "Geographic area code, e.g. 'KHM'."},
                    "geography_name": {"type": "string", "description": "Human-readable geography name, e.g. 'Cambodia'."},
                    "start_period":   {"type": "string", "description": "Start year, default '2015'."},
                    "end_period":     {"type": "string", "description": "End year, default '2023'."},
                },
                "required": ["agency", "dataflow_id", "indicator_id", "geography_id", "geography_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_visualization",
            "description": (
                "Specify how to visualize all the data collected so far. Call this after all "
                "query_sdmx_api calls are done. The UI will combine all fetched datasets and render "
                "the chart you describe. Choose chart type, axes, and color grouping based on what "
                "best answers the user's question (e.g. multi-country → color by geography; "
                "single country over time → color by disaggregation like sex)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["line", "bar"],
                        "description": "Use 'line' for time trends, 'bar' for comparisons at a point in time.",
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column name for the x-axis, e.g. 'TIME_PERIOD' or 'Geographic area'.",
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column name for the y-axis, almost always 'OBS_VALUE'.",
                    },
                    "title": {"type": "string", "description": "Chart title."},
                    "color_by": {
                        "type": "string",
                        "description": "Column to split into separate lines/bars, e.g. 'Geographic area' or 'Sex'.",
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Column-value pairs to filter rows before plotting, e.g. "
                            "{\"SEX\": \"_T\", \"WEALTH_QUINTILE\": \"_T\"}. "
                            "Use this to remove unwanted disaggregations."
                        ),
                    },
                },
                "required": ["chart_type", "x_column", "y_column", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_quality_checks",
            "description": "Run quality checks on SDMX data: completeness, null rate, value plausibility. Returns PASS/PARTIAL/FAIL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_summary":   {"type": "string", "description": "Full string returned by query_sdmx_api."},
                    "indicator_id":   {"type": "string", "description": "Indicator ID being validated."},
                    "indicator_name": {"type": "string", "description": "Human-readable indicator name."},
                    "geography_id":   {"type": "string", "description": "Geographic area code."},
                },
                "required": ["data_summary", "indicator_id", "indicator_name", "geography_id"],
            },
        },
    },
]

_TOOL_MAP = {
    "search_data_dictionary": search_data_dictionary,
    "ask_user_to_select_indicator": ask_user_to_select_indicator,
    "query_sdmx_api": query_sdmx_api,
    "create_visualization": create_visualization,
    "run_quality_checks": run_quality_checks,
}

# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a UNICEF data analyst assistant. "
    "Answer the user's question by following these steps in order:\n"
    "1. Call search_data_dictionary to retrieve the top matching indicators.\n"
    "   The results show, for each indicator, which countries it is available for "
    "   and which are NOT available.\n"
    "2. Call ask_user_to_select_indicator to present the options and let the user choose. "
    "   For each option include: indicator name and in parentheses the countries "
    "   where data is available. If some requested countries are not available, note them too. "
    "   Example: '1. Immunization - DTP3 (Thailand, Myanmar) — not available in: Vietnam, Laos'\n"
    "   Do NOT list the category or individual dataflows as separate options.\n"
    "3. Call query_sdmx_api ONLY for countries where the selected indicator is available. "
    "   The tool automatically tests all available dataflows and returns the best data. "
    "   Note the exact column names in each response — you will need them for the chart.\n"
    "   If a country has no data, mention it clearly in your final answer.\n"
    "4. Call create_visualization to specify the chart. "
    "   First inspect the column names and unique values returned by query_sdmx_api, then apply these rules:\n"
    "   • Multiple countries, single indicator over time → line chart, color by geography column.\n"
    "   • Single country, single indicator over time, one disaggregation of interest (e.g. Sex with 2–3 values) → line chart, color by that disaggregation.\n"
    "   • Comparison at a single point in time → bar chart, x = geography or category.\n"
    "   • CRITICAL — avoid color_by on high-cardinality columns: if a column has more than 8 unique values (e.g. age groups, wealth quintiles), do NOT use it as color_by. Instead:\n"
    "     - Use filters to pick the most relevant value (e.g. a specific age group or total).\n"
    "     - Or use it as the x-axis in a bar chart to show distribution at the most recent time point (add a filter on TIME_PERIOD for the latest year).\n"
    "   • Always filter out irrelevant disaggregations to total/aggregate values (e.g. SEX='_T', AGE='_T') unless the question specifically asks about breakdowns.\n"
    "5. Call run_quality_checks for each dataset fetched.\n"
    "6. Summarise the findings clearly, including key statistics and any quality concerns. "
    "   If some countries had no data for the selected indicator, say so.\n"
    "Always ask the user to choose an indicator — never pick one on their behalf."
)


def run_sdmx_agent(user_question: str) -> str:
    """Orchestrate the three tools via Azure OpenAI function calling."""
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_LLM_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_question},
    ]

    print(f"\n{'='*60}\nSDMX Agent — {user_question}\n{'='*60}")

    while True:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print(msg.content)
            return msg.content

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(f"[tool] {fn_name}({fn_args})")
            result = _TOOL_MAP[fn_name](**fn_args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the under-five mortality rate in Cambodia?"
    answer = run_sdmx_agent(question)
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
