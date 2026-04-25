# CLAUDE.md — project context for AI assistants

## What this project is

A Streamlit chatbot that lets users query UNICEF's SDMX data in plain English. The agent searches a pre-built indicator dictionary, calls the SDMX REST API, runs quality checks, and renders interactive charts.

## Run command

```bash
conda run -n genai_vector streamlit run app.py
```

## Key files

| File | Role |
|---|---|
| `app.py` | Streamlit UI, chart rendering (`_render_chart`), agent loop wrapper |
| `agents/sdmx_agent.py` | All agent tools, session state (`AgentSession`), system prompt (`SYSTEM_PROMPT`), tool schemas (`TOOLS`) |
| `scripts/build_data_dictionary.py` | Full pipeline to rebuild `data/data_dictionary.parquet` |
| `scripts/embed_indicators.py` | Recompute indicator embeddings |
| `data/data_dictionary.parquet` | 1.3M rows — one row per (indicator × geography × dataflow). Columns: `indicator_id`, `indicator`, `geography_id`, `geography`, `definition`, `concatenated_sources`, `agency`, `dataflow_id`, `dataflow_name`, `category`, `national`, `national_country`, `subnational_code_retrieved`, `subnational_country_retrieved`, `subnational_country_retrieved_from_code`, `country`, `year_min`, `year_max` |
| `data/embeddings/indicator_embeddings.parquet` | Pre-computed embeddings (dim 3072) for semantic search |

## Architecture

```
User question
  → search_data_dictionary   (semantic search, embedding cosine similarity)
  → ask_user_to_select_indicator
  → query_sdmx_api           (tries all cached dataflows, picks best)
  → get_data_dimensions      (reports breakdowns + time range in fetched data)
  → create_visualization     (stores viz spec in session)
  → run_quality_checks
  → answer
```

## Agent mode selection

The system prompt has an explicit decision tree at the top. Always choose before acting:

- **Mode A (re-fetch)** — any of: no data yet, different indicator/country, user asks about data recency or a time period not in the fetched data
- **Mode B (re-visualise)** — all of: data fetched, same indicator+country, user only wants a different breakdown/view

## Session state

`AgentSession` (in `sdmx_agent.py`) is per-user. Key fields:
- `fetched_dfs` — list of DataFrames currently in memory
- `viz_spec` — latest chart spec dict
- `dataflow_cache` — maps `(indicator_id, geography_id)` → list of candidate 4-tuples
- `unicef_fallbacks` — same key → UNICEF-specific flows for cross-country borrowing
- `current_query_vec` — embedding of the current user query (used for response validation)
- `fetch_log` — list of dicts describing each dataflow attempt (shown in debug expander)

## Chart rendering (app.py)

`_render_chart(dfs, spec, key)` handles:
- Applying `filters` from the viz spec (including TIME_PERIOD for year-specific views)
- Skipping filters on the `color_by` column
- Robust geo filter aliases when the LLM uses a variant column name
- `_prefer_label_col` — swaps code columns (`WEALTH_QUINTILE`) for label columns (`Wealth Quintile`)
- `_resolve_duplicates` — collapses extra disaggregation dimensions to their Total value
- `_NOT_APPLICABLE_VALUES` — strips `_Z` / "Not applicable" from breakdown dimensions
- High-cardinality cap at 8 series

## Data dictionary rebuild

```bash
# Full rebuild (downloads all 136 SDMX dataflows)
python scripts/build_data_dictionary.py

# Resume flags
--skip-download   reuse data/build_cache/raw_mapping.parquet
--skip-classify   reuse data/build_cache/classified.parquet (merge categories, no API calls)
--skip-embed      skip running embed_indicators.py at the end
```

Raw CSVs are cached per-dataflow in `data/build_cache/` (gitignored, ~8 GB).

## Environment variables (`.env`)

```
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_LLM_DEPLOYMENT      # gpt-4.1-mini
AZURE_OPENAI_LLM_API_VERSION     # 2025-04-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT  # text-embedding-3-large
AZURE_OPENAI_EMBEDDING_API_VERSION # 2024-02-01
APP_PASSWORD                     # optional — leave empty for open access
DEBUG                            # set to 1 to show dataflow selection log in UI
```

## Important conventions

- **No date filters on SDMX queries** — always fetch full history; filter by TIME_PERIOD in `create_visualization` if a specific year is needed
- **Year range is per (indicator × country × best-dataflow)** — `year_min`/`year_max` reflect the dataflow that will actually be used, not the global union across all dataflows
- **`get_data_dimensions` output drives what breakdowns are offered** — the LLM must never suggest a dimension not listed under "Breakdowns available" in its output
- **`_NON_DISAGG_COLS`** — defined in both `app.py` and `sdmx_agent.py`; keep them in sync if adding columns
- **`_TOTAL_CODES` (agent) and `_TOTAL_VALUES` (app)** — intentionally different: `_TOTAL_CODES` is used to detect non-breakable dimensions (includes `_Z`); `_TOTAL_VALUES` is used to pick which row to KEEP when deduplicating (does not include `_Z`)

## Tests

```bash
conda run -n genai_vector python -m pytest tests/test_unit.py -v
```

Unit tests use no API calls and run in seconds. Integration tests require valid `.env` credentials.
