# UNICEF SDMX Data Explorer

A conversational AI assistant for exploring UNICEF's SDMX data. Ask questions in plain English — the agent finds the right indicator, fetches the data, runs quality checks, and renders an interactive chart.

## What it does

- **Semantic search** across 3,300+ UNICEF indicators using embeddings
- **Automatic dataflow selection** — tries all available SDMX dataflows and picks the best data
- **Interactive charts** via Plotly — line, bar, with breakdown by sex, wealth quintile, residence, etc.
- **Quality checks** on every dataset (completeness, null rate, plausibility)
- **Follow-up questions** — ask for a different breakdown or view without re-fetching

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM + embeddings | Azure OpenAI (GPT-4.1-mini / text-embedding-3-large) |
| Data source | [UNICEF SDMX REST API](https://sdmx.data.unicef.org) |
| Charts | Plotly Express |
| Data | Parquet (pandas + pyarrow) |

## Project structure

```
app.py                          # Streamlit UI and chart rendering
agents/
  sdmx_agent.py                 # Agent tools, session state, system prompt
scripts/
  build_data_dictionary.py      # Rebuild data/data_dictionary.parquet from SDMX
  embed_indicators.py           # Recompute data/embeddings/indicator_embeddings.parquet
data/
  data_dictionary.parquet       # 1.3M rows — indicators × geographies × dataflows
  embeddings/
    indicator_embeddings.parquet  # 3,300+ indicator embeddings (dim 3072)
tests/
  test_unit.py                  # Unit tests (no API calls)
  test_integration.py           # Integration tests
```

## Setup

### 1. Environment

```bash
conda create -n genai_vector python=3.9
conda activate genai_vector
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:

```ini
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2025-04-01-preview

AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_LLM_API_VERSION=2025-04-01-preview

AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-02-01

AZURE_SEARCH_ENDPOINT=...   # optional — Azure AI Search
AZURE_SEARCH_KEY=...
AZURE_SEARCH_INDEX=...

APP_PASSWORD=               # optional — leave empty for open access
DEBUG=1                     # show dataflow selection log in UI
```

### 3. Run

```bash
conda run -n genai_vector streamlit run app.py
```

## Rebuilding the data dictionary

Run this whenever UNICEF publishes new data or adds dataflows (~30–60 min):

```bash
# Full rebuild — downloads all dataflows, classifies, embeds
conda run -n genai_vector python scripts/build_data_dictionary.py

# Skip re-downloading (CSVs already cached in data/build_cache/)
conda run -n genai_vector python scripts/build_data_dictionary.py --skip-download

# Skip re-classifying (reuse cached category assignments)
conda run -n genai_vector python scripts/build_data_dictionary.py --skip-classify

# Rebuild embeddings only
conda run -n genai_vector python scripts/embed_indicators.py
```

The pipeline:
1. Fetches all 136 SDMX dataflows
2. Downloads each as CSV and standardises column names
3. Classifies indicators by thematic category using GPT-4.1-mini
4. Resolves geographies to a canonical country name
5. Writes `data/data_dictionary.parquet`
6. Runs `embed_indicators.py` to regenerate vector embeddings

Intermediate outputs are cached in `data/build_cache/` so interrupted runs can be resumed with `--skip-download` or `--skip-classify`.

## Tests

```bash
conda run -n genai_vector python -m pytest tests/test_unit.py -v
```
