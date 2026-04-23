"""
One-time script to pre-compute embeddings for all unique indicators.

Output: data/embeddings/indicator_embeddings.parquet
  Columns: indicator_id (str), text (str), embedding (list[float])

Run with:
  conda run -n genai_vector python scripts/embed_indicators.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ROOT = Path(__file__).parent.parent
DICTIONARY_FILE = ROOT / "data" / "data_dictionary.parquet"
OUTPUT_DIR = ROOT / "data" / "embeddings"
OUTPUT_FILE = OUTPUT_DIR / "indicator_embeddings.parquet"

BATCH_SIZE = 100  # Azure OpenAI max is 2048; keep lower to avoid timeouts


def build_text(row: pd.Series) -> str:
    name = str(row["indicator"] or "").strip()
    defn = str(row["definition"] or "").strip()
    if defn and defn.lower() not in ("n/a", "nan", "none", ""):
        return f"{name}. {defn}"
    return name


def embed_batch(client: AzureOpenAI, deployment: str, texts: list) -> list:
    response = client.embeddings.create(model=deployment, input=texts)
    return [item.embedding for item in response.data]


def main() -> None:
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    if not DICTIONARY_FILE.exists():
        raise FileNotFoundError(
            f"data_dictionary.parquet not found: {DICTIONARY_FILE}\n"
            "Run: python scripts/partition_by_country.py --convert"
        )

    print(f"Reading {DICTIONARY_FILE} ...")
    all_df = pd.read_parquet(
        DICTIONARY_FILE,
        columns=["indicator_id", "indicator", "definition", "category"],
    )

    unique = (
        all_df.drop_duplicates("indicator_id")[
            ["indicator_id", "indicator", "definition", "category"]
        ]
        .reset_index(drop=True)
    )
    unique["text"] = unique.apply(build_text, axis=1)

    # Drop indicators with no usable text
    embeddable = unique[unique["text"].str.strip() != ""].copy().reset_index(drop=True)
    skipped = len(unique) - len(embeddable)
    if skipped:
        print(f"Skipping {skipped} indicators with no name/definition.")

    print(f"Embedding {len(embeddable):,} unique indicators using {deployment}...")

    texts = embeddable["text"].tolist()
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vecs = embed_batch(client, deployment, batch)
        all_embeddings.extend(vecs)
        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.1)

    print(f"\nDone. Embedding dim: {len(all_embeddings[0])}")

    embeddable["embedding"] = all_embeddings
    unique = embeddable
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unique[["indicator_id", "text", "embedding"]].to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
