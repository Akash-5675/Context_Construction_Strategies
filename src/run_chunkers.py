"""
Phase 3.5 — Run all chunkers on the full corpus.

Reads data/pubmed_corpus.csv, applies every chunking condition, saves each
condition as data/chunks/{condition_key}.pkl, and writes aggregate statistics
to data/chunk_stats.csv.

Usage:
    python src/run_chunkers.py

Skip proposition chunking (saves Groq API calls during testing):
    python src/run_chunkers.py --skip-proposition

Skip semantic chunking (avoids needing sentence-transformers/PyTorch):
    python src/run_chunkers.py --skip-semantic

Skip both:
    python src/run_chunkers.py --skip-proposition --skip-semantic
"""

import argparse
import os
import pickle
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from chunkers import (
    FixedTokenChunker,
    SentenceNLTKChunker,
    SentenceScispaCyChunker,
    SemanticChunker,
    PropositionChunker,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "pubmed_corpus.csv")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
STATS_PATH = os.path.join(DATA_DIR, "chunk_stats.csv")


def build_chunker_list(skip_proposition: bool, skip_semantic: bool = False) -> list:
    chunkers = FixedTokenChunker.all_conditions()  # 12 fixed conditions
    chunkers += [
        SentenceNLTKChunker(),
        SentenceScispaCyChunker(),
    ]
    if not skip_semantic:
        chunkers.append(SemanticChunker())
    if not skip_proposition:
        chunkers.append(PropositionChunker())
    return chunkers


def run_chunker(chunker, df: pd.DataFrame) -> list[dict]:
    all_chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=chunker.condition_key, leave=False):
        doc_id = str(row["pmid"])
        text = str(row["abstract"])
        chunks = chunker.chunk(text, doc_id)
        for c in chunks:
            c["topic"] = row.get("topic", "")
        all_chunks.extend(chunks)
    return all_chunks


def compute_stats(condition_key: str, chunks: list[dict]) -> dict:
    import numpy as np
    word_counts = [c["word_count"] for c in chunks]
    return {
        "condition": condition_key,
        "total_chunks": len(chunks),
        "mean_words": round(float(np.mean(word_counts)), 1),
        "std_words": round(float(np.std(word_counts)), 1),
        "min_words": int(np.min(word_counts)),
        "max_words": int(np.max(word_counts)),
        "median_words": round(float(np.median(word_counts)), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-proposition", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(CORPUS_PATH)
    print(f"Loaded {len(df)} abstracts from corpus.")

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    chunkers = build_chunker_list(args.skip_proposition, args.skip_semantic)
    stats_rows = []

    for chunker in chunkers:
        key = chunker.condition_key
        out_path = os.path.join(CHUNKS_DIR, f"{key}.pkl")

        if os.path.exists(out_path):
            print(f"  [skip] {key} already exists.")
            with open(out_path, "rb") as f:
                chunks = pickle.load(f)
        else:
            print(f"  Chunking: {key}")
            chunks = run_chunker(chunker, df)
            with open(out_path, "wb") as f:
                pickle.dump(chunks, f)
            print(f"    -> {len(chunks)} chunks saved to {out_path}")

        stats_rows.append(compute_stats(key, chunks))

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(STATS_PATH, index=False)
    print(f"\nChunk statistics saved to {STATS_PATH}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
