"""
Phase 7 — Experiment runner.

Executes the full factorial experiment:
    conditions (chunking) × questions × k_values × embedding_models

Part A (fixed-token only, 9,600 rows):
    12 fixed conditions × 100 questions × 4 k-values × 2 embeddings

Part B (all 5 strategies with best fixed config, 4,000 rows):
    5 strategies × 100 questions × 4 k-values × 2 embeddings

Results are appended to outputs/all_results.csv with checkpointing every
CHECKPOINT_EVERY rows so the run can be safely interrupted and resumed.

Usage:
    python src/experiment_runner.py --part A
    python src/experiment_runner.py --part B
    python src/experiment_runner.py --part A --worker 0  # first half of conditions
    python src/experiment_runner.py --part A --worker 1  # second half of conditions
    python src/experiment_runner.py --part A --dry-run   # 5 questions only
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.retriever import retrieve
from pipeline.generator import generate
from pipeline.evaluator import evaluate

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RESULTS_PATH = os.path.join(OUTPUTS_DIR, "all_results.csv")

QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.csv")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")

K_VALUES = [3, 5, 8, 10]
EMBEDDING_MODELS = ["minilm", "pubmedbert"]
# Buffered rows are lost if the process is killed, so keep this small —
# a run can die at any moment when the daily quota runs out.
CHECKPOINT_EVERY = 50

FIXED_CONDITIONS = [
    f"fixed_{size}_{pct}pct"
    for size in [128, 256, 512]
    for pct in [0, 10, 20, 40]
]

ALL_STRATEGY_CONDITIONS = [
    "sent_nltk",
    "sent_scispacy",
    "semantic",
    "proposition",
]


def load_questions(dry_run: bool = False) -> pd.DataFrame:
    df = pd.read_csv(QUESTIONS_PATH)
    required = {"question_id", "question", "gold_doc_ids", "reference_answer", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"questions.csv is missing columns: {missing}")
    if dry_run:
        df = df.head(5)
    return df


def get_completed_keys(results_path: str) -> set[str]:
    if not os.path.exists(results_path):
        return set()
    df = pd.read_csv(results_path, usecols=["run_key"])
    return set(df["run_key"].tolist())


def run_single(
    question_id: str,
    question: str,
    gold_doc_ids: list[str],
    reference_answer: str,
    topic: str,
    condition_key: str,
    model_key: str,
    k: int,
) -> dict:
    chunks = retrieve(question, condition_key, model_key, k=k)
    gen_result = generate(question, chunks)
    metrics = evaluate(
        query=question,
        answer=gen_result["answer"],
        reference_answer=reference_answer,
        gold_doc_ids=gold_doc_ids,
        retrieved_chunks=chunks,
        context_word_count=gen_result["context_word_count"],
        latency_ms=gen_result["latency_ms"],
        model_key=model_key,
    )
    return {
        "run_key": f"{question_id}__{condition_key}__{model_key}__k{k}",
        "question_id": question_id,
        "topic": topic,
        "condition": condition_key,
        "embedding_model": model_key,
        "k": k,
        "answer": gen_result["answer"],
        "prompt_tokens": gen_result["prompt_tokens"],
        "completion_tokens": gen_result["completion_tokens"],
        **metrics,
    }


def run_experiment(conditions: list[str], questions_df: pd.DataFrame):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    completed = get_completed_keys(RESULTS_PATH)
    print(f"Already completed: {len(completed)} runs. Resuming...")

    buffer = []
    total = len(conditions) * len(questions_df) * len(K_VALUES) * len(EMBEDDING_MODELS)
    pbar = tqdm(total=total, desc="runs")

    try:
        for cond in conditions:
            for _, row in questions_df.iterrows():
                gold_ids = [g.strip() for g in str(row["gold_doc_ids"]).split("|")]
                for model_key in EMBEDDING_MODELS:
                    for k in K_VALUES:
                        run_key = f"{row['question_id']}__{cond}__{model_key}__k{k}"
                        pbar.update(1)
                        if run_key in completed:
                            continue
                        try:
                            result = run_single(
                                question_id=str(row["question_id"]),
                                question=str(row["question"]),
                                gold_doc_ids=gold_ids,
                                reference_answer=str(row["reference_answer"]),
                                topic=str(row["topic"]),
                                condition_key=cond,
                                model_key=model_key,
                                k=k,
                            )
                            buffer.append(result)
                            completed.add(run_key)
                        except SystemExit:
                            raise
                        except Exception as e:
                            tqdm.write(f"ERROR {run_key}: {e}")

                        if len(buffer) >= CHECKPOINT_EVERY:
                            _flush(buffer)
                            buffer = []
    finally:
        # Always persist buffered rows — quota aborts and Ctrl+C both land here.
        if buffer:
            _flush(buffer)
        pbar.close()

    print(f"Done. Results at {RESULTS_PATH}")


def _flush(rows: list[dict]):
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(RESULTS_PATH)
    df.to_csv(RESULTS_PATH, mode="a", index=False, header=write_header)
    tqdm.write(f"  Checkpoint: {len(rows)} rows written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A", "B"], required=True,
                        help="A = fixed-token only; B = all non-fixed strategies")
    parser.add_argument("--best-fixed", default="fixed_256_20pct",
                        help="Best fixed config from Part A (used as anchor in Part B)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", type=int, choices=[0, 1], default=None,
                        help="0 = first half of conditions, 1 = second half (for parallel runs)")
    parser.add_argument("--cf-token", default=None,
                        help="Override CLOUDFLARE_API_TOKEN env var")
    parser.add_argument("--cf-account", default=None,
                        help="Override CLOUDFLARE_ACCOUNT_ID env var")
    args = parser.parse_args()

    # Override CF credentials if provided
    if args.cf_token:
        os.environ["CLOUDFLARE_API_TOKEN"] = args.cf_token
    if args.cf_account:
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = args.cf_account

    questions_df = load_questions(dry_run=args.dry_run)
    print(f"Running {'DRY RUN' if args.dry_run else 'FULL'} with {len(questions_df)} questions.")

    if args.part == "A":
        conditions = FIXED_CONDITIONS
        print(f"Part A: {len(conditions)} fixed conditions")
    else:
        conditions = [args.best_fixed] + ALL_STRATEGY_CONDITIONS
        print(f"Part B: {len(conditions)} strategies (anchor + {len(ALL_STRATEGY_CONDITIONS)} advanced)")

    # Drop conditions whose chunk file was never built (e.g. proposition,
    # which needs an LLM pass over the corpus first).
    missing = [c for c in conditions
               if not os.path.exists(os.path.join(CHUNKS_DIR, f"{c}.pkl"))]
    if missing:
        print(f"WARNING: no chunk file for {missing} -- skipping.")
        conditions = [c for c in conditions if c not in missing]

    # Split conditions between two parallel workers
    if args.worker is not None:
        mid = len(conditions) // 2
        if args.worker == 0:
            conditions = conditions[:mid]
        else:
            conditions = conditions[mid:]
        print(f"Worker {args.worker}: running conditions {conditions}")

    run_experiment(conditions, questions_df)


if __name__ == "__main__":
    main()
