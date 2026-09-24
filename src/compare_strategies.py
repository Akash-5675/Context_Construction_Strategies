"""Compare completed chunking strategies against the best fixed-token anchor."""

import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "outputs", "all_results.csv")
KEEP = ["fixed_128_0pct", "sent_nltk", "sent_scispacy", "semantic", "proposition"]

df = pd.read_csv(RESULTS)
df = df[df["condition"].isin(KEEP)].copy()
df["refused"] = df["answer"].str.contains("nsufficient", na=False)

summary = df.groupby("condition").agg(
    n=("run_key", "size"),
    answer_rate=("refused", lambda s: 1 - s.mean()),
    rouge_all=("rouge_l", "mean"),
    relevance=("answer_relevance", "mean"),
    faithfulness=("faithfulness", "mean"),
    recall=("retrieval_recall", "mean"),
    ctx_words=("context_words", "mean"),
)
summary["rouge_answered"] = df[~df["refused"]].groupby("condition")["rouge_l"].mean()

complete = summary[summary["n"] >= 800].sort_values("rouge_all", ascending=False)
partial = summary[summary["n"] < 800]

print("Complete conditions, ranked by mean ROUGE-L:\n")
print(complete.round(4).to_string())
if not partial.empty:
    print("\nStill running (not comparable yet):")
    print(partial.round(4).to_string())
