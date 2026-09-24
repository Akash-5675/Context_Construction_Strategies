"""Identify the best-performing fixed-token condition from Part A.

Part B anchors on this condition to compare fixed-token chunking against the
sentence / semantic / proposition strategies, so it must be chosen from data
rather than assumed.

Because ~63% of rows are refusals, a raw mean of rouge_l mostly measures how
often a condition refused. Both components are reported separately:
  - answer_rate: how often the model produced an answer at all
  - rouge_answered: answer quality conditional on having answered
"""

import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "outputs", "all_results.csv")

df = pd.read_csv(RESULTS)
df = df[df["condition"].str.startswith("fixed_")].copy()
df["refused"] = df["answer"].str.contains("nsufficient", na=False)

summary = df.groupby("condition").agg(
    n=("run_key", "size"),
    answer_rate=("refused", lambda s: 1 - s.mean()),
    rouge_all=("rouge_l", "mean"),
    relevance=("answer_relevance", "mean"),
    faithfulness=("faithfulness", "mean"),
    recall=("retrieval_recall", "mean"),
)
answered = df[~df["refused"]].groupby("condition")["rouge_l"].mean()
summary["rouge_answered"] = answered

summary = summary.sort_values("rouge_all", ascending=False).round(4)
print("Fixed-token conditions, ranked by overall mean ROUGE-L:\n")
print(summary.to_string())

best = summary.index[0]
print(f"\nBest by overall ROUGE-L:      {best}")
print(f"Best by answer rate:          {summary['answer_rate'].idxmax()}")
print(f"Best by quality when answered:{summary['rouge_answered'].idxmax():>20}")
print(f"\nSuggested --best-fixed anchor for Part B: {best}")
