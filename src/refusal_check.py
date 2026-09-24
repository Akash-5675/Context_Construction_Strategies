"""Break down refusal rate by condition, k, and embedding model."""

import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "outputs", "all_results.csv")

df = pd.read_csv(RESULTS)
df["refused"] = df["answer"].str.contains("nsufficient", na=False)

print("Refusal rate by chunk size:")
df["size"] = df["condition"].str.extract(r"fixed_(\d+)_")
print((df.groupby("size")["refused"].mean() * 100).round(1).to_string())

print("\nRefusal rate by k:")
print((df.groupby("k")["refused"].mean() * 100).round(1).to_string())

print("\nRefusal rate by embedding model:")
print((df.groupby("embedding_model")["refused"].mean() * 100).round(1).to_string())

print("\nRefusal rate by condition:")
print((df.groupby("condition")["refused"].mean() * 100).round(1).sort_values().to_string())

print("\nMean scores, refused vs answered:")
print(df.groupby("refused")[["rouge_l", "answer_relevance", "faithfulness",
                             "retrieval_recall", "context_words"]].mean().round(3).to_string())
