"""Quick progress report on the experiment results file."""

import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "outputs", "all_results.csv")

PART_A = [f"fixed_{a}_{b}pct" for a in (128, 256, 512) for b in (0, 10, 20, 40)]
PART_B = ["sent_nltk", "sent_scispacy", "semantic", "proposition"]

df = pd.read_csv(RESULTS)
counts = df.groupby("condition").size()

for label, keys in [("Part A (fixed-token)", PART_A), ("Part B (strategies)", PART_B)]:
    got = counts.reindex(keys).fillna(0).astype(int)
    print(f"{label}: {int(got.sum())} / {800*len(keys)}")
    for name, n in got.items():
        flag = "  DONE" if n >= 800 else ""
        print("  %-20s %4d / 800%s" % (name, n, flag))
    print()

print(f"Total rows: {len(df)}")
print("Refusal rate: %.1f%%" % (100 * df["answer"].str.contains("nsufficient", na=False).mean()))
