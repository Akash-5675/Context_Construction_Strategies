"""Test whether chunking condition and embedding model interact.

The novelty claim "chunking x embedding interaction" needs an interaction term,
not two separate main effects. This fits condition + embedding + their product.
"""

import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

RESULTS = os.path.join(os.path.dirname(__file__), "..", "outputs", "all_results.csv")
df = pd.read_csv(RESULTS)

model = ols("rouge_l ~ C(condition) * C(embedding_model)", data=df).fit()
aov = sm.stats.anova_lm(model, typ=2)
print("Two-way ANOVA with interaction (rouge_l):\n")
print(aov.round(6).to_string())

p_int = aov.loc["C(condition):C(embedding_model)", "PR(>F)"]
print(f"\nInteraction p = {p_int:.6g}")
print("Interaction is", "SIGNIFICANT" if p_int < 0.05 else "NOT significant", "at alpha=0.05")

print("\nPer-condition means by embedding model:")
piv = df.groupby(["condition", "embedding_model"])["rouge_l"].mean().unstack()
piv["diff (minilm - pubmedbert)"] = piv["minilm"] - piv["pubmedbert"]
print(piv.round(4).to_string())
