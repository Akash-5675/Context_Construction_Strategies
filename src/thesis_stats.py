"""Gather every statistic the thesis cites into one JSON file.

Everything the thesis reports numerically should trace back to this file, so
the document can be regenerated from data rather than from memory.
"""

import io
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs", "thesis_stats.json")

corpus = pd.read_csv(os.path.join(ROOT, "data", "pubmed_corpus.csv"))
qs = pd.read_csv(os.path.join(ROOT, "data", "questions.csv"))
df = pd.read_csv(os.path.join(ROOT, "outputs", "all_results.csv"))
df["refused"] = df["answer"].str.contains("nsufficient", na=False)
df["answered"] = ~df["refused"]
df["chunk_size"] = df["condition"].str.extract(r"fixed_(\d+)_")[0].astype(float)
df["overlap_pct"] = df["condition"].str.extract(r"fixed_\d+_(\d+)pct")[0].astype(float)

S = {}

# ---------------------------------------------------------------- corpus
corpus["words"] = corpus["abstract"].astype(str).str.split().str.len()
S["corpus"] = {
    "n": int(len(corpus)),
    "per_topic": corpus.groupby("topic").size().to_dict(),
    "words_mean": round(float(corpus["words"].mean()), 1),
    "words_median": float(corpus["words"].median()),
    "words_min": int(corpus["words"].min()),
    "words_max": int(corpus["words"].max()),
    "words_std": round(float(corpus["words"].std()), 1),
    "total_words": int(corpus["words"].sum()),
}

# ---------------------------------------------------------------- questions
qs["q_words"] = qs["question"].str.split().str.len()
qs["a_words"] = qs["reference_answer"].str.split().str.len()
S["questions"] = {
    "n": int(len(qs)),
    "per_topic": qs.groupby("topic").size().to_dict(),
    "q_words_mean": round(float(qs["q_words"].mean()), 1),
    "a_words_mean": round(float(qs["a_words"].mean()), 1),
    "a_words_min": int(qs["a_words"].min()),
    "a_words_max": int(qs["a_words"].max()),
    "samples": qs.groupby("topic").head(1)[["question_id", "topic", "question", "reference_answer"]].to_dict("records"),
}

# ---------------------------------------------------------------- chunk stats
chunk_stats = {}
cdir = os.path.join(ROOT, "data", "chunks")
for f in sorted(os.listdir(cdir)):
    if not f.endswith(".pkl"):
        continue
    with open(os.path.join(cdir, f), "rb") as fh:
        chunks = pickle.load(fh)
    texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
    w = np.array([len(t.split()) for t in texts])
    chunk_stats[f[:-4]] = {
        "n_chunks": int(len(w)),
        "words_mean": round(float(w.mean()), 1),
        "words_std": round(float(w.std()), 1),
        "words_min": int(w.min()),
        "words_max": int(w.max()),
        "chunks_per_doc": round(len(w) / len(corpus), 2),
    }
S["chunks"] = chunk_stats

# ---------------------------------------------------------------- overall
S["overall"] = {
    "n_runs": int(len(df)),
    "n_conditions": int(df["condition"].nunique()),
    "refusal_rate": round(float(df["refused"].mean()), 4),
    "n_refused": int(df["refused"].sum()),
    "n_answered": int(df["answered"].sum()),
    "rouge_mean": round(float(df["rouge_l"].mean()), 4),
    "rouge_answered_mean": round(float(df.loc[df["answered"], "rouge_l"].mean()), 4),
    "relevance_mean": round(float(df["answer_relevance"].mean()), 4),
    "faith_mean": round(float(df["faithfulness"].mean()), 4),
    "recall_mean": round(float(df["retrieval_recall"].mean()), 4),
    "context_mean": round(float(df["context_words"].mean()), 1),
    "latency_mean": round(float(df["latency_ms"].mean()), 1),
    "latency_median": round(float(df["latency_ms"].median()), 1),
}

# ---------------------------------------------------------------- per condition
def cond_table(frame):
    g = frame.groupby("condition")
    t = pd.DataFrame({
        "n": g.size(),
        "answer_rate": g["answered"].mean(),
        "rouge_all": g["rouge_l"].mean(),
        "rouge_answered": frame[frame["answered"]].groupby("condition")["rouge_l"].mean(),
        "relevance": g["answer_relevance"].mean(),
        "faithfulness": g["faithfulness"].mean(),
        "recall": g["retrieval_recall"].mean(),
        "context_words": g["context_words"].mean(),
        "latency_ms": g["latency_ms"].mean(),
    }).round(4)
    return t

ct = cond_table(df)
S["per_condition"] = ct.reset_index().to_dict("records")
S["best_condition"] = ct["rouge_all"].idxmax()
S["worst_condition"] = ct["rouge_all"].idxmin()

# ---------------------------------------------------------------- H1
fixed = df[df["chunk_size"].notna()]
h1 = fixed.groupby("overlap_pct").agg(n=("run_key", "size"), answer_rate=("answered", "mean"),
                                       rouge=("rouge_l", "mean"), recall=("retrieval_recall", "mean"),
                                       faith=("faithfulness", "mean"), context=("context_words", "mean")).round(4)
h1["rouge_answered"] = fixed[fixed["answered"]].groupby("overlap_pct")["rouge_l"].mean().round(4)
S["h1_by_overlap"] = h1.reset_index().to_dict("records")

size_tbl = fixed.groupby("chunk_size").agg(n=("run_key", "size"), answer_rate=("answered", "mean"),
                                            rouge=("rouge_l", "mean"), recall=("retrieval_recall", "mean"),
                                            faith=("faithfulness", "mean"), context=("context_words", "mean")).round(4)
size_tbl["rouge_answered"] = fixed[fixed["answered"]].groupby("chunk_size")["rouge_l"].mean().round(4)
S["by_chunk_size"] = size_tbl.reset_index().to_dict("records")

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

m = ols("rouge_l ~ C(chunk_size) + C(overlap_pct)", data=fixed).fit()
a = sm.stats.anova_lm(m, typ=2)
S["h1_anova"] = {
    "size_F": round(float(a.loc["C(chunk_size)", "F"]), 3), "size_p": float(a.loc["C(chunk_size)", "PR(>F)"]),
    "overlap_F": round(float(a.loc["C(overlap_pct)", "F"]), 3), "overlap_p": float(a.loc["C(overlap_pct)", "PR(>F)"]),
    "r2": round(float(m.rsquared), 4),
}
# eta squared
ss = a["sum_sq"]
S["h1_anova"]["size_eta2"] = round(float(ss["C(chunk_size)"] / ss.sum()), 4)
S["h1_anova"]["overlap_eta2"] = round(float(ss["C(overlap_pct)"] / ss.sum()), 4)

tk = pairwise_tukeyhsd(fixed["rouge_l"], fixed["overlap_pct"].astype(int), alpha=0.05)
S["h1_tukey"] = [dict(g1=int(r[0]), g2=int(r[1]), diff=round(float(r[2]), 4), p=round(float(r[3]), 4),
                      lo=round(float(r[4]), 4), hi=round(float(r[5]), 4), reject=bool(r[6]))
                 for r in tk.summary().data[1:]]
tks = pairwise_tukeyhsd(fixed["rouge_l"], fixed["chunk_size"].astype(int), alpha=0.05)
S["size_tukey"] = [dict(g1=int(r[0]), g2=int(r[1]), diff=round(float(r[2]), 4), p=round(float(r[3]), 4),
                        lo=round(float(r[4]), 4), hi=round(float(r[5]), 4), reject=bool(r[6]))
                   for r in tks.summary().data[1:]]

# ---------------------------------------------------------------- H2
strat = df[df["condition"].isin(["sent_nltk", "sent_scispacy", "semantic", S["best_condition"]])]
h2 = cond_table(strat)
S["h2_table"] = h2.reset_index().to_dict("records")
three = df[df["condition"].isin(["sent_nltk", "sent_scispacy", "semantic"])]
F, p = stats.f_oneway(*[g["rouge_l"].values for _, g in three.groupby("condition")])
m2 = ols("rouge_l ~ C(condition) + context_words", data=three).fit()
a2 = sm.stats.anova_lm(m2, typ=2)
S["h2_anova"] = {"F": round(float(F), 3), "p": float(p),
                 "ancova_condition_p": float(a2.loc["C(condition)", "PR(>F)"]),
                 "ancova_context_p": float(a2.loc["context_words", "PR(>F)"]),
                 "ancova_context_F": round(float(a2.loc["context_words", "F"]), 3)}
tk2 = pairwise_tukeyhsd(three["rouge_l"], three["condition"], alpha=0.05)
S["h2_tukey"] = [dict(g1=str(r[0]), g2=str(r[1]), diff=round(float(r[2]), 4), p=round(float(r[3]), 4),
                      lo=round(float(r[4]), 4), hi=round(float(r[5]), 4), reject=bool(r[6]))
                 for r in tk2.summary().data[1:]]
# four-way incl. anchor
four = strat
F4, p4 = stats.f_oneway(*[g["rouge_l"].values for _, g in four.groupby("condition")])
S["h2_anova"]["with_anchor_F"] = round(float(F4), 3)
S["h2_anova"]["with_anchor_p"] = float(p4)
tk4 = pairwise_tukeyhsd(four["rouge_l"], four["condition"], alpha=0.05)
S["h2_tukey_with_anchor"] = [dict(g1=str(r[0]), g2=str(r[1]), diff=round(float(r[2]), 4), p=round(float(r[3]), 4),
                                  reject=bool(r[6])) for r in tk4.summary().data[1:]]

# ---------------------------------------------------------------- H3
d = df.copy()
d["approx_tokens"] = d["context_words"] / 0.75
bins = [0, 500, 1000, 1500, 2000, 2500, 3000, np.inf]
labels = ["<500", "500-1k", "1k-1.5k", "1.5k-2k", "2k-2.5k", "2.5k-3k", ">3k"]
d["bin"] = pd.cut(d["approx_tokens"], bins=bins, labels=labels)
h3 = d.groupby("bin", observed=True).agg(n=("run_key", "size"), answer_rate=("answered", "mean"),
                                         rouge=("rouge_l", "mean"), faith=("faithfulness", "mean"),
                                         recall=("retrieval_recall", "mean")).round(4)
h3["rouge_answered"] = d[d["answered"]].groupby("bin", observed=True)["rouge_l"].mean().round(4)
S["h3_bins"] = h3.reset_index().assign(bin=lambda x: x["bin"].astype(str)).to_dict("records")
r, pr = stats.pearsonr(d["approx_tokens"], d["rouge_l"])
rs, prs = stats.spearmanr(d["approx_tokens"], d["rouge_l"])
ra, pra = stats.pearsonr(d.loc[d["answered"], "approx_tokens"], d.loc[d["answered"], "rouge_l"])
rr, prr = stats.pearsonr(d["approx_tokens"], d["answered"].astype(float))
S["h3_corr"] = {"pearson_r": round(float(r), 4), "pearson_p": float(pr),
                "spearman_rho": round(float(rs), 4), "spearman_p": float(prs),
                "answered_only_r": round(float(ra), 4), "answered_only_p": float(pra),
                "answer_rate_r": round(float(rr), 4), "answer_rate_p": float(prr)}

# ---------------------------------------------------------------- k effect
kt = df.groupby("k").agg(n=("run_key", "size"), answer_rate=("answered", "mean"), rouge=("rouge_l", "mean"),
                         recall=("retrieval_recall", "mean"), faith=("faithfulness", "mean"),
                         context=("context_words", "mean"), latency=("latency_ms", "mean")).round(4)
kt["rouge_answered"] = df[df["answered"]].groupby("k")["rouge_l"].mean().round(4)
S["by_k"] = kt.reset_index().to_dict("records")

# ---------------------------------------------------------------- embeddings
et = df.groupby("embedding_model").agg(n=("run_key", "size"), answer_rate=("answered", "mean"),
                                       rouge=("rouge_l", "mean"), recall=("retrieval_recall", "mean"),
                                       faith=("faithfulness", "mean"), relevance=("answer_relevance", "mean"),
                                       latency=("latency_ms", "mean")).round(4)
et["rouge_answered"] = df[df["answered"]].groupby("embedding_model")["rouge_l"].mean().round(4)
S["by_embedding"] = et.reset_index().to_dict("records")
a_ = df[df["embedding_model"] == "minilm"]["rouge_l"]; b_ = df[df["embedding_model"] == "pubmedbert"]["rouge_l"]
t, pt = stats.ttest_ind(a_, b_, equal_var=False)
S["embedding_ttest"] = {"t": round(float(t), 3), "p": float(pt)}
ra_, rb_ = df[df["embedding_model"] == "minilm"]["retrieval_recall"], df[df["embedding_model"] == "pubmedbert"]["retrieval_recall"]
t2, pt2 = stats.ttest_ind(ra_, rb_, equal_var=False)
S["embedding_recall_ttest"] = {"t": round(float(t2), 3), "p": float(pt2)}
mi = ols("rouge_l ~ C(condition) * C(embedding_model)", data=df).fit()
ai = sm.stats.anova_lm(mi, typ=2)
S["interaction"] = {"condition_F": round(float(ai.loc["C(condition)", "F"]), 3),
                    "condition_p": float(ai.loc["C(condition)", "PR(>F)"]),
                    "embedding_F": round(float(ai.loc["C(embedding_model)", "F"]), 3),
                    "embedding_p": float(ai.loc["C(embedding_model)", "PR(>F)"]),
                    "interaction_F": round(float(ai.loc["C(condition):C(embedding_model)", "F"]), 3),
                    "interaction_p": float(ai.loc["C(condition):C(embedding_model)", "PR(>F)"])}
piv = df.groupby(["condition", "embedding_model"])["rouge_l"].mean().unstack().round(4)
piv["diff"] = (piv["minilm"] - piv["pubmedbert"]).round(4)
S["embedding_by_condition"] = piv.reset_index().to_dict("records")

# ---------------------------------------------------------------- refusal breakdown
S["refusal"] = {
    "by_size": fixed.groupby("chunk_size")["refused"].mean().round(4).to_dict(),
    "by_k": df.groupby("k")["refused"].mean().round(4).to_dict(),
    "by_embedding": df.groupby("embedding_model")["refused"].mean().round(4).to_dict(),
    "by_condition": df.groupby("condition")["refused"].mean().round(4).to_dict(),
    "by_topic": df.groupby("topic")["refused"].mean().round(4).to_dict(),
    "refused_vs_answered": df.groupby("refused")[["rouge_l", "answer_relevance", "faithfulness",
                                                    "retrieval_recall", "context_words"]].mean().round(4).to_dict("index"),
    "terse": int((df["answer"].astype(str).str.strip() == "Insufficient context.").sum()),
}

# ---------------------------------------------------------------- per topic
tt = df.groupby("topic").agg(n=("run_key", "size"), answer_rate=("answered", "mean"), rouge=("rouge_l", "mean"),
                             recall=("retrieval_recall", "mean"), faith=("faithfulness", "mean"),
                             context=("context_words", "mean")).round(4)
tt["rouge_answered"] = df[df["answered"]].groupby("topic")["rouge_l"].mean().round(4)
S["by_topic"] = tt.reset_index().to_dict("records")
# does chunk size effect hold within every topic?
sz_topic = fixed.groupby(["topic", "chunk_size"])["rouge_l"].mean().unstack().round(4)
S["size_by_topic"] = sz_topic.reset_index().to_dict("records")
Ft, pt_ = stats.f_oneway(*[g["rouge_l"].values for _, g in df.groupby("topic")])
S["topic_anova"] = {"F": round(float(Ft), 3), "p": float(pt_)}

# ---------------------------------------------------------------- recall vs rouge divergence
top = df[df["answered"]].sort_values("rouge_l", ascending=False).head(50)
S["top50_recall_zero_frac"] = round(float((top["retrieval_recall"] == 0).mean()), 3)
S["recall_rouge_corr"] = round(float(df["retrieval_recall"].corr(df["rouge_l"])), 4)

# ---------------------------------------------------------------- latency
S["latency_by_condition"] = df.groupby("condition")["latency_ms"].agg(["mean", "median"]).round(1).reset_index().to_dict("records")

# ---------------------------------------------------------------- example answers
ans = df[df["answered"]].sort_values("rouge_l", ascending=False).drop_duplicates("question_id")
ref = df[df["refused"] & (df["answer"].str.len() > 60)].drop_duplicates("question_id")
def ex(row):
    q = qs.set_index("question_id").loc[row["question_id"]]
    return {"run_key": row["run_key"], "question": q["question"], "reference": q["reference_answer"],
            "answer": row["answer"][:600], "rouge_l": row["rouge_l"], "recall": row["retrieval_recall"],
            "context_words": int(row["context_words"])}
S["examples"] = {"answered": [ex(r) for _, r in ans.head(3).iterrows()],
                 "refused": [ex(r) for _, r in ref.head(3).iterrows()]}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(S, fh, indent=1, default=str)
print("wrote", OUT)
print(json.dumps({k: S[k] for k in ["corpus", "overall", "h1_anova", "h2_anova", "h3_corr", "interaction",
                                     "embedding_ttest", "topic_anova", "top50_recall_zero_frac", "recall_rouge_corr"]},
                 indent=1, default=str))
