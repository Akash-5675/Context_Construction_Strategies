"""Additional figures for the thesis, drawn from outputs/thesis_stats.json and all_results.csv."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "outputs", "figures")
S = json.load(open(os.path.join(ROOT, "outputs", "thesis_stats.json"), encoding="utf-8"))
df = pd.read_csv(os.path.join(ROOT, "outputs", "all_results.csv"))
df["refused"] = df["answer"].str.contains("nsufficient", na=False)

CRIMSON, NAVY, TEAL, GREY = "#A5194D", "#1F2A44", "#0E9A8E", "#8A8F9C"
STYLE = {"axes.spines.top": False, "axes.spines.right": False, "font.size": 11}


def nice(c):
    if c.startswith("fixed_"):
        s, o = c.replace("fixed_", "").replace("pct", "").split("_")
        return f"{s}/{o}%"
    return {"sent_nltk": "Sent-NLTK", "sent_scispacy": "Sent-sciSpaCy", "semantic": "Semantic"}[c]


# ---------------------------------------------------------------- 1. chunks per abstract
with plt.rc_context(STYLE):
    order = list(S["chunks"].keys())
    vals = [S["chunks"][k]["chunks_per_doc"] for k in order]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = [CRIMSON if v > 1.05 else GREY for v in vals]
    ax.bar([nice(k) for k in order], vals, color=colors)
    ax.axhline(1.0, color=NAVY, linestyle="--", linewidth=1.2)
    ax.text(len(order) - 0.5, 1.03, "one chunk = whole abstract", ha="right", va="bottom", fontsize=10, color=NAVY)
    ax.set_ylabel("Chunks per abstract")
    ax.set_title("Granularity actually achieved by each condition (abstracts average 238 words)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_chunks_per_abstract.png"), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- 2. refusal structure
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    bs = S["refusal"]["by_size"]; bk = S["refusal"]["by_k"]; be = S["refusal"]["by_embedding"]
    axes[0].bar([str(int(float(k))) for k in bs], [100 * v for v in bs.values()], color=CRIMSON)
    axes[0].set_title("By chunk size (words)"); axes[0].set_ylabel("Refusal rate (%)")
    axes[1].bar([str(k) for k in bk], [100 * v for v in bk.values()], color=CRIMSON)
    axes[1].set_title("By retrieval depth k")
    axes[2].bar(["MiniLM", "PubMedBERT"], [100 * be["minilm"], 100 * be["pubmedbert"]], color=CRIMSON)
    axes[2].set_title("By embedding model")
    for ax in axes:
        ax.set_ylim(50, 72)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
    fig.suptitle("Refusal rate (\"Insufficient context\") across design factors", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_refusal_structure.png"), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- 3. per-topic
with plt.rc_context(STYLE):
    t = pd.DataFrame(S["by_topic"]).set_index("topic")
    fig, ax1 = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(t)); w = 0.38
    ax1.bar(x - w / 2, t["rouge"], w, color=CRIMSON, label="ROUGE-L (all runs)")
    ax1.bar(x + w / 2, t["rouge_answered"], w, color=TEAL, label="ROUGE-L (answered only)")
    ax1.set_ylabel("ROUGE-L"); ax1.set_xticks(x); ax1.set_xticklabels(t.index)
    ax2 = ax1.twinx(); ax2.plot(x, 100 * t["answer_rate"], marker="o", color=NAVY, linewidth=2, label="Answer rate (%)")
    ax2.set_ylabel("Answer rate (%)"); ax2.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax1.set_title("Per-topic performance")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_per_topic.png"), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- 4. size effect within each topic
with plt.rc_context(STYLE):
    st = pd.DataFrame(S["size_by_topic"]).set_index("topic")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for topic, row in st.iterrows():
        ax.plot([128, 256, 512], [row["128.0"], row["256.0"], row["512.0"]], marker="o", linewidth=2, label=topic)
    ax.set_xticks([128, 256, 512]); ax.set_xlabel("Chunk size (words)"); ax.set_ylabel("Mean ROUGE-L")
    ax.set_title("Chunk-size effect replicates within every topic"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_size_by_topic.png"), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- 5. recall vs rouge divergence
with plt.rc_context(STYLE):
    ans = df[~df["refused"]]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for rec, lab, col in [(1.0, "Gold PMID retrieved (recall = 1)", TEAL), (0.0, "Gold PMID not retrieved (recall = 0)", CRIMSON)]:
        sub = ans[ans["retrieval_recall"] == rec]["rouge_l"]
        ax.hist(sub, bins=30, alpha=0.6, color=col, label=f"{lab}  (n={len(sub)}, mean={sub.mean():.3f})")
    ax.set_xlabel("ROUGE-L of answered runs"); ax.set_ylabel("Runs"); ax.legend(fontsize=9)
    ax.set_title("Answer quality is only weakly tied to retrieving the annotated gold source")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_recall_vs_rouge.png"), dpi=300); plt.close(fig)

print("saved 5 thesis figures")
