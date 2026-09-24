"""Render the 8-phase methodology pipeline diagram with final corrected figures.

Replaces the Phase 1 diagram, which still showed 982 abstracts, 16 conditions,
32 indexes, Gemini as the generator, and 13,600 runs.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs",
                   "figures", "methodology_pipeline.png")

CRIMSON = "#A5194D"
GOLD = "#F2B705"
BAR_FILL = "#FCE9EF"
WHITE = "#FFFFFF"

PHASES = [
    ("Corpus Collection",   "981 PubMed abstracts",          "5 topics via Entrez API"),
    ("Question Annotation", "100 QA pairs",                  "20 per topic, annotated"),
    ("Chunking",            "4 strategies, 15 conditions",   "Fixed / Sentence / Semantic"),
    ("Embedding & Indexing", "MiniLM + PubMedBERT",          "30 FAISS indexes built"),
    ("Retrieval",           "FAISS exact top-k search",      "k ∈ {3, 5, 8, 10}"),
    ("Generation",          "Llama 3.1 8B · Cloudflare", "temperature 0, grounded only"),
    ("Experiment Runner",   "12,000 runs",                   "checkpointing every 50"),
    ("Analysis",            "ANOVA · Tukey HSD · ANCOVA", "6 publication figures"),
]

SUMMARY = ("15 conditions   ×   100 questions   ×   4 k-values   "
           "×   2 embedding models   =   12,000 total runs")

# geometry (figure units)
BOX_W, BOX_H = 3.20, 1.70
GAP_X = 0.55
X0, Y_TOP, Y_BOT = 0.55, 4.30, 2.05

fig, ax = plt.subplots(figsize=(15.5, 7.4))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 7.0)
ax.axis("off")

positions = []
for i in range(8):
    if i < 4:                      # top row, left to right
        col, y = i, Y_TOP
    else:                          # bottom row, right to left (serpentine)
        col, y = 7 - i, Y_BOT
    positions.append((X0 + col * (BOX_W + GAP_X), y))


def draw_box(x, y, number, title, line1, line2):
    ax.add_patch(FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=CRIMSON, edgecolor="none", zorder=2))

    ax.add_patch(Circle((x + 0.34, y + BOX_H - 0.34), 0.235,
                        facecolor=GOLD, edgecolor="none", zorder=3))
    ax.text(x + 0.34, y + BOX_H - 0.34, str(number), ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=CRIMSON, zorder=4)

    ax.text(x + BOX_W / 2, y + BOX_H - 0.66, title, ha="center", va="center",
            fontsize=12.0, fontweight="bold", color=WHITE, zorder=4)
    ax.text(x + BOX_W / 2, y + 0.56, line1, ha="center", va="center",
            fontsize=9.8, style="italic", color=WHITE, zorder=4)
    ax.text(x + BOX_W / 2, y + 0.30, line2, ha="center", va="center",
            fontsize=9.8, style="italic", color=WHITE, zorder=4)


def arrow(p1, p2):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=17,
        linewidth=1.5, color="#4A4A4A", zorder=1,
        shrinkA=0, shrinkB=0))


for i, (x, y) in enumerate(positions):
    draw_box(x, y, i + 1, *PHASES[i])

# horizontal arrows along the top row
for i in range(3):
    x, y = positions[i]
    arrow((x + BOX_W + 0.06, y + BOX_H / 2), (x + BOX_W + GAP_X - 0.06, y + BOX_H / 2))

# vertical drop from phase 4 down to phase 5
x4, y4 = positions[3]
arrow((x4 + BOX_W / 2, y4 - 0.06), (x4 + BOX_W / 2, Y_BOT + BOX_H + 0.06))

# horizontal arrows along the bottom row, flowing right to left
for i in range(4, 7):
    x, y = positions[i]
    arrow((x - 0.06, y + BOX_H / 2), (x - GAP_X + 0.06, y + BOX_H / 2))

# summary bar
bar_w = 4 * BOX_W + 3 * GAP_X
ax.add_patch(FancyBboxPatch(
    (X0, 0.90), bar_w, 0.72,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    facecolor=BAR_FILL, edgecolor=CRIMSON, linewidth=1.1, zorder=2))
ax.text(X0 + bar_w / 2, 1.26, SUMMARY, ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=CRIMSON, zorder=3)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", os.path.abspath(OUT))
