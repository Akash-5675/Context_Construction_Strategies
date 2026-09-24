# Context Construction Strategies in Medical RAG

A systematic empirical study comparing five chunking strategies for
Retrieval-Augmented Generation on biomedical (PubMed) text, motivated by the
research gap that general-domain chunking best practices have not been validated
on biomedical corpora.

## Research Hypotheses

| ID | Hypothesis | Origin | Biomedical angle |
|----|-----------|--------|-----------------|
| **H1** | Overlap between chunks adds no retrieval benefit | Moslonka et al. 2026 on Natural Questions | Clinical sentences span drug-disease-dosage relationships across boundaries — overlap may recover cross-boundary evidence |
| **H2** | Sentence chunking matches semantic chunking | General-domain finding up to ~5k tokens | NLTK punkt creates false boundaries on `i.v.`, `p.o.`, `Dr.` — scispaCy or semantic chunking may win |
| **H3** | Answer quality degrades at ~2,500 context tokens | Wikipedia-corpus finding | PubMed abstracts are denser per token — the cliff may occur at a different threshold |

## Strategies Evaluated

| Strategy | Key | Overlap | Part |
|----------|-----|---------|------|
| Fixed-token 128/256/512 words × 0/10/20/40% overlap | `fixed_*` | Yes | A + B |
| Sentence boundary — NLTK punkt | `sent_nltk` | Carry-forward | B |
| Sentence boundary — scispaCy biomedical | `sent_scispacy` | Carry-forward | B |
| Semantic (embedding similarity threshold) | `semantic` | None | B |
| Proposition (atomic facts via Groq LLM) | `proposition` | None | B |

Total conditions: **~16** (12 fixed + 4 advanced).
Full factorial: conditions × 100 questions × 4 k-values × 2 embeddings ≈ **13,600 runs**.

## Embedding Models

| Key | Model | Dim | Purpose |
|-----|-------|-----|---------|
| `minilm` | `all-MiniLM-L6-v2` | 384 | General-domain baseline |
| `pubmedbert` | `NeuML/pubmedbert-base-embeddings` | 768 | Biomedical domain |

Testing both models answers the additional question: *does embedding domain interact
with chunking strategy?*

---

## Project Structure

```
context_construction_strategies/
│
├── src/                        # Runnable scripts (phases 1, 3–8)
│   ├── collect.py              # Phase 1  — PubMed corpus collection
│   ├── run_chunkers.py         # Phase 3.5 — Apply all chunkers to corpus
│   ├── experiment_runner.py    # Phase 7  — Full factorial experiment
│   ├── analyze.py              # Phase 8  — Statistics and figures
│   └── chunkers/               # Five chunking strategy modules
│       ├── fixed_token.py
│       ├── sentence_nltk.py
│       ├── sentence_scispacy.py
│       ├── semantic.py
│       └── proposition.py
│
├── pipeline/                   # Reusable modules (phases 4–6)
│   ├── embedder.py             # Phase 4  — Load models, embed text
│   ├── indexer.py              # Phase 4  — Build FAISS indexes
│   ├── retriever.py            # Phase 5  — Query → top-k chunks
│   ├── generator.py            # Phase 5  — Groq API answer generation
│   └── evaluator.py            # Phase 6  — Six evaluation metrics
│
├── data/                       # Corpus, questions, and chunk cache
│   ├── pubmed_corpus.csv       # ~1000 PubMed abstracts (generated)
│   ├── questions.csv           # 100 annotated questions (manual)
│   ├── questions_template.csv  # Template/example for questions.csv
│   ├── chunk_stats.csv         # Chunk descriptive statistics (generated)
│   └── chunks/                 # One .pkl per chunking condition (generated)
│
├── indexes/                    # FAISS indexes (generated, gitignored)
│   └── {condition}__{model}.index + .meta.pkl
│
├── outputs/                    # Experiment results and figures
│   ├── all_results.csv         # ~13,600 rows — full run results
│   ├── summary.csv             # Per-condition means and SDs
│   ├── summary_anova.csv       # ANOVA F-stats and p-values
│   └── figures/                # 6 publication-ready PNG figures
│
├── requirements.txt
├── .env.example                # Copy to .env and fill in API keys
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install scispaCy biomedical model

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### 3. Configure API keys

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` and add:
- `GROQ_API_KEY` — free at [console.groq.com](https://console.groq.com)
- `ENTREZ_EMAIL` — any valid email (required by PubMed API)

---

## Running the Pipeline

Run phases in order. Each phase saves its output so you can resume safely.

### Phase 1 — Collect corpus (~5 min)

```bash
python src/collect.py
```

Produces `data/pubmed_corpus.csv` with ~1000 abstracts.

### Phase 2 — Annotate questions (6–8 hours, manual)

Copy `data/questions_template.csv` to `data/questions.csv` and write 20 questions
per topic. See [data/README.md](data/README.md) for the annotation guide.

### Phase 3 — Chunk corpus (~30–40 min)

```bash
# All strategies including proposition (requires Groq)
python src/run_chunkers.py

# Skip proposition during development
python src/run_chunkers.py --skip-proposition
```

Produces `data/chunks/*.pkl` and `data/chunk_stats.csv`.

### Phase 4 — Build indexes (~1.5 hours, can run overnight)

```bash
python pipeline/indexer.py
```

Produces `indexes/*.index` + `*.meta.pkl` for all condition × model combinations.

### Phase 5–7 — Run experiments (~2.5 hours total)

```bash
# Part A: fixed-token strategies only
python src/experiment_runner.py --part A

# Check Part A results, then identify best fixed config (e.g. fixed_256_20pct)
# Part B: all advanced strategies vs best fixed anchor
python src/experiment_runner.py --part B --best-fixed fixed_256_20pct
```

Results append to `outputs/all_results.csv` with checkpointing every 500 rows.

### Phase 8 — Analyse and generate figures

```bash
python src/analyze.py
```

Produces `outputs/summary.csv`, `outputs/summary_anova.csv`, and 6 figures in
`outputs/figures/`.

---

## Resource requirements

| Resource | Requirement |
|----------|------------|
| RAM | ~1.5 GB peak (both embedding models loaded simultaneously) |
| GPU | Not required — all inference runs on CPU |
| Disk | ~500 MB (indexes + chunk pickles) |
| Internet | PubMed API (Phase 1) and Groq API (Phase 5, 7) |

Tested on 8 GB RAM, no GPU.

---

## Evaluation metrics

See [pipeline/README.md](pipeline/README.md) for full definitions.

| Metric | Tests |
|--------|-------|
| Answer relevance | Does the answer address the question? |
| Reference similarity | Does the answer match the gold reference? |
| Faithfulness | Is the answer grounded in the retrieved context? |
| Retrieval recall | Was the correct source document retrieved? |
| Context words | How many words did the model receive? (H3 variable) |
| Latency ms | End-to-end wall-clock time |

---

## Key papers

| Paper | Relevance |
|-------|-----------|
| PMC12649634 — MDPI Bioengineering (Nov 2025) | Medical RAG chunking — closest prior work; adaptive chunking 87% vs 13% fixed baseline |
| arXiv:2410.13070 — Qu et al. NAACL 2025 | Semantic chunking not consistently worth cost |
| arXiv:2406.00456 — MoG, Shanghai AI Lab | Dynamic granularity routing — motivates our fixed baseline study |
| arXiv:2504.19754 — Merola & Singh, Bologna 2025 | Late chunking and contextual retrieval |
| arXiv:2606.00881 — LumberChunker survey Jun 2025 | Explicitly states no comprehensive biomedical chunking study exists |

---

## Folder READMEs

Each subfolder contains its own README with file-level documentation:

- [src/README.md](src/README.md)
- [src/chunkers/README.md](src/chunkers/README.md)
- [pipeline/README.md](pipeline/README.md)
- [data/README.md](data/README.md)
- [indexes/README.md](indexes/README.md)
- [outputs/README.md](outputs/README.md)
