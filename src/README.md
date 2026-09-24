# src/

Source scripts for corpus collection, chunking, and experiment execution.

## Scripts

### collect.py — Phase 1: Corpus collection

Fetches ~1000 PubMed abstracts (200 per topic) via Biopython Entrez and saves
them to `data/pubmed_corpus.csv`.

```bash
python src/collect.py
```

Requires `ENTREZ_EMAIL` in `.env`. Optionally `NCBI_API_KEY` for higher rate limits.
Runtime: ~5 minutes.

---

### run_chunkers.py — Phase 3.5: Apply all chunkers

Loads `data/pubmed_corpus.csv`, applies every chunking condition, and saves
results to `data/chunks/{condition_key}.pkl` plus `data/chunk_stats.csv`.

```bash
python src/run_chunkers.py

# Skip proposition (no Groq API calls — useful for testing)
python src/run_chunkers.py --skip-proposition
```

Skips conditions whose `.pkl` already exists (safe to resume). Runtime: ~30 min
for all non-proposition strategies; add ~8 min for proposition.

---

### experiment_runner.py — Phase 7: Run the full experiment

Executes the factorial experiment (conditions × questions × k × embeddings),
appending rows to `outputs/all_results.csv` with checkpointing every 500 rows.

```bash
# Part A: fixed-token conditions only (9,600 rows)
python src/experiment_runner.py --part A

# Part B: all 5 strategies with best fixed config as anchor (4,000 rows)
python src/experiment_runner.py --part B --best-fixed fixed_256_20pct

# Quick smoke test (5 questions only)
python src/experiment_runner.py --part A --dry-run
```

Requires `GROQ_API_KEY` in `.env`. Estimated runtime: ~2.5 hours total on CPU.

---

### analyze.py — Phase 8: Statistics and figures

Reads `outputs/all_results.csv`, computes summary statistics, runs ANOVA + Tukey
HSD, and generates 6 publication-ready figures in `outputs/figures/`.

```bash
python src/analyze.py
```

---

## Subdirectory

- **chunkers/** — Five chunking strategy modules. See [chunkers/README.md](chunkers/README.md).
