# Appendix A Full Per-Condition Results

Table A.1 reproduces the complete results for all fifteen conditions on all six metrics, together with answer rate and conditional ROUGE-L. Each row summarises 800 runs (100 questions × 4 retrieval depths × 2 embedding models). Table A.2 gives the refusal rate per condition and Table A.3 the latency per condition.

{{table:per_condition|Complete per-condition results.}}

{{table:refusal_by_condition|Refusal rate per condition.}}

{{table:latency|Generation latency per condition. Latency includes network round-trip to the remote inference service.}}

# Appendix B Statistical Output

The following is the verbatim output of the analysis script (`src/analyze.py`) run on the final 12,000-row results file. It contains the ANOVA tables, Tukey HSD contrasts, ANCOVA and correlation results reported in Chapter 4.

{{file:outputs/hypotheses.txt}}

The chunking × embedding interaction test (Section 4.7.2) was run separately by `src/interaction.py`; its ANOVA table is reproduced in Table 4.13 and its per-condition breakdown in Table 4.14.

# Appendix C Question Set

Table C.1 lists all 100 questions with their reference answers and gold source PMIDs. Questions are identified by a topic prefix (HYP: hypertension; DIA: type 2 diabetes; LCA: lung cancer; COV: COVID-19; ALZ: Alzheimer's disease) and a sequence number.

{{table:all_questions|The complete question set.}}

# Appendix D Generator Configuration and Prompt

## D.1 Configuration

| Parameter | Value |
|---|---|
| Model | Llama 3.1 8B Instruct (`@cf/meta/llama-3.1-8b-instruct`) |
| Provider | Cloudflare Workers AI |
| Temperature | 0.0 |
| Maximum output tokens | 512 |
| Retry policy | Up to 5 attempts on HTTP 429 / 5xx with backoff of 2, 5, 10, 20 s |
| Inter-call delay | 1 s |
| Abort condition | 10 consecutive failed generations (daily quota exhausted) |
Table: Generator configuration used for all 12,000 runs.

## D.2 System prompt

> You are a precise biomedical assistant. Answer the question using ONLY the provided context. If the context does not contain enough information to answer, say 'Insufficient context.' Do not use external knowledge.

## D.3 User message template

> Context:
> [Source 1] {text of first retrieved chunk}
>
> [Source 2] {text of second retrieved chunk}
>
> … (one block per retrieved chunk, k blocks in total)
>
> Question: {question text}

## D.4 Embedding configuration

| Parameter | MiniLM | PubMedBERT |
|---|---|---|
| Model identifier | `sentence-transformers/all-MiniLM-L6-v2` | `NeuML/pubmedbert-base-embeddings` |
| Dimensions | 384 | 768 |
| Normalisation | L2 (unit vectors) | L2 (unit vectors) |
| Index type | FAISS IndexFlatIP | FAISS IndexFlatIP |
| Similarity | Cosine (exact) | Cosine (exact) |
| Device | CPU | CPU |
Table: Embedding and index configuration.

## D.5 Chunker parameters

| Chunker | Parameter | Value |
|---|---|---|
| Fixed window | Window sizes | 128, 256, 512 words |
| Fixed window | Overlap ratios | 0%, 10%, 20%, 40% of window |
| Fixed window | Unit | Whitespace-delimited word |
| Sentence (NLTK) | Tokeniser | NLTK punkt, English model |
| Sentence (NLTK) | Accumulation target | 256 words |
| Sentence (NLTK) | Carry-forward | 1 sentence |
| Sentence (scispaCy) | Tokeniser | `en_core_sci_sm` v0.5.4, sentence segmenter only |
| Sentence (scispaCy) | Accumulation target | 256 words |
| Sentence (scispaCy) | Carry-forward | 1 sentence |
| Semantic | Sentence source | scispaCy `en_core_sci_sm` |
| Semantic | Embedding | all-MiniLM-L6-v2 |
| Semantic | Boundary rule | Cosine similarity of adjacent sentences < 0.75 |
Table: Chunker parameters.

# Appendix E Code Structure

The released codebase is organised into three packages and two data directories.

## E.1 Directory layout

| Path | Contents |
|---|---|
| `data/pubmed_corpus.csv` | 981 abstracts: PMID, title, abstract, MeSH terms, topic |
| `data/questions.csv` | 100 questions: ID, topic, question, gold PMID, reference answer |
| `data/chunks/*.pkl` | 15 chunk files, one per condition |
| `indexes/` | 30 FAISS indexes with metadata (15 conditions × 2 encoders) |
| `src/chunkers/fixed_token.py` | Fixed-window chunker, all 12 conditions |
| `src/chunkers/sentence_nltk.py` | NLTK sentence-accumulation chunker |
| `src/chunkers/sentence_scispacy.py` | scispaCy sentence-accumulation chunker |
| `src/chunkers/semantic.py` | Cosine-threshold semantic chunker |
| `src/chunkers/proposition.py` | LLM proposition decomposer (designed, not executed) |
| `pipeline/embedder.py` | Loads and caches the two encoders; L2-normalised embedding |
| `pipeline/indexer.py` | Builds one IndexFlatIP per (condition, encoder) |
| `pipeline/retriever.py` | Top-k cosine retrieval against a named index |
| `pipeline/generator.py` | Prompt assembly, remote generation, retry and abort logic |
| `pipeline/evaluator.py` | The six metrics |
| `src/collect.py` | Entrez corpus collection |
| `src/fill_pmids.py` | Automated gold-PMID assignment |
| `src/run_chunkers.py` | Applies every chunker to the corpus |
| `src/experiment_runner.py` | The 12,000-run factorial loop with checkpointing and sharding |
| `src/validate.py` | Data-integrity checks on the results file |
| `src/analyze.py` | Hypothesis tests and the six analysis figures |
| `src/interaction.py` | Chunking × embedding interaction test |
| `src/thesis_stats.py` | Generates the statistics file from which every table in this thesis is built |
| `src/thesis_figures.py` | Generates the additional thesis figures |
| `outputs/all_results.csv` | All 12,000 runs, one row each |
| `outputs/thesis_stats.json` | Every statistic cited in this thesis |
| `outputs/hypotheses.txt` | Verbatim statistical output (Appendix B) |
| `outputs/figures/` | All figures |
Table: Layout of the released codebase.

## E.2 Execution order

1. `python src/collect.py` — fetch the corpus (requires an Entrez e-mail address).
2. `python src/fill_pmids.py` — assign gold PMIDs to the question set.
3. `python src/run_chunkers.py` — produce the fifteen chunk files.
4. `python pipeline/indexer.py` — build the thirty indexes.
5. `python src/experiment_runner.py --part A` — the 9,600 Part A runs (resumable; shard with `--worker 0|1`).
6. `python src/experiment_runner.py --part B --best-fixed fixed_128_0pct` — the 2,400 Part B runs.
7. `python src/validate.py` — integrity check.
8. `python src/analyze.py` — hypothesis tests and figures.
9. `python src/thesis_stats.py` and `python src/thesis_figures.py` — thesis tables and figures.

## E.3 Run key

Every run is identified by the key `{question_id}__{condition}__{embedding_model}__k{k}`, for example `HYP_003__fixed_128_20pct__pubmedbert__k3`. The experiment runner reads the results file at start-up, builds the set of completed keys, and skips any run whose key is present, which is what makes execution resumable across quota cycles and across two concurrently running shards.

# Appendix F Example Generated Answers

The following are verbatim generated answers from the results file, selected as the three highest-scoring answered runs and three explained refusals. Each is shown with its question, the reference answer against which it was scored, its ROUGE-L, whether the annotated gold source was among the retrieved chunks, and the size of the context the generator received.

{{examples}}

The refusal examples illustrate the behaviour described in Section 4.2: the generator does not merely decline but states what the retrieved context did contain and why it does not answer the question. The answered examples are the highest-scoring answer for each of three distinct questions. The first illustrates the finding of Section 4.10: the single highest-scoring answer in the experiment (ROUGE-L 0.909) was produced without the annotated gold source having been retrieved, from another abstract in the corpus that states the same clinical fact.
