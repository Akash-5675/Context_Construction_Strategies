# data/

This directory holds all corpus and annotation files produced during the study.

## Files

| File | Created by | Description |
|------|-----------|-------------|
| `pubmed_corpus.csv` | `src/collect.py` | ~1000 PubMed abstracts across 5 medical topics. Columns: `pmid`, `title`, `abstract`, `mesh_terms`, `topic`. |
| `questions.csv` | Manual (Phase 2) | 100 annotated questions — 20 per topic. Columns: `question_id`, `topic`, `question`, `gold_doc_ids` (pipe-separated PMIDs), `reference_answer`. |
| `questions_template.csv` | Provided | Example rows showing the expected format for `questions.csv`. Copy and extend this file. |
| `chunk_stats.csv` | `src/run_chunkers.py` | Descriptive statistics per chunking condition: total chunks, mean/SD/min/max/median word count. |

## Subdirectory: chunks/

One `.pkl` file per chunking condition. Each file is a Python list of dicts with keys:

```
chunk_id       str   Unique ID: "{pmid}__{condition}_{index}"
doc_id         str   Source PMID
text           str   Chunk text
word_count     int   Number of words in chunk
strategy       str   Condition key (e.g. "fixed_256_20pct")
chunk_index    int   Position of this chunk within its source document
topic          str   Medical topic inherited from the parent abstract
```

### Condition keys produced

| Key | Description |
|-----|-------------|
| `fixed_128_0pct` … `fixed_512_40pct` | 12 fixed-token conditions: 3 sizes × 4 overlaps |
| `sent_nltk` | Sentence-boundary (NLTK punkt) |
| `sent_scispacy` | Sentence-boundary (scispaCy biomedical) |
| `semantic` | Embedding-similarity boundary (MiniLM, threshold=0.75) |
| `proposition` | Atomic propositions via Groq API |

## Manual annotation guide (questions.csv)

1. Open `data/pubmed_corpus.csv` and read 20–30 abstracts per topic.
2. Write a question whose answer is explicitly stated in at least one abstract.
3. Record the PMID(s) of the supporting abstract(s) in `gold_doc_ids` (pipe-separated if multiple).
4. Write a concise `reference_answer` (1–2 sentences) drawn directly from the abstract text.
5. Aim for a mix of factoid, causal, and comparative question types.

Budget: approximately 6–8 hours for 100 questions.
