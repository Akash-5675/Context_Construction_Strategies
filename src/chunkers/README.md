# src/chunkers/

Five chunking strategy implementations for the biomedical RAG study.

Each chunker exposes a single `chunk(text: str, doc_id: str) -> list[dict]` method
and a `condition_key` attribute used as the filename stem in `data/chunks/`.

## Strategies

### fixed_token.py — `FixedTokenChunker`

Splits text into windows of exactly `chunk_size` words with a word-level `overlap`.

**12 conditions:** 3 sizes × 4 overlap percentages

| Condition key | Chunk size | Overlap |
|---------------|-----------|---------|
| `fixed_128_0pct` | 128 words | 0 |
| `fixed_128_10pct` | 128 words | 13 words (~10%) |
| `fixed_128_20pct` | 128 words | 26 words (~20%) |
| `fixed_128_40pct` | 128 words | 51 words (~40%) |
| `fixed_256_*` | 256 words | same ratios |
| `fixed_512_*` | 512 words | same ratios |

**Role in study:** Baseline. Tests H1 (does overlap help?). Anchor for Part B comparison.

---

### sentence_nltk.py — `SentenceNLTKChunker`

Accumulates sentences (NLTK punkt tokeniser) until ~256 words, then carries
forward the last sentence as overlap. Avoids mid-sentence splits.

**Condition key:** `sent_nltk`

**Role in study:** Tests H2. Represents naive sentence tokenisation that may fail
on biomedical abbreviations (i.v., p.o., Fig.).

---

### sentence_scispacy.py — `SentenceScispaCyChunker`

Identical accumulation logic to NLTK, but uses scispaCy `en_core_sci_sm` for
sentence segmentation, which is trained on biomedical text and handles
domain-specific abbreviations correctly.

**Condition key:** `sent_scispacy`

**Role in study:** Tests H2 counterpart. If scispaCy outperforms NLTK, it confirms
that domain-specific tokenisation matters for biomedical RAG.

**Dependency:**
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

---

### semantic.py — `SemanticChunker`

Uses scispaCy sentences as atomic units, embeds each with `all-MiniLM-L6-v2`,
and inserts a boundary whenever adjacent sentence similarity drops below 0.75.
Groups semantically coherent sentences regardless of word count.

**Condition key:** `semantic`

**Role in study:** Tests whether meaning-based boundaries outperform size-based
ones on biomedical text (H2 extended).

---

### proposition.py — `PropositionChunker`

Calls Groq API (`llama-3.1-8b-instant`) once per abstract to decompose the text
into a list of atomic, self-contained factual propositions. Most granular strategy.

**Condition key:** `proposition`

**Role in study:** Highest retrieval precision hypothesis. Tests whether
fine-grained atomic facts improve clinical QA faithfulness.

**Note:** Requires `GROQ_API_KEY` in `.env`. Makes ~1000 API calls at index time
(~8 min). Results are cached in `data/chunks/proposition.pkl`.

---

## Output format

Every chunker returns a list of dicts:

```python
{
    "chunk_id":    str,   # "{pmid}__{condition}_{index}"
    "doc_id":      str,   # source PMID
    "text":        str,   # chunk text
    "word_count":  int,
    "strategy":    str,   # condition_key
    "chunk_index": int,   # position within source document
}
```
