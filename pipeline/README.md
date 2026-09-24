# pipeline/

Reusable modules for embedding, indexing, retrieval, generation, and evaluation.
These are imported by `src/experiment_runner.py` and can also be used interactively.

## Modules

### embedder.py

Loads and caches the two embedding models. Normalises all vectors to unit length
so FAISS inner-product search equals cosine similarity.

```python
from pipeline.embedder import embed, embed_single

# Embed a batch (returns float32 numpy array, shape [N, dim])
vectors = embed(["text one", "text two"], model_key="minilm")

# Embed a single query
q_vec = embed_single("What causes hypertension?", model_key="pubmedbert")
```

| `model_key` | Model | Dim | Use |
|-------------|-------|-----|-----|
| `minilm` | `all-MiniLM-L6-v2` | 384 | General-domain baseline |
| `pubmedbert` | `NeuML/pubmedbert-base-embeddings` | 768 | Biomedical domain |

---

### indexer.py

Builds FAISS `IndexFlatIP` indexes from pre-chunked `.pkl` files.
Can be run as a script or imported as a function.

```bash
python pipeline/indexer.py
python pipeline/indexer.py --condition fixed_256_20pct --model minilm
```

Output files: `indexes/{condition}__{model}.index` + `.meta.pkl`

---

### retriever.py

Loads an index pair (lazy, cached after first load), embeds the query, and
returns top-k chunk dicts sorted by cosine similarity.

```python
from pipeline.retriever import retrieve

chunks = retrieve(
    query="What is the mechanism of metformin?",
    condition_key="fixed_256_20pct",
    model_key="minilm",
    k=5,
)
# Returns list of chunk dicts with added key: retrieval_score (float)
```

---

### generator.py

Calls Groq API with retrieved chunks as context. Returns the answer string,
context word count, latency, and token usage.

```python
from pipeline.generator import generate

result = generate(query="...", chunks=[...])
# result keys: answer, context_word_count, latency_ms, prompt_tokens, completion_tokens
```

Requires `GROQ_API_KEY` in `.env`.

---

### evaluator.py

Computes the 6 evaluation metrics for a single (question, answer, chunks) triple.

```python
from pipeline.evaluator import evaluate

metrics = evaluate(
    query="...",
    answer="...",
    reference_answer="...",
    gold_doc_ids=["12345678"],
    retrieved_chunks=[...],
    context_word_count=450,
    latency_ms=1200.0,
    model_key="minilm",
)
# Keys: answer_relevance, reference_similarity, faithfulness,
#       retrieval_recall, context_words, latency_ms
```

## Metric definitions

| Metric | Formula | What it tests |
|--------|---------|---------------|
| `answer_relevance` | cosine(query_emb, answer_emb) | Answer addresses the question |
| `reference_similarity` | cosine(answer_emb, ref_emb) | Agreement with gold answer |
| `faithfulness` | token F1(answer ∩ context) / len(answer) | No hallucination |
| `retrieval_recall` | \|gold_doc_ids ∩ retrieved_doc_ids\| / \|gold_doc_ids\| | Correct doc retrieved |
| `context_words` | word_count(concatenated chunks) | Context window size (H3 variable) |
| `latency_ms` | wall-clock ms | Practical efficiency |
