# 3 Methodology

## 3.1 Overview of the Experimental Design

The study is a controlled factorial experiment. A single RAG pipeline — corpus, question set, embedding procedure, retrieval engine, generator and evaluation — is held fixed, and three factors are varied: the chunking condition (15 levels), the embedding model (2 levels) and the retrieval depth k (4 levels). Every one of the 100 questions is evaluated under every combination of factor levels, giving 15 × 2 × 4 × 100 = 12,000 runs, with exactly 800 runs per chunking condition. Because nothing else changes between conditions, any systematic difference in outcome is attributable to the factors under study.

Figure 3.1 shows the eight phases of the pipeline. Phases 1–4 are executed once and produce static artefacts — the corpus, the question set, fifteen sets of chunks and thirty FAISS indexes. Phases 5–7 are executed once per run. Phase 8 is executed once on the full results file.

![The eight-phase experimental pipeline. Phases 1–4 build static artefacts; phases 5–7 run 12,000 times; phase 8 runs once.](methodology_pipeline.png)

The experiment was organised in two parts. **Part A** evaluated the twelve fixed-window conditions and tests H1 (overlap) and the window-size question. **Part B** evaluated the three segmentation strategies against the best fixed-window condition identified in Part A, and tests H2. H3 (context size) is tested on the pooled results of both parts, since context volume varies across every condition and depth. The Part A anchor was selected by mean ROUGE-L across all 800 of its runs; the selection was made after Part A completed and before Part B began.

## 3.2 Corpus Construction

### 3.2.1 Source and topics

The corpus consists of PubMed abstracts retrieved through the NCBI Entrez API using the Biopython interface [27]. Five disease areas were chosen to span chronic cardiovascular disease, metabolic disease, oncology, infectious disease and neurodegeneration, and each was defined by a MeSH-tagged query so that topic membership rests on curated indexing rather than free-text matching. The queries were:

- `hypertension[MeSH] AND (treatment OR pathophysiology OR epidemiology)`
- `type 2 diabetes mellitus[MeSH] AND (management OR complications OR therapy)`
- `lung neoplasms[MeSH] AND (diagnosis OR treatment OR prognosis)`
- `COVID-19[MeSH] AND (clinical features OR treatment OR outcomes)`
- `Alzheimer disease[MeSH] AND (pathology OR biomarkers OR treatment)`

Two hundred records were requested per topic. Records without an abstract were discarded, yielding 981 abstracts (Table 3.1). For each record the PMID, title, abstract text, MeSH descriptor list and topic label were stored.

{{table:corpus_topics|Corpus composition by topic.}}

### 3.2.2 Corpus statistics

Abstracts average 238.2 words (SD 79.6; median 238; range 52–903), for a total of 233,668 words. The length distribution is the single most important property of the corpus for interpreting the results: a 256-word window covers the majority of abstracts entirely, and a 512-word window covers all but a handful. Section 3.4 returns to this when describing how many chunks each strategy actually produced.

### 3.2.3 Question set

One hundred question–answer pairs were constructed, twenty per topic. Each pair consists of a question, a reference answer of one or two sentences, and a gold source PMID. Questions were written to be answerable from the kind of clinical fact that appears in abstracts — first-line therapies, diagnostic thresholds, mechanisms of action, pathological hallmarks, common presentations — and reference answers were written as a domain reader would expect them to be stated. Questions average 11.8 words; reference answers average 29.7 words (range 16–47).

Gold source PMIDs were assigned by an automated procedure: for each question, the corpus was searched for the abstract with the greatest overlap between question keywords and abstract text within the question's topic, and that abstract's PMID was recorded. Ninety-six questions were matched this way; four questions for which no abstract exceeded the keyword threshold were assigned a representative abstract from the topic. Each question therefore carries exactly one gold PMID. This single-source design is the simplest possible annotation and was chosen for feasibility; its consequences for the retrieval-recall metric are quantified in Chapter 4 and discussed in Chapter 5. Table 3.2 shows one question from each topic.

{{table:question_samples|One question–answer pair from each topic.}}

## 3.3 Chunking Strategies

All chunkers expose the same interface: given an abstract's text and PMID, return a list of chunk records, each carrying the chunk text, its word count, its source PMID, its position within the abstract and the condition key. This uniformity is what allows the retrieval, generation and evaluation stages to be identical across conditions. Word counts throughout are whitespace-token counts — the text is split on whitespace and each resulting token counts as one word. The window sizes 128, 256 and 512 are therefore word counts, not subword-token counts; at roughly 1.3 subword tokens per English word they correspond to approximately 170, 340 and 680 model tokens.

### 3.3.1 Fixed-window chunking

The fixed-window chunker slides a window of `chunk_size` words across the abstract with a step of `chunk_size − overlap` words, where `overlap = chunk_size × overlap_pct / 100`. The final window is truncated at the end of the abstract. Three window sizes (128, 256, 512 words) were crossed with four overlap percentages (0%, 10%, 20%, 40%) to give the twelve conditions shown in Figure 3.2. At 128 words, the four overlaps correspond to 0, 12, 25 and 51 shared words between neighbours; at 512 words, to 0, 51, 102 and 204.

![The twelve fixed-window conditions: three window sizes crossed with four overlap ratios.](fixed_token_factorial.png)

### 3.3.2 Sentence-boundary chunking

Two sentence chunkers share identical accumulation logic and differ only in the tokeniser that supplies sentence boundaries. The text is segmented into sentences; sentences are appended to a buffer until the buffer's word count reaches a target of 256 words, at which point the buffer is emitted as a chunk; the last sentence of the emitted chunk is carried forward as the start of the next buffer, giving a one-sentence overlap between consecutive chunks; and any sentences remaining at the end of the abstract are emitted as a final chunk. The 256-word target was chosen to match the middle fixed-window size so that sentence and fixed chunks would be comparable in length.

The **NLTK** variant (`sent_nltk`) uses the punkt sentence tokeniser [2] with its distributed English model. The **scispaCy** variant (`sent_scispacy`) uses the `en_core_sci_sm` model [6] with the named-entity, tagger and lemmatiser components disabled, so that only the dependency-parse-based sentence segmenter runs.

### 3.3.3 Semantic chunking

The semantic chunker (`semantic`) segments the abstract into sentences using scispaCy, embeds each sentence with all-MiniLM-L6-v2, and computes the cosine similarity between each consecutive pair of sentence embeddings. A chunk boundary is inserted wherever that similarity falls below a threshold of 0.75; sentences on the same side of every boundary are concatenated into one chunk. Chunk length is therefore not controlled — it is whatever the similarity structure of the abstract produces. The threshold follows the most widely used open-source implementation of the method and the value cited by Śmigielski et al. [3].

### 3.3.4 Proposition chunking (designed, not executed)

A proposition chunker was implemented that would send each abstract to an LLM with an instruction to decompose it into self-contained atomic facts, one per line, following Chen et al. [8]. It was not run. Decomposing 981 abstracts requires 981 LLM calls with input and output of a few hundred tokens each — approximately 700,000 tokens — which exceeded the daily free-tier quota of every provider used and would have competed directly with the 12,000 generation calls on which the experiment depended. The condition is excluded from every analysis in this thesis and is discussed as future work.

## 3.4 Chunk Statistics

Table 3.3 reports, for each of the fifteen conditions, the number of chunks produced from the 981 abstracts, the mean number of chunks per abstract, and the distribution of chunk lengths.

{{table:chunk_stats|Chunk statistics for the fifteen conditions applied to the 981-abstract corpus.}}

Two features of this table govern the interpretation of every result that follows.

First, **only the 128-word window sub-divides abstracts to any meaningful degree**. At 128 words the corpus yields 2,293–2,879 chunks, or 2.3–2.9 per abstract. At 256 words it yields about 1.35 per abstract — most abstracts fit in one window, and only those above 256 words are split. At 512 words every abstract fits in a single window: all four 512-word conditions produce 984 or 985 chunks from 981 abstracts, one per abstract with a handful of exceptions for the longest records, and their mean chunk length (237.5–238.1 words) is simply the mean abstract length. The "512-word" condition is, on this corpus, whole-abstract retrieval, and the four overlap levels at 512 words are four copies of the same segmentation.

Second, **the sentence and semantic strategies also operate at close to whole-abstract granularity**. The 256-word accumulation target is longer than most abstracts, so the sentence chunkers emit 1.35 chunks per abstract — the same figure as the 256-word fixed window — and the semantic chunker emits 1.26. The three "advanced" strategies and the 256-word fixed window are therefore all variants of near-whole-abstract retrieval, differing in where the occasional boundary falls.

The two sentence chunkers produced identical chunk counts (1,329) and identical mean lengths (183.0 words), which prompted a direct comparison: of the 1,329 chunks, 1,291 (97.1%) are character-for-character identical between the NLTK and scispaCy conditions, and 38 differ. On this corpus the two tokenisers disagree on the boundaries of fewer than 3% of chunks. This measurement was made after chunking and before any generation run, and it is reported here rather than in the results because it is a property of the materials, not of the experiment's outcome; its implication for H2 is drawn out in Chapter 5.

Figure 3.3 plots chunks per abstract for all fifteen conditions against the one-chunk-per-abstract line.

![Chunks per abstract achieved by each condition. Only the 128-word windows sub-divide abstracts substantially; every other condition sits close to one chunk per abstract.](fig_chunks_per_abstract.png)

## 3.5 Embedding and Indexing

### 3.5.1 Embedding models

Every chunk set was embedded twice, once with each of two sentence-embedding models.

**all-MiniLM-L6-v2** [25] is a six-layer general-purpose sentence encoder producing 384-dimensional vectors, trained on over one billion sentence pairs drawn from general web and QA sources. It is the most widely used encoder in open-source RAG and the default in most tutorials.

**NeuML/pubmedbert-base-embeddings** is a 768-dimensional encoder built on PubMedBERT [19], which was pretrained from scratch on PubMed abstracts and PubMed Central full text, and subsequently fine-tuned contrastively on biomedical query–passage pairs for dense retrieval. It represents the domain-adapted alternative.

Both models were run through the Sentence-Transformers library [25] on CPU with L2 normalisation applied to every output vector, so that the inner product of two vectors equals their cosine similarity. Queries were embedded with the same model as the index they were searched against.

### 3.5.2 Indexing

For each of the 15 × 2 = 30 (condition, model) pairs, a FAISS [26] `IndexFlatIP` index was built over the normalised chunk vectors and written to disk together with a metadata file mapping index positions back to chunk records. `IndexFlatIP` performs exhaustive inner-product search; with normalised vectors this is exact cosine-similarity retrieval, with no approximation and no tuning parameters. Exactness was preferred over speed because index sizes (984–2,879 vectors) are small enough that exhaustive search completes in milliseconds, and because approximate indexes introduce a source of variance unrelated to the factors under study.

## 3.6 Retrieval

For a run at retrieval depth k, the question is embedded with the run's embedding model, the corresponding index is searched, and the k highest-scoring chunks are returned with their similarity scores and source PMIDs. Four depths were tested: k ∈ {3, 5, 8, 10}. No re-ranking, query expansion, hybrid lexical search or deduplication was applied; the retrieved set is exactly the top-k by cosine similarity.

Because chunk lengths differ across conditions, the same k delivers different amounts of text: at k = 5, the 128-word conditions deliver roughly 500 words of context while the 512-word conditions deliver roughly 1,200. This is intentional — k is the practitioner's control, and the experiment measures what happens when it is held constant across strategies — but it means that strategy comparisons at fixed k are confounded with context volume. Section 3.9 describes how that confound is handled statistically.

## 3.7 Generation

### 3.7.1 Generator

All answers were produced by **Llama 3.1 8B Instruct** [24], served through Cloudflare Workers AI. The generator was held constant across every run: the same model, the same system prompt, temperature 0.0, and a maximum output length of 512 tokens. Temperature zero makes decoding greedy and therefore deterministic, so that the same context always yields the same answer and differences between runs are attributable to the context and not to sampling.

### 3.7.2 Prompt

The system prompt instructs the model to answer only from the supplied context and to decline when the context is insufficient:

> You are a precise biomedical assistant. Answer the question using ONLY the provided context. If the context does not contain enough information to answer, say 'Insufficient context.' Do not use external knowledge.

The user message concatenates the retrieved chunks, each prefixed with a source label, followed by the question:

> Context:
> [Source 1] {chunk 1 text}
>
> [Source 2] {chunk 2 text}
> …
> Question: {question}

The explicit permission to refuse is a design decision with consequences examined at length in Chapter 4. Without it, a model asked a question whose answer is not in the context will typically answer anyway from parametric memory, and the resulting answer cannot be attributed to the retrieval. With it, a refusal is an observable outcome that carries information about the context — and the proportion of refusals becomes a measurement in its own right.

### 3.7.3 Infrastructure

Generation ran on two Cloudflare Workers AI free-tier accounts, each allowing 10,000 "neurons" of inference per day — approximately 1,500 runs at the observed average context size. The experiment runner was sharded by chunking condition so that the two accounts could work through disjoint halves of the design simultaneously, and was made resumable: results were appended to a single file with a checkpoint every fifty runs, each run was keyed by its (question, condition, model, k) tuple so that completed runs were skipped on restart, and sustained API failure — the signature of an exhausted daily quota — caused a clean abort with all buffered results flushed to disk. Under these arrangements the 12,000 runs completed over several daily quota cycles at no cost. Part A completed on 7 August 2026 and Part B on 8 August 2026. Earlier attempts using the Groq and Google Gemini free tiers were abandoned after both exhausted their daily token allowances within hours; those runs were discarded and every result reported here was generated by the Llama 3.1 8B configuration described above.

## 3.8 Evaluation Metrics

Six quantities were recorded for every run. Four measure quality; two measure cost.

**Answer relevance.** The cosine similarity between the embedding of the question and the embedding of the generated answer, computed with the run's embedding model. It measures whether the answer is *about* the question, independently of correctness. Range [−1, 1]; in practice values near zero or below indicate an answer that does not engage with the question, such as a refusal.

**ROUGE-L F1.** The F-measure of the longest common subsequence between the generated answer and the reference answer [21], computed with Porter stemming. If LCS is the length of the longest common subsequence, P = LCS / |answer| and R = LCS / |reference|, ROUGE-L is 2PR / (P + R). It is the primary quality metric: it rewards answers that reproduce the reference's content in order without requiring identical wording, and it is the standard reference-based measure for generated text. Range [0, 1].

**Faithfulness.** The proportion of distinct alphanumeric tokens in the answer that also appear in the concatenated retrieved context: |tokens(answer) ∩ tokens(context)| / |tokens(answer)|. It is a lexical approximation to the claim-level faithfulness of RAGAS [22] and detects answers that introduce material not present in the context. Range [0, 1]. A refusal scores low because the words "insufficient context" rarely appear in the context.

**Retrieval recall.** The proportion of the question's gold PMIDs whose abstracts contributed at least one retrieved chunk. Since every question has exactly one gold PMID, this is binary: 1 if any retrieved chunk came from the annotated source, 0 otherwise. It measures the retriever, not the generator.

**Context words.** The total word count of the concatenated retrieved chunks — the amount of text the generator received, and the independent variable for H3. Reported in words; approximate model-token counts are obtained by dividing by 0.75.

**Latency.** Wall-clock milliseconds from dispatch of the generation request to receipt of the answer. Because generation was served remotely, this includes network time and reflects the service's load as well as the context size; it is reported as a cost indicator, not as a precise measurement of compute.

### 3.8.1 The refusal outcome

The prompt permits the generator to answer "Insufficient context," and it frequently does. Every such answer scores near zero on ROUGE-L, relevance and faithfulness — not because the generator failed, but because it correctly declined. A condition's mean ROUGE-L is therefore a mixture of two things: how often it produced an answer at all, and how good the answers were when it did. Averaging them together hides which of the two is driving a difference.

The analysis accordingly treats refusal as a distinct outcome. For every condition and factor level, two quantities are reported: the **answer rate** (the proportion of runs that produced a substantive answer) and the **conditional quality** (mean ROUGE-L over answered runs only), alongside the unconditional mean. A run is classified as a refusal if its answer contains the phrase "insufficient context" (case-insensitive), which captures both the bare phrase and the explained refusals in which the model states what the context did and did not contain.

## 3.9 Statistical Analysis

All tests were two-sided at α = 0.05 and were specified before results were examined. ROUGE-L is the dependent variable for hypothesis tests unless stated otherwise.

**H1** was tested with a two-way analysis of variance on the 9,600 Part A runs, with window size (3 levels) and overlap (4 levels) as factors. The test of interest is the main effect of overlap after window size is accounted for. Pairwise contrasts between overlap levels were tested with Tukey's HSD, which controls the family-wise error rate across all six pairwise comparisons. Partial η² is reported as the effect size.

**H2** was tested with a one-way ANOVA across the three segmentation strategies (NLTK, scispaCy, semantic) on their 2,400 runs, with Tukey HSD contrasts. Because the strategies deliver different context volumes at the same k, the comparison was repeated as an analysis of covariance with context words as a covariate, so that the strategy effect is estimated at matched context size. A four-group ANOVA including the Part A anchor was also run to place the strategies relative to the best fixed window.

**H3** was tested on all 12,000 runs by the Pearson and Spearman correlations between context size (in approximate tokens) and ROUGE-L, supplemented by a binned analysis in which runs were grouped into seven context-size bands and the mean of each band was inspected for the presence of a threshold. The correlation was also computed on answered runs only, to separate the effect of context on conditional quality from its effect on answer rate.

**Secondary analyses.** The window-size main effect from the H1 model was followed with Tukey contrasts between sizes. The embedding models were compared with Welch's t-test on ROUGE-L and on recall. The chunking × embedding interaction was tested by adding an interaction term to a two-way ANOVA over all fifteen conditions and both models. Per-topic robustness was assessed by repeating the window-size comparison within each topic and by a one-way ANOVA across topics.

With 800 runs per condition, the design has power above 0.99 to detect a difference in mean ROUGE-L of 0.03 between any two conditions at the observed within-condition standard deviation of approximately 0.135. Null results in this thesis are therefore interpretable as evidence that any effect present is smaller than that, not as failures to detect.

All analyses were performed in Python 3.12 with pandas, SciPy and statsmodels; the analysis script is included in the released code.

## 3.10 Data Integrity

Before analysis, the results file was checked for rows with no generator response (identifiable by a latency of zero, since latency is recorded only on a successful response), blank answers, duplicate run keys, and metric values outside their valid ranges. None were found in the final 12,000-row file. During execution, transient API failures had at one point been recorded as rows carrying the refusal phrase as a fallback value; the generator was modified to raise an exception on failure so that no row is written at all, and every row in the final file corresponds to a completed generation call.

## 3.11 Reproducibility

The pipeline runs end-to-end on a CPU-only laptop with 8 GB of memory. The corpus, question set, chunk files, indexes, the complete results file, the statistics file from which every table in this thesis is generated, and all scripts — collection, annotation, chunking, indexing, the experiment runner, the analysis, and the figure and table generators — are released. The only external dependency at run time is an inference endpoint serving Llama 3.1 8B Instruct; any provider serving the same weights with greedy decoding should reproduce the answers.
