# 2 Literature Review

This chapter surveys the work on which the thesis builds. Section 2.1 establishes the RAG paradigm and locates chunking within it. Sections 2.2 to 2.7 review the principal families of chunking strategy in roughly the order in which they entered the literature — fixed-window, sentence-boundary, semantic, LLM-based, hierarchical and query-adaptive, and finally late chunking and contextual retrieval — with attention to what each has been shown to achieve, at what cost, and on which kinds of text. Section 2.8 reviews what is known about how LLMs behave as context grows, the basis of the third hypothesis. Section 2.9 turns to the biomedical domain specifically: the one prior chunking study on clinical text, and the case for domain-adapted embedding models. Section 2.10 reviews how RAG systems are evaluated. Section 2.11 synthesises the review into a statement of the research gap that the experiment addresses.

## 2.1 Retrieval-Augmented Generation

Lewis et al. [1] introduced RAG as a way to combine a dense passage retriever with a sequence-to-sequence generator so that the generator conditions on retrieved evidence rather than on parametric memory alone. The original formulation treated the retriever and generator as jointly trainable, but the pattern that has become dominant in practice is simpler: a frozen embedding model indexes a corpus, a nearest-neighbour search returns the top-k units for a query, and a frozen LLM is prompted with the query and the retrieved units concatenated into a context window. This "frozen RAG" configuration requires no training, works with any generator exposed through an API, and is the configuration studied here.

Within that configuration, the corpus must be divided into units before it can be indexed. Lewis et al. used 100-word passages of Wikipedia, a choice inherited from the dense passage retrieval work that preceded them and adopted without evaluation of alternatives. The choice matters because the retrieval unit is simultaneously the unit of embedding, of ranking and of context assembly: it determines what a single vector represents, what a single search hit returns, and how much text each hit contributes to the generator's input. A large body of subsequent work has examined retrievers and generators; the unit itself has been examined far less.

## 2.2 Fixed-Window Chunking and the Overlap Question

Fixed-window chunking partitions a document into segments of a specified length — measured in characters, words or subword tokens — with an optional overlap between adjacent segments. It requires no linguistic analysis, behaves identically on any language, and runs in negligible time. The `RecursiveCharacterTextSplitter` in LangChain [4] popularised the pattern in 2022 and it has since become the default in most open-source RAG implementations. Window sizes of 128, 256 and 512 tokens recur across the literature [3, 5]; overlaps of 10–20% of the window are typical.

The theoretical case for overlap is that semantically related content may straddle a boundary; if adjacent chunks share text, both flanks carry partial evidence from the boundary region and the retriever has two chances to find it. The empirical case is weaker than the theoretical one. Śmigielski et al. [3], in the most comprehensive recent benchmark, find that fixed-window chunking at 512 tokens with a 50-token overlap is the most robust configuration across eight datasets — but they do not isolate the overlap from the window size, so the contribution of the overlap itself cannot be read from their result. Earlier observations on Natural Questions suggest that overlap adds little on short factoid queries, but that finding has not been tested on text where cross-sentence evidence is common.

Biomedical abstracts are precisely such text. A drug may be named in one sentence and its adverse effect reported in the next; an effect size may appear in one sentence and its confidence interval in the following one. If overlap has value anywhere, it should have value here. This is the basis for treating H1 as a genuine test rather than a foregone conclusion: the general-domain prior is that overlap does little, but the biomedical prior points the other way.

Qu et al. [5] provide the most directly relevant evidence on window size. Comparing fixed and semantic chunking across document types, they find that the advantage of content-aware splitting is largest on long documents with coherent topical sections and shrinks to within noise on short texts. They note explicitly that on abstracts — texts of a few hundred words — a well-tuned fixed splitter may be as good as anything more elaborate. That observation motivates including several window sizes in the present design: on short documents, window size may matter more than segmentation method.

## 2.3 Sentence-Boundary Segmentation

Sentence-boundary chunking uses detected sentence boundaries as candidate split points, either treating each sentence as a unit or accumulating sentences until a target length is reached. Its advantage over fixed windows is that it never cuts mid-sentence, so predicate–argument structure survives. Its risk is tokeniser error: a false boundary produces a degenerate fragment, and a missed boundary produces an oversized unit.

NLTK's punkt tokeniser [2] is the most widely used sentence segmenter in English NLP. It is an unsupervised model that learns abbreviation lists and collocation statistics from training text, and its distributed model was trained on general-domain material. Applied to biomedical prose, it encounters abbreviations — "i.v.", "p.o.", "vs.", "et al.", "Fig." — that are rare or absent from its training data, and the concern raised repeatedly in the applied literature is that it will over-segment at these tokens, producing fragments such as "Patients received 10 mg metformin i." and "v." that carry no retrievable meaning.

scispaCy [6], developed at the Allen Institute for AI, provides a spaCy pipeline trained on biomedical corpora including MedMentions and BC5CDR. Its sentence segmentation is dependency-parse-based and is claimed to respect domain abbreviations and to handle structured abstract headings correctly. Whether that claimed advantage translates into better retrieval and better answers is the open question that H2 addresses. No prior study has compared the two tokenisers under identical retrieval–generation conditions on PubMed text.

It is worth noting a structural feature of sentence-accumulation chunkers that becomes important in Chapter 5: when the target chunk length is comparable to the length of the documents being chunked, the accumulator may absorb an entire document into a single chunk before the target is reached. On a corpus of short abstracts, a sentence chunker with a 256-word target is therefore largely a whole-document chunker with occasional splits.

## 2.4 Semantic and Embedding-Based Chunking

Semantic chunking replaces the fixed-length rule with a data-driven boundary criterion: adjacent text units are embedded, and a split is inserted wherever the cosine similarity between neighbouring embeddings falls below a threshold θ. A sharp drop in similarity is taken to signal a topic transition. The idea has a long ancestry — Hearst's TextTiling [7] used lexical-cohesion valleys for the same purpose in 1997 — and has been reformulated for dense embeddings in several open-source frameworks, the most common implementation using all-MiniLM-L6-v2 at θ ≈ 0.75.

Śmigielski et al. [3] evaluate several semantic variants, including recursive semantic splitting, sequential hierarchical agglomerative clustering, a max–min criterion that maximises minimum within-chunk similarity, and GraphSeg, which partitions a sentence-similarity graph into communities. Their findings are consistent: recursive semantic splitting matches the best fixed-window baseline on dense-retrieval benchmarks at two to three orders of magnitude greater processing time; on literary text (GutenQA) it gains 4–6 F1 points over fixed windows, and on short factoid QA (Natural Questions) the gain disappears. The graph-based variants, which require pairwise similarity across all sentences and scale quadratically, do not outperform the recursive method. The authors' conclusion — that chunking complexity should be matched to document complexity — is the general-domain prior behind H2: on short abstracts, semantic chunking is not expected to beat simpler methods.

Qu et al. [5] reach the same conclusion from a different direction, asking directly whether semantic chunking is worth its computational cost and finding that it is only on long, topically structured documents.

An aspect of semantic chunking that has not been studied is the interaction between the embedding model used to *segment* and the one used to *retrieve*. If both share a representation space, split points align with retrieval-relevant features; if they differ — MiniLM to segment, PubMedBERT to retrieve — boundaries may be placed suboptimally for the retrieval task. The present design embeds every condition, including the semantic one, with both retrieval models, and so provides a first observation of this cross-model effect.

## 2.5 LLM-Based Chunking: Propositions, Boundary Detection and Granularity Routing

A more recent family uses an LLM at index time to decide how text should be divided. Chen et al. [8] proposed Dense X Retrieval, in which a document is decomposed by an LLM into atomic *propositions* — minimal, self-contained factual statements — each embedded and indexed separately; at query time, retrieved propositions are re-expanded to their surrounding passage before generation. The claim is that proposition-level indexing raises retrieval precision by removing noise from co-located but irrelevant sentences, and the original paper reports gains on factoid QA.

The cost is severe. Śmigielski et al. [3] reproduce Dense X and report processing times exceeding fifteen hours per dataset, against under a second for fixed windows; on five of eight datasets the method fails outright through API timeouts or memory exhaustion, and where it completes it does not consistently beat fixed windows. LumberChunker [9], which uses an LLM to detect semantic-shift points rather than to decompose sentences, times out on several datasets and performs comparably to recursive semantic chunking where it finishes. Both methods require one or more LLM calls per document, which makes them sensitive to rate limits, pricing and the non-determinism of the chunking model.

Meta-Chunking [10] and Mix-of-Granularity [11] represent a second direction. Meta-Chunking assigns boundary probabilities from the perplexity gradient of a small language model and merges chunks dynamically at query time; it gains 6.2 accuracy points over fixed windows on the RULER long-document benchmark, evaluated on general-domain corpora. Mix-of-Granularity trains a router to send factoid queries to proposition-level units, thematic queries to sentence-level units and survey queries to paragraph-level units, and improves on any single granularity — but the router must be trained on domain-specific data, an obstacle for zero-shot biomedical deployment.

Proposition-level chunking was included in the original design of this study and a decomposer was implemented. It was not executed, because the inference quota it would have consumed — one LLM call per abstract across 981 abstracts — was the same quota on which the twelve thousand generation calls depended. The benchmark evidence reviewed here suggests the omission is unlikely to have changed the conclusions: on short documents, proposition decomposition has not been shown to outperform fixed windows even where it completes.

## 2.6 Hierarchical and Query-Adaptive Chunking

Lu et al. [12] name the tension that runs through all of the above: small chunks give high retrieval precision but insufficient context for generation; large chunks give context but dilute the retrieval signal. HiChunk resolves it by modelling document hierarchy explicitly. A fine-tuned Qwen3-4B assigns each segment to a structural level (section, paragraph, clause), and an Auto-Merge retrieval step promotes clusters of matched fine-grained chunks to their coarser parent before generation. On the evidence-dense task of the authors' HiCBench benchmark, HiChunk with Auto-Merge reaches 81.0% evidence recall against 72.1% for 200-token fixed windows; on the evidence-sparse task the gap narrows to about three points. Processing costs roughly 5.75 seconds per document, which is tractable on CPU for a corpus of a thousand abstracts but still requires a fine-tuned model that may not transfer to abstract-only text.

Rastogi's Query-Adaptive Semantic Chunking (QASC) [13] departs from every method above by moving chunking from index time to query time: candidate sentences are scored against the query, seed sentences above a threshold are identified, and a window of neighbouring sentences is assembled around each seed. On 100 technical documents and 200 queries of four types, QASC reports F1 = 0.85, a relative improvement of 18–27% over fixed chunking and 8–12% over semantic and agentic alternatives, corroborated by a three-annotator human evaluation. The cost is that no static index can be built: every query scans every document. For the present design — 100 questions × 15 conditions × 4 depths × 2 encoders — that would mean hundreds of thousands of full-corpus scans, and the method has not been evaluated on biomedical text.

These methods define the frontier of the field and the direction it is moving: from choosing one granularity, to preserving structure, to choosing granularity per query. They also define what this thesis does *not* do. The experiment reported here is a controlled comparison of static strategies, intended to establish which of the simpler methods works on biomedical abstracts before more elaborate ones are considered. The literature above supplies the reason that ordering is sensible: elaborate methods have repeatedly failed to beat simple ones on short documents.

## 2.7 Late Chunking and Contextual Retrieval

Two further techniques improve chunk *embeddings* rather than chunk *boundaries*. Late chunking [14] reverses the conventional order: the whole document is passed through a long-context encoder first, and per-token representations that already carry document-level context are then pooled into chunks post hoc. Contextual retrieval [16] prepends to each chunk a short LLM-generated description situating it within its source document before embedding, so that entity mentions and pronominal references can be resolved.

Merola and Singh [15] evaluate both in a common framework on NFCorpus (a biomedical retrieval collection) and MS MARCO. Late chunking with Jina-V3 reaches NDCG@5 = 0.309 on NFCorpus, narrowly below contextual retrieval at 0.317; late chunking is cheaper but behaves inconsistently across encoders, improving with long-context models and degrading with short-context ones. Contextual retrieval combined with BM25 fusion and a cross-encoder reranker gives the best scores in their study but requires one LLM call per chunk and roughly 20 GB of GPU memory for the context-generation step. Neither technique was feasible within the compute constraints of this study, and neither addresses the boundary question that the hypotheses target; they are noted here as the natural next step once boundaries are settled.

## 2.8 Long-Context Behaviour in Language Models

The third hypothesis descends from a body of work on how LLMs use long inputs. Liu et al. [20], in the study from which the phrase "lost in the middle" comes, show that performance on multi-document QA depends strongly on *where* in the context the relevant passage sits: models are best when it is at the beginning or end and markedly worse when it is in the middle, and the degradation grows with context length. Shi et al. [17] show, separately, that LLMs are easily distracted by irrelevant material in the prompt — adding unrelated sentences to an arithmetic problem lowers accuracy — even when the model is capable of solving the problem in isolation.

Together these findings establish that more context is not automatically better: past some point, adding retrieved units adds distraction faster than it adds evidence. In the applied RAG literature this has been summarised as a "context cliff" at roughly 2,500 tokens on general-domain corpora, beyond which answer quality falls sharply. Whether that summary transfers to biomedical abstracts is unclear on two counts. Abstracts are denser than Wikipedia paragraphs — a 250-word abstract may carry as much distinct information as a 600-word encyclopaedia section — so if attention is limited by information rather than by word count, the cliff might come earlier. Conversely, IMRAD structure may give the generator anchors that let it locate relevant content in longer contexts, pushing the cliff later. H3 was formulated to find out; the result, reported in Chapter 4, is that on this corpus the relationship is not a cliff at all.

## 2.9 Chunking and Retrieval in the Biomedical Domain

### 2.9.1 Evidence density

Clinical and biomedical text has a property that general-domain chunking benchmarks do not exercise: evidence density. A single abstract reporting a randomised trial may state the intervention, comparator, primary endpoint, hazard ratio, confidence interval and p-value in two consecutive sentences. A boundary between those sentences leaves each chunk technically relevant but individually insufficient. Lu et al. [12] operationalise this concern in HiCBench's evidence-dense task and show that the gap between fixed and hierarchical chunking is largest there.

### 2.9.2 The one prior clinical study

Gomez-Cabello et al. [18] report what is, to the best of this author's knowledge, the only published chunking comparison on medical text. They compare an adaptive method that adjusts chunk boundaries by medical-entity density against a fixed-window baseline on physician-validated clinical questions, and report answer accuracy of 87% for the adaptive method against 13% for the baseline. The magnitude of that gap far exceeds anything reported on general-domain benchmarks and would, if it generalised, indicate that biomedical text poses a fundamentally different retrieval problem. The result is important for this thesis in two ways. It is the closest prior work, and the present study's findings — that simple fixed windows perform well and that sophisticated segmentation does not help — point the other way. Chapter 5 takes up the comparison directly; the short version is that the two studies differ in corpus, question type, generator and metric, and that both may be right about the text they studied.

### 2.9.3 Domain-adapted embedding models

Most chunking studies use general-purpose encoders — BERT-base, all-MiniLM-L6-v2, GTE, E5 — for both segmentation and retrieval. In biomedical NLP, models pretrained from scratch on PubMed text have consistently outperformed general BERT on named-entity recognition, relation extraction and question answering. Gu et al. [19] established this with PubMedBERT, and the NeuML `pubmedbert-base-embeddings` variant adapts the encoder for dense retrieval by contrastive fine-tuning on biomedical query–passage pairs, producing 768-dimensional vectors against MiniLM's 384. The expectation in the field is that these representations capture drug–gene–disease relations more faithfully and should retrieve biomedical passages better.

Whether that advantage persists across chunking strategies is unknown, and there is a plausible argument that it might not be uniform: a domain encoder might matter more for fragmented chunks, where biomedical co-reference is needed to infer relevance, than for coherent ones, where both encoders agree. That possibility — a chunking × embedding interaction — has not been tested in any published study, and its presence or absence determines whether chunking recommendations can be stated independently of encoder choice.

## 2.10 Evaluating RAG Systems

A RAG system can fail at retrieval, at generation, or at the interface between them, and a single end-to-end score cannot say which. The evaluation literature has therefore converged on measuring several properties separately.

*Retrieval* is assessed by whether the units returned contain the evidence a question needs — recall at k against annotated gold sources, or ranking measures such as MRR and nDCG where relevance grades are available. *Answer quality* is assessed against a reference answer, most commonly with ROUGE [21], whose longest-common-subsequence variant ROUGE-L rewards in-order overlap without requiring contiguous n-grams and is the standard for generated text with a reference. *Relevance* — whether the answer addresses the question at all — is assessed by embedding similarity between question and answer. *Faithfulness* — whether the answer is supported by the retrieved context rather than by the generator's own knowledge — is assessed by the proportion of answer content that can be located in the context; the RAGAS framework [22] formalises this as a claim-level check with an LLM judge, and a cheaper lexical approximation measures token-level precision of answer against context. *Efficiency* is assessed by context size and latency.

Two properties of these metrics matter for what follows. First, reference-based metrics such as ROUGE-L measure surface agreement with one reference phrasing, not correctness; two correct answers phrased differently score differently. Second, gold-source recall depends entirely on how gold sources were annotated. If each question is annotated with a single source but several documents in the corpus state the same fact, recall penalises the retriever for finding a correct source that is not the annotated one. The present study encountered exactly this, and Chapter 4 quantifies it.

## 2.11 Synthesis and Research Gap

Table 2.1 summarises the strategies reviewed by boundary criterion, cost, the document types on which they have been shown to work, and whether they have been validated on biomedical text.

| Method | Boundary criterion | Cost per document | Shown to work on | Biomedical validation | Refs |
|---|---|---|---|---|---|
| Fixed window | Word/token count | O(1) | General; short docs | Partial | [3, 5] |
| Sentence – NLTK | punkt boundaries | O(n) | General | None | [2] |
| Sentence – scispaCy | Dependency parse | O(n) | — | None in RAG | [6] |
| Semantic (cosine θ) | Embedding similarity drop | O(n·d) | Long, topical docs | None | [3, 5, 7] |
| Proposition (Dense X) | LLM decomposition | O(n)·LLM | Factoid QA | None | [3, 8] |
| LumberChunker | LLM shift detection | O(n)·LLM | Long narrative | None | [3, 9] |
| Meta-Chunking | Perplexity gradient | O(n)·LM | Long docs (RULER) | None | [10] |
| Mix-of-Granularity | Trained router | O(n)·LLM | Mixed query types | None | [11] |
| HiChunk | LLM hierarchy + Auto-Merge | O(n)·LLM | Evidence-dense docs | None | [12] |
| QASC | Query-time seeding | O(n·q) per query | Technical docs | None | [13] |
| Late chunking | Post-hoc pooling | O(n·d) | Long docs | Partial (NFCorpus) | [14, 15] |
| Contextual retrieval | LLM prefix per chunk | O(n)·LLM | Precision-sensitive | Partial (NFCorpus) | [15, 16] |
| Adaptive (entity density) | Medical entity density | O(n) | Clinical QA | One study | [18] |
Table: Chunking strategies surveyed, by boundary criterion, cost and validation status.

Three patterns emerge. Cost and performance do not scale together: the cheapest method is also the most robust across datasets, and the most expensive methods fail on several. Performance depends on document type: methods that help on long, structured documents help little or not at all on short ones. And almost none of this has been established on biomedical text — one study, on a different kind of clinical text with a different design, is the sum of the domain-specific evidence.

Table 2.2 states the specific open questions.

| Research axis | General-domain status | Biomedical status |
|---|---|---|
| Overlap in fixed windows | Studied on NQ, SQuAD, not isolated from window size | Not evaluated on PubMed |
| NLTK vs biomedical sentence segmenter | NLTK is the de facto standard | scispaCy exists; never benchmarked in RAG |
| Sentence vs semantic chunking | Comparable on short documents | Unknown on ~240-word abstracts |
| Window size on short documents | Suggested to dominate segmentation method | Not tested |
| Context-length effect | Position and length effects established; "cliff" at ~2,500 tokens | Shape and location unknown on dense text |
| Embedding × chunking interaction | Not studied | Not studied |
| LLM-based chunking at scale | Fails or times out on several benchmarks | Not evaluated |
Table: Open questions addressed by, or delimited from, the present study.

The first six rows are the gap this thesis addresses: a controlled, multi-metric comparison of fixed-window, sentence-boundary and semantic chunking on PubMed abstracts, across two embedding models and four retrieval depths, with hypotheses about overlap, segmentation and context size stated in advance. The seventh row — LLM-based chunking — is delimited from the study for the cost reasons the review has documented, and is returned to in Chapter 6.
