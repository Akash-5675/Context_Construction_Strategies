# 1 Introduction

## 1.1 Background

Large language models (LLMs) answer questions fluently, but the knowledge they draw on is frozen at training time, is not attributable to any source, and is not guaranteed to be correct. In domains where a wrong answer has consequences — clinical decision support, drug information, evidence synthesis — none of those properties is acceptable. Retrieval-Augmented Generation (RAG), introduced by Lewis et al. [1], addresses all three at once: rather than answering from parametric memory, the model is first given a set of documents retrieved from a curated knowledge base and instructed to answer only from what it has been shown. The answer becomes grounded in evidence that can be inspected, updated and cited.

A RAG system has two stages. The *retrieval* stage embeds the user's question into a vector space, searches an index of pre-embedded text units, and returns the most similar ones. The *generation* stage assembles those units into a context window, prepends the question, and asks an LLM to produce an answer. Almost all attention in the literature has gone to the components at either end — better embedding models, better retrievers, better generators. The step that connects them has received far less: before any of this can happen, the source documents have to be cut into the units that will be embedded, indexed, retrieved and assembled. That step is called *chunking*, and the family of decisions it involves — how large the units are, whether adjacent units share text, and whether boundaries follow the structure of the text or simply a word count — is what this thesis calls *context construction*.

Chunking is consequential because it fixes what the retriever can see. If a chunk is too small, the evidence needed to answer a question may be split across two units, and retrieving one without the other yields an incomplete answer. If a chunk is too large, a single unit carries the relevant sentence together with several irrelevant ones, so the retrieval signal is diluted and the generator's context fills with noise. Between those failure modes lies a region in which chunks are small enough to be retrieved precisely and large enough to preserve the evidence a question needs. Where that region lies, and how sensitive downstream answer quality is to landing outside it, is an empirical question — and one whose answer is likely to depend on the kind of text being chunked.

## 1.2 Motivation

In practice, chunking decisions are rarely made empirically. The dominant pattern in deployed systems is a fixed window of a few hundred tokens with a modest overlap between neighbours, a configuration popularised by the `RecursiveCharacterTextSplitter` in LangChain [4] and reproduced in tutorials, templates and default settings ever since. Overlap is included because it feels safer: if a fact straddles a boundary, both flanking chunks carry part of it. Window sizes of 256 or 512 tokens are chosen because they are round numbers that fit comfortably in an embedding model's input limit. Neither choice is usually validated on the corpus in question.

Where chunking *has* been evaluated, the evaluation has been on general-domain text: Wikipedia, news, web crawl, or question-answering benchmarks such as Natural Questions and SQuAD built on top of them [3, 5]. Biomedical text differs from that material in ways that bear directly on chunking. PubMed abstracts are short and dense: a typical abstract of around 240 words reports a study population, an intervention, a comparator, a primary endpoint and a statistical result, often with several named entities per sentence. They use abbreviations — "i.v.", "p.o.", "vs.", "et al." — that a general-purpose sentence tokeniser may misread as sentence terminals, fragmenting the text at exactly the wrong places. And they are structured: most follow an implicit or explicit IMRAD organisation, so that a boundary falling between a Results sentence and its Methods context severs information that a reader would treat as a single unit.

These properties motivate three specific doubts about received chunking practice. First, the benefit of overlap has been argued from intuition rather than measured, and on short, dense abstracts it is not obvious that duplicating boundary text buys anything. Second, the general-purpose sentence tokeniser NLTK punkt [2] is the de facto standard for sentence-boundary chunking, while a biomedical alternative, scispaCy [6], exists and is claimed to handle domain abbreviations correctly — yet the two have never been compared inside a retrieval–generation pipeline on biomedical text. Third, it is well documented on general-domain corpora that answer quality degrades once the context handed to the generator grows past a certain size [20], but whether that degradation follows the same shape, or occurs at the same point, on information-dense biomedical abstracts is unknown.

The practical stakes are not small. A RAG system deployed for clinical question answering that uses an oversized window, an unnecessary overlap and a retrieval depth chosen by habit may be delivering worse answers at higher cost than a simpler configuration would — and nobody would know, because the configuration was never tested against alternatives on the text it actually serves.

## 1.3 Problem Statement

The retrieval unit in a biomedical RAG system is chosen by convention rather than evidence. Chunk size, overlap and segmentation strategy are all known to affect what the retriever can find and how much text the generator receives, but their effect on answer quality has not been measured systematically on PubMed abstracts, and general-domain findings cannot be assumed to transfer to text that is shorter, denser and more abbreviation-laden than the corpora on which those findings were established.

This thesis addresses that gap with a controlled factorial experiment. The rest of the RAG pipeline — corpus, question set, embedding procedure, retrieval engine, generator and evaluation metrics — is held fixed while chunking strategy, embedding model and retrieval depth are varied, so that any difference in outcome can be attributed to the factors under study.

## 1.4 Research Questions and Hypotheses

The study is organised around three falsifiable hypotheses, formulated before any results were examined.

**H1 — Overlap.** Adding overlap between adjacent fixed-window chunks does not improve retrieval or answer quality on biomedical abstracts. The prediction is that, once window size is controlled for, overlap levels of 0%, 10%, 20% and 40% will be statistically indistinguishable.

**H2 — Segmentation.** Simple sentence-boundary chunking matches embedding-based semantic chunking on biomedical abstracts. The prediction is that NLTK sentence segmentation, scispaCy sentence segmentation and cosine-similarity semantic chunking will not differ significantly in answer quality, and that the biomedical tokeniser will offer no measurable advantage over the general-purpose one.

**H3 — Context size.** Answer quality degrades as the amount of retrieved context grows. As originally formulated in the literature [20] this was expressed as a threshold — a "context cliff" at roughly 2,500 tokens beyond which quality falls sharply. The hypothesis as tested here is directional: that quality declines with context size, with the shape and location of any threshold left as an empirical question.

Three secondary questions accompany the hypotheses. Does window size itself have an effect, independent of overlap? Does a biomedical embedding model (PubMedBERT) outperform a general-purpose one (MiniLM), and does the choice of embedding model interact with the choice of chunking strategy? And how does retrieval depth k trade off retrieval coverage against context volume?

## 1.5 Objectives

The objectives of the work were as follows.

1. To construct a biomedical evaluation corpus of PubMed abstracts spanning several disease areas, together with a question set carrying reference answers and gold source identifiers.
2. To implement a family of chunking strategies — fixed-window at three sizes and four overlap levels, sentence-boundary segmentation with two tokenisers, and semantic chunking — under a common interface so that they can be substituted without changing anything else in the pipeline.
3. To build a complete, reproducible RAG pipeline — embedding, indexing, retrieval, generation and evaluation — that runs on CPU-only hardware and free-tier inference, so that the experiment can be repeated without institutional compute.
4. To execute a full factorial experiment across chunking condition, embedding model and retrieval depth, with every question evaluated under every combination.
5. To test the three hypotheses with appropriate inferential statistics — analysis of variance, Tukey's honestly significant difference procedure for pairwise contrasts, and analysis of covariance to control for context volume — and to report effect sizes alongside significance.
6. To translate the findings into practical recommendations for practitioners building biomedical RAG systems under resource constraints.

## 1.6 Scope and Delimitations

The study is deliberately bounded in several respects, each of which is discussed further in Chapter 5.

The corpus consists of PubMed *abstracts*, not full-text articles. Abstracts are the unit most biomedical retrieval systems actually index, and they are freely available at scale; but they are short, and — as the results will show — their length interacts with chunk size in ways that constrain what can be concluded about strategies designed for longer documents.

The generator is a single model, Llama 3.1 8B Instruct, held constant at temperature zero across all conditions. This is a deliberate control, not a limitation of interest: the experiment isolates the effect of context construction, and a second generator would have doubled the cost of an experiment already constrained by inference quota. It does mean that findings about how the generator responds to context size are findings about this generator.

Chunking strategies that require an LLM pass over the corpus at index time — proposition-level decomposition in the style of Dense X Retrieval [8], and hierarchical methods such as HiChunk [12] — were designed for inclusion but not executed, because the same inference quota that limited the generator would have had to be spent on chunking. They are treated as related work and future work rather than as conditions.

Finally, the evaluation is fully automatic. Answer quality is measured by ROUGE-L against a reference answer, by embedding similarity between question and answer, and by lexical overlap between answer and context. No human judgement of correctness was collected. Automatic metrics are appropriate for a factorial comparison of this size — twelve thousand runs are beyond what human evaluation could cover — but they measure surface agreement rather than clinical correctness, and the thesis is careful not to claim more than they support.

## 1.7 Contributions

The thesis makes the following contributions.

1. **The first systematic, multi-metric comparison of chunking strategies on PubMed abstracts.** Fifteen conditions, two embedding models, four retrieval depths and one hundred questions — 12,000 controlled runs — evaluated on six metrics with pre-registered hypotheses and inferential statistics. Prior work has evaluated chunking on general-domain corpora or, in one case, on a small set of clinical questions; no previous study has run a factorial design on biomedical abstracts.
2. **The first head-to-head comparison of NLTK and scispaCy sentence segmentation inside a RAG pipeline on biomedical text.** The comparison is accompanied by a direct measurement of how often the two tokenisers actually disagree on this corpus, which explains the result mechanistically.
3. **The first test of chunking × embedding-model interaction on biomedical text.** The finding — no interaction — means that chunking recommendations derived here do not depend on which embedding model a practitioner uses.
4. **A refinement of the context-cliff hypothesis.** On biomedical abstracts, the relationship between context size and answer quality is a continuous decline from the smallest contexts tested, not a threshold. The thesis also separates two effects that a single averaged metric conflates — the generator's willingness to answer, which rises with context, and the quality of the answers it gives, which falls — and argues that both should be reported.
5. **A reproducible, zero-cost experimental framework.** The entire pipeline runs on a CPU-only laptop and free-tier inference; the corpus, question set, chunk files, indexes, results and analysis scripts are released so that the experiment can be repeated or extended.

## 1.8 Organisation of the Thesis

Chapter 2 reviews the literature on chunking for RAG — fixed-window, sentence-boundary, semantic, proposition-level, hierarchical and query-adaptive methods — together with work on long-context behaviour in LLMs, biomedical embedding models and RAG evaluation, and closes with a statement of the research gap. Chapter 3 describes the methodology in full: corpus construction, question annotation, each chunking strategy as implemented, embedding and indexing, retrieval, generation, the six evaluation metrics, the factorial design, the statistical analysis plan and the infrastructure on which the experiment ran. Chapter 4 reports the results, beginning with descriptive statistics and the structure of refusals, then testing each hypothesis in turn, followed by the secondary analyses of window size, embedding model, retrieval depth and per-topic robustness. Chapter 5 interprets the results, explains the mechanisms behind them, compares them with prior work, sets out practical recommendations and discusses threats to validity. Chapter 6 concludes and identifies future work. The appendices contain the full per-condition results table, the complete statistical output, sample question–answer pairs, the generator configuration and prompt, the structure of the codebase, and example generated answers.
