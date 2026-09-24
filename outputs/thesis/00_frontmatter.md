# CONTEXT CONSTRUCTION STRATEGIES FOR BIOMEDICAL RETRIEVAL-AUGMENTED GENERATION

**An Empirical Study of Chunking and Retrieval Strategies on PubMed**

A thesis submitted in partial fulfilment of the requirements for the degree of

**Master of Technology in Artificial Intelligence and Machine Learning**

by

**Akash Ramesh**
**AM.SC.P2AML25002**

Under the guidance of

**Dr. Aswathy Mohan**

Department of Computer Science
School of Computing
Amrita Vishwa Vidyapeetham, Amritapuri Campus

2026

<<pagebreak>>

## Certificate

This is to certify that the thesis entitled **"Context Construction Strategies for Biomedical Retrieval-Augmented Generation: An Empirical Study of Chunking and Retrieval Strategies on PubMed"** submitted by **Akash Ramesh (AM.SC.P2AML25002)** in partial fulfilment of the requirements for the award of the degree of Master of Technology in Artificial Intelligence and Machine Learning is a bona fide record of the work carried out by him under my supervision and guidance at the Department of Computer Science, School of Computing, Amrita Vishwa Vidyapeetham, Amritapuri Campus.

The contents of this thesis have not been submitted, in full or in part, to any other university or institution for the award of any degree or diploma.

Place: Amritapuri

Date: 19 September 2026

**Dr. Aswathy Mohan**
Project Guide
Department of Computer Science

**Head of Department**
Department of Computer Science

<<pagebreak>>

## Declaration

I, **Akash Ramesh (AM.SC.P2AML25002)**, hereby declare that the thesis entitled **"Context Construction Strategies for Biomedical Retrieval-Augmented Generation: An Empirical Study of Chunking and Retrieval Strategies on PubMed"** is a record of original work carried out by me under the guidance of Dr. Aswathy Mohan, Department of Computer Science, School of Computing, Amrita Vishwa Vidyapeetham, Amritapuri Campus.

I further declare that this work has not formed the basis for the award of any degree, diploma, associateship, fellowship or other similar title to any candidate of any university. All sources of information and assistance have been acknowledged, and all material quoted or paraphrased from other work has been cited.

Place: Amritapuri

Date: 19 September 2026

**Akash Ramesh**
AM.SC.P2AML25002

<<pagebreak>>

## Acknowledgements

I express my sincere gratitude to my project guide, **Dr. Aswathy Mohan**, Department of Computer Science, School of Computing, for her guidance, encouragement and critical feedback throughout this work. Her insistence on rigorous evaluation shaped the statistical design of this study and the way its results are reported.

I thank the Head of the Department and the faculty of the Department of Computer Science, Amrita Vishwa Vidyapeetham, Amritapuri Campus, for providing the academic environment and the review structure within which this project was carried out.

I am grateful to the maintainers of the open-source tools on which this work depends: the Biopython Entrez interface, NLTK, scispaCy, Sentence-Transformers, FAISS, the rouge-score library, statsmodels and Matplotlib. I also acknowledge Cloudflare Workers AI, whose free-tier inference made it possible to complete twelve thousand generation calls without institutional compute funding.

Finally, I thank my family and friends for their patience during the months in which this experiment ran overnight on a laptop.

Akash Ramesh

<<pagebreak>>

## Abstract

Retrieval-Augmented Generation (RAG) has become the standard architecture for grounding large language model (LLM) answers in external evidence. Every RAG system must first divide its source documents into retrievable units — a step known as *chunking* — and the size, overlap and segmentation rule chosen at this step determine what the retriever can find and how much text the generator receives. In practice these choices are made by convention: fixed windows of a few hundred tokens, copied from general-purpose tutorials, with an overlap added on intuition. Whether those conventions hold for dense, abbreviation-rich biomedical text has not been tested systematically.

This thesis reports a controlled factorial study of context-construction strategies on a corpus of 981 PubMed abstracts spanning five MeSH-defined disease areas — hypertension, type 2 diabetes, lung neoplasms, COVID-19 and Alzheimer's disease — evaluated against 100 question–answer pairs with reference answers and gold source identifiers. Fifteen chunking conditions were compared: twelve fixed-window configurations formed by crossing three window sizes (128, 256 and 512 words) with four overlap ratios (0%, 10%, 20% and 40%), together with NLTK sentence segmentation, scispaCy biomedical sentence segmentation and embedding-similarity semantic chunking. Each condition was indexed with two embedding models (all-MiniLM-L6-v2 and PubMedBERT) and queried at four retrieval depths (k = 3, 5, 8, 10), with generation held constant using Llama 3.1 8B Instruct at temperature zero. The design yielded 12,000 evaluation runs, each scored on answer relevance, ROUGE-L, faithfulness, retrieval recall, context size and latency.

Three pre-registered hypotheses were tested with two-way ANOVA, Tukey HSD and ANCOVA. **H1** (overlap adds no benefit) was supported: overlap had no significant effect on answer quality once window size was controlled (F = 0.59, p = 0.62), and no pairwise contrast between overlap levels was significant. **H2** (sentence segmentation matches semantic chunking) was supported: NLTK, scispaCy and semantic strategies were statistically indistinguishable (F = 0.38, p = 0.68), a result that survived controlling for context volume (p = 0.83) and is explained mechanistically by the two sentence segmenters producing identical chunks for 97% of the corpus. **H3** (a context-length cliff) was supported in direction but refined in form: answer quality declined monotonically with context size from the smallest contexts tested (r = −0.150, p < 10⁻⁶⁰), with no threshold effect. The dominant factor was window size — 128-word chunks outperformed 256- and 512-word chunks (F = 57.7, p < 10⁻²⁴) — and the biomedical embedding model did not outperform the general-purpose one (p = 0.07), with no chunking × embedding interaction (p = 0.99).

Two further observations shape the interpretation. First, 62.6% of runs ended in a grounded refusal ("insufficient context"), so that aggregate quality scores conflate *whether* the system answered with *how well* it answered; the thesis therefore reports answer rate and conditional quality as separate outcomes and finds that larger contexts raise the former while lowering the latter. Second, because PubMed abstracts average 238 words, the sentence, semantic and 512-word conditions all reduce to whole-abstract retrieval; only the 128-word condition sub-divides abstracts, so the observed advantage of small windows is an advantage of sub-document over document-level retrieval units.

The study contributes the first systematic, multi-metric comparison of chunking strategies on PubMed abstracts, the first head-to-head test of NLTK against scispaCy segmentation inside a RAG pipeline, the first test of chunking × embedding interaction on biomedical text, and a refinement of the context-cliff hypothesis. Its practical recommendation for resource-constrained biomedical RAG is direct: use small fixed windows, omit overlap, retrieve fewer chunks, and do not assume that domain-specific segmenters or encoders will pay for themselves.

**Keywords:** Retrieval-Augmented Generation; chunking; biomedical NLP; PubMed; sentence segmentation; semantic chunking; context length; FAISS; PubMedBERT; evaluation.

<<pagebreak>>

## Table of Contents

<<toc>>

<<pagebreak>>

## List of Abbreviations

| Abbreviation | Expansion |
|---|---|
| ANCOVA | Analysis of covariance |
| ANOVA | Analysis of variance |
| API | Application programming interface |
| BERT | Bidirectional Encoder Representations from Transformers |
| CI | Confidence interval |
| CPU | Central processing unit |
| FAISS | Facebook AI Similarity Search |
| FWER | Family-wise error rate |
| HSD | Honestly significant difference (Tukey) |
| IMRAD | Introduction, Methods, Results and Discussion |
| LCS | Longest common subsequence |
| LLM | Large language model |
| MeSH | Medical Subject Headings |
| MiniLM | all-MiniLM-L6-v2 sentence-embedding model |
| NLTK | Natural Language Toolkit |
| PMID | PubMed identifier |
| QA | Question answering |
| RAG | Retrieval-Augmented Generation |
| ROUGE-L | Recall-Oriented Understudy for Gisting Evaluation, longest-common-subsequence variant |
| SD | Standard deviation |
| TPD / TPM | Tokens per day / tokens per minute (API quota units) |
Table: Abbreviations used in this thesis.
