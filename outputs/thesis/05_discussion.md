# 5 Discussion

The results chapter established what happened. This chapter asks why, what it means, and what it does not mean. Sections 5.1 to 5.3 interpret the three hypotheses in turn, drawing on the chunk statistics of Section 3.4 to explain the findings mechanistically. Section 5.4 addresses the finding that was not hypothesised — the dominance of window size — and argues that it and the three hypotheses are one result seen from four angles. Section 5.5 sets the results against prior work, including the one study whose conclusions point the other way. Section 5.6 states the threats to validity, including one discovered during analysis. Section 5.7 translates the findings into recommendations for practice.

## 5.1 Overlap: Why Nothing Happened

The theoretical case for overlap is that evidence straddling a chunk boundary is recovered by whichever flanking chunk the retriever finds. For that mechanism to matter, three things must be true: boundaries must fall inside abstracts; they must fall between sentences that jointly carry an answer; and the retriever must find the chunk on one side of the boundary but not the other. On this corpus the first condition largely fails.

Section 3.4 showed that at 512 words there are no internal boundaries at all — every abstract is one chunk — and at 256 words only about a third of abstracts are split. Overlap at those sizes has nothing to overlap. Only at 128 words are abstracts routinely divided, into two or three chunks, and there the overlap of 12–51 words is being asked to recover evidence that spans a boundary placed by word count rather than by structure. It could in principle help; the results say it does not. Recall at 128 words does not rise with overlap, and neither does anything else.

There is also a reason to expect overlap to be neutral even where boundaries exist. The retrieval unit is scored by the cosine similarity of its embedding to the query. Adding the last few sentences of the previous chunk to the start of the current one changes the current chunk's embedding — it now represents slightly different content — but it does not make that embedding more similar to a query about the content of the *previous* chunk, because the previous chunk still exists and still embeds its own content more faithfully. Overlap duplicates text without duplicating retrievability. The only mechanism by which it could help is the case where the two halves of a fact are each individually too weak a signal to retrieve but together are strong enough, and the overlapping chunk happens to contain both halves; the results suggest that case is rare.

The finding matches the general-domain prior from Natural Questions and extends it to text where the biomedical prior — that cross-sentence evidence is common — pointed the other way. The practical implication is unambiguous. Overlap enlarges the index (by up to 26% at 128 / 40%), enlarges every retrieved context (by 8% at 40%), and slows every generation call proportionally, in exchange for no measurable benefit. It should be omitted.

## 5.2 Segmentation: Why Three Strategies Tied, and Why All Three Lost

### 5.2.1 The sentence tokenisers agree

H2 included scispaCy on the strength of a concern raised repeatedly in the applied literature: that NLTK's general-purpose tokeniser over-segments biomedical text at abbreviations such as "i.v." and "et al.", producing fragments that destroy meaning. The concern is reasonable in principle. On this corpus it did not materialise. The two tokenisers produced character-identical chunks for 1,291 of 1,329 chunks (97.1%). The remaining 38 chunks — fewer than 3% — are the entire population on which the tokenisers could have produced a different retrieval outcome, and 38 chunks spread across 800 runs at four retrieval depths is not enough to move a mean.

Two features of the materials explain the agreement. PubMed abstracts, unlike clinical notes or full-text methods sections, contain relatively few of the abbreviations that trip punkt; they are written for a general biomedical readership and tend to spell out "intravenous" and "versus." And the sentence chunkers do not treat each sentence as a unit — they *accumulate* sentences to a 256-word target, so that even where the two tokenisers place a boundary differently, the sentences on both sides usually end up in the same chunk anyway. A tokeniser disagreement changes the chunk only when it falls exactly at the point where the accumulator crosses 256 words. The biomedical tokeniser's advantage, whatever it may be on other text, was neutralised by the corpus and by the chunker design before any retrieval took place.

This is the mechanistic content of the H2 null. It is not that the tokenisers are equivalent in general; it is that on abstracts, accumulated to a target longer than most abstracts, they produce the same chunks.

### 5.2.2 All three strategies are whole-abstract retrieval

The semantic chunker differs from the sentence chunkers in principle — its boundaries follow embedding similarity rather than word count — but Section 3.4 showed it produces 1.26 chunks per abstract against the sentence chunkers' 1.35. All three strategies leave most abstracts whole and split the rest once. They are, functionally, three ways of retrieving whole abstracts with an occasional internal boundary, and the results show what one would expect of three such methods: near-identical answer rate, near-identical quality, near-identical recall, and context volumes that differ only because the semantic chunker's variable-length chunks run slightly longer.

This also explains why all three lost to the 128-word fixed window by 0.042–0.047 ROUGE-L. The fixed window at 128 words is the only strategy in the design that retrieves *sub-abstract* units. The comparison in Part B was framed as fixed versus sentence versus semantic; in practice it was sub-abstract versus whole-abstract, and sub-abstract won. Section 5.4 develops this.

### 5.2.3 What the null does and does not say

The ANCOVA result is the important guard here. The three strategies delivered different amounts of context — the semantic chunker about 8% more than the sentence chunkers — and since context volume is itself associated with outcome, an uncontrolled comparison could have hidden a real difference behind a volume difference, or manufactured one. Controlling for context words, the strategy effect is p = 0.83; the covariate is p < 0.0001. The strategies do not differ once volume is held constant, and volume is what matters.

The null is bounded, not absolute. With 800 runs per condition, the design has 84% power to detect a difference of 0.02 ROUGE-L and 99% power for 0.03; the noise floor from serving non-determinism (Section 4.3.1) is about 0.013. The three strategies fall within 0.005 of each other. The correct reading is that any difference between them on this corpus is smaller than 0.02, and probably smaller than the noise floor — not that no difference could exist on any corpus.

## 5.3 Context Size: A Slope, Not a Cliff

### 5.3.1 The shape

H3 was formulated as a threshold because that is how the effect has been described in applied work on general-domain corpora: quality holds, then falls. The results show a different shape. Quality declines from the smallest contexts tested — the drop from the under-500-token band to the 500–1,000 band is the largest single step in the curve — flattens through the middle bands, and falls again at the top. There is no band at which quality is stable before falling. Both the unconditional and the conditional metric decline monotonically or nearly so across the full range, and the Spearman correlation of −0.198 is stronger than the Pearson of −0.150, which is what one expects when the relationship is monotonic but not linear.

The mechanism suggested by Liu et al. [20] — that a generator attends less reliably to evidence buried in a long context — is consistent with a continuous decline. Each additional retrieved chunk that does not contain the answer is a distractor; each one that does is diluted by the others. Shi et al. [17] showed that a single irrelevant sentence lowers accuracy on tasks the model can otherwise solve. There is no reason such an effect would wait for a threshold; each added distractor should cost something. On abstracts, where the retrieved units are themselves dense, the cost per added unit appears to be paid from the first.

### 5.3.2 Willingness and accuracy

The decomposition into answer rate and conditional quality yields the most useful finding of the study for practitioners. As context grows, the generator answers more often and answers less well. Between k = 3 and k = 10, answer rate rises from 32% to 40% and conditional ROUGE-L falls from 0.212 to 0.188. The reason is structural. A larger context is more likely to contain *something* that addresses the question — so the generator, instructed to answer only from context, finds licence to answer — but it is also more likely to contain material that addresses a neighbouring question, and the generator draws on that too. The refusal that a smaller context would have produced is replaced by an answer that is partly right.

A deployed system's user sees only the unconditional average, and the average declines. But the decomposition shows that the decline is a fall in quality that a rise in willingness partly masks, which is the more alarming reading: a system tuned for a high answer rate is being tuned toward confident partial answers. In a clinical setting a grounded refusal is a safer outcome than a plausible half-answer, and the results suggest that the two are traded against each other by the single knob of context size.

### 5.3.3 The 2,500-token threshold

The threshold as stated in the literature is not visible in these data, and the reason may simply be that this experiment operates in a different regime. Most runs delivered under 2,000 model tokens; the largest band, above 3,000, contains 625 runs. General-domain studies reporting a cliff at 2,500 tokens typically retrieve from long documents at high k, delivering contexts of many thousands of tokens, and the "cliff" may be the point at which a slope that was already present becomes visible against the noise. This study cannot rule out steeper degradation above 3,000 tokens; it can say that below that point the degradation is already present and already continuous.

## 5.4 Why Small Windows Win

The window-size effect was not a hypothesis. It was included as a design factor because H1 required window size to be controlled, and it turned out to be the largest effect in the study — larger than any of the three hypothesised factors, significant at p < 10⁻²⁴, and replicated in direction in all five topics.

The chunk statistics make the finding legible. At 128 words, the retriever's unit is a third of an abstract — roughly the Background, the Methods-and-Results, or the Conclusion of a structured abstract. At 256 words, the unit is most of an abstract. At 512 words, it is the whole abstract, every time. The three window sizes are three granularities of retrieval unit, and the results say that the finest granularity is best on every quality metric while being worst on answer rate.

Two mechanisms combine. The first is retrieval precision: a 128-word chunk that matches the query is a 128-word chunk *about the query*, whereas a 512-word chunk that matches may match on one sentence and carry four hundred words of something else. The second is context economy: at fixed k, three 128-word chunks deliver about 400 words and three 512-word chunks deliver about 700, and Section 5.3 has established that fewer words is better. Both mechanisms point the same way, and the design cannot separate them; a follow-up that fixes context volume rather than k would.

The finding reframes the three hypotheses. H1 asked about overlap, H2 about segmentation method, and H3 about context volume, and the answers were "no effect," "no effect," and "continuous decline." Window size is the factor that determines granularity, and granularity determines context volume, and context volume is what drives quality. Overlap and segmentation method do not change granularity on this corpus — they produce whole-abstract units either way — and so they do not change the outcome. The one factor that does change granularity is the one that mattered. Seen this way, the four results are one result: on short biomedical abstracts, retrieve sub-document units and as few of them as possible.

## 5.5 Relation to Prior Work

The general-domain benchmarks anticipated most of what was found. Śmigielski et al. [3] identified fixed windows as the most robust method across eight datasets; this study finds the same on a ninth. Qu et al. [5] predicted that semantic chunking's advantage would vanish on short documents; it vanished. The Natural Questions evidence that overlap adds little transferred intact.

The domain-specific priors did not. The concern that punkt over-segments biomedical text produced no measurable effect, for the reasons given in Section 5.2.1. The expectation that a biomedical encoder would retrieve biomedical passages better was reversed: PubMedBERT retrieved the annotated source less often than MiniLM (22.5% against 29.3%, p < 10⁻¹⁶) and produced no better answers. Several explanations are possible and the design cannot choose between them. The NeuML embeddings are fine-tuned for retrieval on biomedical query–passage pairs whose queries may resemble literature search strings more than the clinical questions used here; MiniLM's training on over a billion general pairs may make it better at matching a short natural-language question to a passage regardless of domain; or the questions may simply not exercise the terminology on which a domain encoder would gain. What can be said is that the assumption "biomedical text needs a biomedical encoder" did not hold on this task, and that the absence of any chunking × encoder interaction (p = 0.99) means the chunking findings do not depend on the choice.

The one prior clinical study points the other way and deserves direct treatment. Gomez-Cabello et al. [18] report 87% answer accuracy for an adaptive entity-density chunker against 13% for a fixed baseline on physician-validated questions. This study finds fixed windows performing well and no advantage for any content-aware strategy. Four differences between the studies could account for the divergence. The corpora differ: clinical decision-support documents are longer and less uniformly structured than PubMed abstracts, and Section 5.4 has argued that the advantage of any content-aware method depends on there being internal structure to find. The adaptive method differs: entity-density segmentation is not among the strategies tested here, and it may succeed where similarity-based boundaries do not. The questions differ: physician-validated clinical questions may demand multi-sentence evidence assembly that keyword-matched factual questions do not. And the metric differs: human-judged accuracy against automatic ROUGE-L. The honest conclusion is that both studies may be correct about the text and task they examined, and that the transfer of chunking findings across biomedical sub-domains is itself an open question — one on which this study's result is that abstracts behave like short general-domain documents, not like clinical notes.

## 5.6 Threats to Validity

### 5.6.1 Internal validity

**Serving non-determinism.** The generator was run at temperature zero, which should make decoding deterministic, but the remote inference service did not return identical outputs for identical inputs: 41% of paired runs with identical context produced different answer text (Section 4.3.1). The effect on the analysis is additive noise with a measured floor of about 0.013 ROUGE-L at the condition-mean level. It does not bias any comparison — it affects every condition equally — but it means that condition differences below roughly 0.015 are uninterpretable, and all null results in this thesis should be read with that floor in mind. The window-size and strategy-versus-anchor effects (0.027–0.047) are well above it; the overlap and strategy-versus-strategy differences (under 0.006) are well below it. A replication on locally hosted weights with fixed seeds would remove the issue.

**Single gold source.** Each question is annotated with one gold PMID, assigned by keyword matching. Section 4.10 showed the consequence: 62% of the fifty best answers were produced without retrieving the annotated source, and answered runs that missed the gold source scored *higher* than those that found it. The retrieval-recall metric is measuring agreement with one annotation on a corpus where many abstracts state the same fact. The hypothesis tests do not depend on recall — they use ROUGE-L — but every statement in this thesis about retrieval success should be read as a statement about exact-source agreement. A pooled-relevance annotation, in which every abstract that answers a question is marked relevant, would repair the metric; Chapter 6 describes it.

**Confounding of k with context volume.** Strategies were compared at fixed k, and different strategies deliver different volumes at the same k. The ANCOVA controls for this statistically, and the conclusion survived, but a design that fixed context volume directly and let k vary would be cleaner.

### 5.6.2 Construct validity

**ROUGE-L measures surface agreement.** Two correct answers phrased differently score differently; a fluent wrong answer that reuses the reference's vocabulary can score well. ROUGE-L is the standard reference-based metric and is appropriate for a comparison of this size, but the absolute values in this thesis — a best condition mean of 0.126 — should not be read as accuracy percentages, and the findings are findings about relative performance across conditions, where the metric's biases apply equally to all.

**Faithfulness is lexical.** The token-overlap measure detects answers that introduce vocabulary absent from the context; it does not detect answers that recombine context vocabulary into unsupported claims. A claim-level judge would be stronger.

**Refusal detection is by phrase.** A run is classed as a refusal if the answer contains "insufficient context." An answer that declined in other words would be classed as an answer; inspection of samples found no such cases, but the classification was not exhaustive.

### 5.6.3 External validity

**One generator.** Every finding about how the generator responds to context is a finding about Llama 3.1 8B Instruct. The willingness–accuracy trade-off in particular may be specific to how this model handles a "refuse if insufficient" instruction. Replication with a second generator is the most valuable single extension of the work.

**Abstracts, not full text.** The corpus is short by construction, and Section 5.4 has argued that the shortness is what made window size dominant and made the segmentation strategies indistinguishable. On full-text articles, where abstracts' 238 words become thousands, the strategies would produce genuinely different segmentations and the results could differ. This thesis makes claims about PubMed abstracts, which are what most biomedical retrieval systems index, and does not extend them to full text.

**Five topics, one hundred questions, automated annotation.** The question set is small by benchmark standards and its gold sources were assigned automatically rather than by domain experts. The per-topic analysis shows that the main effect holds in every topic, which addresses generalisation across disease areas within the corpus; it does not address generalisation to question types — multi-hop, comparative, numerical — that the set does not contain.

**No human evaluation.** Twelve thousand runs are beyond human evaluation, and automatic metrics were the only feasible choice. A human check on a sample — correctness, relevance and evidence support judged by a clinician — would establish how well ROUGE-L tracks clinical correctness on this material and is a natural next step.

## 5.7 Recommendations for Practice

For a practitioner building a RAG system over PubMed abstracts or similar short biomedical documents under resource constraints, the results support the following.

1. **Use small fixed windows.** A 128-word window outperformed every alternative tested on every quality metric. It is also the cheapest strategy to implement and the fastest to run.
2. **Omit overlap.** It had no measurable effect on any outcome at any window size and inflates index size and context volume. Set it to zero.
3. **Retrieve few chunks.** The best quality in the design was at k = 3, the shallowest depth tested. Every increase in k raised answer rate and lowered answer quality. Choose the smallest k that yields an acceptable answer rate for the use case, and treat any increase as a trade of accuracy for coverage.
4. **Do not assume that domain-specific components help.** The biomedical sentence tokeniser produced the same chunks as the general one; the biomedical encoder retrieved worse and answered no better. Test before adopting.
5. **Do not pay for semantic chunking on short documents.** It requires an embedding pass at index time and offered nothing over sentence splitting, which offered nothing over fixed windows.
6. **Report answer rate and answer quality separately.** A system's average score conflates how often it answers with how well; the two move in opposite directions as context grows; and in a clinical setting the difference between a refusal and a partial answer matters. Any evaluation that reports one number is hiding which of the two it is measuring.
7. **Treat retrieval as the bottleneck.** Sixty-three per cent of runs were refusals, and refusals tracked retrieval misses. Improvements to retrieval — better questions-to-corpus matching, hybrid lexical search, re-ranking — would raise the answer rate without the quality cost that more context imposes.

Every one of these recommendations rests on 800 measured runs per condition, and each is bounded by the limitations of Section 5.6. The first five are recommendations about what to omit rather than what to add, which is the shape a negative result takes when it is useful: the practitioner saves the cost of a component that would not have helped.
