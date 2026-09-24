# 4 Results

This chapter reports the outcome of the 12,000 runs. Section 4.1 gives the overall picture and the per-condition results table. Section 4.2 examines the refusal outcome, which shapes the interpretation of everything after it. Sections 4.3 to 4.5 test the three hypotheses. Sections 4.6 to 4.9 report the secondary analyses — window size, embedding model and interaction, retrieval depth, and per-topic robustness. Section 4.10 quantifies the divergence between retrieval recall and answer quality, and Section 4.11 summarises the findings. Every number in this chapter is drawn from the statistics file generated from the results by the released analysis script; the complete per-condition table and the full statistical output are reproduced in Appendices A and B.

## 4.1 Overall Results

All 12,000 runs completed: 15 conditions × 100 questions × 4 retrieval depths × 2 embedding models, with exactly 800 runs per condition. Across the full set, the mean ROUGE-L was 0.098, mean answer relevance 0.387, mean faithfulness 0.465 and mean retrieval recall 0.259. The mean context delivered to the generator was 1,131 words (approximately 1,500 model tokens); the median generation latency was 585 ms.

Those aggregate figures are low, and the reason is the subject of Section 4.2: 7,510 of the 12,000 runs (62.6%) ended with the generator declining to answer. Restricted to the 4,490 runs that produced a substantive answer, mean ROUGE-L was 0.198 — twice the unconditional figure.

Table 4.1 reports every condition. Conditions are listed alphabetically; the best and worst are identified below the table.

{{table:per_condition|Results for all fifteen conditions (800 runs each). "ROUGE-L (all)" is the unconditional mean; "ROUGE-L (answered)" is the mean over runs that produced a substantive answer.}}

The best condition by unconditional ROUGE-L was **Fixed 128 / 0%** at 0.126; the worst was **Fixed 512 / 20%** at 0.074. The full ordering places all four 128-word conditions and two of the 256-word conditions in the top six, all four 512-word conditions in the bottom four, and the three segmentation strategies between the 256- and 512-word groups. That ordering is the descriptive form of the two main findings — window size dominates, and the segmentation strategies do not separate from each other — which the hypothesis tests below examine formally.

## 4.2 The Refusal Outcome

### 4.2.1 Prevalence

The generator answered "Insufficient context" — either as the bare phrase or followed by an explanation of what the context lacked — in 7,510 runs. Of these, 5,948 were the bare phrase and 1,562 were explained refusals. Every refusal is a genuine generation: all 7,510 carry a non-zero latency, none is an API failure, and the explained refusals describe the retrieved context accurately (examples in Appendix F).

A refusal rate of 62.6% is high, and it is a finding about the retrieval, not the generator. Table 4.2 compares the runs that refused with those that answered.

{{table:refused_vs_answered|Mean metrics for answered and refused runs.}}

Refused runs retrieved the gold source less than half as often as answered runs (recall 0.173 against 0.404) while receiving nearly as much context (1,096 against 1,190 words). The generator declined because the context it was shown did not contain the answer — which is what the prompt instructed it to do — and it did so in proportion to how often retrieval missed. Faithfulness among answered runs was 0.869, meaning that when the generator did answer, 87% of the distinct tokens in its answer could be located in the retrieved text.

### 4.2.2 Structure of refusals across factors

Figure 4.1 shows how refusal rate varies with the three design factors.

![Refusal rate across window size, retrieval depth and embedding model.](fig_refusal_structure.png)

Refusal decreases as the window grows (64.9% at 128 words, 61.4% at 256, 61.3% at 512) and as retrieval depth grows (67.6% at k = 3, 63.4% at k = 5, 59.1% at k = 8, 60.2% at k = 10). Both gradients run in the expected direction: more retrieved text means more chance that the answer is somewhere in it. The embedding models differ by more than either of the other factors — MiniLM refused 59.9% of the time and PubMedBERT 65.3% — which is examined in Section 4.7.

Refusal also varies substantially by topic (Table 4.3): from 51.2% for Alzheimer's disease to 70.8% for type 2 diabetes. Topic is not a manipulated factor, but the variation shows that some question sets were harder to ground in this corpus than others, and Section 4.9 checks whether the main effects hold within each.

| Topic | Refusal rate |
|---|---|
| Alzheimer's disease | 51.2% |
| Lung cancer | 58.0% |
| COVID-19 | 65.0% |
| Hypertension | 67.9% |
| Type 2 diabetes | 70.8% |
Table: Refusal rate by topic.

### 4.2.3 Consequence for the analysis

Because refusals score near zero on every quality metric, a condition's unconditional mean is dominated by its refusal rate. Two conditions with identical answer quality but different refusal rates will show different means; two with the same mean may differ in opposite directions on rate and quality. All results below therefore report the unconditional mean (which is what a deployed system's user experiences), the answer rate, and the conditional mean over answered runs, and note where the three disagree.

## 4.3 H1: Overlap Adds No Benefit

**H1** predicts that adding overlap between adjacent fixed-window chunks does not improve retrieval or answer quality once window size is controlled.

### 4.3.1 Descriptive results

Table 4.4 reports the 9,600 Part A runs grouped by overlap level, pooling across window sizes.

{{table:h1_overlap|Part A results by overlap ratio, pooled over window sizes (2,400 runs per level).}}

Unconditional ROUGE-L is flat: 0.106, 0.103, 0.102, 0.101 for 0%, 10%, 20% and 40% overlap. Conditional ROUGE-L is flatter still: 0.201, 0.202, 0.200, 0.206. Answer rate moves by two percentage points in no consistent direction. Retrieval recall — the metric that overlap is theoretically supposed to improve, by giving the retriever two chances at boundary-spanning evidence — is 0.261, 0.257, 0.267, 0.260: no trend. Context volume rises modestly with overlap (1,084 to 1,166 words), as it must, since overlapping chunks are larger; the generator received about 8% more text at 40% overlap for no change in any outcome.

Figure 4.2 plots ROUGE-L and answer rate against overlap separately for each window size.

![Overlap ratio against ROUGE-L (left) and answer rate (right), by window size. No consistent slope at any size.](fig1_overlap.png)

Within each window size the lines are essentially horizontal. The 512-word conditions deserve a separate comment. As Section 3.4 showed, all four are the same segmentation for every abstract shorter than 512 words — which is nearly all of them — so their retrieved contexts are identical for most runs, and their ROUGE-L means (0.087, 0.084, 0.082, 0.074) have no relationship to overlap. Their spread is instead an empirical measurement of run-to-run noise. Pairing the 512 / 0% and 512 / 10% conditions run by run, 738 of 800 pairs received context of identical size; of those, 306 (41%) nonetheless produced different answer text, with a mean absolute ROUGE-L difference of 0.047 per pair. The generator was run at temperature zero, but the remote inference service did not return bit-identical outputs for identical inputs. The consequence for the analysis is a known noise floor: the range across four near-replicate conditions is 0.013 ROUGE-L, so differences between conditions smaller than roughly 0.01–0.015 cannot be distinguished from serving non-determinism, whereas the window-size differences of Section 4.6 (0.027–0.036) lie well outside it. Section 5.6 returns to this as a threat to validity.

### 4.3.2 Inferential test

A two-way ANOVA on ROUGE-L with window size and overlap as factors gives the result in Table 4.5.

| Source | Sum of squares | df | F | p | Partial η² |
|---|---|---|---|---|---|
| Window size | 2.297 | 2 | 57.72 | 1.2 × 10⁻²⁵ | 0.0119 |
| Overlap | 0.035 | 3 | 0.587 | 0.623 | 0.0002 |
| Residual | 190.857 | 9,594 | | | |
Table: Two-way ANOVA on ROUGE-L for the Part A runs.

The main effect of overlap is not significant (F(3, 9594) = 0.587, p = 0.623). Its partial η² of 0.0002 means that overlap accounts for two hundredths of one per cent of the variance in ROUGE-L. Tukey HSD contrasts between every pair of overlap levels are reported in Table 4.6.

{{table:h1_tukey|Tukey HSD contrasts between overlap levels on ROUGE-L (FWER = 0.05).}}

No contrast is significant; the smallest adjusted p is 0.617. Every 95% confidence interval contains zero, and the widest interval spans −0.016 to +0.006: the data are consistent with a true effect of overlap anywhere in that range, and inconsistent with any effect larger than about 0.016 in either direction. Welch's t-test comparing 0% directly with 40% gives t = 1.21, p = 0.226.

### 4.3.3 Verdict

**H1 is supported.** Overlap has no detectable effect on answer quality, answer rate or retrieval recall on this corpus, at any window size, and the confidence intervals bound any undetected effect below 0.016 ROUGE-L. The cost of overlap — 8% more context at 40% and a proportionally larger index — buys nothing.

## 4.4 H2: Sentence Segmentation Matches Semantic Chunking

**H2** predicts that NLTK sentence segmentation, scispaCy biomedical sentence segmentation and embedding-similarity semantic chunking do not differ in answer quality.

### 4.4.1 Descriptive results

Table 4.7 reports the three strategies alongside the Part A anchor, Fixed 128 / 0%.

{{table:h2_table|Part B results: the three segmentation strategies and the Part A anchor (800 runs each).}}

The three strategies are within 0.005 ROUGE-L of each other unconditionally (0.079, 0.081, 0.084) and within 0.006 conditionally (0.181, 0.187, 0.184). Their answer rates (37.4%, 36.6%, 38.0%), recall (0.250, 0.253, 0.254) and faithfulness (0.452, 0.443, 0.457) are likewise indistinguishable. They deliver similar context volumes — 1,163 words for both sentence strategies and 1,256 for semantic — which are all far larger than the anchor's 677.

Figure 4.3 shows the four conditions on all four quality metrics.

![The three segmentation strategies against the best fixed-window condition on four metrics.](fig3_strategies.png)

### 4.4.2 Inferential tests

A one-way ANOVA across the three strategies gives F(2, 2397) = 0.381, p = 0.683. Because the semantic condition delivers about 8% more context than the sentence conditions, and because context volume is itself associated with outcome (Section 4.5), the comparison was repeated as an ANCOVA with context words as covariate: the strategy effect remains non-significant (p = 0.828), while the covariate is strongly significant (F = 16.7, p = 4.5 × 10⁻⁵). Controlling for how much text each strategy delivers does not reveal a hidden difference between them; it confirms that the volume of context, not the method of segmentation, is what is associated with variation in the outcome.

Tukey contrasts including the anchor are in Table 4.8.

{{table:h2_tukey|Tukey HSD contrasts among the three strategies and the anchor (FWER = 0.05).}}

The three strategy-versus-strategy contrasts have adjusted p-values of 0.976, 0.858 and 0.982. The NLTK-versus-scispaCy contrast — the specific comparison that the biomedical tokeniser was included to test — is the least different of the three, at a mean difference of 0.0024. All three strategy-versus-anchor contrasts are significant at p < 0.001, with the anchor ahead by 0.042–0.047; a four-group ANOVA including the anchor gives F(3, 3196) = 25.7, p = 1.9 × 10⁻¹⁶.

### 4.4.3 Verdict

**H2 is supported.** The three segmentation strategies are statistically indistinguishable on every metric, with and without controlling for context volume, and the biomedical tokeniser offers no advantage over the general-purpose one. The comparison also yields a result the hypothesis did not predict: all three strategies are significantly worse than the best fixed window. Chapter 5 explains both findings from the chunk statistics in Section 3.4.

## 4.5 H3: Answer Quality Declines with Context Size

**H3** predicts that answer quality degrades as the context delivered to the generator grows. The original formulation in the literature describes a threshold — a cliff — at roughly 2,500 tokens.

### 4.5.1 Binned results

Table 4.9 groups all 12,000 runs into seven bands of approximate context size (words ÷ 0.75).

{{table:h3_bins|All runs grouped by approximate context size in model tokens.}}

Unconditional ROUGE-L falls from 0.161 in the smallest band to 0.067 in the largest — a 58% decline. The decline is present in the conditional metric as well: among answered runs, ROUGE-L falls from 0.253 to 0.170. Answer rate moves in the opposite direction, rising from 29.8% in the smallest band to 43.8% at 2.5k–3k tokens before falling back to 38.7% above 3k. Retrieval recall rises with context throughout, from 0.148 to 0.328, as it must — more retrieved chunks means more chance that one of them is the gold source.

Figure 4.4 plots ROUGE-L and answer rate against context band.

![Mean ROUGE-L (with standard error) and answer rate by context-size band. Quality declines from the smallest band; answer rate rises.](fig4_context_cliff.png)

### 4.5.2 Shape of the relationship

The decline is monotonic in the unconditional metric across the first three bands (0.161 → 0.119 → 0.088), essentially flat across the next three (0.089, 0.078, 0.080) and falls again in the last (0.067). It is monotonic without exception in the conditional metric (0.253 → 0.214 → 0.208 → 0.188 → 0.179 → 0.176 → 0.170). The steepest drop in both is between the first two bands — below 1,000 tokens. By 2,500 tokens, where the hypothesised cliff sits, the unconditional curve has already flattened and the conditional curve is declining gently. There is no band at which quality falls sharply after having been stable.

The Pearson correlation between context tokens and ROUGE-L over all runs is r = −0.150 (p = 4.5 × 10⁻⁶¹); the Spearman rank correlation, which does not assume linearity, is ρ = −0.198 (p = 4.7 × 10⁻¹⁰⁶). Restricted to answered runs, r = −0.134 (p = 1.5 × 10⁻¹⁹). The correlation between context tokens and the binary answer indicator is positive, r = +0.076 (p = 1.2 × 10⁻¹⁶). The relationship is therefore not an artefact of refusals: context reduces the quality of the answers the generator gives even as it increases the generator's willingness to give one.

### 4.5.3 Verdict

**H3 is supported in direction and refined in form.** Answer quality is negatively associated with context size, decisively so. But the shape is a continuous decline from the smallest contexts tested, steepest below 1,000 tokens and flattening thereafter, not a threshold. On this corpus there is no context budget below which quality is safe and above which it collapses; every increase in context is associated with some loss of quality. The practical implication — retrieve less rather than more — holds across the full range tested.

The opposite movement of answer rate and conditional quality is a finding in its own right. A larger context makes the generator more likely to attempt an answer and less likely to get it right. A single averaged metric conceals this: between the 500–1k and 2.5k–3k bands, unconditional ROUGE-L falls by 0.039 while answer rate rises by ten percentage points and conditional quality falls by 0.038, and only the decomposition shows that the fall in the average is entirely a fall in quality that the rise in rate partially masks.

## 4.6 Window Size

Window size was the largest effect in the experiment. Table 4.10 pools the Part A runs by window size.

{{table:chunk_size|Part A results by window size, pooled over overlap ratios (3,200 runs per size).}}

Unconditional ROUGE-L falls from 0.118 at 128 words to 0.109 at 256 and 0.082 at 512; conditional ROUGE-L from 0.224 to 0.209 to 0.175. Answer rate moves the other way, from 35.1% to 38.6% to 38.7%. The same willingness–accuracy pattern seen for context size in Section 4.5 appears here, which is unsurprising: window size is the main determinant of context volume at a fixed k (706, 1,165 and 1,475 words for the three sizes at the pooled k).

The two-way ANOVA in Table 4.5 gave a main effect of window size of F(2, 9594) = 57.7, p = 1.2 × 10⁻²⁵, partial η² = 0.0119. Tukey contrasts (Table 4.11) show every pair of sizes significantly different.

{{table:size_tukey|Tukey HSD contrasts between window sizes on ROUGE-L (FWER = 0.05).}}

The 128-versus-512 difference of 0.036 is the largest between-level difference for any factor in the study. Figure 4.5 plots the three metrics against window size.

![ROUGE-L (unconditional and conditional) and answer rate against window size.](fig2_chunk_size.png)

The η² of 0.012 deserves a comment. The window-size effect is highly significant and is the largest in the design, yet it explains only 1.2% of the total variance in ROUGE-L. The remaining 98.8% is variation between questions, between topics and between individual generations — a characteristic of RAG evaluation in which the difficulty of the question dominates the configuration of the system. The effect is real, consistent and replicable (Section 4.9 shows it in every topic), but it is a shift of a few hundredths in a metric whose per-run standard deviation is 0.135.

## 4.7 Embedding Model and Interaction

### 4.7.1 Main effect

Table 4.12 compares the two embedding models across all 6,000 runs each.

{{table:by_embedding|Results by embedding model, pooled over all conditions and depths (6,000 runs each).}}

MiniLM, the general-purpose encoder, scored higher than PubMedBERT on unconditional ROUGE-L (0.101 against 0.096), answer rate (40.1% against 34.7%), retrieval recall (0.293 against 0.225), faithfulness and relevance. PubMedBERT scored higher only on conditional ROUGE-L (0.203 against 0.194) — when it answered, it answered slightly better, but it answered less often because it retrieved the gold source less often.

Welch's t-test on unconditional ROUGE-L gives t = 1.81, p = 0.071: not significant at α = 0.05. On retrieval recall the difference is significant (t = 8.52, p = 1.7 × 10⁻¹⁷): MiniLM retrieved the annotated source in 29.3% of runs and PubMedBERT in 22.5%. The biomedical encoder was worse at finding the annotated source and no better at producing answers.

### 4.7.2 Interaction with chunking condition

A two-way ANOVA over all fifteen conditions and both encoders with an interaction term gives the result in Table 4.13.

| Source | df | F | p |
|---|---|---|---|
| Chunking condition | 14 | 13.81 | 2.4 × 10⁻³³ |
| Embedding model | 1 | 3.31 | 0.069 |
| Condition × embedding | 14 | 0.31 | 0.993 |
| Residual | 11,970 | | |
Table: Two-way ANOVA with interaction on ROUGE-L over all 12,000 runs.

The interaction is as close to zero as a test of this size can show: F(14, 11970) = 0.31, p = 0.993. Table 4.14 gives the per-condition difference between encoders.

{{table:embedding_by_condition|Mean ROUGE-L by condition and embedding model, with the MiniLM − PubMedBERT difference.}}

The difference ranges from −0.004 to +0.012 across fifteen conditions with no pattern by window size or strategy. Figure 4.6 shows the same data as paired bars.

![Mean ROUGE-L by condition for the two embedding models. The gap is small and constant.](fig5_embeddings.png)

The absence of interaction means that the ranking of chunking conditions is the same under both encoders. A practitioner who follows the chunking recommendations of this thesis need not condition them on which embedding model they use.

## 4.8 Retrieval Depth

Table 4.15 pools all runs by k.

{{table:by_k|Results by retrieval depth k, pooled over all conditions and encoders (3,000 runs each).}}

Retrieval depth shows the willingness–accuracy pattern most cleanly of any factor. As k rises from 3 to 10: retrieval recall rises from 0.188 to 0.317; answer rate rises from 32.4% to 39.8%; unconditional ROUGE-L falls from 0.132 to 0.079; conditional ROUGE-L falls from 0.212 to 0.188; context rises from 517 to 1,746 words; and latency rises from 818 to 1,178 ms. Retrieving more finds the gold source more often, makes the generator answer more often, makes each answer worse, and takes longer. Figure 4.7 plots ROUGE-L and answer rate against k for the four Part B conditions.

![ROUGE-L (left) and answer rate (right) against retrieval depth for the three segmentation strategies and the anchor.](fig6_k_effect.png)

The best unconditional ROUGE-L in the entire design is obtained at k = 3, the shallowest depth tested. Whether an even shallower depth would be better still is a question the design cannot answer.

## 4.9 Per-Topic Robustness

Topic was not a manipulated factor, but the five question sets differ substantially in difficulty, and a main effect that held only in some topics would be a weaker finding than one that held in all. Table 4.16 reports results by topic.

{{table:by_topic|Results by topic, pooled over all conditions, encoders and depths (2,400 runs each).}}

Topics differ strongly: a one-way ANOVA on ROUGE-L across topics gives F(4, 11995) = 38.6, p = 4.1 × 10⁻³². Alzheimer's disease questions were answered most often (48.8%) and diabetes questions least (29.2%); hypertension questions, when answered, were answered best (conditional ROUGE-L 0.232) and diabetes questions worst (0.151). Figure 4.8 shows the per-topic results.

![Per-topic ROUGE-L (unconditional and conditional) and answer rate.](fig_per_topic.png)

Table 4.17 tests whether the window-size effect — the largest in the study — replicates within each topic.

{{table:size_by_topic|Mean ROUGE-L by window size within each topic.}}

In four of five topics the ordering 128 > 256 > 512 holds exactly. In COVID-19 the 256-word window edges the 128-word window by 0.002, with 512 well below both. The direction of the effect — smaller windows better — holds in every topic, and the 512-word window is last in every topic. Figure 4.9 plots the five lines.

![The window-size effect within each topic. Smaller windows are better in all five; 512 words is worst in all five.](fig_size_by_topic.png)

## 4.10 Retrieval Recall and Answer Quality Diverge

Retrieval recall — whether the annotated gold PMID was among the retrieved sources — is only weakly related to answer quality. Across all runs, the correlation between recall and ROUGE-L is r = 0.080. Of the fifty highest-scoring answers in the experiment, 31 (62%) were produced without the gold source having been retrieved. Among answered runs, those in which the gold source was *not* retrieved scored higher (mean ROUGE-L 0.213, n = 2,678) than those in which it was (0.176, n = 1,812). Figure 4.10 shows the two distributions.

![ROUGE-L of answered runs, split by whether the annotated gold source was retrieved. Runs that did not retrieve the gold source score higher on average.](fig_recall_vs_rouge.png)

This is not a paradox once the annotation is recalled. Each question has one gold PMID assigned by keyword matching, but the corpus has roughly two hundred abstracts per topic and many of them state the same clinical facts. The retriever frequently returns an abstract that answers the question and is not the annotated one; the generator answers correctly from it; and recall scores the run zero. The recall metric measures agreement with a single annotated source, not whether retrieval succeeded, and on a topically dense corpus the two come apart. Recall is reported throughout this chapter, but it is treated as a diagnostic of exact-source agreement rather than as a measure of retrieval success, and Chapter 5 discusses the annotation design that would repair it.

## 4.11 Summary of Findings

| Hypothesis or question | Result | Key statistic | Verdict |
|---|---|---|---|
| H1: overlap adds no benefit | No effect at any window size | F = 0.59, p = 0.623; all Tukey CIs include zero | Supported |
| H2: sentence matches semantic | Three strategies indistinguishable; survives ANCOVA | F = 0.38, p = 0.683; ANCOVA p = 0.828 | Supported |
| H3: quality declines with context | Continuous decline from smallest contexts; no threshold | r = −0.150, p < 10⁻⁶⁰ | Supported in direction; refined in form |
| Window size | 128 > 256 > 512, replicated in every topic | F = 57.7, p < 10⁻²⁴; η² = 0.012 | Dominant factor |
| Embedding model | Biomedical encoder not better; worse at recall | ROUGE-L p = 0.071; recall p < 10⁻¹⁶ | No advantage |
| Chunking × embedding | No interaction | F = 0.31, p = 0.993 | Recommendations are encoder-independent |
| Retrieval depth | More depth: more recall, more answers, worse answers | Best ROUGE-L at k = 3 | Retrieve less |
| Refusal | 62.6% of runs; tracks retrieval miss | Recall 0.17 (refused) vs 0.40 (answered) | Retrieval is the bottleneck |
Table: Summary of results.

Three findings organise the rest of the thesis. The configuration choices that practitioners spend effort on — overlap, biomedical tokenisers, semantic boundaries, domain encoders — made no detectable difference. The one choice that did, window size, favoured the smallest option and the least context. And the effect that both of those point to — less text to the generator is better — operates continuously, so that there is no safe amount of extra context to add.
