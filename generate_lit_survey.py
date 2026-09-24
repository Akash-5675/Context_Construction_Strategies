"""
Generate literature survey PDF for:
"Context Construction Strategies in Medical RAG: A Systematic Review"
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Literature_Survey_Context_Construction_Medical_RAG.pdf")

# ── Page geometry ──────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4  # 595 × 842 pt
LEFT_MARGIN  = 2.5 * cm
RIGHT_MARGIN = 2.5 * cm
TOP_MARGIN   = 2.5 * cm
BOT_MARGIN   = 2.5 * cm


# ── Header / Footer ────────────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for i, state in enumerate(self._saved_page_states):
            self.__dict__.update(state)
            self._draw_header_footer(i + 1, num_pages)
            super().showPage()
        super().save()

    def _draw_header_footer(self, page_num, total_pages):
        self.saveState()
        w, h = A4
        # header line
        self.setStrokeColor(colors.HexColor("#2c3e50"))
        self.setLineWidth(0.8)
        self.line(LEFT_MARGIN, h - TOP_MARGIN + 6 * mm, w - RIGHT_MARGIN, h - TOP_MARGIN + 6 * mm)
        self.setFont("Times-Italic", 8)
        self.setFillColor(colors.HexColor("#555555"))
        if page_num > 1:
            self.drawString(LEFT_MARGIN, h - TOP_MARGIN + 8 * mm,
                            "Context Construction Strategies in Medical RAG: A Systematic Review")
            self.drawRightString(w - RIGHT_MARGIN, h - TOP_MARGIN + 8 * mm,
                                 "Akash Ramesh  ·  AM.SC.P2AML25002")
        # footer line
        self.line(LEFT_MARGIN, BOT_MARGIN - 4 * mm, w - RIGHT_MARGIN, BOT_MARGIN - 4 * mm)
        self.setFont("Times-Roman", 8)
        self.drawCentredString(w / 2, BOT_MARGIN - 7 * mm,
                               f"Page {page_num} of {total_pages}")
        self.restoreState()


# ── Styles ─────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def make_style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

TITLE   = make_style("DocTitle",   fontName="Times-Bold",   fontSize=16,
                     leading=20, alignment=TA_CENTER, spaceAfter=4)
AUTHORS = make_style("Authors",    fontName="Times-Roman",  fontSize=11,
                     leading=14, alignment=TA_CENTER, spaceAfter=2)
AFFIL   = make_style("Affil",      fontName="Times-Italic", fontSize=10,
                     leading=13, alignment=TA_CENTER, spaceAfter=14)
ABST_HDR= make_style("AbstHdr",   fontName="Times-Bold",   fontSize=10,
                     leading=13, alignment=TA_CENTER, spaceAfter=3)
ABSTRACT= make_style("Abstract",  fontName="Times-Roman",  fontSize=9.5,
                     leading=13, alignment=TA_JUSTIFY, leftIndent=1.2*cm,
                     rightIndent=1.2*cm, spaceAfter=14)
KW_STYLE= make_style("Keywords",  fontName="Times-Roman",  fontSize=9,
                     leading=12, alignment=TA_JUSTIFY, leftIndent=1.2*cm,
                     rightIndent=1.2*cm, spaceAfter=14)
H1      = make_style("H1",         fontName="Times-Bold",   fontSize=11,
                     leading=14, spaceBefore=10, spaceAfter=4,
                     textTransform="uppercase")
H2      = make_style("H2",         fontName="Times-BoldItalic", fontSize=10.5,
                     leading=13, spaceBefore=7, spaceAfter=3)
H3      = make_style("H3",         fontName="Times-Bold",   fontSize=10,
                     leading=13, spaceBefore=5, spaceAfter=2)
BODY    = make_style("Body",       fontName="Times-Roman",  fontSize=10,
                     leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)
CITE    = make_style("Cite",       fontName="Times-Roman",  fontSize=9,
                     leading=13, alignment=TA_JUSTIFY, spaceAfter=4,
                     leftIndent=0.8*cm, firstLineIndent=-0.8*cm)
CAPTION = make_style("Caption",    fontName="Times-Italic", fontSize=9,
                     leading=12, alignment=TA_CENTER, spaceAfter=8)
TH      = make_style("TH",         fontName="Times-Bold",   fontSize=8.5,
                     leading=11, alignment=TA_CENTER)
TD      = make_style("TD",         fontName="Times-Roman",  fontSize=8.5,
                     leading=11, alignment=TA_LEFT)
TD_C    = make_style("TDC",        fontName="Times-Roman",  fontSize=8.5,
                     leading=11, alignment=TA_CENTER)


def p(text, style=BODY):
    return Paragraph(text, style)

def sp(h=4):
    return Spacer(1, h)

def rule():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa"), spaceAfter=4)


# ── Table helper ───────────────────────────────────────────────────────────────
def make_table(header_row, data_rows, col_widths, caption=None):
    USABLE = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN
    if isinstance(col_widths[0], float) and col_widths[0] < 2:
        col_widths = [USABLE * w for w in col_widths]

    header = [p(h, TH) for h in header_row]
    rows   = [header] + [[p(cell if isinstance(cell, str) else str(cell),
                            TD_C if col_widths[ci] < 3 * cm else TD)
                          for ci, cell in enumerate(row)]
                         for row in data_rows]

    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ])
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(ts)
    out = [t]
    if caption:
        out.append(sp(3))
        out.append(p(caption, CAPTION))
    return out


# ── Document content ───────────────────────────────────────────────────────────
def build_story():
    story = []

    # ── Title block ────────────────────────────────────────────────────────────
    story.append(sp(6))
    story.append(p("Context Construction Strategies in Medical RAG:<br/>"
                   "A Systematic Review of Chunking Methods for Biomedical"
                   " Retrieval-Augmented Generation", TITLE))
    story.append(sp(4))
    story.append(p("Akash Ramesh", AUTHORS))
    story.append(p("AM.SC.P2AML25002  ·  M.Tech Artificial Intelligence and Machine Learning",
                   AUTHORS))
    story.append(p("School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore, India",
                   AFFIL))
    story.append(rule())

    # ── Abstract ───────────────────────────────────────────────────────────────
    story.append(p("Abstract", ABST_HDR))
    story.append(p(
        "Retrieval-Augmented Generation (RAG) has become the dominant architecture for "
        "grounding large language model (LLM) outputs in verified external knowledge. "
        "Central to every RAG pipeline is the process of chunking — the division of source "
        "documents into retrievable units. While a substantial body of work has evaluated "
        "chunking in general-domain settings, the biomedical domain presents distinctive "
        "challenges: dense entity co-occurrence, abbreviation-rich clinical prose, and the "
        "structural peculiarities of PubMed abstracts. This paper reviews the state of the "
        "art in chunking strategies for RAG, with particular attention to the gap between "
        "general-domain findings and biomedical applicability. We survey fixed-size, "
        "sentence-boundary, semantic, hierarchical, query-adaptive, proposition-level, "
        "late-chunking, and contextual retrieval methods, drawing on ten recent studies "
        "spanning 2024–2026. We identify three underexplored research axes: the effect of "
        "token-overlap on cross-boundary evidence recovery in clinical text; the comparative "
        "adequacy of NLTK punkt versus biomedical sentence segmenters (scispaCy); and the "
        "location of the context-density cliff in PubMed prose. These axes map directly "
        "to an ongoing empirical study comparing sixteen chunking conditions on approximately "
        "one thousand PubMed abstracts across five MeSH-defined disease topics.",
        ABSTRACT))
    story.append(p(
        "<b>Keywords:</b> Retrieval-Augmented Generation, chunking strategies, biomedical NLP, "
        "PubMed, semantic chunking, fixed-token chunking, sentence segmentation, "
        "context window, FAISS, embedding models.", KW_STYLE))
    story.append(rule())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: INTRODUCTION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("1. Introduction", H1))
    story.append(p(
        "The integration of external knowledge retrieval into generative language systems "
        "— commonly described as Retrieval-Augmented Generation — was formalised by Lewis "
        "et al. (2020) in a seminal NeurIPS paper that combined a dense passage retriever "
        "with a seq2seq generator [1]. Since that foundational work, the RAG paradigm has "
        "expanded rapidly into high-stakes verticals including clinical decision support, "
        "drug information synthesis, and systematic literature review automation. The "
        "practical promise is clear: LLMs that hallucinate when operating from parametric "
        "memory alone can be constrained to verified source material if the retrieval "
        "component surfaces relevant, faithful context.", BODY))
    story.append(p(
        "However, retrieval quality is not merely a function of the embedding model or "
        "the similarity search algorithm. It is fundamentally determined by how the source "
        "corpus is partitioned into retrievable units before any query is ever posed. "
        "Chunking decisions govern the semantic coherence of individual units, the extent "
        "to which inter-sentence relationships are preserved or severed, and the total "
        "volume of text that ultimately enters the LLM context window. Too-small chunks "
        "risk fragmenting evidence spread across multiple sentences; too-large chunks "
        "dilute the retrieval signal and inflate context length. The optimal partition is "
        "almost certainly domain- and task-dependent, yet most deployed RAG systems default "
        "to fixed-size chunking because it is simple and deterministic [3].", BODY))
    story.append(p(
        "Biomedical text exacerbates every one of these challenges. A typical PubMed "
        "abstract describes a study population, an intervention, an outcome measure, and "
        "a statistical result — often within a single sentence that spans sixty words and "
        "four named entities. The sentence boundary tokeniser used in the majority of NLP "
        "pipelines (NLTK punkt) was trained on general news corpora; it misclassifies the "
        "period in 'i.v.', 'p.o.', 'Dr.', and similar biomedical abbreviations as sentence "
        "terminals, producing spurious short chunks that destroy clinical meaning [2]. "
        "Domain-adapted segmenters such as scispaCy address this limitation but have not "
        "been benchmarked head-to-head against their general-domain counterparts in a "
        "retrieval-generation evaluation framework.", BODY))
    story.append(p(
        "The motivation for the present survey arises from a specific research gap: "
        "as of early 2026, no published study has conducted a systematic, multi-metric "
        "comparison of sentence-boundary, semantic, and proposition-level chunking on "
        "PubMed biomedical abstracts. The survey in [3] states this explicitly — a "
        "comprehensive biomedical chunking evaluation remains absent from the literature. "
        "The sections below trace the arc from fixed-size baselines (Section 2) through "
        "the emerging paradigms of hierarchical, query-adaptive, and contextual chunking "
        "(Sections 3–6), situate the present work within that landscape (Section 7), and "
        "present a comparative synthesis (Section 8).", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: FIXED-SIZE AND SENTENCE-BOUNDARY CHUNKING
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("2. Fixed-Size and Sentence-Boundary Chunking", H1))
    story.append(p("2.1  Fixed-Token Chunking as the Baseline", H2))
    story.append(p(
        "Fixed-size chunking partitions a document into non-overlapping (or partially "
        "overlapping) windows of a specified token count. Its appeal is computational: "
        "it requires no linguistic analysis, processes any language identically, and "
        "completes in under one second per document for corpora of several thousand "
        "abstracts [3]. The LangChain RecursiveCharacterTextSplitter, released in 2022, "
        "popularised the pattern among practitioners and has since become the de facto "
        "default in open-source RAG implementations [4].", BODY))
    story.append(p(
        "The primary variable in fixed-size chunking is window size, typically expressed "
        "in tokens or words. Common choices in the literature are 128, 256, and 512 tokens "
        "[3, 5]. A secondary variable is overlap — the number of tokens shared between "
        "adjacent chunks. Overlap is theoretically motivated by the concern that "
        "semantically related content may straddle a chunk boundary; a non-zero overlap "
        "allows both flanking chunks to carry partial evidence from the boundary region. "
        "However, the empirical evidence for this intuition is equivocal. Smigielski et al. "
        "(2026) find that fixed-size chunking with window 512 and 50-token overlap is the "
        "most computationally robust method across eight datasets, but do not isolate the "
        "overlap contribution from the window-size effect [3]. The hypothesis that overlap "
        "provides no additional retrieval benefit — derived from the Natural Questions "
        "benchmark — may not transfer to biomedical text, where a single drug-disease-dosage "
        "relationship routinely spans two or three sentences.", BODY))
    story.append(p(
        "Qu et al. (NAACL 2025) examine whether semantic chunking — which dynamically "
        "identifies split points using embedding similarity — justifies its computational "
        "overhead compared to the fixed baseline [5]. Their finding is nuanced: semantic "
        "chunking improves retrieval F1 on long-form documents with coherent topical "
        "sections, but on shorter texts — including abstracts — the improvement shrinks "
        "to within noise. This result motivates the inclusion of PubMed abstracts (median "
        "length approximately 250 words) as a test domain: they are short enough that "
        "semantic boundary detection may not outperform a well-tuned fixed splitter.", BODY))

    story.append(p("2.2  Sentence-Boundary Segmentation", H2))
    story.append(p(
        "Sentence-boundary chunking treats each detected sentence as an atomic retrieval "
        "unit, occasionally merging short sentences to reach a minimum token threshold. "
        "The principal advantage over fixed-size splitting is that it never severs a "
        "sentence mid-clause, preserving predicate-argument structure. The principal risk "
        "is tokeniser error: false boundary detections create degenerate chunks while "
        "missed boundaries produce oversized ones.", BODY))
    story.append(p(
        "NLTK's punkt tokeniser implements an unsupervised algorithm that learns "
        "abbreviation lists and collocations from training text. When applied to PubMed "
        "content, it encounters a large set of biomedical abbreviations — 'i.v.' "
        "(intravenous), 'p.o.' (per os), 'vs.' (versus), 'et al.' — that are absent from "
        "or underrepresented in its training corpus. The consequence is a systematic "
        "over-segmentation at these tokens, producing fragments such as 'Patients received "
        "10 mg metformin i.' and 'v.' as separate chunks — neither of which is retrievable "
        "in isolation.", BODY))
    story.append(p(
        "scispaCy, developed by Neumann et al. at the Allen Institute for Artificial "
        "Intelligence, provides a sentence segmenter trained on biomedical corpora including "
        "MedMentions and BC5CDR [6]. Its dependency parse–based segmentation respects "
        "domain abbreviations and handles structured abstract sections (Background, Methods, "
        "Results, Conclusions) as distinct paragraph units. No published RAG study has "
        "compared NLTK punkt and scispaCy sentence chunking under identical retrieval-"
        "generation conditions on PubMed text, leaving open the question of whether the "
        "segmenter choice materially affects downstream answer quality.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: SEMANTIC CHUNKING
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("3. Semantic and Embedding-Based Chunking", H1))
    story.append(p("3.1  Embedding-Similarity Boundary Detection", H2))
    story.append(p(
        "Semantic chunking replaces the fixed-window rule with a data-driven criterion: "
        "a split is inserted wherever the cosine similarity between the embeddings of "
        "adjacent text units falls below a threshold θ. The intuition is that a sharp "
        "similarity drop signals a genuine topic transition. The approach was implicit in "
        "TextTiling (Hearst, 1997) — which used term-frequency vectors rather than dense "
        "embeddings — and has been updated for the dense-vector era by several recent "
        "frameworks [7]. The most widely used open-source implementation employs "
        "all-MiniLM-L6-v2 (sentence-transformers) at θ ≈ 0.75.", BODY))
    story.append(p(
        "Smigielski et al. (2026) evaluate two embedding-based semantic chunking variants "
        "— Recursive Semantic and Sequential HAC (hierarchical agglomerative clustering) "
        "— across eight benchmark datasets including NaturalQuestions, SQuAD, and Qasper "
        "[3]. Recursive Semantic achieves results comparable to the best fixed-size "
        "baseline on dense-retrieval benchmarks but at processing times that are two to "
        "three orders of magnitude longer. On the GutenQA dataset (literary text), semantic "
        "chunking outperforms fixed chunking by 4–6 F1 points; on NaturalQuestions (short "
        "factoid questions), the advantage disappears. The authors conclude that chunking "
        "complexity should be matched to document complexity — a principle with direct "
        "implications for short biomedical abstracts.", BODY))
    story.append(p(
        "An underexplored dimension is the interaction between the chunker's embedding "
        "model and the retriever's embedding model. If both use the same representation "
        "space, semantic split points align with retrieval-relevant features; if they "
        "differ — for example, MiniLM is used to segment while PubMedBERT is used to "
        "retrieve — the boundary placements may be suboptimal for the retrieval task. No "
        "prior work has examined this cross-model interaction in a biomedical setting.", BODY))

    story.append(p("3.2  Max-Min Semantic and Graph-Based Variants", H2))
    story.append(p(
        "Beyond the pairwise adjacency criterion, Smigielski et al. (2026) also evaluate "
        "Max-Min Semantic chunking (which selects split points to maximise the minimum "
        "within-chunk similarity) and GraphSeg (which builds a sentence similarity graph "
        "and partitions it into communities) [3]. Both require pairwise similarity "
        "computations across all sentence pairs, scaling as O(n²) in the number of "
        "sentences. On the corpora examined, neither outperforms Recursive Semantic "
        "consistently, yet both incur substantially higher computational costs. For "
        "abstracts of 10–20 sentences, the O(n²) cost is negligible; but at the "
        "document level, these methods become impractical without GPU acceleration.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: ADVANCED CHUNKING — PROPOSITION AND LLM-BASED
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("4. Proposition-Level and LLM-Based Chunking", H1))
    story.append(p("4.1  Dense Passage Retrieval at Proposition Granularity", H2))
    story.append(p(
        "Chen et al. (2024) proposed Dense X Retrieval, in which a source document is "
        "decomposed into atomic propositions — minimal, self-contained factual statements "
        "— using an LLM as the decomposer [8]. Each proposition is then embedded and "
        "indexed independently. At query time, retrieval operates over propositions rather "
        "than passages; a post-retrieval step reassembles the original passage context "
        "around each matched proposition before passing it to the generator. The key claim "
        "is that proposition-level indexing increases retrieval precision by eliminating "
        "the noise from co-located but irrelevant sentences.", BODY))
    story.append(p(
        "The cost, however, is substantial. Smigielski et al. (2026) reproduce DenseX "
        "(which they label as their 'proposition' condition) and report average processing "
        "times exceeding fifteen hours per dataset — versus under one second for fixed-size "
        "chunking [3]. On five of eight evaluated datasets, DenseX fails entirely due to "
        "API timeouts or memory overflow. On the datasets where it completes, it does not "
        "consistently outperform fixed-size chunking. The authors characterise this as "
        "evidence that 'chunk quality matters more than chunk quantity' — the granularity "
        "advantage of propositions is not sufficient to compensate for the noise introduced "
        "by LLM decomposition errors and the increased retrieval surface area.", BODY))
    story.append(p(
        "LumberChunker (Duarte et al., 2024) adopts a related but less granular approach: "
        "it uses an LLM to detect semantic shift points in running text and inserts splits "
        "at those points rather than decomposing each sentence into sub-propositional "
        "units [9]. Smigielski et al. evaluate LumberChunker and report that it also "
        "times out on several benchmark datasets [3]. Where it does complete, LumberChunker "
        "performs comparably to Recursive Semantic chunking at substantially higher "
        "computational cost. Both DenseX and LumberChunker require LLM API calls per "
        "document, making them sensitive to rate limits, API pricing, and non-determinism "
        "in the chunker LLM — a practical concern for large-scale biomedical corpora.", BODY))

    story.append(p("4.2  Meta-Chunking and Mixture-of-Granularities", H2))
    story.append(p(
        "Zhao et al. (2025) introduce Meta-Chunking, which operates at two levels: "
        "a Margin Sampling criterion assigns each split candidate a boundary probability "
        "derived from the perplexity gradient of a small language model, while a "
        "Dynamic Combination strategy merges adjacent chunks based on their embedding "
        "similarity to the query at run time [10]. The approach is positioned as a "
        "middle ground between static proposition-level splitting and fully dynamic "
        "query-adaptive methods. On the RULER long-document benchmark, Meta-Chunking "
        "outperforms fixed-size chunking by 6.2 points in answer accuracy, though "
        "the evaluation is restricted to general-domain corpora.", BODY))
    story.append(p(
        "Zhong et al. (2025) propose a Mixture-of-Granularities (MoG) framework that "
        "routes each query to the chunking granularity best matched to the query type "
        "[11]. Factoid queries are routed to proposition-level chunks; thematic queries "
        "to sentence-level chunks; broad survey queries to paragraph-level chunks. The "
        "routing decision is made by a lightweight classifier trained on the query "
        "embeddings. MoG achieves consistent improvements over any single granularity "
        "in isolation, but adds a routing component that must be trained on domain-specific "
        "data — an obstacle for zero-shot biomedical deployment.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: HIERARCHICAL AND QUERY-ADAPTIVE CHUNKING
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("5. Hierarchical and Query-Adaptive Chunking", H1))
    story.append(p("5.1  HiChunk: Hierarchical Structure with Auto-Merge Retrieval", H2))
    story.append(p(
        "Lu et al. (2025) at Tencent Youtu Lab identify a fundamental tension in RAG "
        "chunking: small chunks yield high retrieval precision but insufficient context "
        "for generation; large chunks provide context but dilute the retrieval signal [12]. "
        "They propose HiChunk, a framework that resolves this tension by explicitly "
        "modelling document hierarchy. A fine-tuned Qwen3-4B model assigns each segment "
        "to one of three hierarchical levels (section, paragraph, clause) based on its "
        "structural role in the document. At retrieval time, the Auto-Merge algorithm "
        "identifies clusters of matched fine-grained chunks and promotes them to the "
        "coarser parent context before passing to the generator.", BODY))
    story.append(p(
        "To evaluate HiChunk fairly, Lu et al. construct HiCBench — a manually annotated "
        "benchmark with two tasks: T1 (evidence-dense retrieval, where a single document "
        "contains multiple relevant pieces) and T2 (evidence-sparse retrieval, where "
        "relevant evidence is scattered across the corpus) [12]. On T1 with Qwen3-8B as "
        "generator, HC200+AM achieves evidence recall of 81.03%, compared to 72.1% for "
        "fixed-size 200-token chunking and 68.4% for LumberChunker. On T2, the gap "
        "narrows considerably — HiChunk adds about 3 points over fixed baseline. This "
        "task-dependency is instructive: PubMed abstracts, which are structured syntheses "
        "of study findings, are closer to evidence-dense documents than evidence-sparse "
        "ones, suggesting HiChunk's hierarchical approach may be well-suited to the "
        "biomedical domain.", BODY))
    story.append(p(
        "Processing speed, however, distinguishes the approaches sharply. Fixed-size "
        "chunking processes a document in under one second; HiChunk requires approximately "
        "5.75 seconds per document owing to the LLM inference step; LumberChunker "
        "processes the same document in several minutes [12]. For a corpus of one thousand "
        "PubMed abstracts, these figures translate to roughly sixteen minutes (HiChunk) "
        "versus hours (LumberChunker), making HiChunk the only LLM-based chunker with "
        "practical throughput on a CPU-only research cluster.", BODY))

    story.append(p("5.2  QASC: Query-Adaptive Semantic Chunking", H2))
    story.append(p(
        "Rastogi (2025) introduces a conceptually distinct departure from all prior "
        "methods: Query-Adaptive Semantic Chunking (QASC) decouples the chunking process "
        "from the indexing process entirely [13]. Rather than segmenting at index time and "
        "retrieving at query time, QASC integrates the user query into the segmentation "
        "decision itself. The pipeline operates in three stages: (i) each candidate "
        "sentence in the source document is scored by cosine similarity to the query "
        "embedding using all-MiniLM-L6-v2; (ii) seed sentences (those exceeding a "
        "similarity threshold) are identified as anchors; (iii) a contextual window of "
        "m = 3 sentences around each seed is assembled into the retrieved chunk, with "
        "chunk scores aggregated by maximum seed similarity.", BODY))
    story.append(p(
        "QASC is evaluated on 100 technical documents with 40 queries spanning four "
        "types: factoid, topical, comparative, and multi-hop [13]. QASC achieves F1 = 0.85 "
        "versus 0.76 for standard semantic chunking and 0.72 for the best fixed-size "
        "baseline — improvements of 8–12 percentage points and 18–27 percentage points "
        "respectively. Faithfulness reaches 0.87, and inter-annotator agreement on human "
        "evaluation reaches Fleiss' κ = 0.74–0.80 across evaluator panels. Query latency "
        "is 380 ms versus 2,850 ms for agentic RAG pipelines.", BODY))
    story.append(p(
        "The main limitation of QASC for the present research context is practical: "
        "because chunking is performed at query time rather than index time, QASC cannot "
        "pre-compute and cache a static index. Each query incurs a full document scan. "
        "For a corpus of one thousand abstracts and one hundred questions at four k-values, "
        "this results in 400,000 individual cosine-similarity scans, which may be "
        "computationally prohibitive without vector acceleration. Additionally, QASC's "
        "evaluation does not include biomedical text, leaving open the question of whether "
        "the MiniLM similarity threshold is well-calibrated for clinical vocabulary.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: LATE CHUNKING AND CONTEXTUAL RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("6. Late Chunking and Contextual Retrieval", H1))
    story.append(p("6.1  Late Chunking: Embed First, Segment Later", H2))
    story.append(p(
        "Günther et al. (2024) propose a reversal of the conventional chunking order: "
        "instead of segmenting a document and then embedding each chunk independently, "
        "late chunking first passes the entire document through the encoder as a single "
        "sequence, extracting per-token or per-sentence representations that incorporate "
        "full-document context [14]. Segmentation is then applied to these contextualised "
        "representations post-hoc, producing chunks whose embeddings reflect not only their "
        "local content but also the document's overall argumentative context.", BODY))
    story.append(p(
        "Merola and Singh (2025) evaluate late chunking against contextual retrieval in "
        "a unified experimental framework, testing four embedding models: Stella-V5, "
        "Jina-V3, Jina-V2, and BGE-M3 [15]. Their dataset combination includes NFCorpus "
        "(biomedical retrieval) and MS MARCO (general-domain question answering). On "
        "NFCorpus, late chunking with Jina-V3 achieves NDCG@5 = 0.309, narrowly below "
        "contextual retrieval at 0.317. Late chunking is more computationally efficient — "
        "requiring no LLM context-generation step — but shows inconsistent behaviour "
        "across embedding models: it improves over standard chunking with long-context "
        "encoders (Jina-V3, Stella-V5) but degrades with shorter-context models (BGE-M3). "
        "This model-dependency makes late chunking harder to deploy in settings where the "
        "embedding model may change.", BODY))

    story.append(p("6.2  Contextual Retrieval: LLM-Prepended Context Prefixes", H2))
    story.append(p(
        "Anthropic (2024) introduced Contextual Retrieval as an enhancement to the "
        "standard chunking pipeline: before embedding each chunk, a short LLM-generated "
        "context description (50–100 tokens) is prepended to the chunk text, situating "
        "the chunk within its source document [16]. The hypothesis is that isolated chunks "
        "lose the document-level cues needed to disambiguate entity mentions and "
        "pronominal references; the prepended context restores that information without "
        "increasing the retrieval granularity.", BODY))
    story.append(p(
        "Merola and Singh (2025) combine contextual retrieval with BM25 rank fusion and "
        "a cross-encoder reranker, testing multiple embedding models on NFCorpus and "
        "MS MARCO [15]. The full pipeline — contextual retrieval + BM25 fusion + reranking "
        "— achieves the highest NDCG@5 scores in the study (0.317 with Jina-V3 on "
        "NFCorpus). The computational cost, however, is substantial: LLM context-"
        "generation on GPU requires approximately 20 GB of device memory, and the "
        "contextualization step adds roughly one API call per chunk. On a CPU-only system "
        "with a corpus of 50,000 chunks, this step alone may require several hours. "
        "Within the scope of an abstract-level PubMed corpus (~15,000 chunks), the "
        "overhead is more manageable but still non-trivial.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: BIOMEDICAL DOMAIN SPECIFICS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("7. Chunking Strategies in the Biomedical Domain", H1))
    story.append(p("7.1  The Evidence-Density Problem in Medical RAG", H2))
    story.append(p(
        "Clinical and biomedical text exhibits a property that has received insufficient "
        "attention in the chunking literature: evidence density. A PubMed abstract "
        "reporting a randomised controlled trial may pack the intervention, comparator, "
        "primary endpoint, hazard ratio, confidence interval, and p-value into two "
        "consecutive sentences. If fixed-size chunking places a boundary between those "
        "sentences, a retrieval query for the efficacy outcome will recover the effect "
        "size but not the confidence interval — or vice versa. The chunk is technically "
        "relevant but insufficiently complete for accurate answer generation.", BODY))
    story.append(p(
        "Lu et al. (2025) operationalise this concern through HiCBench Task T1 (evidence-"
        "dense queries) and demonstrate that the gap between fixed-size and hierarchical "
        "chunking is larger on dense-evidence tasks [12]. Their finding aligns with the "
        "prior observation by Shi et al. (2023) that irrelevant context in the LLM input "
        "degrades performance — a phenomenon they termed 'lost in the middle' [17]. If "
        "dense evidence is spread across multiple retrieved chunks, some of which are "
        "marginally relevant, the generator's attention may not span all the needed "
        "information, leading to incomplete answers.", BODY))
    story.append(p(
        "A 2025 study published in MDPI Bioengineering (PMC12649634) conducted what is "
        "arguably the closest prior work to the present research: a comparison of adaptive "
        "versus fixed chunking on medical question answering [18]. The adaptive method — "
        "which dynamically adjusts chunk boundaries based on medical entity density — "
        "achieved 87% answer accuracy versus 13% for a fixed-size baseline on "
        "physician-validated clinical questions. The magnitude of this difference far "
        "exceeds what general-domain benchmarks would predict, suggesting that biomedical "
        "text poses a fundamentally different retrieval problem than Wikipedia or news "
        "corpora.", BODY))

    story.append(p("7.2  Embedding Model Choice in Biomedical Retrieval", H2))
    story.append(p(
        "Most RAG chunking studies use general-domain embedding models (BERT-base, "
        "all-MiniLM-L6-v2, GTE, E5) as both the chunking signal and the retrieval "
        "encoder. In the biomedical domain, domain-adapted models such as PubMedBERT "
        "(Gu et al., 2021) — trained from scratch on PubMed abstracts and PubMed Central "
        "full texts — have consistently outperformed general-domain BERT on downstream "
        "biomedical NLP tasks including named entity recognition, relation extraction, "
        "and question answering [19]. The NeuML/pubmedbert-base-embeddings variant "
        "adapts this encoder for dense retrieval via contrastive fine-tuning on "
        "biomedical query-passage pairs, producing 768-dimensional representations that "
        "may capture drug–gene–disease relationships more faithfully than the 384-"
        "dimensional all-MiniLM-L6-v2 embeddings.", BODY))
    story.append(p(
        "Whether this embedding advantage persists across different chunking strategies "
        "is unknown. It is conceivable that the benefit of PubMedBERT over MiniLM "
        "diminishes for semantically coherent chunks (where both models agree on relevance) "
        "but amplifies for fragmented chunks (where biomedical co-reference is needed "
        "to infer relevance). This embedding × chunking interaction has not been examined "
        "in any published study.", BODY))

    story.append(p("7.3  The Context-Length Cliff in Biomedical Abstractive QA", H2))
    story.append(p(
        "A well-documented finding in general-domain RAG is that answer quality reaches "
        "a plateau — or even declines — once the retrieved context exceeds a threshold "
        "of approximately 2,500 words [20]. The proposed mechanism is that LLM attention "
        "diffuses over longer contexts, reducing the probability of attending to any "
        "single relevant passage. This 'context cliff' is relevant to the choice of k "
        "(the number of retrieved chunks): larger k yields higher retrieval recall but "
        "may flood the context with irrelevant material.", BODY))
    story.append(p(
        "Whether this cliff occurs at the same threshold in biomedical text is unclear. "
        "PubMed abstracts are significantly denser than Wikipedia paragraphs — a 250-word "
        "abstract may carry the information equivalent of a 600-word Wikipedia section. "
        "If LLM attention is governed by information density rather than word count, the "
        "biomedical context cliff may occur at a lower word threshold than the 2,500 "
        "observed in general-domain studies. Alternatively, the structured format of "
        "IMRAD-style abstracts (Introduction, Methods, Results, and Discussion) may "
        "provide salient structural anchors that help the LLM locate relevant content "
        "even in longer contexts, pushing the cliff to a higher threshold.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8: COMPARATIVE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("8. Comparative Analysis", H1))
    story.append(p("8.1  Summary of Key Methods", H2))

    tbl1_header = ["Method", "Boundary\nCriterion", "Complexity", "Best\nDomain",
                   "Biomedical\nValidated", "Key Ref"]
    tbl1_data = [
        ["Fixed-Token",        "Token count",            "O(1)/doc",   "General",    "Partially", "[3,5]"],
        ["Sentence-NLTK",      "Punkt tokeniser",        "O(n)",       "General",    "No",        "[2]"],
        ["Sentence-scispaCy",  "Dep. parse (biomedical)","O(n)",       "Biomedical", "No",        "[6]"],
        ["Semantic (cos θ)",   "Embedding similarity",   "O(n·d)",     "Mixed",      "No",        "[3,7]"],
        ["TextTiling",         "Term freq. valleys",     "O(n²)",      "Long docs",  "No",        "[7]"],
        ["LumberChunker",      "LLM shift detection",    "O(n)·API",   "Long docs",  "No",        "[3,9]"],
        ["DenseX / Proposition","LLM decomposition",     "O(n)·API",   "Factoid QA", "No",        "[3,8]"],
        ["HiChunk",            "LLM hierarchy + AM",     "O(n)·LLM",   "Evidence-dense","No",    "[12]"],
        ["QASC",               "Query-at-segment time",  "O(n·q)/query","Technical", "No",        "[13]"],
        ["Late Chunking",      "Post-hoc segment of doc embedding","O(n·d)","Long docs","Partial","[14,15]"],
        ["Contextual Retrieval","LLM prefix per chunk",  "O(n)·API",   "Precision-sensitive","Partial","[15,16]"],
        ["Adaptive (medical)", "Entity density",         "O(n)",       "Biomedical", "Yes",       "[18]"],
    ]
    ew = (PAGE_W - LEFT_MARGIN - RIGHT_MARGIN)
    story += make_table(tbl1_header, tbl1_data,
                        [ew*0.17, ew*0.17, ew*0.12, ew*0.12, ew*0.12, ew*0.08],
                        "Table 1: Comparative overview of chunking strategies surveyed.")

    story.append(sp(8))
    story.append(p("8.2  Computational Cost vs. Performance Trade-Off", H2))
    story.append(p(
        "A consistent finding across the surveyed literature is that computational "
        "complexity and downstream performance do not scale proportionally. Fixed-size "
        "chunking — the least expensive method — demonstrates robust performance across "
        "the widest range of datasets in Smigielski et al.'s 2026 benchmark [3]. "
        "LumberChunker and DenseX, which rank among the most computationally expensive "
        "methods, fail entirely on five of eight datasets due to resource exhaustion. "
        "HiChunk achieves the strongest evidence recall on dense-retrieval tasks but "
        "requires LLM inference at chunking time, a cost that most research groups can "
        "only partially absorb through free-tier API access.", BODY))
    story.append(p(
        "The practical implication for resource-constrained research — including the "
        "present study, which runs entirely on CPU without GPU acceleration — is that "
        "proposition-level and LLM-based chunkers must be evaluated against their API "
        "costs and latency constraints, not merely their retrieval scores. A method that "
        "achieves 5 additional F1 points at 300× the compute cost does not represent a "
        "Pareto improvement for most deployment scenarios.", BODY))

    story.append(p("8.3  The Research Gap in Biomedical Chunking", H2))
    story.append(p(
        "The following dimensions of the chunking problem remain either unaddressed or "
        "only partially addressed in the biomedical domain:", BODY))

    tbl2_header = ["Research Axis", "Status in General-Domain Lit.", "Status in Biomedical Lit."]
    tbl2_data = [
        ["Token overlap in fixed chunking",
         "Partially studied on NQ, SQuAD",
         "Not systematically evaluated on PubMed"],
        ["NLTK vs. biomedical sentence segmenter",
         "NLTK is de facto standard; no comparison",
         "scispaCy available but not benchmarked in RAG"],
        ["Semantic vs. sentence chunking",
         "Comparable on short docs (Qu et al., 2025)",
         "Unknown on 250-word PubMed abstracts"],
        ["Proposition/LLM chunking scalability",
         "Fails at scale (Smigielski et al., 2026)",
         "Not evaluated on biomedical text"],
        ["Context-length cliff threshold",
         "~2,500 words (Wikipedia corpora)",
         "Unknown for denser biomedical text"],
        ["Embedding model × chunking interaction",
         "Not studied",
         "Not studied (MiniLM vs. PubMedBERT)"],
        ["Adaptive/entity-aware chunking",
         "MoG, Meta-Chunking exist",
         "One study (PMC12649634); needs replication"],
        ["Query-adaptive chunking (QASC)",
         "Evaluated on technical docs",
         "Not evaluated on PubMed"],
    ]
    story += make_table(tbl2_header, tbl2_data,
                        [ew*0.25, ew*0.375, ew*0.375],
                        "Table 2: Open research questions in biomedical RAG chunking.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9: THE PRESENT STUDY
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("9. The Present Study: Experimental Framework", H1))
    story.append(p("9.1  Corpus and Question Set", H2))
    story.append(p(
        "The empirical work reviewed in this survey motivates an ongoing study that "
        "addresses the open questions in Table 2. The corpus consists of approximately "
        "one thousand PubMed abstracts collected via the Entrez API, distributed across "
        "five MeSH-defined disease topics: hypertension, type 2 diabetes mellitus, lung "
        "neoplasms, COVID-19, and Alzheimer's disease. These topics were selected to span "
        "chronic disease, infectious disease, oncology, neurology, and metabolic medicine — "
        "covering a representative cross-section of the clinical literature. Abstracts "
        "shorter than fifty words are excluded to eliminate single-sentence records. "
        "One hundred annotated questions (twenty per topic) are paired with gold reference "
        "answers and source PMIDs, enabling both retrieval and generation evaluation.", BODY))

    story.append(p("9.2  Chunking Conditions", H2))
    story.append(p(
        "Sixteen chunking conditions are evaluated in a two-phase design. Phase A (Part A) "
        "evaluates twelve fixed-token conditions formed by the Cartesian product of three "
        "window sizes (128, 256, 512 words) and four overlap percentages (0%, 10%, 20%, "
        "40%). The overlap manipulation directly tests Hypothesis H1 — whether overlap "
        "provides a retrievability benefit in biomedical text. Phase B (Part B) adds four "
        "advanced conditions: NLTK sentence chunking, scispaCy sentence chunking, "
        "embedding-similarity semantic chunking (threshold θ = 0.75, all-MiniLM-L6-v2), "
        "and proposition-level chunking (atomic decomposition via Groq API Llama-3-70B).", BODY))

    story.append(p("9.3  Embedding and Retrieval", H2))
    story.append(p(
        "Each chunking condition is indexed using two embedding models: all-MiniLM-L6-v2 "
        "(384 dimensions, general-domain) and NeuML/pubmedbert-base-embeddings (768 "
        "dimensions, biomedical domain). Indexes are built using FAISS IndexFlatIP on "
        "L2-normalised vectors, enabling cosine similarity search without GPU. Retrieval "
        "is performed at four k-values (1, 3, 5, 10), yielding a full factorial of "
        "16 conditions × 2 embeddings × 4 k-values × 100 questions = 12,800 evaluation "
        "runs, plus 800 additional runs for the best fixed-token anchor in Part B.", BODY))

    story.append(p("9.4  Generation and Evaluation", H2))
    story.append(p(
        "Retrieved chunks are assembled into a context string and passed to "
        "Gemini-2.0-Flash-Lite (temperature = 0.0, maximum 512 output tokens) with a "
        "constrained system prompt requiring the model to answer from provided context "
        "only. A four-second inter-call sleep respects the free-tier rate limit of "
        "fifteen requests per minute. Six metrics are recorded for each run: "
        "answer relevance (cosine similarity of query and answer embeddings), ROUGE-L "
        "F1 against the gold reference, faithfulness (token-level precision overlap "
        "between answer and context), retrieval recall (proportion of gold PMIDs "
        "recovered), context word count (the H3 independent variable), and end-to-end "
        "latency in milliseconds.", BODY))

    tbl3_header = ["Hypothesis", "Variable Tested", "Metric", "Expected Finding"]
    tbl3_data = [
        ["H1: Overlap benefit",
         "Overlap % (0, 10, 20, 40)",
         "Retrieval recall, ROUGE-L",
         "Overlap improves recall on PubMed due to cross-boundary clinical evidence"],
        ["H2: NLTK vs. scispaCy vs. semantic",
         "Chunker type (fixed, sent_nltk, sent_scispacy, semantic)",
         "Answer relevance, faithfulness",
         "scispaCy and/or semantic outperform NLTK on biomedical text"],
        ["H3: Context-length cliff",
         "Context words (varies with k and window size)",
         "Answer relevance at increasing context word count",
         "Cliff at a different threshold than general-domain 2,500 words"],
    ]
    story += make_table(tbl3_header, tbl3_data,
                        [ew*0.14, ew*0.22, ew*0.20, ew*0.44],
                        "Table 3: Research hypotheses and their operationalisation.")

    story.append(sp(8))
    story.append(p("9.5  Novelty and Contributions", H2))
    story.append(p(
        "Relative to the prior work surveyed in Sections 2–7, the present study makes "
        "the following contributions. First, it provides the first head-to-head comparison "
        "of NLTK punkt and scispaCy sentence chunking within a retrieval-generation "
        "evaluation framework on PubMed text — addressing a gap explicitly noted by "
        "Smigielski et al. (2026) [3]. Second, it introduces PubMedBERT as a second "
        "embedding model alongside MiniLM, enabling direct measurement of the embedding × "
        "chunking interaction that has not been studied in any prior chunking benchmark. "
        "Third, it tests proposition-level chunking on biomedical abstracts — a domain "
        "where atomic facts are both more precise (drug-disease-dosage triplets) and more "
        "fragile under LLM decomposition (complex statistical reporting) than in general-"
        "domain text. Fourth, the context-cliff hypothesis (H3) is operationalised at the "
        "abstract level, where the density of clinical information per word differs "
        "markedly from Wikipedia or news corpora.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10: DISCUSSION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("10. Discussion", H1))
    story.append(p(
        "The trajectory of chunking research from 2020 to 2026 reveals a familiar "
        "pattern: simple baselines established early, more complex alternatives proposed "
        "and partially validated, and the field now entering a phase of critical "
        "reassessment. The reassessment is driven by two convergent findings: that "
        "LLM-based chunking methods (DenseX, LumberChunker) fail at scale despite "
        "theoretical appeal [3], and that the domain-specificity of chunking decisions "
        "has been systematically underweighted in benchmark design [18].", BODY))
    story.append(p(
        "For the biomedical domain specifically, the literature points toward a "
        "productive middle ground: methods that are smarter than fixed-size splitting "
        "but more tractable than full LLM decomposition. Semantic chunking satisfies "
        "this criterion computationally, though its performance advantage on short "
        "abstracts remains to be demonstrated [5]. scispaCy sentence chunking may "
        "provide the biomedical adaptation needed without the API cost of LLM-based "
        "methods [6]. HiChunk's Auto-Merge retrieval is conceptually appealing for "
        "evidence-dense biomedical documents but requires a fine-tuned model that "
        "may not generalise to the abstract-only setting [12].", BODY))
    story.append(p(
        "The overlap hypothesis (H1) deserves particular attention. The clinical "
        "literature contains many instances where a drug is named in one sentence and "
        "its side effect reported in the next — a cross-boundary evidence pattern that "
        "overlap is designed to address. Whether overlap improves retrieval in this "
        "setting by a statistically significant margin is an empirical question that "
        "the present study is positioned to answer for the first time.", BODY))
    story.append(p(
        "The context-cliff hypothesis (H3) has practical relevance beyond academic "
        "benchmarking. If the cliff occurs at 1,500 words in PubMed text (rather than "
        "2,500 in Wikipedia), then deploying k = 10 with 256-word chunks would routinely "
        "exceed the optimal context length. Clinicians using RAG-based decision-support "
        "tools would receive answers generated from over-loaded contexts, with attendant "
        "risks of information omission or attention diffusion. Identifying the correct "
        "cliff threshold is therefore a safety-relevant finding, not merely a performance "
        "metric.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 11: CONCLUSION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("11. Conclusion", H1))
    story.append(p(
        "This survey has reviewed the principal chunking strategies proposed for "
        "Retrieval-Augmented Generation, tracing their development from fixed-size "
        "windowing through sentence-boundary segmentation, semantic boundary detection, "
        "hierarchical and LLM-based decomposition, query-adaptive segmentation, late "
        "chunking, and contextual retrieval. Ten studies from 2024–2026 have been "
        "examined in detail, and their findings synthesised with respect to the biomedical "
        "domain.", BODY))
    story.append(p(
        "The central conclusion is that the biomedical domain remains significantly under-"
        "represented in chunking research: only one published study (PMC12649634) has "
        "evaluated chunking strategies specifically on medical text, and none has conducted "
        "a multi-method, multi-metric comparison on PubMed abstracts covering diverse "
        "disease topics. The present study addresses this gap through a sixteen-condition, "
        "two-embedding, full-factorial experiment on one thousand PubMed abstracts, "
        "with hypotheses specifically designed to surface the mechanisms through which "
        "chunking choices interact with biomedical text properties.", BODY))
    story.append(p(
        "The findings — when complete — will provide practitioners with actionable guidance "
        "on chunking strategy selection for clinical RAG applications, and will contribute "
        "to the broader literature by grounding general-domain findings in a rigorous "
        "biomedical evaluation. The dataset, chunking code, and evaluation framework "
        "are designed to be fully reproducible on CPU-only hardware, lowering the barrier "
        "for future biomedical RAG research.", BODY))

    # ══════════════════════════════════════════════════════════════════════════
    # REFERENCES
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(p("References", H1))
    story.append(rule())

    refs = [
        ("[1]", "P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, et al., "
                "'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,' "
                "<i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 33, "
                "pp. 9459–9474, 2020."),
        ("[2]", "E. Loper and S. Bird, 'NLTK: The Natural Language Toolkit,' in <i>Proceedings of "
                "the ACL Workshop on Effective Tools and Methodologies for Teaching NLP and "
                "Computational Linguistics</i>, Philadelphia, PA, 2002, pp. 63–70."),
        ("[3]", "P. Smigielski, M. Wrobel, A. Jakubowski, and R. Czarnecki, 'Chunking Methods on "
                "RAG: Evaluating Effectiveness Against Computational Cost,' <i>Proceedings of the "
                "20th International Conference on Knowledge-Based and Intelligent Information "
                "and Engineering Systems (KES 2026)</i>, Wrocław, Poland, 2026. "
                "arXiv:2606.00881."),
        ("[4]", "H. Chase, 'LangChain,' GitHub repository, 2022. [Online]. Available: "
                "https://github.com/hwchase17/langchain"),
        ("[5]", "J. Qu, K. Qian, S. Ye, R. Bhuiyan, H. Zhang, and P. Yu, 'Is Semantic Chunking "
                "Worth the Computational Cost?' in <i>Proceedings of NAACL 2025</i>, "
                "Albuquerque, NM, 2025. arXiv:2410.13070."),
        ("[6]", "M. Neumann, D. King, I. Beltagy, and W. Ammar, 'ScispaCy: Fast and Robust "
                "Models for Biomedical Natural Language Processing,' in <i>Proceedings of the "
                "18th BioNLP Workshop at ACL</i>, Florence, Italy, 2019, pp. 319–327."),
        ("[7]", "M. A. Hearst, 'TextTiling: Segmenting Text into Multi-Paragraph Subtopic Passages,' "
                "<i>Computational Linguistics</i>, vol. 23, no. 1, pp. 33–64, 1997."),
        ("[8]", "T. Chen, H. Wang, S. Chen, W. Yu, K. Ma, X. Zhao, et al., 'Dense X Retrieval: "
                "What Retrieval Granularity Should We Use?' in <i>Proceedings of ACL 2024</i>, "
                "Bangkok, Thailand, 2024. arXiv:2312.06648."),
        ("[9]", "A. Duarte, J. Marques, A. J. Pinto, A. Lourenco, F. Bettencourt, C. Pinto, and "
                "H. Antunes, 'LumberChunker: Long-Form Narrative Document Segmentation,' "
                "<i>arXiv preprint</i> arXiv:2406.17998, 2024."),
        ("[10]", "Z. Zhao, Z. Wu, B. Gao, Y. Lu, X. Sun, and H. Zhang, 'Meta-Chunking: "
                 "Learning Efficient Text Segmentation via Logical Perception,' "
                 "<i>arXiv preprint</i> arXiv:2410.12788, 2025."),
        ("[11]", "Q. Zhong, S. Zheng, X. Li, Y. Lao, H. Li, Z. Liu, and J. Chen, "
                 "'Mix-of-Granularity: Optimize the Chunking Granularity for RAG,' "
                 "<i>arXiv preprint</i> arXiv:2406.00456, 2025."),
        ("[12]", "W. Lu, S. Mao, K. Yu, Q. Fang, C. Gu, and Y. Liang, 'HiChunk: Evaluating and "
                 "Enhancing RAG with Hierarchical Chunking,' Tencent Youtu Lab, "
                 "<i>arXiv preprint</i> arXiv:2509.11552v3, September 2025."),
        ("[13]", "M. Rastogi, 'QASC: Query-Adaptive Semantic Chunking for Efficient and Accurate "
                 "Retrieval-Augmented Generation,' Independent Researcher, 2025."),
        ("[14]", "M. Günther, J. Milliken, J. Koukounas, M. Geuter, X. Wang, H. Yüksel, et al., "
                 "'Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models,' "
                 "<i>arXiv preprint</i> arXiv:2409.04701, 2024."),
        ("[15]", "M. Merola and D. Singh, 'Reconstructing Context: Evaluating Advanced Chunking "
                 "Strategies for Retrieval-Augmented Generation,' University of Bologna, "
                 "<i>arXiv preprint</i> arXiv:2504.19754, April 2025."),
        ("[16]", "Anthropic, 'Introducing Contextual Retrieval,' Anthropic Research Blog, "
                 "September 2024. [Online]. Available: https://www.anthropic.com/news/"
                 "contextual-retrieval"),
        ("[17]", "F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. Chi, et al., "
                 "'Large Language Models Can Be Easily Distracted by Irrelevant Context,' "
                 "in <i>Proceedings of ICML 2023</i>, Honolulu, HI, 2023. arXiv:2302.00093."),
        ("[18]", "C. Gomez-Cabello, B. Shivakumar, and A. Rajpurkar, 'Adaptive Chunking for "
                 "Medical Retrieval-Augmented Generation,' <i>Bioengineering</i>, MDPI, "
                 "vol. 12, no. 11, 2025. PMC12649634."),
        ("[19]", "Y. Gu, R. Tinn, H. Cheng, M. Lucas, N. Usuyama, X. Liu, et al., "
                 "'Domain-Specific Language Model Pretraining for Biomedical Natural Language "
                 "Processing,' <i>ACM Transactions on Computing for Healthcare</i>, "
                 "vol. 3, no. 1, pp. 1–23, 2021."),
        ("[20]", "N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and "
                 "P. Liang, 'Lost in the Middle: How Language Models Use Long Contexts,' "
                 "<i>Transactions of the Association for Computational Linguistics</i>, "
                 "vol. 12, pp. 157–173, 2024."),
    ]

    for num, text in refs:
        row_text = f"<b>{num}</b>&nbsp;&nbsp;{text}"
        story.append(p(row_text, CITE))
        story.append(sp(2))

    return story


# ── Build PDF ──────────────────────────────────────────────────────────────────
def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN + 8 * mm,
        bottomMargin=BOT_MARGIN + 6 * mm,
        title="Context Construction Strategies in Medical RAG: A Systematic Review",
        author="Akash Ramesh",
        subject="Literature Survey — Chunking Strategies for Biomedical RAG",
        keywords="RAG, chunking, biomedical NLP, PubMed, literature survey",
    )
    story = build_story()
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
