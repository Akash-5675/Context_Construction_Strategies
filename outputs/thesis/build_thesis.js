/**
 * Thesis builder: converts chapter markdown files + thesis_stats.json into .docx.
 *
 * Supported markup (deliberately small):
 *   # / ## / ### / ####   headings (numbered by the writer in the text)
 *   blank-line paragraphs, **bold**, *italic*, `code`
 *   - bullet             bullets
 *   1. item              numbered
 *   | a | b |            pipe tables, first row is header, |---| separator optional
 *   ![Caption](path)     figure, centred, with caption
 *   {{table:key}}        table generated from thesis_stats.json (see GEN_TABLES)
 *   <<pagebreak>>        page break
 *   > quote              indented block
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  WidthType, AlignmentType, BorderStyle, ShadingType, LevelFormat, HeadingLevel,
  PageBreak, VerticalAlign, TableOfContents, Header, Footer, PageNumber,
} = require("docx");

const ROOT = "F:/context_construction_strategies";
const SRC = path.join(ROOT, "outputs", "thesis");
const FIG = path.join(ROOT, "outputs", "figures");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs", "thesis_stats.json"), "utf8"));

const FONT = "Times New Roman";
const BODY = 24;          // 12pt
const W = 9026;           // A4 usable width @ 1" margins (DXA)
const NAVY = "1F2A44";

// ----------------------------------------------------------------- generated tables
const f = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const pct = (x) => (x === null || x === undefined) ? "—" : (100 * x).toFixed(1) + "%";
const pval = (p) => p < 0.001 ? (p < 1e-10 ? "< 10⁻¹⁰" : p.toExponential(2)) : f(p, 3);
const nice = (c) => c.replace(/^fixed_(\d+)_(\d+)pct$/, "Fixed $1 / $2%")
                     .replace("sent_nltk", "Sentence–NLTK").replace("sent_scispacy", "Sentence–scispaCy")
                     .replace("semantic", "Semantic");

const GEN_TABLES = {
  corpus_topics: () => [["Topic", "MeSH query root", "Abstracts"],
    ...Object.entries({hypertension: "Hypertension", diabetes: "Diabetes Mellitus, Type 2",
      lung_cancer: "Lung Neoplasms", covid19: "COVID-19", alzheimers: "Alzheimer Disease"})
      .map(([k, v]) => [v, k, String(S.corpus.per_topic[k])]),
    ["Total", "", String(S.corpus.n)]],

  chunk_stats: () => [["Condition", "Chunks", "Chunks / abstract", "Mean words", "SD", "Min", "Max"],
    ...Object.entries(S.chunks).map(([k, v]) => [nice(k), String(v.n_chunks), f(v.chunks_per_doc, 2),
      f(v.words_mean, 1), f(v.words_std, 1), String(v.words_min), String(v.words_max)])],

  per_condition: () => [["Condition", "Answer rate", "ROUGE-L (all)", "ROUGE-L (answered)", "Relevance", "Faithfulness", "Recall", "Context words", "Latency (ms)"],
    ...S.per_condition.map(r => [nice(r.condition), pct(r.answer_rate), f(r.rouge_all), f(r.rouge_answered),
      f(r.relevance), f(r.faithfulness), f(r.recall), f(r.context_words, 0), f(r.latency_ms, 0)])],

  h1_overlap: () => [["Overlap", "n", "Answer rate", "ROUGE-L (all)", "ROUGE-L (answered)", "Recall", "Faithfulness", "Context words"],
    ...S.h1_by_overlap.map(r => [`${r.overlap_pct}%`, String(r.n), pct(r.answer_rate), f(r.rouge), f(r.rouge_answered),
      f(r.recall), f(r.faith), f(r.context, 0)])],

  h1_tukey: () => [["Comparison", "Mean diff.", "Adj. p", "95% CI lower", "95% CI upper", "Significant"],
    ...S.h1_tukey.map(r => [`${r.g1}% vs ${r.g2}%`, f(r.diff, 4), f(r.p, 3), f(r.lo, 4), f(r.hi, 4), r.reject ? "Yes" : "No"])],

  chunk_size: () => [["Chunk size (words)", "n", "Answer rate", "ROUGE-L (all)", "ROUGE-L (answered)", "Recall", "Faithfulness", "Context words"],
    ...S.by_chunk_size.map(r => [String(r.chunk_size), String(r.n), pct(r.answer_rate), f(r.rouge), f(r.rouge_answered),
      f(r.recall), f(r.faith), f(r.context, 0)])],

  size_tukey: () => [["Comparison", "Mean diff.", "Adj. p", "95% CI lower", "95% CI upper", "Significant"],
    ...S.size_tukey.map(r => [`${r.g1} vs ${r.g2}`, f(r.diff, 4), r.p === 0 ? "< 0.001" : f(r.p, 3), f(r.lo, 4), f(r.hi, 4), r.reject ? "Yes" : "No"])],

  h2_table: () => [["Condition", "Answer rate", "ROUGE-L (all)", "ROUGE-L (answered)", "Relevance", "Faithfulness", "Recall", "Context words"],
    ...S.h2_table.map(r => [nice(r.condition), pct(r.answer_rate), f(r.rouge_all), f(r.rouge_answered), f(r.relevance),
      f(r.faithfulness), f(r.recall), f(r.context_words, 0)])],

  h2_tukey: () => [["Comparison", "Mean diff.", "Adj. p", "Significant"],
    ...S.h2_tukey_with_anchor.map(r => [`${nice(r.g1)} vs ${nice(r.g2)}`, f(r.diff, 4), r.p === 0 ? "< 0.001" : f(r.p, 3), r.reject ? "Yes" : "No"])],

  h3_bins: () => [["Approx. context tokens", "n", "Answer rate", "ROUGE-L (all)", "ROUGE-L (answered)", "Faithfulness", "Recall"],
    ...S.h3_bins.map(r => [r.bin, String(r.n), pct(r.answer_rate), f(r.rouge), f(r.rouge_answered), f(r.faith), f(r.recall)])],

  by_k: () => [["k", "n", "Answer rate", "ROUGE-L (all)", "ROUGE-L (answered)", "Recall", "Faithfulness", "Context words", "Latency (ms)"],
    ...S.by_k.map(r => [String(r.k), String(r.n), pct(r.answer_rate), f(r.rouge), f(r.rouge_answered), f(r.recall),
      f(r.faith), f(r.context, 0), f(r.latency, 0)])],

  by_embedding: () => [["Embedding model", "n", "Answer rate", "ROUGE-L (all)", "ROUGE-L (answered)", "Relevance", "Recall", "Faithfulness"],
    ...S.by_embedding.map(r => [r.embedding_model === "minilm" ? "MiniLM (all-MiniLM-L6-v2)" : "PubMedBERT", String(r.n),
      pct(r.answer_rate), f(r.rouge), f(r.rouge_answered), f(r.relevance), f(r.recall), f(r.faith)])],

  embedding_by_condition: () => [["Condition", "MiniLM", "PubMedBERT", "Difference"],
    ...S.embedding_by_condition.map(r => [nice(r.condition), f(r.minilm, 4), f(r.pubmedbert, 4), (r.diff >= 0 ? "+" : "") + f(r.diff, 4)])],

  by_topic: () => [["Topic", "n", "Answer rate", "ROUGE-L (all)", "ROUGE-L (answered)", "Recall", "Faithfulness", "Context words"],
    ...S.by_topic.map(r => [r.topic, String(r.n), pct(r.answer_rate), f(r.rouge), f(r.rouge_answered), f(r.recall), f(r.faith), f(r.context, 0)])],

  size_by_topic: () => [["Topic", "128 words", "256 words", "512 words"],
    ...S.size_by_topic.map(r => [r.topic, f(r["128.0"], 4), f(r["256.0"], 4), f(r["512.0"], 4)])],

  refusal_by_condition: () => [["Condition", "Refusal rate"],
    ...Object.entries(S.refusal.by_condition).map(([k, v]) => [nice(k), pct(v)])],

  refused_vs_answered: () => {
    const a = S.refusal.refused_vs_answered["false"], r = S.refusal.refused_vs_answered["true"];
    return [["Metric", "Answered runs", "Refused runs"],
      ["ROUGE-L", f(a.rouge_l), f(r.rouge_l)], ["Answer relevance", f(a.answer_relevance), f(r.answer_relevance)],
      ["Faithfulness", f(a.faithfulness), f(r.faithfulness)], ["Retrieval recall", f(a.retrieval_recall), f(r.retrieval_recall)],
      ["Context words", f(a.context_words, 0), f(r.context_words, 0)]];
  },

  latency: () => [["Condition", "Mean latency (ms)", "Median latency (ms)"],
    ...S.latency_by_condition.map(r => [nice(r.condition), f(r.mean, 0), f(r.median, 0)])],

  all_questions: () => {
    const csv = fs.readFileSync(path.join(ROOT, "data", "questions.csv"), "utf8");
    const rows = []; let cur = [], field = "", q = false;
    for (let i = 0; i < csv.length; i++) {
      const ch = csv[i];
      if (q) { if (ch === '"') { if (csv[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += ch; }
      else if (ch === '"') q = true;
      else if (ch === ",") { cur.push(field); field = ""; }
      else if (ch === "\n") { cur.push(field); rows.push(cur); cur = []; field = ""; }
      else if (ch !== "\r") field += ch;
    }
    if (field || cur.length) { cur.push(field); rows.push(cur); }
    const hdr = rows[0]; const iq = hdr.indexOf("question"), ia = hdr.indexOf("reference_answer"), ii = hdr.indexOf("question_id"), ig = hdr.indexOf("gold_doc_ids");
    return [["ID", "Question", "Reference answer", "Gold PMID"],
      ...rows.slice(1).filter(r => r.length >= hdr.length).map(r => [r[ii], r[iq], r[ia], r[ig]])];
  },

  question_samples: () => [["ID", "Topic", "Question", "Reference answer"],
    ...S.questions.samples.map(q => [q.question_id, q.topic, q.question, q.reference_answer])],
};

// ----------------------------------------------------------------- inline formatting
function inline(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: BODY, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), font: FONT, size: BODY, bold: true, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: "Consolas", size: BODY - 2, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), font: FONT, size: BODY, italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: BODY, ...base }));
  return out;
}

const para = (text, opts = {}) => new Paragraph({
  children: inline(text, opts.run || {}),
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: 160, line: 360 },   // 1.5 spacing
  ...opts.p,
});

const HEAD = {
  1: { size: 32, level: HeadingLevel.HEADING_1, before: 480, after: 240, pageBreak: true },
  2: { size: 28, level: HeadingLevel.HEADING_2, before: 360, after: 160 },
  3: { size: 26, level: HeadingLevel.HEADING_3, before: 240, after: 120 },
  4: { size: 24, level: HeadingLevel.HEADING_4, before: 200, after: 100 },
};
let firstHeading = true;
const heading = (text, lvl) => new Paragraph({
  heading: HEAD[lvl].level,
  pageBreakBefore: !!HEAD[lvl].pageBreak && !(firstHeading && (firstHeading = false)),
  spacing: { before: HEAD[lvl].before, after: HEAD[lvl].after },
  children: [new TextRun({ text, font: FONT, size: HEAD[lvl].size, bold: true, color: NAVY })],
});

const thin = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: thin, bottom: thin, left: thin, right: thin };

let tableNo = 0, figNo = 0;
let chapterNo = 0;
let chapterLabel = "";
const capNo = (n) => chapterLabel ? `${chapterLabel}.${n}` : String(n);

function makeTable(rows, caption) {
  const cols = rows[0].length;
  const widths = colWidths(rows);
  const fs_ = cols >= 8 ? BODY - 8 : cols >= 6 ? BODY - 6 : BODY - 4;   // 8pt / 9pt / 10pt
  const pad = cols >= 8 ? 40 : 80;
  const cell = (t, i, isHead) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: pad, right: pad },
    shading: isHead ? { type: ShadingType.CLEAR, fill: "E7E9EF", color: "auto" } : undefined,
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { after: 0 },
      children: inline(String(t), { size: fs_, bold: isHead }) })],
  });
  const tbl = new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, ri) => new TableRow({ children: r.map((c, ci) => cell(c, ci, ri === 0)), tableHeader: ri === 0 })),
  });
  const out = [];
  if (caption) {
    tableNo += 1;
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true,
      children: [new TextRun({ text: `Table ${capNo(tableNo)}: `, font: FONT, size: BODY - 2, bold: true }),
                 ...inline(caption, { size: BODY - 2 })] }));
  }
  out.push(tbl, new Paragraph({ spacing: { after: 160 }, children: [] }));
  return out;
}

function colWidths(rows) {
  const cols = rows[0].length;
  const maxLen = Array(cols).fill(6);
  rows.forEach(r => r.forEach((c, i) => { maxLen[i] = Math.max(maxLen[i], Math.min(String(c).length, 40)); }));
  const total = maxLen.reduce((a, b) => a + b, 0);
  const w = maxLen.map(l => Math.floor(W * l / total));
  w[cols - 1] += W - w.reduce((a, b) => a + b, 0);   // hand rounding remainder to last column
  return w;
}

function figure(caption, file) {
  const p = path.isAbsolute(file) ? file : path.join(FIG, file);
  if (!fs.existsSync(p)) return [para(`[missing figure: ${file}]`)];
  const data = fs.readFileSync(p);
  // read PNG dimensions
  const wpx = data.readUInt32BE(16), hpx = data.readUInt32BE(20);
  const maxW = 560, maxH = 380;
  let w = maxW, h = Math.round(maxW * hpx / wpx);
  if (h > maxH) { h = maxH; w = Math.round(maxH * wpx / hpx); }
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width: w, height: h } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: `Figure ${capNo(figNo)}: `, font: FONT, size: BODY - 2, bold: true }),
                 ...inline(caption, { size: BODY - 2 })] }),
  ];
}

// ----------------------------------------------------------------- markdown parser
function parse(md) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0, paraBuf = [];
  const flush = () => { if (paraBuf.length) { out.push(para(paraBuf.join(" "))); paraBuf = []; } };

  while (i < lines.length) {
    const ln = lines[i];
    if (/^\s*$/.test(ln)) { flush(); i++; continue; }
    if (ln.startsWith("<<pagebreak>>")) { flush(); out.push(new Paragraph({ children: [new PageBreak()] })); i++; continue; }
    if (ln.startsWith("<<toc>>")) { flush(); out.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" })); i++; continue; }

    let m;
    if ((m = ln.match(/^(#{1,4})\s+(.*)$/))) {
      flush();
      const lvl = m[1].length;
      if (lvl === 1) {
        const t = m[2].trim();
        let lm;
        if ((lm = t.match(/^Appendix\s+([A-Z])/i))) { chapterLabel = lm[1].toUpperCase(); tableNo = 0; figNo = 0; }
        else if ((lm = t.match(/^(?:Chapter\s+)?(\d+)/))) { chapterLabel = lm[1]; tableNo = 0; figNo = 0; }
        else { chapterLabel = ""; }
      }
      out.push(heading(m[2].trim(), lvl)); i++; continue;
    }
    if ((m = ln.match(/^!\[(.*?)\]\((.*?)\)\s*$/))) { flush(); out.push(...figure(m[1], m[2])); i++; continue; }
    if ((m = ln.match(/^\{\{table:(\w+)(?:\|(.*?))?\}\}\s*$/))) {
      flush();
      const gen = GEN_TABLES[m[1]];
      if (!gen) { out.push(para(`[unknown table ${m[1]}]`)); }
      else out.push(...makeTable(gen(), m[2] || ""));
      i++; continue;
    }
    if ((m = ln.match(/^\{\{file:(.+?)\}\}\s*$/))) {
      flush();
      const fp = path.join(ROOT, m[1]);
      const txt = fs.existsSync(fp) ? fs.readFileSync(fp, "utf8") : `[missing file ${m[1]}]`;
      for (const l of txt.replace(/\r/g, "").split("\n"))
        out.push(new Paragraph({ spacing: { after: 0, line: 240 },
          children: [new TextRun({ text: l.length ? l : " ", font: "Consolas", size: 16 })] }));
      out.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
      i++; continue;
    }
    if (ln.startsWith("{{examples}}")) {
      flush();
      const block = (label, ex) => {
        out.push(new Paragraph({ spacing: { before: 200, after: 80 }, keepNext: true,
          children: [new TextRun({ text: `${label} — ${ex.run_key}`, font: FONT, size: BODY, bold: true })] }));
        for (const [k, v] of [["Question", ex.question], ["Reference answer", ex.reference], ["Generated answer", ex.answer],
                              ["ROUGE-L", String(ex.rouge_l)], ["Gold source retrieved", ex.recall ? "yes" : "no"], ["Context words", String(ex.context_words)]])
          out.push(new Paragraph({ spacing: { after: 60, line: 300 }, alignment: AlignmentType.JUSTIFIED,
            children: [new TextRun({ text: k + ": ", font: FONT, size: BODY - 2, bold: true }), new TextRun({ text: v, font: FONT, size: BODY - 2 })] }));
      };
      S.examples.answered.forEach((e, n) => block(`Answered example ${n + 1}`, e));
      S.examples.refused.forEach((e, n) => block(`Refused example ${n + 1}`, e));
      i++; continue;
    }
    if (ln.startsWith("|")) {
      flush();
      const rows = [];
      let caption = "";
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
        if (!cells.every(c => /^:?-{2,}:?$/.test(c))) rows.push(cells);
        i++;
      }
      if (i < lines.length && lines[i].startsWith("Table:")) { caption = lines[i].slice(6).trim(); i++; }
      out.push(...makeTable(rows, caption)); continue;
    }
    if (/^- /.test(ln)) {
      flush();
      while (i < lines.length && /^- /.test(lines[i])) {
        out.push(new Paragraph({ children: inline(lines[i].slice(2)), numbering: { reference: "bul", level: 0 },
          spacing: { after: 80, line: 336 } }));
        i++;
      }
      continue;
    }
    if (/^\d+\. /.test(ln)) {
      flush();
      while (i < lines.length && /^\d+\. /.test(lines[i])) {
        out.push(new Paragraph({ children: inline(lines[i].replace(/^\d+\. /, "")), numbering: { reference: "num", level: 0 },
          spacing: { after: 80, line: 336 } }));
        i++;
      }
      out.push(new Paragraph({ children: [], spacing: { after: 40 } }));
      continue;
    }
    if (/^> /.test(ln)) {
      flush();
      const q = [];
      while (i < lines.length && /^> /.test(lines[i])) { q.push(lines[i].slice(2)); i++; }
      out.push(new Paragraph({ children: inline(q.join(" "), { italics: true }), indent: { left: 720, right: 720 },
        spacing: { after: 160, line: 336 }, alignment: AlignmentType.JUSTIFIED }));
      continue;
    }
    paraBuf.push(ln.trim()); i++;
  }
  flush();
  return out;
}

// ----------------------------------------------------------------- assemble
const CHAPTERS = process.argv.slice(2).length ? process.argv.slice(2)
  : ["00_frontmatter.md", "01_introduction.md", "02_literature.md", "03_methodology.md",
     "04_results.md", "05_discussion.md", "06_conclusion.md", "07_references.md", "08_appendices.md"];

const children = [];
for (const c of CHAPTERS) {
  const p = path.join(SRC, c);
  if (!fs.existsSync(p)) { console.warn("skip missing", c); continue; }
  children.push(...parse(fs.readFileSync(p, "utf8")));
}

const doc = new Document({
  features: { updateFields: true },
  creator: "Akash Ramesh",
  title: "Context Construction Strategies for Biomedical Retrieval-Augmented Generation",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 480, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 3 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) },
    children,
  }],
});

const outName = process.env.OUT || "Thesis_Context_Construction_Biomedical_RAG.docx";
const outPath = path.join(ROOT, "outputs", outName);
Packer.toBuffer(doc).then(b => { fs.writeFileSync(outPath, b); console.log(`wrote ${outPath} (${(b.length / 1024).toFixed(0)} KB), ${children.length} blocks`); });
