"""
Matches each question in questions.csv to a real PMID from pubmed_corpus.csv
using keyword search, replacing FILL_AFTER_COLLECT placeholders in-place.
"""
import pandas as pd

corpus = pd.read_csv("data/pubmed_corpus.csv")
questions = pd.read_csv("data/questions.csv")

search_map = {
    "HYP_001": ["first-line","thiazide","calcium channel blocker","ACE inhibitor","angiotensin receptor"],
    "HYP_002": ["stage 1 hypertension","130","systolic","diastolic 80"],
    "HYP_003": ["chronic kidney disease","CKD","130/80","target blood pressure"],
    "HYP_004": ["renin-angiotensin","RAAS","aldosterone","pathophysi"],
    "HYP_005": ["myocardial infarction","stroke","heart failure","complication","uncontrolled"],
    "HYP_006": ["resistant hypertension","three antihypertensive","three agents"],
    "HYP_007": ["DASH","sodium restriction","lifestyle","aerobic exercise","weight loss"],
    "HYP_008": ["ACE inhibitor","angiotensin-converting enzyme","mechanism","vasoconstrict"],
    "HYP_009": ["gestational","preeclampsia","pregnancy hypertension"],
    "HYP_010": ["white-coat","white coat hypertension","ambulatory blood pressure"],
    "HYP_011": ["sodium intake","dietary sodium","salt intake","sodium retention"],
    "HYP_012": ["calcium channel blocker","vasodilation","peripheral vascular resistance"],
    "HYP_013": ["left ventricular hypertrophy","LVH","afterload","ventricular hypertrophy"],
    "HYP_014": ["primary hypertension","secondary hypertension","renal artery stenosis","pheochromocytoma"],
    "HYP_015": ["masked hypertension","ambulatory blood pressure monitoring","home monitoring"],
    "HYP_016": ["sleep apnoea","sleep apnea","obstructive sleep","sympathetic nervous"],
    "HYP_017": ["HYVET","elderly hypertension","80 years","octogenarian"],
    "HYP_018": ["diabetes hypertension","ACE inhibitor","ARB","renal protection","both conditions"],
    "HYP_019": ["spironolactone","mineralocorticoid antagonist","resistant hypertension","fourth-line"],
    "HYP_020": ["hypertensive nephropathy","proteinuria","glomerular filtration","arteriolosclerosis"],
    "DIA_001": ["metformin","AMPK","gluconeogenesis","hepatic glucose"],
    "DIA_002": ["HbA1c","7%","53 mmol","glycated haemoglobin target"],
    "DIA_003": ["SGLT2 inhibitor","empagliflozin","canagliflozin","heart failure","cardiovascular"],
    "DIA_004": ["GLP-1","liraglutide","semaglutide","weight loss","appetite"],
    "DIA_005": ["diabetic nephropathy","microalbuminuria","albumin excretion","GFR"],
    "DIA_006": ["lifestyle intervention","prevention","impaired glucose tolerance","physical activity"],
    "DIA_007": ["insulin resistance","skeletal muscle","hepatic glucose output","hyperinsulinaemia"],
    "DIA_008": ["diabetic retinopathy","neovascularisation","proliferative","microaneurysm"],
    "DIA_009": ["sulfonylurea","hypoglycaemia","insulin secretion","elderly"],
    "DIA_010": ["UKPDS","metformin","overweight","myocardial infarction"],
    "DIA_011": ["DPP-4","dipeptidyl peptidase","sitagliptin","saxagliptin"],
    "DIA_012": ["diagnosis","fasting plasma glucose","oral glucose tolerance","criteria"],
    "DIA_013": ["bariatric surgery","gastric bypass","remission","obesity diabetes"],
    "DIA_014": ["peripheral neuropathy","polyol pathway","advanced glycation","axonal"],
    "DIA_015": ["ACE inhibitor","ARB","diabetic nephropathy","intraglomerular","efferent"],
    "DIA_016": ["ketoacidosis","DKA","ketonaemia","insulin infusion"],
    "DIA_017": ["cardiovascular risk","type 2 diabetes","dyslipidaemia","endothelial dysfunction"],
    "DIA_018": ["EMPA-REG","empagliflozin","cardiovascular mortality","heart failure hospitalisation"],
    "DIA_019": ["diabetic foot","foot ulcer","Charcot","peripheral arterial disease"],
    "DIA_020": ["non-alcoholic fatty liver","NAFLD","NASH","cirrhosis"],
    "LCA_001": ["adenocarcinoma","squamous cell carcinoma","histological subtype","NSCLC"],
    "LCA_002": ["EGFR","exon 19","L858R","gefitinib","osimertinib","erlotinib"],
    "LCA_003": ["PD-L1","pembrolizumab","tumour proportion score","TPS"],
    "LCA_004": ["radon","asbestos","non-smoking","risk factor lung","air pollution"],
    "LCA_005": ["TNM staging","stage I","stage II","metastasis","nodal"],
    "LCA_006": ["ALK","crizotinib","alectinib","rearrangement","fusion"],
    "LCA_007": ["cough","haemoptysis","dyspnoea","lung cancer symptom","presentation"],
    "LCA_008": ["low-dose CT","NLST","screening","mortality reduction"],
    "LCA_009": ["cisplatin","carboplatin","pemetrexed","paclitaxel","platinum chemotherapy"],
    "LCA_010": ["small cell lung cancer","SCLC","etoposide","chemosensitive"],
    "LCA_011": ["T790M","resistance","EGFR TKI","acquired resistance"],
    "LCA_012": ["KRAS G12C","sotorasib","adagrasib"],
    "LCA_013": ["paraneoplastic","SIADH","PTHrP","hypercalcaemia","ectopic ACTH"],
    "LCA_014": ["KEYNOTE-024","pembrolizumab","TPS 50","first-line immunotherapy"],
    "LCA_015": ["stage IV","overall survival","targeted therapy prognosis","EGFR","ALK"],
    "LCA_016": ["tobacco carcinogen","G-to-T","TP53","smoking mutation","malignant transformation"],
    "LCA_017": ["lobectomy","VATS","surgical resection","stage I stage II","lung surgery"],
    "LCA_018": ["ROS1","entrectinib","ROS1 rearrangement","ROS1 inhibitor"],
    "LCA_019": ["tumour mutational burden","TMB","neoantigen","mutations per megabase"],
    "LCA_020": ["mesothelioma","pemetrexed mesothelioma","nivolumab ipilimumab","pleural mesothelioma"],
    "COV_001": ["fever","cough","anosmia","COVID-19 symptom","ageusia","fatigue"],
    "COV_002": ["ACE2","spike protein","TMPRSS2","viral entry","receptor binding domain"],
    "COV_003": ["dexamethasone","RECOVERY trial","mechanical ventilation","mortality"],
    "COV_004": ["risk factor","severe COVID","obesity","diabetes","older age"],
    "COV_005": ["mRNA vaccine","lipid nanoparticle","spike protein","BNT162","mRNA-1273"],
    "COV_006": ["cytokine storm","IL-6","TNF","hyperinflammatory","endothelial injury"],
    "COV_007": ["D-dimer","coagulation","thrombosis","pulmonary embolism","hypercoagulable"],
    "COV_008": ["long COVID","post-COVID","brain fog","cognitive impairment","12 weeks"],
    "COV_009": ["remdesivir","RNA polymerase","nucleoside analogue","hospitalised"],
    "COV_010": ["Omicron","immune evasion","spike mutation","upper respiratory","severity"],
    "COV_011": ["tocilizumab","IL-6 receptor","mechanical ventilation","inflammatory"],
    "COV_012": ["ARDS","pulmonary complication","aspergillosis","organising pneumonia"],
    "COV_013": ["vaccine efficacy","90%","mRNA vaccine","hospitalisation","severe disease"],
    "COV_014": ["nirmatrelvir","Paxlovid","outpatient","89%","antiviral"],
    "COV_015": ["neurological","encephalopathy","anosmia","stroke COVID","Guillain"],
    "COV_016": ["prone positioning","proning","ARDS","mortality","ventral"],
    "COV_017": ["cardiovascular disease","myocardial injury","arrhythmia","ACE2 cardiac"],
    "COV_018": ["acute kidney injury","AKI","tubular","renal COVID","ACE2 kidney"],
    "COV_019": ["anticoagulation","heparin","venous thromboembolism","bleeding","therapeutic anticoagulation"],
    "COV_020": ["MIS-C","multisystem inflammatory","children","Kawasaki","immunoglobulin"],
    "ALZ_001": ["amyloid plaque","neurofibrillary tangles","tau","hallmark Alzheimer","amyloid beta"],
    "ALZ_002": ["donepezil","rivastigmine","galantamine","cholinesterase inhibitor","acetylcholine"],
    "ALZ_003": ["APOE","apolipoprotein E","epsilon 4","ε4","genetic risk factor"],
    "ALZ_004": ["amyloid cascade hypothesis","APP","amyloid precursor protein","tau phosphorylation"],
    "ALZ_005": ["Lewy body","clinical diagnosis","episodic memory","biomarker","distinguish"],
    "ALZ_006": ["neuroinflammation","microglia","astrogliosis","inflammatory cytokine"],
    "ALZ_007": ["CSF amyloid","phosphorylated tau","amyloid PET","biomarker diagnosis"],
    "ALZ_008": ["lecanemab","CLARITY AD","27%","protofibrils","amyloid PET"],
    "ALZ_009": ["tau propagation","entorhinal cortex","Braak","spread","neurofibrillary"],
    "ALZ_010": ["physical exercise","cognitive training","dementia prevention","non-pharmacological"],
    "ALZ_011": ["familial Alzheimer","PSEN1","PSEN2","autosomal dominant","APP mutation"],
    "ALZ_012": ["memantine","NMDA receptor","glutamate","excitotoxicity","moderate severe"],
    "ALZ_013": ["agitation","neuropsychiatric","apathy","psychosis","antipsychotic","behaviour"],
    "ALZ_014": ["hippocampal atrophy","MRI","cortical thinning","temporoparietal","white matter"],
    "ALZ_015": ["sleep","glymphatic","amyloid clearance","sleep deprivation","slow-wave"],
    "ALZ_016": ["APOE e4","amyloid clearance","aggregation","plaque accumulation","APOE epsilon"],
    "ALZ_017": ["mild cognitive impairment","MCI","episodic memory","conversion rate","10%"],
    "ALZ_018": ["plasma biomarker","blood-based","p-tau 217","amyloid 42 40 ratio"],
    "ALZ_019": ["vascular pathology","cerebrovascular","blood-brain barrier","mixed pathology"],
    "ALZ_020": ["lecanemab","donanemab","anti-amyloid","ARIA","disease-modifying","phase 3"],
}

topic_map = {
    "HYP": "hypertension",
    "DIA": "diabetes",
    "LCA": "lung_cancer",
    "COV": "covid19",
    "ALZ": "alzheimers",
}

def find_pmid(qid, keywords, corpus_sub):
    text_col = (corpus_sub["title"].fillna("") + " " + corpus_sub["abstract"].fillna("")).str.lower()
    for kw in keywords:
        mask = text_col.str.contains(kw.lower(), regex=False)
        hits = corpus_sub[mask]
        if not hits.empty:
            return str(hits.iloc[0]["pmid"])
    return "FILL_AFTER_COLLECT"

results = {}
for qid, keywords in search_map.items():
    prefix = qid[:3]
    topic = topic_map[prefix]
    sub = corpus[corpus["topic"] == topic]
    pmid = find_pmid(qid, keywords, sub)
    results[qid] = pmid
    status = "OK " if pmid != "FILL_AFTER_COLLECT" else "MISS"
    print(f"[{status}] {qid}: {pmid}")

questions["gold_doc_ids"] = questions["question_id"].map(results).fillna(questions["gold_doc_ids"])
questions.to_csv("data/questions.csv", index=False)

matched = sum(1 for v in results.values() if v != "FILL_AFTER_COLLECT")
missed  = sum(1 for v in results.values() if v == "FILL_AFTER_COLLECT")
print(f"\nMatched: {matched}/100   Unmatched: {missed}/100")
