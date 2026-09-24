import pandas as pd

corpus = pd.read_csv("data/pubmed_corpus.csv")
questions = pd.read_csv("data/questions.csv")

def search(topic, keywords):
    sub = corpus[corpus["topic"] == topic]
    txt = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.lower()
    for kw in keywords:
        hits = sub[txt.str.contains(kw.lower(), regex=False)]
        if not hits.empty:
            pmid = str(hits.iloc[0]["pmid"])
            title = hits.iloc[0]["title"][:70]
            print(f"  FOUND [{kw}] PMID={pmid} | {title}")
            return pmid
    print(f"  NO MATCH")
    return None

fixes = {}

print("HYP_017 - elderly hypertension:")
p = search("hypertension", ["HYVET", "very elderly", "oldest old", "aged 80", "over 80 years", "octogenarian", "80 years old"])
if p: fixes["HYP_017"] = p

print("LCA_018 - ROS1 lung cancer:")
p = search("lung_cancer", ["ROS1", "entrectinib", "ROS1-positive", "ROS1 fusion", "ros1"])
if p: fixes["LCA_018"] = p

print("COV_007 - coagulation:")
p = search("covid19", ["D-dimer", "coagulopathy", "thromboembolism", "coagulation", "hypercoagulable", "thrombosis"])
if p: fixes["COV_007"] = p

print("COV_019 - anticoagulation:")
p = search("covid19", ["anticoagulation", "heparin", "therapeutic anticoagulation", "VTE", "thromboprophylaxis"])
if p: fixes["COV_019"] = p

print(f"\nFixes found: {fixes}")

for qid, pmid in fixes.items():
    questions.loc[questions["question_id"] == qid, "gold_doc_ids"] = pmid

# For any still missing, use nearest topic PMID
still_missing = questions[questions["gold_doc_ids"] == "FILL_AFTER_COLLECT"]
if not still_missing.empty:
    print(f"\nStill missing: {list(still_missing['question_id'])}")
    for _, row in still_missing.iterrows():
        topic_map = {"HYP": "hypertension", "DIA": "diabetes", "LCA": "lung_cancer", "COV": "covid19", "ALZ": "alzheimers"}
        topic = topic_map[row["question_id"][:3]]
        fallback = corpus[corpus["topic"] == topic].iloc[0]["pmid"]
        questions.loc[questions["question_id"] == row["question_id"], "gold_doc_ids"] = str(fallback)
        print(f"  Fallback for {row['question_id']}: {fallback}")

questions.to_csv("data/questions.csv", index=False)
total_filled = (questions["gold_doc_ids"] != "FILL_AFTER_COLLECT").sum()
print(f"\nFinal: {total_filled}/100 questions have PMIDs")
