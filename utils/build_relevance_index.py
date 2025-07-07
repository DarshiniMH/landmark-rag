import json, pathlib,yaml, re
from collections import defaultdict

CHUNK_PATH = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = yaml.safe_load(open("manifests/landmarks.yaml"))["evaluation_questions"]  

truth = defaultdict(set)

with CHUNK_PATH.open('r', encoding = 'utf-8') as f:
    chunks = [json.loads(l) for l in f]

for q in QUESTIONS:
    pats = [re.compile(r, re.I) for r in q["expected_answer_contains"]]
    for ch in chunks:
        if q.get("landmark_refs") and ch["landmark_id"] not in q["landmark_refs"]:
            continue
        if any(p.search(ch["text"]) for p in pats):
            truth[q["id"]].add(ch["chunk_id"])

json.dump({k: list(v) for k,v in truth.items()},
          open("data/relevance_index.json", "w", encoding = "utf-8"),
          indent = 2, ensure_ascii = False)