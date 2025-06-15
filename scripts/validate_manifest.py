import yaml, pathlib, sys

path = pathlib.Path("manifests/landmarks.yaml")
data = yaml.safe_load(path.read_text())

# 1. unique landmark IDs
ids = [lm["id"] for lm in data["landmarks"]]
if len(ids) != len(set(ids)):
    dupes = {x for x in ids if ids.count(x) > 1}
    sys.exit(f"Duplicate landmark id(s): {dupes}")


# 2. every question references existing IDs
known = set(ids)
valid_types = {"simple", "multi-hop", "temporal"}
for q in data["evaluation_questions"]:
    if q.get("type") not in valid_types:
        sys.exit(f"Unknown question type in {q['id']}: {q.get('type')}")
    
    unknown = [r for r in q.get("landmark_refs", []) if r not in known]
    if unknown:
        sys.exit(f"Question {q['id']} references unknown landmark ID(s): {unknown}")
    
    if not isinstance(q.get("expected_answer_contains"), list) or not q["expected_answer_contains"]:
        sys.exit(f"Question {q['id']} needs a non-empty expected_answer_contains list")


print("✅ manifest passes sanity checks")
