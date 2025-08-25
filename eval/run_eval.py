from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  

from utils.seed import get_run_seed
RUN_SEED = get_run_seed()

import pathlib, json, datetime, statistics, time
from dotenv import load_dotenv
load_dotenv()

from utils.rag_utils import (
    retrieve_context, get_embedder, get_llm_answer, build_prompt, get_collections
)
from .metrics import score_pair

import openai, torch, yaml 

import datetime, zoneinfo
LOCAL = zoneinfo.ZoneInfo("America/Los_Angeles")   # or your tz
ts = datetime.datetime.now(LOCAL).isoformat(timespec="seconds")

#------------ Config ------------

QUEST_YAML = pathlib.Path("manifests/landmarks.yaml")
RESULTS_DIR = pathlib.Path("generated_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDER = get_embedder()
COLLECTION = get_collections()
LLM = openai.OpenAI()

TOP_K = 7
POOL = 80
N_REWRITES = 2
LLM_MODEL = "gpt-4o-mini"  

#------------------- Load Eval Questions -------------------
with open(QUEST_YAML, "r") as f:
    QUESTIONS = yaml.safe_load(f)["evaluation_questions"]

#------------------- Run Evaluation -------------------
rows = []
t_start = time.time()

for q in QUESTIONS:
    qid = q["id"]
    question = q["question"]
    gold = q["expected_answer_contains"]

    docs, ids, metas, _ = retrieve_context(
        question, EMBEDDER, COLLECTION, TOP_K, POOL, N_REWRITES
    )

    context = "\n\n".join(docs)

    prompt = build_prompt(question, [{"text": d, "meta": m} for d, m in zip(docs,metas)])
    answer = get_llm_answer(prompt, LLM, LLM_MODEL)

    scores = score_pair( question, answer, context, gold)

    row = {
        "id": qid,
        "question": question,
        "answer": answer,
        "faithfulness":        scores["faithfulness"],
        "relevance":           scores["relevance"],
        "correctness":         scores["correctness"],
        "correctness_grade":   scores["correctness_grade"],
        "coverage":            scores["coverage"],
        "doc_ids":             ids[:TOP_K],
        "run_seed":            RUN_SEED
    }
    rows.append(row)
    rel_str = "NA" if row["relevance"] is None else f"{row['relevance']:.2f}"
    print(f"✓ {qid}  -  corr={row['correctness_grade']}, "
          f"faith={row['faithfulness']:.2f}, "
          f"rel={rel_str}, ")

ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
out_path = RESULTS_DIR / f"{ts}_{LLM_MODEL}_{N_REWRITES}rewrites_{POOL}pool_gemini_eval.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ── aggregate summary --------------------------------------------------
covered = [r for r in rows if r["coverage"]==1]
if covered:
    avg_corr  = statistics.mean(r["correctness"]  for r in covered)
    avg_faith = statistics.mean(r["faithfulness"] for r in covered)
    avg_rel   = statistics.mean(r["relevance"]    for r in covered)
else:
    avg_corr  = avg_faith = avg_rel = 0.0

coverage_pct = 100* len(covered) / len(rows)

print("\n──────── summary ────────")
print(f"Questions evaluated : {len(rows)}")
print(f"Coverage (answered) : {coverage_pct:.1f}%")
print(f"Avg correctness     : {avg_corr:.3f}")
print(f"Avg faithfulness    : {avg_faith:.3f}")
print(f"Avg relevance       : {avg_rel:.3f}")
print(f"Run seed            : {RUN_SEED}")
print(f"Saved results       : {out_path}")
print(f"Elapsed             : {(time.time()-t_start):.1f}s")