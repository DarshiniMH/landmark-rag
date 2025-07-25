# ----- deterministic seeds -----------------------------------
import os, random, numpy as np
SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

#---------------------------------------------------------------

import csv, datetime
import yaml
import json
import pathlib
from rank_bm25 import BM25Okapi
from utils.rag_utils import get_embedder, get_collections, rerank, expand_query, retrieve_context
from sentence_transformers import SentenceTransformer
import numpy as np

# Load configurations
DB_PATH = "vector_db"
COLL_NAME = "landmarks"
MANIFEST_PATH = "manifests/landmarks.yaml"
REL_PATH = "data/relevance_index.json"
OUTPUT_DIR = "precision_recall_results"
K = 7
CHUNK = 700
OVERLAP = 110
ALPHA = 0.95
RUN_NOTE = "revert to simpler retriever setting"
RERANK = 40



# load embeder and collection
embedder = get_embedder()
collection = get_collections()

# Load evaluation questions and relevance index
questions = yaml.safe_load(open("manifests/landmarks.yaml"))["evaluation_questions"]
ground_truth = json.load(open(REL_PATH, "r", encoding="utf-8"))

# Evaluate retriever
def eval_retriever(k_val: int) -> dict:
    results = {}

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]

        true_relevant_ids = set(ground_truth.get(q_id,[]))

        if not true_relevant_ids:
            results[q_id] = {"precision": 0, "recall": 0, "retrieved_count": 0, "true_count": 0}
            continue
        
        q_text = expand_query(q_text)
        
        # The result from Chroma is a list of lists, we only need the first one
        top_docs, top_ids, metas, ids = retrieve_context(q_text, embedder, collection, k_val, true_relevant_ids,q_id)

        true_positives = len(set(top_ids) & true_relevant_ids)

        precision = true_positives/ k_val
        recall = true_positives / len(true_relevant_ids) if true_relevant_ids else 0

        # Context Precision
        positional_precision = []
        counter = 0
        for pos,cid in enumerate(top_ids, start=1):
            if cid in true_relevant_ids:
                counter += 1
                positional_precision.append(counter / pos)
            else:
                positional_precision.append(0)
        context_precision = np.mean(positional_precision)

        # context recall
        retrieved_relevant = len(true_relevant_ids & set(top_ids))
        total_relevant = len(true_relevant_ids)
        context_recall = retrieved_relevant/ total_relevant

        results[q_id] = {
            "precision": precision,
            "recall": recall,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "retrieved_count": len(top_ids),
            "true_count": len(true_relevant_ids)
        }
    return results

def generate_report(eval_results: dict, k_val: int) -> str:
    report_lines = [f"# Retrieval Evaluation Report (K={k_val})\n"]

    total_precision = 0
    total_recall = 0
    total_context_precision = 0
    total_context_recall = 0

    report_lines.append("| Question ID | Question Text | Precision | Recall | Context-Precision | Context-Recall |")
    report_lines.append("|-------------|---------------|-----------|--------|-------------------|----------------|")

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]
        res = eval_results[q_id]
        precision = res.get("precision", 0)
        recall = res.get("recall", 0)
        context_precision = res.get("context_precision", 0)
        context_recall = res.get("context_recall", 0)

        total_precision += precision
        total_recall += recall
        total_context_precision += context_precision
        total_context_recall += context_recall
        report_lines.append(f"| {q_id} | {q_text} | {precision:.2f} | {recall:.2f} | {context_precision:.2f} | {context_recall:.2f} |")

    num_questions = len(questions)
    avg_precision = total_precision / num_questions if num_questions else 0
    avg_recall = total_recall / num_questions if num_questions else 0
    avg_context_precision = total_context_precision / num_questions if num_questions else 0
    avg_context_recall = total_context_recall / num_questions if num_questions else 0
    
    report_lines.append("\n## ──────── Averages ────────")
    report_lines.append(f"- **Mean Precision@{k_val}:** {avg_precision:.3f}")
    report_lines.append(f"- **Mean Recall@{k_val}:** {avg_recall:.3f}")
    report_lines.append(f"- **Mean Context-Precision@{k_val}:** {avg_context_precision:.3f}")
    report_lines.append(f"- **Mean Context-Recall@{k_val}:** {avg_context_recall:.3f}")
    
    row = {
    "chunk"   : CHUNK,
    "overlap" : OVERLAP,
    "alpha"   : ALPHA,
    "k"       : K,
    "rerank"  : RERANK,
    "ctx_P"   : f"{avg_context_precision:.3f}",
    "ctx_R"   : f"{avg_context_recall:.3f}",
    "id_P"    : f"{avg_precision:.3f}",
    "id_R"    : f"{avg_recall:.3f}",
    "note"    : RUN_NOTE,
    "timestamp": datetime.datetime.utcnow().isoformat(timespec="minutes")
    }

    tsv_path = pathlib.Path("experiments/retriever_results.tsv")
    tsv_path.parent.mkdir(exist_ok=True)

    write_header = not tsv_path.exists()
    with tsv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, row.keys(), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"📊  Logged run → {tsv_path}")

    return "\n".join(report_lines)

if __name__ == "__main__":
    print(f"Evaluating retriever with K={K}...")
    eval_results = eval_retriever(K)

    print("Generating report...")
    report = generate_report(eval_results, K)

    #output_path = pathlib.Path(OUTPUT_DIR) / f"retriever_eval_with_hybrid_alpha_0.8_reranker40_query_expanded_k{K}_chunk{CHUNK}.md"
    #output_path.write_text(report, encoding="utf-8")

    print("\n--- Evaluation Summary ---")
    print(report.split("## ──────── Averages ────────")[-1].strip())
    #print(f"\n✅ Full report saved to: {output_path}")

