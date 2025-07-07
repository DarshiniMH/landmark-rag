import yaml
import json
import pathlib
from utils.rag_utils import get_embedder, get_collections, rerank, expand_query
from sentence_transformers import SentenceTransformer

# Load configurations
DB_PATH = "vector_db"
COLL_NAME = "landmarks"
MANIFEST_PATH = "manifests/landmarks.yaml"
REL_PATH = "data/relevance_index.json"
OUTPUT_DIR = "precision_recall_results"
K = 7
CHUNK = 500



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
        
        query_embedding =  embedder.encode(q_text, normalize_embeddings = True)
        retrieved = collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results = 50,
            include = ["documents","metadatas"]
        )
        docs = retrieved["documents"][0]
        ids = retrieved["ids"][0]
        metas = retrieved["metadatas"][0]
    # The result from Chroma is a list of lists, we only need the first one
        top_docs, top_ids = rerank(q_text, docs, ids, k_val)

        true_positives = len(set(top_ids) & true_relevant_ids)

        precision = true_positives/ k_val
        recall = true_positives / len(true_relevant_ids) if true_relevant_ids else 0

        results[q_id] = {
            "precision": precision,
            "recall": recall,
            "retrieved_count": len(top_ids),
            "true_count": len(true_relevant_ids)
        }
    return results

def generate_report(eval_results: dict, k_val: int) -> str:
    report_lines = [f"# Retrieval Evaluation Report (K={k_val})\n"]

    total_precision = 0
    total_recall = 0

    report_lines.append("| Question ID | Question Text | Precision | Recall |")
    report_lines.append("|-------------|---------------|-----------|--------|")

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]
        res = eval_results[q_id]
        precision = res.get("precision", 0)
        recall = res.get("recall", 0)

        total_precision += precision
        total_recall += recall
        report_lines.append(f"| {q_id} | {q_text} | {precision:.2f} | {recall:.2f} |")

    num_questions = len(questions)
    avg_precision = total_precision / num_questions if num_questions else 0
    avg_recall = total_recall / num_questions if num_questions else 0
    
    report_lines.append("\n## ──────── Averages ────────")
    report_lines.append(f"- **Mean Precision@{k_val}:** {avg_precision:.3f}")
    report_lines.append(f"- **Mean Recall@{k_val}:** {avg_recall:.3f}")
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    print(f"Evaluating retriever with K={K}...")
    eval_results = eval_retriever(K)

    print("Generating report...")
    report = generate_report(eval_results, K)

    output_path = pathlib.Path(OUTPUT_DIR) / f"retriever_eval_with_reranker40_query_expanded_k{K}_chunk{CHUNK}.md"
    output_path.write_text(report, encoding="utf-8")

    print("\n--- Evaluation Summary ---")
    print(report.split("## ──────── Averages ────────")[-1].strip())
    print(f"\n✅ Full report saved to: {output_path}")

