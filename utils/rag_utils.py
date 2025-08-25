import openai
import chromadb, torch
from collections import defaultdict
from functools import lru_cache
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import tiktoken
from typing import List, Dict, Any

DB_PATH = "vector_db"
COLL_NAME = "landmarks"
MODEL = "BAAI/bge-base-en-v1.5"
LLM = openai.OpenAI()
RRF_K = 80
LAMBDA = 60


@lru_cache
def get_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")

@lru_cache
def get_embedder():
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  
    model = SentenceTransformer(MODEL, device = device)
    model.max_seq_length = 512
    return model

@lru_cache
def get_collections():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLL_NAME)
    return collection


def expand_query(query: str, llm_client: openai.OpenAI | None = None) -> str:
    """
    Uses the LLM to expand a user query with thematic keywords for better retrieval.
    """
    # This prompt is designed to get thematic keywords, not just nouns.
    expansion_prompt = f"""
You are an expert search query analyst. Your task is to understand the user's query and generate a list of 5 thematic keywords that capture the core intent of the question. These keywords will be used to improve a vector database search. Do not simply repeat the nouns from the query.

Here is an example:
User Query: "What is the Eiffel Tower made of?"
Thematic Keywords: material, construction, built with, iron, steel

Now, analyze the following user query and provide the thematic keywords.

User Query: "{query}"

Thematic Keywords:"""
    if llm_client is None:
        llm_client = openai.OpenAI()

    try:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo", # Using a fast model is good for this step
            messages=[{"role": "user", "content": expansion_prompt}],
            temperature=0.2
        )
        expanded_terms = response.choices[0].message.content.strip()
        # Combine the original query with the new keywords for a richer search
        return f"{query} {expanded_terms.replace(',', ' ')}"
    except openai.OpenAIError as e:
        # If the LLM call fails for any reason, we safely fall back to the original query
        print(f"Warning: Query expansion failed with error: {e}. Using original query.")
        return query



def rerank(query, docs, ids, top_k):
    """
    Reranks the retrieved documents using a cross-encoder model.
    """
    # Prepare the inputs for the reranker
    rerank_inputs = [[query, doc] for doc in docs]
    
    # Get rerank scores
    rerank_scores = get_reranker().predict(rerank_inputs, convert_to_numpy=True)
    
    # Sort by scores and get top-k results
    sorted_indices = np.argsort(rerank_scores)[::-1][:top_k]
    
    return [docs[i] for i in sorted_indices], [ids[i] for i in sorted_indices]


def _keyword_filter(order: List[int], metas: List[Dict[str, Any]], query_lc: str) -> List[int]:
    """
    Filter a pre‑sorted list of candidate indices to those whose
    `landmark_id` appears (strictly or loosely) in the user’s query.
    """
    strict = [
        i for i in order
        if metas[i]["landmark_id"].replace("_", " ") in query_lc
    ]
    if strict:
        return strict

    # 2) fallback: token overlap
    q_tokens = set(query_lc.split())
    loose = [
        i for i in order
        if any(tok in q_tokens for tok in metas[i]["landmark_id"].split("_"))
    ]
    return loose or order

def alpha_for_query(query: str, dense: np.ndarray) -> float:
    q = query.lower().split()
    short_q = len(q) <= 4
    has_digit = any(ch.isdigit() for ch in query)

    # 1) very confident dense?  keep α high
    dense_gap = dense[0] - dense[1]           # already normalised 0‑1
    if dense_gap < 0.2:
        return 0.85

    # 2) short OR has digits → trust BM25 more
    if short_q or has_digit:
        return 0.75

    return 0.95   

def reformulate_query(q: str, n :int=2)->list[str]:
    prompt = ("Rewrite the following landmark question into "
              "a different phrasing that preserves the meaning.\n"
              f"Q: {q}\n---\nOne rewrite:")
    resp = LLM.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = 0.7,
        messages = [{"role":"user", "content":prompt}],
        n=n
    ) 
    return [c.message.content.strip() for c in resp.choices]   


def _rrf_merge(lists: list[list[str]]) -> list[str]:
    """
    lists – list of ranked ID lists (one per query rewrite)
    returns global list of IDs ranked by Reciprocal‑Rank Fusion
    """
    score = defaultdict(float)
    for ranked in lists:
        for r, cid in enumerate(ranked, start = 1):        # 1‑based rank
            score[cid] += 1.0 / (LAMBDA + r)
    return [cid for cid,_ in sorted(score.items(),
                                    key=lambda x: (-x[1],x[0]))]  


def retrieve_context(query: str, embedder, collection, k: int, pool: int =40, n_rewrites: int = 0) -> list[dict]:
    """
    Embeds the query and retrieves the top-k most relevant document chunks from ChromaDB.
    """
   # ── 1) Build 1 + N rewrites ──────────────────────────────────
    if n_rewrites>=1:
        queries = [query] + reformulate_query(query, n_rewrites)
    else:
        queries = [query]

   # ── 2) retrieve 200 chunks *per* rewrite ─────────────────────────
    per_query_ids, per_query_docs, per_query_meta, per_query_dist = [], [], [], []

    for q in queries:
        emb = embedder.encode(q, normalize_embeddings=True).tolist()
        res = collection.query(
            query_embeddings=[emb],
            n_results=200,
            include=["documents","metadatas","distances"]
        )
        per_query_ids  .append(res["ids"][0])
        per_query_docs .append(res["documents"][0])
        per_query_meta .append(res["metadatas"][0])
        per_query_dist .append(res["distances"][0])

    
    # ---------- 3) fuse ID lists (RRF) ---------------------------------
    fused_ids = _rrf_merge(per_query_ids)[:pool]

    # build a fast lookup:  id  → (doc, meta, dist)
    lookup = {}
    for ids_, docs_, meta_, dist_ in zip(per_query_ids,
                                         per_query_docs,
                                         per_query_meta,
                                         per_query_dist):
        for cid, doc, meta, d in zip(ids_, docs_, meta_, dist_):
            if cid not in lookup:          # first occurrence wins
                lookup[cid] = (doc, meta, d)

    docs   = [lookup[cid][0] for cid in fused_ids]
    metas  = [lookup[cid][1] for cid in fused_ids]
    dist   = [lookup[cid][2] for cid in fused_ids]
    ids    = fused_ids        

    # ── 4) Hybrid scoring ───────────────────────────────────────
    dense_sim = 1.0 - np.array(dist)
    dense     = (dense_sim - dense_sim.min()) / (dense_sim.ptp() + 1e-9)
    tokens    = [d.split() for d in docs]
    bm25      = BM25Okapi(tokens)
    bm_raw    = bm25.get_scores(query.split())
    bm_norm   = (bm_raw - bm_raw.min()) / (bm_raw.ptp()+1e-9)

    alpha      = 0.95
    hybrid = alpha * dense + (1-alpha) * bm_norm

    # deterministic secondary sort on chunk‑ID
    order = sorted(
        range(len(hybrid)),
        key=lambda i: (-hybrid[i], ids[i])
    )

    # ── 5) Landmark‑keyword filter ─────────────────────────────
    keep = _keyword_filter(order, metas, query.lower())
    if keep:
        order = keep

    docs  = [docs[i]  for i in order]
    ids   = [ids[i]   for i in order]
    metas = [metas[i] for i in order]

    # ---- Cross-encoder rerank to final k --------------------------
    top_docs, top_ids = rerank(query, docs, ids, k)

    return top_docs, top_ids, metas, ids

def format_retrieved_chunks(top_docs: List[str],
                            top_ids: List[str],
                            metas: List[dict],
                            all_ids: List[str]) -> List[dict]:
    out = []
    for j, cid in enumerate(top_ids):
        out.append({
            "text": top_docs[j],
            "meta": metas[all_ids.index(cid)]     # same position, still safe
        })
    return out
   

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Composes a clear, instruction-based prompt for the LLM, including the
    retrieved context as numbered sources.
    """
    if not context_chunks:
        # A safe fallback if no context is found
        return "I could not find any relevant information in my documents to answer this question."

    # Format each chunk with its source metadata for the LLM to see
    context_for_prompt = ""
    for i, chunk in enumerate(context_chunks, 1):
        source_info = chunk["meta"].get("landmark_name", "Unknown Source")
        context_for_prompt += f"--- Source [{i}] (from {source_info}):\n{chunk['text']}\n\n"

    # The robust prompt template
    prompt = f"""You are an expert Landmark assistant.

**Grounding rules**
1. Use *only* information found in the **Sources** below.
2. Every factual sentence must end with a citation [source #].
3. If the answer is not explicitly present in Sources, reply exactly: "I don't know."

Sources:
{context_for_prompt}

User query:
{query}

Assistant:"""
    return prompt

def build_prompt_conversational(query: str, 
                                chunks: list[dict],     
                                summary: str, 
                                turns: str) -> str:
    src_lines = []
    for i, ch in enumerate(chunks, 1):
        src_lines.append(f"[source {i}] {ch['text']}")
    sources_block ="\n".join(src_lines)

    promt =f"""You are an expert Landmark assistant.

**Conversation summary**
{summary or '*None so far*'}

**Recent turns**
{turns or '*No recent turns*'}

**Grounding rules**
1. Use *only* information found in the **Sources** section.
2. *Every factual sentence* must end with a citation like [source #].
3. If the answer is not explicitly present in Sources, reply exactly: "I don't know."

Sources:
{sources_block}

User query:
{query}

Assistant:"""
    return promt


def get_llm_answer(prompt: str, llm_client: openai.OpenAI, llm_model: str) -> str:
    """Sends the complete prompt to the LLM and returns the response."""
    enc = tiktoken.encoding_for_model(llm_model)
    tok = enc.encode(prompt)
    print("prompt tokens =", len(tok))

    if len(tok) > 12000:
        print("Prompt > 12k tokens, consider lowering k or chunk size")
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # Lower temperature for more factual answers
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        return f"An error occurred with the OpenAI API: {e}"