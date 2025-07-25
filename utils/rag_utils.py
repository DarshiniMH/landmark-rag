import openai
import chromadb, torch
from functools import lru_cache
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import tiktoken
from typing import List, Dict, Any

DB_PATH = "vector_db"
COLL_NAME = "landmarks"
MODEL = "BAAI/bge-base-en-v1.5"

@lru_cache
def get_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")

@lru_cache
def get_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

def reformulate(q: str, n :int=2)->list[str]:
    prompt = ("Rewqite the following landmark question into "
              "a different phrasing that preserves the meaning.\n"
              f"Q: {q}\n---\nOne rewrite:")
    resp = llm.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = 0.7,
        messages = [{"role":"user", "content":prompt}],
        n=n
    ) 
    return [c.message.content.strip() for c in resp.choices]     

def retrieve_context(query: str, embedder, collection, k: int, true_relevant_ids, qid) -> list[dict]:
    """
    Embeds the query and retrieves the top-k most relevant document chunks from ChromaDB.
    """
    # The BGE model does not use a "query: " prefix.
    query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=200,
        include=["documents", "metadatas", "distances"] # We need both text and metadata
    )
    
    docs = results["documents"][0]
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    dist = np.array(results["distances"][0])

    dense_sim = 1.0 - dist  # Convert distances to similarity scores
    dense   = (dense_sim - dense_sim.min()) / (dense_sim.ptp() + 1e-9)

    # lexical component using BM25
    tokens = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokens)
    bm_raw = bm25.get_scores(query.split())
    bm_norm = (bm_raw - bm_raw.min()) / (bm_raw.ptp()+ 1e-9)

    # Combine BM25 scores with Chroma results
    alpha = 0.95
    hybrid = alpha * np.array(dense)+(1-alpha) * bm_norm

    # Dynamically decide of the pool of candidates based on query tokens
    n_tokens = len(query.split())
    POOL = 100 if n_tokens>12 else 40
    
    order = sorted(
        range(len(hybrid)),
        key=lambda i: (-hybrid[i], ids[i])   # secondary tie-break
    )[:40]      
    
    #------ Debugging ---                           
    
    dbg = []

    hits_100 = len(set(ids[i] for i in order) & true_relevant_ids)
    hits_40  = len(set(ids[i] for i in order[:40]) & true_relevant_ids)
    dbg.append((qid, hits_40, hits_100))

    print(dbg)

    #------ Debugging end ----

    # ------- Keep only the chunks where landmark appears in the i -------
    keep = _keyword_filter(order, metas, query.lower())
    
    if keep:
        docs = [docs[i] for i in keep]
        ids  = [ids[i] for i in keep]
        metas = [metas[i] for i in keep]

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

Use *only* the sources below to answer the user's question.

If you don't know the answer, say "I don't know".

Cite the sources using [source #] in your answers.

Sources:
{context_for_prompt}

User Query: {query}
Assistant:"""
    return prompt

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