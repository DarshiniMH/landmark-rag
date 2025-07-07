import openai
import chromadb, torch
from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

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
            temperature=0.3,
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

def retrieve_context(query: str, embedder, collection, n_results: int) -> list[dict]:
    """
    Embeds the query and retrieves the top-k most relevant document chunks from ChromaDB.
    """
    # The BGE model does not use a "query: " prefix.
    query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=25,
        include=["documents", "metadatas", "ids"] # We need both text and metadata
    )
    
    docs = results["documents"][0]
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    # The result from Chroma is a list of lists, we only need the first one
    top_docs, top_ids = rerank(query, docs, ids, n_results)

    out = []
    for cid in top_ids:
        i = ids.index(cid)
        out.append({
            "text": docs[i],
            "meta": metas[i]  # Include metadata for context
        })
    # Using zip is a clean way to combine the parallel lists from Chroma
    #for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
     #   retrieved_chunks.append({"text": doc, "meta": meta})
        
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
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # Lower temperature for more factual answers
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        return f"An error occurred with the OpenAI API: {e}"