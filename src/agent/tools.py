from typing import List, Dict
from utils.rag_utils import retrieve_context, get_embedder, get_collections, format_retrieved_chunks

def search_docs(query: str, pool: int, n_rewrites: int, k: int = 7) -> List[Dict]:
    """
    Use the hybrid reranker to search for relevant documents
    based on the query.
    """
    embedder = get_embedder()
    collection = get_collections()

    # Retrieve context chunks
    top_docs, top_ids, metas, ids= retrieve_context(query, embedder, collection, k, pool, n_rewrites)

    context_chunks = format_retrieved_chunks(top_docs, top_ids, metas, ids)

    return context_chunks

TOOL_SCHEMA = [{
    "type" : "function",
    "function":{
        "name" : "search_docs",
        "description" : "Retrieve up to k landmark text chunks. Useful for answering questions about landmarks.",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "query" : {"type" : "string"}
                #"k" : {"type" : "integer", "minimum": 3, "maximum" : 15, "default": 7}
            },
            "required" : ["query"]
        }
    }
}]