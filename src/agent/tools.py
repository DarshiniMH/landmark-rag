from typing import List, Dict
from utils.rag_utils import retrieve_context, get_embedder, get_collections, format_retrieved_chunks
from utils.web_search import get_web_chunks, rank_chunks_by_embed
from utils.web_ingest import make_splitter

# ---------- existing DB search ----------
def search_docs(query: str, pool: int, n_rewrites: int, k: int = 7) -> List[Dict]:
    embedder = get_embedder()
    collection = get_collections()
    top_docs, top_ids, metas, ids = retrieve_context(query, embedder, collection, k, pool, n_rewrites)
    context_chunks = format_retrieved_chunks(top_docs, top_ids, metas, ids)
    return context_chunks

# ---------- NEW: web search tool (non-Wikipedia) ----------
def web_search(query: str, k: int = 7, max_urls: int = 5) -> List[Dict]:
    splitter = make_splitter()
    # fetch + extract + chunk
    web_chunks = get_web_chunks(query, splitter, topn_results=max_urls, per_url_chars=3000)
    # rank with the SAME embedder as your corpus
    ranked = rank_chunks_by_embed(query, web_chunks, topn=max(40, k))
    # return k chunks in the same shape as search_docs returns
    out = []
    for c in ranked[:k]:
        # ensure meta shows this is web, with link
        meta = dict(c["meta"])
        meta["source"] = meta.get("source", "web")
        out.append({"text": c["text"], "meta": meta, "id": c["id"]})
    return out

# ---------- tool schemas ----------
TOOL_SCHEMA_DB = [{
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Retrieve up to k landmark text chunks from the local collection.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 3, "maximum": 15, "default": 7}
            },
            "required": ["query"]
        }
    }
}]

TOOL_SCHEMA_WITH_WEB = [
    TOOL_SCHEMA_DB[0],
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Retrieve up to k non-Wikipedia web chunks (title+URL in meta).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 3, "maximum": 15, "default": 7},
                    "max_urls": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]

# ---------- runtime dispatch ----------
TOOL_FUNCS = {
    "search_docs": search_docs,
    "web_search":  web_search
}
