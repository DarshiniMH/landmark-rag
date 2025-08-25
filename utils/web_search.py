# utils/web_search.py
from __future__ import annotations
import os, re, json, time, hashlib, math
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# ---------- HTTP config ----------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT_S = 12

# ---------- Domain policy ----------
BLOCK_DOMAINS = {"wikipedia.org"}  # keep Wiki out (you already have it locally)

def _domain_blocked(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(b in host for b in BLOCK_DOMAINS)

# ---------- Cache (ignore empty payloads) ----------
CACHE_DIR = Path(".web_cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
def _cache_path(key: str) -> Path:
    return CACHE_DIR / (hashlib.md5(key.encode("utf-8")).hexdigest() + ".json")

def _load_cache(key: str, ttl_s: int = 1800):
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > ttl_s:
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        if not data:  # [] or {}
            return None
        return data
    except Exception:
        return None

def _save_cache(key: str, obj) -> None:
    try:
        _cache_path(key).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# ---------- Query normalizer ----------
STOP = set("the a an of to for in on at is are was were i we you they give make tell me please".split())
def normalize_for_search(q: str) -> str:
    # keep keywords; favor nouns/places; remove extra punctuation
    q = re.sub(r"[^\w\s]", " ", q.lower())
    toks = [t for t in q.split() if t and t not in STOP]
    # heuristic: prefer shorter, travel-ish keyword pattern
    if "itinerary" not in toks: toks.insert(0, "itinerary")
    return " ".join(toks[:10])  # keep it short

# ---------- Provider 1: Tavily ----------
def _tavily_search(query: str, topn: int = 5) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        print("Tavily API key not set; skipping Tavily.")
        return []
    payload = {"api_key": key, "query": query, "max_results": topn, "search_depth": "basic"}
    endpoints = [
        os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/v1/search"),
        "https://api.tavily.com/search",
    ]
    for ep in endpoints:
        try:
            r = requests.post(ep, json=payload, headers=HEADERS, timeout=TIMEOUT_S)
            if r.status_code == 404:
                print(f"Tavily 404 at {ep}; trying fallback…")
                continue
            if r.status_code >= 400:
                print(f"Tavily error {r.status_code}: {r.text[:200]}")
                return []
            data = (r.json() or {}).get("results", []) or []
        except Exception as e:
            print(f"Tavily request failed at {ep}: {e}")
            continue

        out = []
        for d in data:
            url = (d.get("url") or "").strip()
            if not url or _domain_blocked(url):
                continue
            out.append({
                "title": (d.get("title") or "").strip(),
                "url": url,
                "snippet": (d.get("content") or "")[:800]
            })
        print(f"tavily search results: {len(out)} (from {ep})")
        return out
    return []

# ---------- Provider 2: DuckDuckGo API (library) ----------
def _ddg_api_search(query: str, topn: int = 5) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return []
    out: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=topn, safesearch="moderate", backend="api"):
                url = (r.get("href") or r.get("url") or "").strip()
                if not url or _domain_blocked(url):
                    continue
                out.append({
                    "title": (r.get("title") or "").strip(),
                    "url": url,
                    "snippet": (r.get("body") or "").strip()
                })
    except Exception as e:
        print(f"DDG API failed: {e}")
        return []
    print(f"ddg-api search results: {len(out)}")
    return out

# ---------- Provider 3: DuckDuckGo HTML (last resort) ----------
def _ddg_html_search(query: str, topn: int = 5) -> list[dict]:
    out: list[dict] = []
    try:
        r = requests.post("https://duckduckgo.com/html/",
                          data={"q": query}, headers=HEADERS, timeout=TIMEOUT_S)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for it in soup.select(".result"):
            a = it.select_one(".result__a")
            if not a:
                continue
            url = (a.get("href") or "").strip()
            if not url or _domain_blocked(url):
                continue
            title = a.get_text(" ", strip=True)
            snip_el = it.select_one(".result__snippet")
            snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
            out.append({"title": title, "url": url, "snippet": snippet})
            if len(out) >= topn:
                break
    except Exception as e:
        print(f"DDG HTML failed: {e}")
        return []
    print(f"ddg-html search results: {len(out)}")
    return out

# ---------- High-level search with cache ----------
def search_web(query: str, topn: int = 6, ttl_s: int = 3600) -> list[dict]:
    q_norm = normalize_for_search(query)
    print(f"Searching web for: {q_norm} (top {topn})")

    cache_key = f"websearch::{q_norm}::{topn}"
    cached = _load_cache(cache_key, ttl_s=ttl_s)
    if isinstance(cached, list) and len(cached) > 0:
        print(f"cache HIT ({len(cached)} results)")
        return cached

    # Try providers in order of reliability
    results = (
        _tavily_search(q_norm, topn=topn)
        or _ddg_api_search(q_norm, topn=topn)
        or _ddg_html_search(q_norm, topn=topn)
    )
    print(f"providers returned: {len(results)} results")

    if results:
        _save_cache(cache_key, results)  # don't cache empty
    return results

# ---------- Fetch + extract ----------
def fetch_and_extract(url: str, max_chars: int = 6000) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT_S)
        r.raise_for_status()
    except Exception as e:
        print(f"[fetch] FAIL {url}: {e}")
        return ""
    html = r.text or ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] if text else ""

# ---------- Split & assemble chunks ----------
def split_into_chunks(text: str, splitter) -> list[str]:
    if not text:
        return []
    try:
        parts = splitter.split_text(text)
        return [p.strip() for p in parts if p.strip()]
    except Exception:
        return [text]

def get_web_chunks(query: str, splitter, topn_results: int = 5, per_url_chars: int = 3000) -> list[dict]:
    results = search_web(query, topn=topn_results)
    chunks: list[dict] = []
    for hit in results:
        url, title, snippet = hit["url"], hit.get("title",""), hit.get("snippet","")
        body = fetch_and_extract(url, max_chars=per_url_chars)

        if not body and snippet:
            # at least emit snippet chunk so model has something
            cid = hashlib.md5((url + "|snippet").encode()).hexdigest()
            chunks.append({
                "id": cid,
                "text": snippet,
                "meta": {"origin_url": url, "source":"web", "title": title, "section": "snippet"}
            })
            continue

        for i, p in enumerate(split_into_chunks(body, splitter)):
            cid = hashlib.md5((url + f"|{i}").encode()).hexdigest()
            chunks.append({
                "id": cid,
                "text": p,
                "meta": {"origin_url": url, "source":"web", "title": title, "section": f"web#{i+1}"}
            })
    print(f"[get_web_chunks] produced {len(chunks)} chunks from {len(results)} URLs")
    return chunks

# ---------- Rank by your existing embedder ----------
from utils.rag_utils import get_embedder
def rank_chunks_by_embed(query: str, chunks: list[dict], topn: int = 40) -> list[dict]:
    if not chunks:
        return []
    EMB = get_embedder()
    qv  = EMB.encode(query, normalize_embeddings=True)
    dv  = EMB.encode([c["text"] for c in chunks], normalize_embeddings=True)
    import numpy as np
    sims = (dv @ qv).tolist()
    order = np.argsort(sims)[::-1][:topn]
    return [chunks[i] for i in order]
