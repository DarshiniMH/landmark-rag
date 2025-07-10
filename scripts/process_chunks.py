#!/usr/bin/env python3
"""
Clean → BM25-prune → recursive-split every raw document and write them
into data/processed/chunks.jsonl.
"""

import json, hashlib, re, yaml
from pathlib import Path
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk

nltk.download("punkt", quiet=True)

# ── Project helpers ─────────────────────────────────────────────
from cleaners import get_cleaner
from cleaners.wikipedia import sections_with_context
from cleaners.common import normalize_whitespace

RAW_DIR   = Path("data/raw")
OUT_PATH  = Path("data/processed/chunks.jsonl")
MANIFEST  = Path("manifests/landmarks.yaml")

KEEP_RATIO     = 0.70
CHUNK_SIZE     = 700
CHUNK_OVERLAP  = 110

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    separators    = ["\n\n", ". ", " ", ""],
)

# ── HTML heading splitter (very simple) ────────────────────────
HEAD_RE = re.compile(r"\n\s*<h[1-6][^>]*>([^<]{3,80})</h[1-6]>", re.I)

def sections_from_html(text: str):
    hits = HEAD_RE.split(text)
    if len(hits) < 3:
        return [("Full text", text)]
    secs = [("Introduction", hits[0].strip())] if hits[0].strip() else []
    for i in range(1, len(hits)-1, 2):
        secs.append((hits[i].strip(), hits[i+1].strip()))
    return secs

# ── BM25 pruning helper ────────────────────────────────────────
def prune_section(article: str, title: str, body: str):
    sents = sent_tokenize(body)
    if len(sents) <= 3:
        return body
    bm25 = BM25Okapi([word_tokenize(s.lower()) for s in sents])
    q    = word_tokenize(f"{article} {title}".lower())
    scores = bm25.get_scores(q)
    k = max(1, int(len(sents)*KEEP_RATIO))
    idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
    return " ".join(sents[i] for i in sorted(idx))

# ── Main ───────────────────────────────────────────────────────
def main():
    print("⏳  processing raw docs …")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = yaml.safe_load(MANIFEST.read_text())["landmarks"]
    id2name  = {row["id"]: row["name"] for row in manifest}

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for lid, title in tqdm(id2name.items(), desc="Landmarks"):
            lm_dir = RAW_DIR / lid
            if not lm_dir.is_dir():
                continue

            for raw_path in lm_dir.iterdir():
                # 1) choose cleaner
                if raw_path.suffix == ".html":
                    raw_text = raw_path.read_text("utf-8")
                    cleaned  = get_cleaner("html")(raw_text)
                    meta     = raw_path.with_suffix(".meta.json")
                    origin   = json.loads(meta.read_text())["url"] if meta.exists() else ""
                    sections = sections_from_html(cleaned)
                    source   = "html"

                elif raw_path.name == "wikipedia.json":
                    raw_json = json.loads(raw_path.read_text("utf-8"))
                    cleaned  = get_cleaner("wikipedia")(raw_json["content"])
                    origin   = raw_json.get("url", "")
                    sections = sections_with_context(cleaned)
                    source   = "wikipedia"

                else:
                    continue

                # 2) section → prune → chunk
                for sec_title, sec_body in sections:
                    pruned = prune_section(title, sec_title, sec_body)
                    if not pruned:
                        continue

                    prefix = f"Context: {title} - {sec_title}.\n\n"
                    for chunk in splitter.split_text(prefix + pruned):
                        url_hash = hashlib.md5(origin.encode()).hexdigest()[:6]  # NEW

                        slug  = re.sub(r"[^a-z0-9]+", "_", sec_title.lower())[:12]
                        cid   = f"{lid}-{source}-{slug}-{url_hash}-{hashlib.md5(chunk.encode()).hexdigest()[:6]}"

                        out_f.write(json.dumps({
                            "chunk_id":       cid,
                            "landmark_id":    lid,
                            "landmark_name":  title,
                            "source":         source,
                            "section":        sec_title,
                            "origin_url":     origin,
                            "text":           chunk
                        }, ensure_ascii=False) + "\n")

    print(f"✅  wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
