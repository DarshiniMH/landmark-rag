#!/usr/bin/env python3
"""
scripts/process_chunks.py

Clean raw docs, prune with BM25, chunk with a recursive splitter,
and write ALL chunks into a single JSONL file.

Run:  python scripts/process_chunks.py
"""

import json, hashlib, yaml, uuid
from pathlib import Path
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk

# Ensure sentence tokenizer is available
nltk.download("punkt", quiet=True)

# ── Project imports ───────────────────────────────────────────────────
from cleaners import get_cleaner
from cleaners.wikipedia import sections_with_context

# ── Config ────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
OUT_PATH      = Path("data/processed/chunks.jsonl")
MANIFEST_FILE = Path("manifests/landmarks.yaml")

KEEP_RATIO     = 0.7    # keep top 70 % sentences in each section
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 75

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    separators    = ["\n\n", ". ", " ", ""],      # removed ", "
)

# ── Helper: BM25 prune one section ────────────────────────────────────
def prune_section(article_title: str, section_title: str, body: str) -> str:
    if not body.strip():
        return ""

    sents = sent_tokenize(body)
    if len(sents) < 4:          # tiny sections → keep all
        return body

    tokens = [word_tokenize(s.lower()) for s in sents]
    bm25   = BM25Okapi(tokens)

    query_tokens = word_tokenize(f"{article_title} {section_title}".lower())
    scores = bm25.get_scores(query_tokens)

    keep_n = int(len(sents) * KEEP_RATIO)
    top_idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:keep_n]
    pruned = " ".join(sents[i] for i in sorted(top_idx))
    return pruned

# ── Main pipeline ─────────────────────────────────────────────────────
def main() -> None:
    print(" Processing raw Wikipedia docs …")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = yaml.safe_load(MANIFEST_FILE.read_text())["landmarks"]

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for lm in tqdm(manifest, desc="Landmarks"):
            lid   = lm["id"]
            title = lm["name"]

            raw_path = RAW_DIR / lid / "wikipedia.json"
            if not raw_path.exists():
                continue                                   # skip missing

            raw = json.loads(raw_path.read_text())
            cleaned = get_cleaner("wikipedia")(raw["content"])
            sections = sections_with_context(cleaned)      # [(title, body),…]

            for sec_title, sec_body in sections:
                pruned = prune_section(title, sec_title, sec_body)
                if not pruned:
                    continue

                text_for_split = f"Context: {title} - {sec_title}.\n\n{pruned}"
                for chunk in splitter.split_text(text_for_split):
                    # deterministic id: landmark + md5 hash of chunk
                    cid = f"{lid}-{hashlib.md5(chunk.encode()).hexdigest()[:8]}"

                    out_f.write(json.dumps({
                        "chunk_id"     : cid,
                        "landmark_id"  : lid,
                        "landmark_name": title,
                        "source"       : "wikipedia",
                        "section"      : sec_title,
                        "origin_url"   : raw.get("url", ""),
                        "text"         : chunk
                    }, ensure_ascii=False) + "\n")

    print(f" Finished → {OUT_PATH}")

if __name__ == "__main__":
    main()
