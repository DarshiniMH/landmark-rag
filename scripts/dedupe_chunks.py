#!/usr/bin/env python3
"""
Remove duplicate chunk_id lines from data/processed/chunks.jsonl
and write a new de‑duplicated file chunks.unique.jsonl
"""

import json, pathlib, shutil

SRC  = pathlib.Path("data/processed/chunks.jsonl")
DEST = SRC.with_name("chunks.unique.jsonl")

seen  = set()
kept  = []

with SRC.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        cid = obj["chunk_id"]
        if cid in seen:
            continue                    # duplicate → skip
        seen.add(cid)
        kept.append(line.rstrip("\n"))  # keep first occurrence only

original = sum(1 for _ in SRC.open("r", encoding="utf-8"))
removed  = original - len(kept)

DEST.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(f"⚙️  Original lines : {original:,}")
print(f"⚙️  Duplicates     : {removed:,}")
print(f"✅  Unique file    : {DEST}")
