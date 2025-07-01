#!/usr/bin/env python3
"""
Embed every row in  data/processed/chunks.jsonl  that is not yet stored in
the Chroma collection landmarks → vector_db/.
"""

import json, pathlib, itertools, sys, torch
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

#  Config 
CHUNK_PATH  = pathlib.Path("data/processed/chunks.jsonl")
DB_PATH     = "vector_db"
COLL_NAME   = "landmarks"
BATCH_SIZE  = 64         # adjust to GPU / CPU RAM

# Select Model & initialise DB 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
model.max_seq_length = 512

client      = chromadb.PersistentClient(path=DB_PATH)
collection  = client.get_or_create_collection(COLL_NAME)
print(f"Loaded Chroma collection “{COLL_NAME}”")

#  1. Gather existing IDs (paged) 
existing_ids = set()
cursor = 0
PAGE   = 10_000
while True:
    page = collection.get(limit=PAGE, offset=cursor, include=[])
    if not page["ids"]:
        break
    existing_ids.update(page["ids"])
    cursor += PAGE

print(f"{len(existing_ids):,} vectors already stored.")

#  2. Load chunks.jsonl 
with CHUNK_PATH.open("r", encoding="utf-8") as f:
    all_chunks = [json.loads(line) for line in f]

chunks_to_add = [c for c in all_chunks if c["chunk_id"] not in existing_ids]
print(f"{len(chunks_to_add):,} new chunks to embed.")

if not chunks_to_add:
    sys.exit(" vector_db is up to date.")

#  3. Batch‑embed & add 
for i in tqdm(range(0, len(chunks_to_add), BATCH_SIZE), desc="Embedding"):
    batch   = chunks_to_add[i : i + BATCH_SIZE]
    texts   = [b["text"] for b in batch]
    ids     = [b["chunk_id"] for b in batch]
    metas   = [
        {
            "landmark_id":  b["landmark_id"],
            "source":       b["source"],          # now matches process_chunks.py
            "section":      b["section"],
        }
        for b in batch
    ]

    vecs = model.encode(texts, normalize_embeddings=True)
    collection.add(ids=ids, embeddings=vecs.tolist(),
                   documents=texts, metadatas=metas)

print(f"✅ Added {len(chunks_to_add):,} new chunks.  Collection total: "
      f"{len(existing_ids) + len(chunks_to_add):,}")
