#!/usr/bin/env python3
"""
Read data/processed/chunks.jsonl → encode with BGE-base in batches → 
store in a persistent Chroma collection under vector_db/.
This version uses batch processing for a significant speedup.
"""

import json
import pathlib
import torch  # FIX: Added the missing torch import
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CHUNK_PATH = pathlib.Path("data/processed/chunks.jsonl")
DB_PATH = "vector_db"
COLL_NAME = "landmarks"
BATCH_SIZE = 64  # Process 64 chunks at a time. Adjust based on your VRAM/RAM.

# --- 1. Load Model (once) ---
# Use a context manager for cleaner device placement
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
model.max_seq_length = 512

# --- 2. Connect to ChromaDB ---
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLL_NAME)
print(f"Chroma collection '{COLL_NAME}' loaded/created.")

# --- 3. Read all chunks and identify which are new ---
print("Reading chunks and identifying new ones to process...")
with open(CHUNK_PATH, "r", encoding="utf-8") as f:
    all_chunks = [json.loads(line) for line in f]

# Get all existing IDs from Chroma in one go. This is more efficient than looping.
existing_ids = set(collection.get(include=[])["ids"]) 
print(f"Found {len(existing_ids)} existing documents in the collection.")

# Filter out the chunks that already exist in the database
chunks_to_add = [chunk for chunk in all_chunks if chunk["chunk_id"] not in existing_ids]

if not chunks_to_add:
    print("✅ No new chunks to add. The database is up to date.")
    exit()

print(f"Found {len(chunks_to_add)} new chunks to embed and add.")

# --- 4. Process New Chunks in Batches ---
# Use tqdm to create a progress bar over the batches
for i in tqdm(range(0, len(chunks_to_add), BATCH_SIZE), desc="Embedding batches"):
    # a. Create a batch of chunks from the list
    batch = chunks_to_add[i : i + BATCH_SIZE]
    
    # b. Extract the parts we need for this batch
    batch_texts = [chunk["text"] for chunk in batch]
    batch_ids = [chunk["chunk_id"] for chunk in batch]
    batch_metadatas = [
        {
            "landmark_id": chunk["landmark_id"],
            "source": chunk.get("source_type", "unknown"), # Use .get for safety
            "section": chunk.get("chunk_index", -1) # Use .get for safety
        } 
        for chunk in batch
    ]
    
    # c. Embed the entire batch of texts in one go
    batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)
    
    # d. Add the entire batch to ChromaDB in one go
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings.tolist(),
        documents=batch_texts,
        metadatas=batch_metadatas
    )

print(f"✅ Embedding complete! Added {len(chunks_to_add)} new documents. Index stored in '{DB_PATH}'")