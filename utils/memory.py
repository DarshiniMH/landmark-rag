from __future__ import annotations
from collections import deque
import re, json, pathlib
from typing import Deque, List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb, hashlib

from utils.rag_utils import get_embedder

#--------------------- Config -----------------------
LLM = OpenAI()
EMBEDDER = get_embedder()

MAX_BUFFER_TURNS = 8
SUMMARY_MAX_WORDS = 75
MEMORY_DB_PATH = "memory_vector_db"
MEMORY_COLL_NAME = "memory"

client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
mem_coll = client.get_or_create_collection(MEMORY_COLL_NAME)

#------------------- Helper Prompts --------------------

def _summarise_prompt(chunk: str, prev_summary: str) -> str:
    return (
        "You are a conversation summariser.\n\n"
        f"Previous summary (keep it concise):\n{prev_summary}\n\n"
        f"New dialogue chunk:\n{chunk}\n\n"
        f"Rewrite a NEW summary of the ENTIRE conversation in <= {SUMMARY_MAX_WORDS} words."
    )

def _rewrite_prompt(history_block: str, user_msg: str) -> str:
    return(
        "Rewrite the final user question so that it is fully self contained, "
        "keeping all references clear. \n\n"
        f"History block: {history_block}\n\n"
        f"User's question: {user_msg}\n\n"
        "Rewrite the question to be standalone:"
    )

#------------------- Memory Management --------------------
class MemoryManager:
    def __init__(self, session_state):
        self.state = session_state
        self.state.setdefault("chat_buffer", deque(maxlen=MAX_BUFFER_TURNS))
        self.state.setdefault("chat_summary", "")
    
    #--------------- Public Properties -------------------
    @property
    def buffer(self) -> Deque[str]:
        return self.state["chat_buffer"]
    
    @property
    def summary(self) ->str:
        return self.state["chat_summary"]

    #--------------- Rewrite user query -------------------
    def rewrite_question(self, user_msg: str) -> str:
        history_block = "\n".join(self.buffer)
        prompt = _rewrite_prompt(history_block, user_msg)
        resp = LLM.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role":"user", "content": prompt}],
            temperature = 0,
            max_tokens = 100
        )
        return resp.choices[0].message.content.strip()

    def update(self, user_msg: str, assistant_msg: str):
        self.buffer.append(f"User: {user_msg}")
        self.buffer.append(f"Assistant: {assistant_msg}")

        if len(self.buffer) == MAX_BUFFER_TURNS:
            chunk = "\n".join([self.buffer.popleft(), self.buffer.popleft()])
            new_summary = self._summarise(chunk)
            self.state["chat_summary"] = new_summary

            self._extract_and_store_facts(chunk)
    
    #--------------- Private Helpers -------------------
    def _summarise(self, chunk:str) -> str:
        prompt = _summarise_prompt(chunk, self.summary)
        resp = LLM.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role":"user", "content": prompt}],
            temperature = 0,
            max_tokens = 100
        )

        return resp.choices[0].message.content.strip()

    def _extract_and_store_facts(self, chunk: str):
        for sent in re.split(r"[.?!]\s+", chunk):
            if re.search(r"\b(I am|I like|I prefer|My)\b", sent, re.I):
                fid = hashlib.md5(sent.encode()).hexdigest()
                if fid not in mem_coll.get(ids=[fid], include=[])["ids"]:
                    vec = EMBEDDER.encode(sent, normalize_embeddings=True)
                    mem_coll.add(ids=[fid], documents=[sent], embeddings=[vec.tolist()],
                                 metadatas=[{"type":"fact"}])

    # ------------ retrieve long‑term memory -----------
    def recall(self, query: str, top_k: int = 3) -> List[str]:
        qvec = EMBEDDER.encode(query, normalize_embeddings=True)
        res = mem_coll.query(query_embeddings=[qvec.tolist()], n_results=top_k)
        return res["documents"][0] if res["ids"] else []