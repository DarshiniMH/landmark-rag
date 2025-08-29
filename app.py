# app.py  ― Interactive Landmark Explorer (chat version)
from __future__ import annotations
from dotenv import load_dotenv; load_dotenv()

# -------------- deterministic seeds -----------------------------------------
import os, random, numpy as np, torch

from utils.seed import get_run_seed
SEED = get_run_seed()
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)

# -------------- std libs & deps ---------------------------------------------
import sys, time, chromadb, openai, streamlit as st, re
from pathlib import Path
from sentence_transformers import SentenceTransformer
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))

# -------------- local helpers ------------------------------------------------
from utils.rag_utils import (
    retrieve_context, build_prompt_conversational,
    get_llm_answer, expand_query, get_embedder, get_collections, get_gemini
)
from src.agent.active_retriever import agent
from utils.memory import MemoryManager

# ---- web fallback helpers (non‑Wikipedia) -----------------------------------
from utils.web_search import get_web_chunks, rank_chunks_by_embed
from utils.web_ingest import make_splitter

# -------------- initialise session‑state objects ----------------------------
if "chat_history" not in st.session_state:          # each item: {role, content, ctx?}
    st.session_state.chat_history = []

mem = MemoryManager(st.session_state)               # rolling summary + buffer

# -------------- Streamlit basics --------------------------------------------
st.set_page_config(page_title="Interactive Landmark Explorer",
                   page_icon="🗺️", layout="centered")
st.title("Interactive Landmark Explorer")

# -------------- lazy‑load embedder + DB + OpenAI -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_db():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set the OPENAI_API_KEY environment variable."); st.stop()

    embedder = get_embedder()
    collection = get_collections()

    return embedder, collection, openai.OpenAI()

embedder, collection, llm = load_model_and_db()

# -------------- sidebar ------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    use_agent  = st.toggle("Use Agentic Reasoning (multi‑step)", value=False)
    use_expand = st.toggle("Query Expansion", value=True)
    use_rewrite = st.toggle("Query Rewriting", value=False,
                            help = "Rewrites the user query for better retrival. Increases response time slightly.")
    top_k      = st.slider("Top‑k Chunks", 3, 15, 7, 1)

    st.markdown("---")
    allow_web  = st.toggle("🌐 Allow web fallback (non‑Wikipedia)", value=True,
                           help="If answer is 'I don't know' or context is weak, fetch a few non‑Wikipedia pages and retry.")
    max_web_urls = st.slider("Max web URLs", 1, 8, 5, 1)

    st.markdown("---")
    llm_model_selection = st.selectbox(
        "Answer model",
        ("gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"),
        index=3
    )

POOL       = 40    # initial candidate pool inside retrieve_context
N_REWRITES = 0     # single-shot by default (you can wire your toggle later)

# -------------- helpers ------------------------------------------------------
def _format_chunks(docs, ids, metas):
    """docs/ids/metas -> list of {text, meta, id} aligned by index."""
    return [{"text": d, "meta": m, "id": cid}
            for d, cid, m in zip(docs, ids, metas)]

IDK_RE = re.compile(r"\b(i\s*don['’]t\s*know|no\s+information|not\s+sure)\b", re.I)
def is_idk(ans: str) -> bool:
    return bool(IDK_RE.search(ans or ""))

# -------------- render previous chat ----------------------------------------
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("ctx"):                            # assistant with sources
            with st.expander("📑 Sources", expanded=False):
                for i, ch in enumerate(turn["ctx"], 1):
                    meta = ch.get("meta", {})
                    src  = meta.get("source", "")
                    sec  = meta.get("section", "")
                    url  = meta.get("origin_url", "")
                    line = f"**{i}.** *{src or 'db'}* — {sec}\n\n> {ch['text']}"
                    if url:
                        line += f"\n\n[{url}]({url})"
                    st.markdown(line)

# -------------- user input ---------------------------------------------------
user_msg = st.chat_input("Ask a question about a landmark…")
if user_msg is None:
    st.stop()

# display the user's bubble immediately
with st.chat_message("user"):
    st.markdown(user_msg)

# -------------- rewrite user query ------------------------------------------
rew_q = mem.rewrite_question(user_msg)

# ---------- optional agentic path -------------------------------------------
if use_agent:
    with st.spinner("Reasoning step‑by‑step …"):
        answer, scratch = agent(rew_q, POOL, N_REWRITES,
                                chat_history=st.session_state.chat_history,
                                return_scratch=True,
                                allow_web=allow_web)
    with st.chat_message("assistant"):
        st.markdown(answer)
        if scratch:
            with st.expander("🛠️ Scratch‑pad", expanded=False):
                st.write(scratch)

    st.session_state.chat_history.extend([
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": answer}
    ])
    mem.update(user_msg, answer)
    st.stop()     # skip single‑shot path below

# ---------- single‑shot RAG path --------------------------------------------

exp_q = expand_query(rew_q, llm) if use_expand else rew_q

with st.spinner("Retrieving context …"):
    t0 = time.time()
    if use_rewrite:
        N_REWRITES = 2
    docs, ids, metas, pool_ids = retrieve_context(
        exp_q, embedder, collection, top_k, POOL, N_REWRITES
    )
    t_retr = (time.time() - t0) * 1000

ctx_chunks = _format_chunks(docs, ids, metas)

# 1st pass prompt & answer
prompt = build_prompt_conversational(
    query   = exp_q,
    chunks  = ctx_chunks,
    summary = mem.summary,
    turns   = "\n".join(mem.buffer)
)

with st.spinner("Generating answer from gpt …"):
    #answer = get_gemini(prompt)
    answer = get_llm_answer(prompt, llm, llm_model_selection)

# ---------- web fallback (non‑Wikipedia), only if allowed & needed ----------
# Trigger if model refused or context is very small (you can tune this gate).
trigger = is_idk(answer) or (len(ctx_chunks) < max(4, top_k // 2))

if allow_web and trigger:
    #st.info("🌐 Fetching additional web sources…")
    with st.spinner("Searching the web …"):
        splitter = make_splitter()
        # 1) fetch & extract readable text for a few URLs, then split into chunks
        web_chunks   = get_web_chunks(exp_q, splitter, topn_results=max_web_urls, per_url_chars=3000)
        print(f" web chunks: {len(web_chunks)}")
        # 2) rank those chunks by similarity to the query, keep the best few
        web_ranked   = rank_chunks_by_embed(exp_q, web_chunks, topn=40)

    if web_ranked:
        #st.info("🌐 Found non‑Wikipedia web sources, retrying with them…")
        # Fuse local context + a small number of top web chunks
        # Keep UI+prompt tidy: add up to half of top_k as web chunks (at least 3, at most 8)
        # extra_web = min(max(3, top_k // 2), 8, len(web_ranked))
        web_ctx = [
            {"text": c["text"], "meta": c["meta"], "id": c["id"]} for c in web_ranked[:top_k]
        ]

        prompt2 = build_prompt_conversational(
            query   = exp_q,
            chunks  = web_ctx,
            summary = mem.summary,
            turns   = "\n".join(mem.buffer)
        )

        with st.spinner("Generating answer with web support using gpt"):
            #answer = get_gemini(prompt2)
            answer = get_llm_answer(prompt2, llm, llm_model_selection)

        ctx_chunks = web_ctx  # so the Sources panel shows web chunks too

        #st.info("🌐 Included non‑Wikipedia web sources.")

# ---------- update memory AFTER final answer is settled ----------------------
mem.update(user_msg, answer)

# ---------- assistant bubble -------------------------------------------------
with st.chat_message("assistant"):
    st.markdown(answer)
    with st.expander("📑 Sources", expanded=False):
        for i, ch in enumerate(ctx_chunks, 1):
            meta = ch.get("meta", {})
            src  = meta.get("source", "")
            sec  = meta.get("section", "")
            url  = meta.get("origin_url", "")
            line = f"**{i}.** *{src or 'db'}* — {sec}\n\n> {ch['text']}"
            if url:
                line += f"\n\n[{url}]({url})"
            st.markdown(line)

# ---------- save to session‑state -------------------------------------------
st.session_state.chat_history.extend([
    {"role": "user",      "content": user_msg},
    {"role": "assistant", "content": answer, "ctx": ctx_chunks}
])

# ---------- footer latency ---------------------------------------------------
st.caption(
    f"⏱️ {t_retr:,.0f} ms • pool={len(pool_ids)} • rewrites={N_REWRITES or 1}"
)
