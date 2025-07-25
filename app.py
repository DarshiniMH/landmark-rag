# app.py  ― Interactive Landmark Explorer
# ----- deterministic seeds -----------------------------------
import os, random, numpy as np
SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

#---------------------------------------------------------------
import sys, os, torch, chromadb, openai, streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer   
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))

# ── local helpers ─────────────────────────────────────────────
from utils.rag_utils import (
    retrieve_context, build_prompt, get_llm_answer, expand_query
)
from src.agent.active_retriever import agent            # NEW – ReAct loop

# ── Streamlit basics ─────────────────────────────────────────
st.set_page_config(page_title="Interactive Landmark Explorer",
                   page_icon="🗺️", layout="centered")
st.title("Interactive Landmark Explorer")
st.write("Ask questions about famous landmarks and get detailed answers.")

# ── lazy-load models & DB │ cached across runs ───────────────
@st.cache_resource(show_spinner=False)
def load_model_and_db():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set the OPENAI_API_KEY environment variable."); st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    embedder.max_seq_length = 512

    client = chromadb.PersistentClient(path="vector_db")
    collection = client.get_collection("landmarks")

    return embedder, collection, openai.OpenAI()

embedder, collection, llm = load_model_and_db()

# ── sidebar settings ─────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    use_agent   = st.toggle("Use Agentic Reasoning (multi-step)", value=False,
                            help="Let the LLM iterate search-→-observe before answering.")
    use_expand  = st.toggle("Query Expansion", value=True)
    top_k       = st.slider("Top-k Chunks (single-shot mode)",
                             min_value=3, max_value=15, value=7, step=1)

    llm_model_selection = st.selectbox(
        "Language Model",
        ("gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"),
        index=0
    )

# ── main input ───────────────────────────────────────────────
query = st.text_input("Ask a landmark question:",
                      placeholder="e.g. Which emperor commissioned the Colosseum?")

if not query:
    st.stop()

# ── agentic path ─────────────────────────────────────────────
if use_agent:
    with st.spinner("Reasoning step-by-step …"):
        answer, scratch = agent(query, chat_history=[], return_scratch = True)

    st.subheader("AI Answer (agentic)")
    st.markdown(answer)

    with st.expander("Thought / Action / Observation"):
        if not scratch:
            st.write("No scratch-pad returned")
        else:
            for i, step in enumerate(scratch, 1):
                st.markdown(f"#### Step {i}")
                st.markdown(f"**Thought {i}** \n{step['thought'] or '*<empty>*'}")
                st.markdown(f"**Action {i}** \n{step['action']}")
                st.markdown(f"**Observtion {i}**")
                st.code(step["obs"], language="json")
    # In this minimal version we don’t show intermediate thoughts;
    # flip the env-flag AGENT_DEBUG=1 if you want terminal prints.

# ── single-shot RAG path ─────────────────────────────────────
else:
    # 1) optional query expansion
    expanded_q = expand_query(query, llm) if use_expand else query
    if use_expand:
        st.info(f"Expanded query → **{expanded_q}**")

    # 2) retrieve context
    with st.spinner("Retrieving context …"):
        top_docs, top_ids, metas, ids = retrieve_context(expanded_q, embedder, collection, top_k)
        ctx = format_retrieved_chunks(top_docs, top_ids, metas, ids)
    # 3) build prompt & ask LLM
    prompt = build_prompt(expanded_q, ctx)
    with st.spinner("Generating answer …"):
        answer = get_llm_answer(prompt, llm, llm_model_selection)

    # 4) display
    st.subheader("AI Answer")
    st.markdown(answer)

    with st.expander("Context chunks"):
        if not ctx:
            st.write("No chunks retrieved.")
        else:
            for i, ch in enumerate(ctx, 1):
                meta = ch["meta"]
                st.markdown(
                    f"**[{i}]** • *{meta.get('source','?')}* — {meta.get('section','?')}"
                )
                st.markdown("> " + ch["text"])

