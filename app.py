import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'utils'))

import os
import torch
import chromadb
import openai
from sentence_transformers import SentenceTransformer   
import streamlit as st

# Importing utility functions from rag_utils
from utils.rag_utils import retrieve_context, build_prompt, get_llm_answer, expand_query

# Streamlit app setup
st.set_page_config(page_title="Interactive Landmark Explorer", page_icon="🗺️", layout= "centered")

st.title("Interactive Landmark Explorer")
st.write("Ask questions about famous landmarks and get detailed answers from AI with sources.")

# configuration and model loading
@st.cache_resource
def load_model_and_db():

    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set the OPENAI_API_KEY environment variable.")
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    embedder.max_seq_length = 512

    client = chromadb.PersistentClient(path="vector_db")
    collection = client.get_collection("landmarks")

    try:
        llm_client = openai.OpenAI()
    except openai.OpenAIError as e:
        st.error(f"Error initializing OpenAI client: {e}")
        llm_client = None
    
    return embedder, collection, llm_client

embedder, collection, llm = load_model_and_db()

options = ("gpt-3.5-turbo", "gpt-4", "gpt-4o")
default_index = options.index("gpt-3.5-turbo")

with st.sidebar:
    st.header("Settings")

    llm_model_selection = st.selectbox(
        "Choose Language Model",
        options,
        index = default_index,
    )

    top_k = st.slider(
        "Number of Context Chunks to retrieve(Top K)",
        min_value = 3,
        max_value = 15,
        value = 7,
        step = 1
    )

    use_expansion = st.toggle(
        "Use Query Expansion",
        value = True,
        help = "Expand your query with thematic keywords to improve search results."
    )


query = st.text_input("Enter your question about landmarks:", placeholder = "e.g., What is the history of the Eiffel Tower?")

if query:
    # Expand Query
    if use_expansion:
        with st.spinner("Expanding your query with thematic keywords..."):
            expanded_query = expand_query(query, llm)
            st.info(f"Expanded Query: {expanded_query}")
    else:
        expanded_query = query

    # Retrieve Context
    with st.spinner("Retrieving relevant context chunks..."):
        context_chunks = retrieve_context(expanded_query, embedder, collection, top_k)
    
    # Build Prompt
    prompt = build_prompt(expanded_query, context_chunks)

    # Get Answer from LLM
    with st.spinner("Generating answer from AI..."):
        answer = get_llm_answer(prompt, llm, llm_model_selection)

    # Display the answer
    st.subheader("AI Answer")
    if answer:
        st.markdown(answer)
    
    # Display the context chunks used
    with st.expander("View Context Chunks Used"):
        if not context_chunks:
            st.write("No context chunks were retrieved.")
        else:
            for i, chunk in enumerate(context_chunks, 1):
                meta = chunk.get("meta", {})
                source_name = meta.get("source", "Unknown")
                section = meta.get("section", "N/A")

                st.markdown(f"**Source [{i}] | From:** '{source_name}' | **Section:** '{section}' ")
                st.markdown(f">{chunk['text']}")
                


