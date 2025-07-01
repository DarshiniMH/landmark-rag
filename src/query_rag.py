import os, sys, torch, chromadb, openai
from sentence_transformers import SentenceTransformer

#--Configuration--

DB_PATH = "vector_db"
COLLECTION = "landmarks"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "gpt-3.5-turbo"
TOP_K = 7

#-- Guards --
# This part runs once when the script starts.
print("Loading models and connecting to the database...")

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Error: OPENAI_API_KEY environment variable is not set.")

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION)

#Initialize the OpenAI client
llm = openai.OpenAI()

print("Setup complete. You can now ask questions.")

def expanded_query_with_llm(query: str) -> str:
    prompt = f"""
    You are an expert search query analyst. Your task is to understand the user's query and generate a list of 5 thematic keywords that capture the core intent of the question. These keywords will be used to improve a vector database search. Do not simply repeat the nouns from the query.

Here is an example:
User Query: "What is the Eiffel Tower made of?"
Thematic Keywords: material, construction, built with, iron, steel

Now, analyze the following user query and provide the thematic keywords.

User Query: "{query}"

Thematic Keywords:"""

    try:
        response = llm.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [{"role": "user", "content": prompt}],                
            temperature = 0.3,
        )
        expanded_terms = response.choices[0].message.content
        return f"{query} {expanded_terms.replace(',', ' ')}"    
    except openai.OpenAIError as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return query  # Fallback to original query if there's an error

#-- The main function to get answers --
def get_answer(query: str):
    print(" Expanding the query and searching for relevant context...")
    expanded_query = expanded_query_with_llm(query)
    print(f" Expanded Query: {expanded_query}")

    print(" Step 1: Embedding the query...")
    query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()

    print(" Step 2: Searching the database...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )

    context_chunks_text = results["documents"][0]
    context_chunks_meta = results["metadatas"][0]

    #===== Inspect retrieved chunks =====
    print("\n\033[95m--- DEBUG: CONTEXT RETRIEVED FOR LLM --- \033[0m")
    
    if not context_chunks_text:
        print("\033[93mNo context was retrieved from the database.\033[0m")
    else:
        # Use zip to loop through the parallel lists of text and metadata
        for i, (text, meta) in enumerate(zip(context_chunks_text, context_chunks_meta), 1):
            
            # Correctly access the metadata fields
            landmark_name = meta.get("landmark_name", "N/A")
            section_title = meta.get("section", "N/A")
            chunk_id = meta.get("chunk_id", "N/A")

            print(f"\033[94m--- Chunk [{i}] | From: {landmark_name} - '{section_title}' section ---\033[0m")
            print(f"  \033[96mID:   \033[0m {chunk_id}")
            # This line now prints the FULL, untruncated text of the chunk
            print(f"  \033[96mText: \033[0m {text}") 
    
    print("\033[95m--- END OF DEBUG CONTEXT ---\033[0m\n")
    #===================================

    if not context_chunks_text:
        return "No relevant information found."

    print(" Step 3: Building the prompt for the AI...")
    context_for_prompt = ""
    for i, text in enumerate(context_chunks_text):
        metadata = context_chunks_meta[i]
        source_info = metadata.get("source", "Unknown Source")
        context_for_prompt += f"--- Source [{i+1}] (from {source_info}):\n{text}\n\n"

    # Create the final prompt for clear instructions
    prompt = f"""
    you are an expert Landmark assistant.

Use *only* the sources below to answer the user's question.

if you don't know the answer, say "I don't know".

Cite the sources using [source #] in your answers
    Sources:
    {context_for_prompt}

    User Query: {query}
    Assistant:"""

    print(" Step 4: Generating the answer using the LLM...")

    try:
        response = llm.chat.completions.create(
            model = LLM_MODEL,
            messages = [{"role": "user", "content": prompt}],                
            temperature = 0.2,
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"An error occurred with the OpenAI API: {e}"

#--Main Loop --
if __name__ == "__main__":
    while True:
        user_query = input("Ask a question about landmarks (or type 'exit' to quit): ")
        if user_query.lower() in {"quit", "exit"}:
            print("Assistant: Goodbye!")
            break
        
        # call our single function and print the result
        answer = get_answer(user_query)
        print(f"Assistant: {answer}\n")