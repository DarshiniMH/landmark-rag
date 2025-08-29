from openai import OpenAI
from typing import List, Dict, Tuple
from .tools import search_docs, TOOL_SCHEMA_DB, TOOL_SCHEMA_WITH_WEB, TOOL_FUNCS
import json, os

llm = OpenAI()
MAX_STEPS = 4
DEBUG = bool(os.getenv("AGENT_DEBUG"))
MAX_OBS_CHARS = 2500

def agent(
    question: str,
    pool: int,
    n_rewrites: int,
    chat_history: List[Dict],
    return_scratch: bool = False,
    allow_web: bool = True        # NEW
) -> Tuple[str, List[Dict]] | str:

    scratch: List[Dict] = []

    # Choose which tool schema to expose
    tool_schema = TOOL_SCHEMA_WITH_WEB if allow_web else TOOL_SCHEMA_DB

    # Give the model clear guidance to switch tools if context looks thin
    sys_prompt = (
        "You are an expert landmark assistant following the ReAct cycle (Thought, Action, Observation).\n\n"
        "Follow this process strictly:\n"
        "1. Analyze the Question and previous Observations.\n"
        "2. Generate a detailed 'Thought' explaining your analysis and plan. This is MANDATORY.\n"
        "3. Execute an Action (using search_docs or web_search) OR provide the final answer.\n\n"
        
        "Strategy:\n"
        "- Start with 'search_docs' (local collection).\n"
        "- CRITICAL: If the Observation from 'search_docs' is irrelevant or insufficient, you MUST analyze why in your Thought and switch to 'web_search' (web) in the next step.\n"
        "- Do NOT repeat the same search query if it did not yield relevant results.\n"
        "Always cite sources in the final answer like [source #]."

        "Final Answer Format:\n"
        "- When you have enough information, provide the final answer DIRECTLY and factually.\n"
        "- DO NOT include any preamble, meta-commentary, or summary of your actions (e.g., avoid phrases like 'To address the question...', 'I searched the web and found...', or 'Based on the observations').\n"
        "- Start the answer immediately.\n"
        "- Always cite sources using the indices provided in the Observations, like [1], [2]."
    )

    while len(scratch) < MAX_STEPS:
        chain = ""
        for i, s in enumerate(scratch, 1):
            chain += (f"Thought {i}: {s['thought']}\n"
                      f"Action {i}: {s['action']}\n"
                      f"Observation {i}: {s['obs']}\n")

        user_prompt = (
            f"{chain}\nQuestion: {question}\nThought {len(scratch)+1}:"
        )

        resp = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            tools=tool_schema
        )

        msg = resp.choices[0].message

        # Tool call?
        if msg.tool_calls:
            call = msg.tool_calls[0]
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            
            # --- Capture or Synthesize Thought ---
            thought = (msg.content or "").strip()
            
            # Fallback: Synthesize a thought if the model didn't provide one
            if not thought:
                thought = f"I need more information to answer the question. I will use the '{name}' tool."

            # extract args for either tool
            q2 = args.get("query", "") or question
            k2 = int(args.get("k", 7))
            max_urls = int(args.get("max_urls", 5))  # safe even if not used

            # dispatch to the requested tool
            func = TOOL_FUNCS.get(name)
            if not func:
                obs = f"Unknown tool: {name}"
                scratch.append({
                    "thought": thought,
                    "action":  f"{name}({q2[:40]}…)",
                    "obs":     obs[:800]
                })
                continue

            if name == "search_docs":
                chunks = func(q2, pool, n_rewrites, k2)
            else:  # web_search
                chunks = func(q2, k=k2, max_urls=max_urls)
            
            if name == "search_docs" and len(chunks) < max(3, k2 // 2) and allow_web:
                # force a web_search observation immediately
                web_chunks = TOOL_FUNCS["web_search"](q2, k=k2, max_urls=5)
                web_obs = "\n\n".join(c["text"] for c in web_chunks[:k2])[:1500]
                scratch.append({
                    "thought": "Local context was thin; switching to web_search.",
                    "action":  f"web_search({q2[:40]}…)",
                    "obs":     web_obs
                })
                continue

            # build a compact observation summary
            k_to_show = min(len(chunks), k2)

            if k_to_show == 0:
                obs = "No results found."
            else:
                # Dynamically calculate per-chunk limit
                # Divide the total budget by the number of chunks, leaving a buffer for formatting
                per_chunk_limit = (MAX_OBS_CHARS // k_to_show) - 15 
                
                # Ensure a minimum readability (e.g., at least 250 chars if possible)
                per_chunk_limit = max(250, per_chunk_limit)

                obs_texts = []
                # Iterate through the chunks we intend to show
                for idx, chunk in enumerate(chunks[:k_to_show], 1):
                    text = chunk.get("text", "").strip()
                    
                    if len(text) > per_chunk_limit:
                        # Truncate and add ellipsis
                        text = text[:per_chunk_limit] + "…"
                    
                    # Format with index for better readability by the LLM
                    obs_texts.append(f"[{idx}]: {text}")
                
                obs = "\n\n".join(obs_texts)
                
                # Final safety truncation
                if len(obs) > MAX_OBS_CHARS:
                   obs = obs[:MAX_OBS_CHARS] + "..."

            scratch.append({
                "thought": thought,
                "action":  f"{name}({q2[:40]}…)",
                "obs":     obs
            })

            if DEBUG:
                print(f"[AGENT] STEP {len(scratch)} · tool={name} · returned={len(chunks)}")

            # continue loop; model will see the Observation and decide next
            continue

        # otherwise, final answer
        final = (msg.content or "").strip()
        return (final, scratch) if return_scratch else final

    fallback = "I couldn’t answer within four search steps."
    return (fallback, scratch) if return_scratch else fallback
