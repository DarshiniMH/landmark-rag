from openai import OpenAI
from typing import List, Dict, Tuple
from .tools import search_docs, TOOL_SCHEMA
import json, os

llm = OpenAI()
MAX_STEPS = 4
DEBUG = bool(os.getenv("AGENT_DEBUG"))  # set env var to print debug


def agent(
    question: str, 
    chat_history: List[Dict],
    return_scratch: bool = False
) -> Tuple[str, List[Dict]] | str:

    scratch: List[Dict] = []
    
    while len(scratch) < MAX_STEPS:
        
        chain = ""
        for i, s in enumerate(scratch, 1):
            chain += (f"Thought {i}: {s['thought']}\n"
                        f"Action {i}: {s['action']}\n"
                        f"Observation {i}: {s['obs']}\n")
        user_prompt = (f"{chain}\nQuestion: {question}\nThought {len(scratch)+1}:"
                        "\n(If the Observation already answers the question, "
                        "respond with the final answer instead of calling the tool.)"
        )
        print(f"chain: {chain}\n")

        resp = llm.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "system",
                        "content": "You are an expert landmark assistant."},
                        {"role": "user", "content": user_prompt}],
            temperature = 0.2,
            tools = TOOL_SCHEMA
        )

        print(f"resp: {resp}\n")

        msg = resp.choices[0].message
        print(f"message: {msg}\n")

        if msg.tool_calls:
            call = msg.tool_calls[0]
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            q2 = args.get("query", "")
            k2   =int(args.get("k", 7))
            
            chunks = search_docs(q2, k2)

            obs_text = "\n\n".join(c["text"] for c in chunks[:k2])[:1500]

            scratch.append({
                "thought": (msg.content or "").strip(),
                "action":  f"search_docs({q2[:40]}…)",
                "obs":     json.dumps(chunks, ensure_ascii=False)[:800]
            })

            if DEBUG:
                print(f"STEP {len(scratch)}  ·  docs={len(chunks)}")

            continue
        
        # 4) Otherwise final answer
        final =  (msg.content or "").strip()
        return (final, scratch) if return_scratch else final
    
    fallback = "I couldn’t answer within four search steps."
    return (fallback, scratch) if return_scratch else fallback

if __name__ == "__main__":
    q = ("Which emperor commissioned the Colosseum and "
         "in which century was construction begun?")
    ans, sc = agent(q, [], return_scratch = True)
    print(ans)
    print(json.dumps(sc, indent=2)[:1200])