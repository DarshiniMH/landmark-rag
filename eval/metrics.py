from dotenv import load_dotenv
load_dotenv()  

from utils.seed import get_run_seed
RUN_SEED = get_run_seed()

import json, random, pathlib
from typing import List
from openai import OpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer,util
from utils.rag_utils import get_embedder
import os
import re

LLM = OpenAI()
PROMPTS = pathlib.Path(__file__).parent/"prompts"
GRADE_TO_SCORE = { "Good": 1.0, "Bad": 0.0, "Average": 0.5 }
IDK_RE = re.compile(
    r"\b(i\s+don['’]t\s+know|unknown|no\s+information|not\s+sure)\b", re.I
)

# EVAL_MODEL and EVAL_PROVIDER are set in .env
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "openai").lower()
#-------------- helper: fill prompt templates ----------------------------------

def _p(name) : return (PROMPTS/name).read_text()

def _ask_openai(prompt, seed: int, model: str | None = None, **kw) -> str:
    model = model or EVAL_MODEL
    kw.setdefault("temperature", 0)
    kw.setdefault("seed", seed)
    msg = [{"role":"user", "content":prompt}]
    return LLM.chat.completions.create(model = model, messages = msg, **kw)\
            .choices[0].message.content.strip()

def _ask_gemini(prompt: str, seed: int, model: str | None = None, 
                json_mode: bool = False, max_output_tokens: int = 512) -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set; cannot use Gemini for eval.")
    genai.configure(api_key=key)
    model = model or os.getenv("EVAL_MODEL", "gemini-1.5-pro")
    gen_cfg = {
        "temperature": 0,
        "top_p": 0,
        "top_k": 1,
        "candidate_count": 1,
        "max_output_tokens": max_output_tokens,
    }
    # Gemini supports seed in 1.5; if client version lacks it, ignore quietly
    """try:
        gen_cfg["seed"] = seed
    except Exception:
        pass"""

    if json_mode:
        # Ask for JSON back; this greatly reduces parse issues
        gen_cfg["response_mime_type"] = "application/json"

    m = genai.GenerativeModel(model_name=model)
    # single-turn generation; we already craft full prompts
    resp = m.generate_content(prompt, generation_config=gen_cfg)
    # Handle finish reasons / empty outputs defensively
    text = getattr(resp, "text", "") or ""
    if not text:
        try:
            parts = []
            for c in (resp.candidates or []):
                for p in getattr(c.content, "parts", []) or []:
                    parts.append(getattr(p, "text", "") or "")
            text = "\n".join([t for t in parts if t]).strip()
        except Exception:
            text = ""
    return text.strip()

def _ask_llm(prompt: str, model: str | None = None, json_mode: bool = False, **kw) -> str:
    """
    Unified entrypoint for evaluator calls. Chooses Gemini or OpenAI based on env.
    """
    if EVAL_PROVIDER == "gemini":
        print(f"Using Gemini model: {model or EVAL_MODEL}")
        return _ask_gemini(prompt, RUN_SEED, model or EVAL_MODEL, json_mode=json_mode)
    # default: OpenAI
    print(f"Using OpenAI model: {model or EVAL_MODEL}")
    return _ask_openai(prompt, RUN_SEED, model or EVAL_MODEL, **kw)
#-------------------------------------------------------------------------------
EMBEDDER = get_embedder()

#------------------------------- Faithfulness ----------------------------------

def extract_claims(answer:str) -> List[str]:
    prompt = _p("extract_claims.txt").format(answer=answer)
    txt = _ask_llm(prompt, json_mode=True)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [l.lstrip("1234567890. ").strip() 
                for l in txt.splitlines() if l.strip()]

def claim_supported(claim: str, question:str, context: str) -> bool:
    prompt = _p("check_entailment.txt").format(
        claim = claim, question = question, context = context[:3500]
    )
    resp = _ask_llm(prompt)
    return resp.upper().startswith("Y")

def faithfulness(question: str, answer: str, context:str) -> float:
    claims = extract_claims(answer)
    if not claims:
        return 0.0
    supported = sum(claim_supported(c, question, context) for c in claims)
    return supported/len(claims)

#---------------------------- Answer Relevance ---------------------------------

def _embed(text: str):
    return EMBEDDER.encode(text, normalize_embeddings = True)

def answer_relevance(answer:str, original_q:str, n: int = 3) -> float:
    prompt = _p("gen_questions.txt").format(answer = answer, n= n)
    qlist = _ask_llm(prompt, json_mode=True)
    gen_qs = [l.lstrip("1234567890.- ").strip()
              for l in qlist.splitlines() if l.strip()]
    gen_qs = gen_qs[:n] if len(gen_qs) >= n else gen_qs
    if not gen_qs:
        return 0.0
    
    emb_orig = _embed(original_q)
    sims = []
    for q in gen_qs:
        print(f"Comparing: {original_q} vs generated question: {q}")
        sim = float(util.cos_sim(emb_orig, _embed(q)))
        sims.append(sim)
    sims_01 = [(s+1)/2 for s in sims]
    return sum(sims_01) / len(sims_01)

#-------------------------------- Correctness -----------------------------------

def judge_correctness_grade(question: str,
                           answer: str, 
                           gold_phrases: List[str]) -> dict:
    prompt = _p("correctness.txt").format(
        question = question,
        answer = answer,
        gold = ", ".join(gold_phrases)
    )
    raw = _ask_llm(prompt, json_mode=True)

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = {"grade": "Bad", "notes": "Could not parse JSON"}

    grade = (obj.get("grade") or "").strip()
    if grade not in GRADE_TO_SCORE:
        grade, obj["notes"] = "Bad", "Invalid grade label"
    
    obj["score"] = GRADE_TO_SCORE[grade]
    obj["grade"] = grade
    return obj

def lexical_correct(answer: str, gold_phrases: list[str]) -> bool:
    """True iff every gold phrase appears (case‑insensitively) in the answer."""
    a = answer.lower()
    return all(p.lower() in a for p in gold_phrases)
    
def correctness(question: str,
                answer: str,
                gold_phrases: list[str]) -> tuple[float, str]:
    
    if lexical_correct(answer, gold_phrases):
        return 1.0, "good"

    judged = judge_correctness_grade(question, answer, gold_phrases)
    return judged["score"], judged["grade"]

#----------------------------- Convenience wrapper ------------------------------

def is_idk(ans: str) -> bool:
    return bool(IDK_RE.search(ans))

def score_pair(q: str, ans: str, ctx: str, gold: list[str]) -> dict:
    """
    Convenience function to score a question-answer-context triplet.
    Returns a dictionary with faithfulness and relevance scores.
    """
    if is_idk(ans):
        return {
            "faithfulness": 1.0,     # no claims → fully faithful
            "relevance":    None,    # excluded from averages later
            "correctness":  0.0,
            "correctness_grade": "N/A",
            "coverage": 0            # refused
        }

    corr_num, corr_grade = correctness(q, ans, gold)

    return {
        "faithfulness": faithfulness(q, ans, ctx),
        "relevance": answer_relevance(ans, q),
        "correctness":  corr_num,
        "correctness_grade": corr_grade,
        "coverage": 1
    }
#---------------------------- Debug faithfulness --------------------------------

def debug_faithfulness(question, answer, context):
    claims = extract_claims(answer)
    print("\n--- CLAIM DEBUG ---")
    for c in claims:
        sup = claim_supported(c, question, context)
        print("✓" if sup else "✗", c)
    print("-------------------")

#-------------------------------------------------------------------------------

if __name__ =="__main__":
    q  = "What was the outcome of the Battle of Waterloo?"
    ctx = ("The Battle of Waterloo, fought on 18 June 1815, marked the final "
           "defeat of Napoleon Bonaparte. The battle resulted in a decisive "
           "victory for the Seventh Coalition.")
    gold = ["decisive victory for the Seventh Coalition",
            "final defeat of Napoleon", "1815"]
    ans_good = ("The Battle of Waterloo resulted in a decisive victory for "
                "the Seventh Coalition on 18 June 1815, marking Napoleon’s "
                "final defeat.")
    ans_bad  = ("The Battle of Waterloo was a decisive French victory in 1815 "
                "that ended the Napoleonic Wars.")
    
    debug_faithfulness(q, ans_good, ctx)
    print("GOOD →", score_pair(q, ans_good, ctx, gold))
    debug_faithfulness(q, ans_bad, ctx)
    print("BAD  →", score_pair(q, ans_bad,  ctx, gold))






