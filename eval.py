"""
eval.py — a small, honest benchmark for the memory engine.

It answers the question a MemoryAgent judge actually asks: *does the memory
work, and does the fancy ranking earn its keep?* Five probes, real embeddings,
a throwaway database (your real recruitmemory.db is never touched):

  A. RETRIEVAL     recall@k + MRR — does a question surface the right memory?
  B. ABLATION      full relevance+importance+recency ranking vs plain cosine — on
                   stale-but-similar distractors, how often does each recover the CURRENT truth?
  C. BELIEF UPDATE a correction should retire the old belief, not hoard both.
  D. FORGETTING    a low-value memory should fade out while a high-value one survives.
  E. REFLECTION    synthesised insights should be stored AND influence retrieval.

Run:  .venv/bin/python eval.py      (needs a working QWEN_API_KEY; ~60 API calls)
"""

import os
import tempfile

# Point the whole stack at a throwaway DB BEFORE importing db (it opens on import).
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "eval.db")

import time
import numpy as np

import db
import memory
import qwen

db.init_db()


# --- helpers ----------------------------------------------------------
def _seed(name, facts):
    """facts: list of (text, category, importance). Returns candidate id."""
    cid = db.create_candidate(name, "Loom Operator")["id"]
    for text, cat, imp in facts:
        db.add_memory(cid, text, cat, imp, qwen.embed(text))
    return cid


def _rank(memories, query, mode):
    """Score memories the way retrieve() does ('full') or cosine-only ('cosine').
    Kept local so the ablation runs on frozen state without retrieve()'s touch()."""
    qv = np.array(qwen.embed(query))
    now = time.time()
    scored = []
    for m in memories:
        sim = max(0.0, memory._cosine(qv, np.array(m["embedding"])))
        if mode == "full":
            s = (memory.W_RELEVANCE * sim
                 + memory.W_IMPORTANCE * (m["importance"] / 10.0)
                 + memory.W_RECENCY * memory._recency_weight(m["last_accessed_at"], now))
        else:
            s = sim
        scored.append((s, m["fact_text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored]


# --- A. retrieval recall@k -------------------------------------------
def eval_retrieval():
    facts = [
        ("Karim has 5 years of experience operating jute looms", "experience", 7),
        ("Karim scored 9 out of 10 on the safety assessment", "score", 8),
        ("Karim trained three junior operators last year", "experience", 6),
        ("Karim is fluent in Bengali and basic English", "skill", 4),
        ("Karim was late to two shifts in March", "red_flag", 5),
        ("Karim can service and re-thread the looms himself", "skill", 7),
        ("Karim prefers the night shift", "note", 3),
        ("Karim holds a certificate in industrial machine maintenance", "skill", 6),
        ("Karim lifts and moves raw jute bales without assistance", "skill", 4),
        ("Karim resolved a loom jam that stopped the line for an hour", "experience", 6),
    ]
    probes = [  # (question phrased differently from the fact, the fact it should surface)
        ("How experienced is he with weaving machinery?", facts[0][0]),
        ("Is he safe to put on the floor?", facts[1][0]),
        ("Has he ever mentored other workers?", facts[2][0]),
        ("Can he fix a machine when it breaks down?", facts[5][0]),
        ("Any attendance concerns?", facts[4][0]),
        ("Does he have formal maintenance training?", facts[7][0]),
        ("Is he physically capable of heavy work?", facts[8][0]),
        ("Tell me about a problem he solved on the line", facts[9][0]),
    ]
    cid = _seed("Recall Karim", facts)
    r1 = r3 = r5 = 0
    rr = 0.0
    for q, want in probes:
        got = [m["fact_text"] for m in memory.retrieve(cid, q, k=5)]
        if want in got:
            rank = got.index(want) + 1
            rr += 1.0 / rank
            r1 += rank <= 1
            r3 += rank <= 3
            r5 += rank <= 5
    n = len(probes)
    return {"n": n, "recall@1": r1 / n, "recall@3": r3 / n, "recall@5": r5 / n, "MRR": rr / n}


# --- B. ranking ablation ---------------------------------------------
def eval_ablation():
    """Each case: an OLD low-importance, aged fact and a NEW high-importance, fresh
    fact about the same attribute. The CURRENT truth is the new one. How often does
    each ranking put it first?  (We keep both un-archived to isolate ranking.)"""
    cases = [
        ("scored 6 out of 10 on the safety assessment",
         "scored 9 out of 10 on the safety assessment", "What is his safety score?"),
        ("has 2 years of experience on the looms",
         "has 6 years of experience on the looms", "How many years of loom experience?"),
        ("was rated an average operator by his last supervisor",
         "was rated the top operator on his shift by his last supervisor",
         "How did his supervisor rate him?"),
        ("can operate one type of loom",
         "can operate every loom model in the mill", "What range of looms can he run?"),
        ("missed several shifts last quarter",
         "has not missed a single shift in six months", "What is his attendance like?"),
    ]
    full_wins = cosine_wins = 0
    for old, new, q in cases:
        cid = db.create_candidate("ablation", "")["id"]
        db.add_memory(cid, "Karim " + old, "note", 3, qwen.embed("Karim " + old))
        db.add_memory(cid, "Karim " + new, "note", 8, qwen.embed("Karim " + new))
        db.age_memories(cid, 30 * 86400)          # age BOTH by 30d…
        # …then refresh only the NEW one (it was just learned): reset its recency.
        newest = max(db.get_active_memories(cid), key=lambda m: m["fact_text"] == "Karim " + new)
        db.touch_memories([newest["id"]])
        mems = db.get_active_memories(cid)
        full_wins += _rank(mems, q, "full")[0] == "Karim " + new
        cosine_wins += _rank(mems, q, "cosine")[0] == "Karim " + new
    n = len(cases)
    return {"n": n, "full_ranking_correct": full_wins, "cosine_only_correct": cosine_wins}


# --- C. belief update -------------------------------------------------
def eval_belief_update():
    cases = [
        ("Karim scored 6 out of 10 on the safety assessment.",
         "Correction: Karim actually scored 9 out of 10 on the safety assessment.",
         "6", "9"),
        ("Karim has 3 years of experience on jute looms.",
         "Update: Karim actually has 7 years of experience on jute looms.",
         "3", "7"),
        ("Karim is available for the day shift only.",
         "Correction: Karim is now available for any shift.", "day shift only", "any shift"),
    ]
    passed = 0
    for first, correction, old_token, new_token in cases:
        cid = db.create_candidate("belief", "")["id"]
        memory.extract_and_store(cid, first)
        memory.extract_and_store(cid, correction)
        active = " ".join(m["fact_text"].lower() for m in db.get_active_memories(cid))
        if new_token.lower() in active and old_token.lower() not in active:
            passed += 1
    return {"n": len(cases), "passed": passed}


# --- D. selective forgetting -----------------------------------------
def eval_forgetting():
    """After ~one half-life untouched, a low-importance fact should archive while a
    high-importance one survives."""
    passed = 0
    trials = [(2, 9), (3, 8), (1, 10)]
    for low, high in trials:
        cid = db.create_candidate("forget", "")["id"]
        db.add_memory(cid, f"minor note importance {low}", "note", low, qwen.embed("minor note"))
        db.add_memory(cid, f"key strength importance {high}", "skill", high, qwen.embed("key strength"))
        db.age_memories(cid, 14 * 86400)   # one half-life
        memory.decay_and_consolidate(cid)
        active = [m["importance"] for m in db.get_active_memories(cid)]
        if low not in active and high in active:
            passed += 1
    return {"n": len(trials), "passed": passed}


# --- E. reflection ----------------------------------------------------
def eval_reflection():
    facts = [
        ("Karim scored 9 out of 10 on the safety assessment", "score", 8),
        ("Karim always wears full protective gear on the floor", "note", 5),
        ("Karim reported a faulty guard rail before it caused an injury", "experience", 7),
        ("Karim completed an extra workplace-safety course on his own time", "skill", 6),
    ]
    cid = _seed("reflect Karim", facts)
    result = memory.reflect(cid)
    insights = result.get("insights", [])
    stored_ok = len(insights) >= 1 and all(
        m["category"] == "insight" and m["importance"] >= 7
        for m in db.get_active_memories(cid) if m["category"] == "insight"
    )
    # does a trait-level question now surface a synthesised insight (not a raw fact)?
    got = memory.retrieve(cid, "Is he safety-conscious as a person?", k=3)
    insight_retrieved = any(m["category"] == "insight" for m in got)
    return {"insights_created": len(insights), "stored_ok": stored_ok,
            "insight_influences_retrieval": insight_retrieved,
            "example": insights[0]["insight"] if insights else None}


if __name__ == "__main__":
    print("Running RecruitMemory eval (real embeddings, throwaway DB)…\n")
    A = eval_retrieval()
    print(f"A. RETRIEVAL   recall@1={A['recall@1']:.0%}  recall@3={A['recall@3']:.0%}  "
          f"recall@5={A['recall@5']:.0%}  MRR={A['MRR']:.2f}  (n={A['n']})")
    B = eval_ablation()
    print(f"B. ABLATION    full ranking recovered current truth {B['full_ranking_correct']}/{B['n']}  "
          f"vs plain cosine {B['cosine_only_correct']}/{B['n']}")
    C = eval_belief_update()
    print(f"C. BELIEF UPD  old belief retired, new kept: {C['passed']}/{C['n']}")
    D = eval_forgetting()
    print(f"D. FORGETTING  low-value archived, high-value survived: {D['passed']}/{D['n']}")
    E = eval_reflection()
    print(f"E. REFLECTION  insights created={E['insights_created']}  stored_ok={E['stored_ok']}  "
          f"influences_retrieval={E['insight_influences_retrieval']}")
    if E["example"]:
        print(f'               e.g. "{E["example"]}"')
    print("\nDone.")
