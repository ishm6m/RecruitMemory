"""
memory.py — the memory engine. This is the heart of the MemoryAgent track.

Three mechanisms, each its own function:

  1. EXTRACTION   -> extract_and_store()   : pull structured facts from a chat
                                              exchange and store each as a row
                                              with an embedding vector.
  2. RETRIEVAL    -> retrieve()            : cosine-similarity search over a
                                              candidate's memories, weighted by
                                              importance x recency, return top-k.
  3. DECAY +      -> decay_and_consolidate(): shrink importance over time,
     CONSOLIDATION                          archive faded memories, and merge
                                            old/low ones into one summary row.
"""

import json
import time
import math

import numpy as np

import db
import qwen

# --- tuning knobs (all in one place so they're easy to demo/adjust) ---
TOP_K = 5                 # how many memories to inject into the prompt
HALF_LIFE_DAYS = 14       # importance halves every 14 days of no access
ARCHIVE_THRESHOLD = 1.5   # decayed importance below this -> archive it
CONSOLIDATE_LIMIT = 15    # more than this many active memories -> consolidate
CONSOLIDATE_BATCH = 8     # how many old/low memories to merge into one summary
DUP_THRESHOLD = 0.78      # new fact this similar to an existing one -> skip it
                          # (measured: paraphrases ~0.81, distinct facts ~0.55)
SUPERSEDE_LOW = 0.62      # same-topic band: related enough that a new fact might be
                          # UPDATING an old belief (e.g. "scored 6/10" -> "scored 9/10")


# =====================================================================
# 1. EXTRACTION
# =====================================================================
def extract_and_store(candidate_id, interviewer_msg):
    """
    Ask Qwen to read the interviewer's latest message and return NEW candidate
    facts as structured JSON: [{fact, category, importance}]. Each fact is
    embedded and stored as its own memory row.

    We extract from the interviewer's message only (not the agent's reply): the
    reply just parrots back recalled facts, which would create duplicates.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "You extract durable facts about a job candidate from an "
                "interview conversation. Return ONLY a JSON array. Each item: "
                '{"fact": str, "category": one of '
                '["skill","score","red_flag","experience","note"], '
                '"importance": integer 1-10}. '
                "Only include NEW, concrete, candidate-specific facts. "
                "If nothing worth remembering, return []."
            ),
        },
        {"role": "user", "content": interviewer_msg},
    ]
    raw = qwen.chat(prompt, temperature=0)
    facts = _parse_json_array(raw)

    # Load existing memories once (full rows) so we can both skip duplicates and
    # detect when a new fact UPDATES an old belief instead of just adding to it.
    existing = db.get_active_memories(candidate_id)

    stored = []
    for f in facts:
        fact_text = str(f.get("fact", "")).strip()
        if not fact_text:
            continue
        category = f.get("category", "note")
        # clamp importance into 1..10 in case the model returns something odd
        importance = max(1, min(10, int(f.get("importance", 5))))
        vec = np.array(qwen.embed(fact_text))

        # Find the most-similar thing we already know. High cosine means same topic:
        # either a plain duplicate OR a change to an existing belief.
        match, sim = _most_similar(vec, existing)
        replaces = None
        if match and sim >= SUPERSEDE_LOW:
            if match["id"] is not None and _is_update(match["fact_text"], fact_text):
                # BELIEF UPDATE: the agent changes its mind. Retire the stale fact so
                # it can't contradict the new one at retrieval time.
                db.archive_memory(match["id"])
                existing = [m for m in existing if m["id"] != match["id"]]
                replaces = match["fact_text"]
            elif sim >= DUP_THRESHOLD:
                continue  # genuine duplicate — already known, skip

        emb_list = vec.tolist()
        db.add_memory(candidate_id, fact_text, category, importance, emb_list)
        # reflect this new fact in `existing` so later facts in the same batch see it
        existing.append({"id": None, "fact_text": fact_text, "category": category,
                         "importance": importance, "embedding": emb_list})
        item = {"fact": fact_text, "category": category, "importance": importance}
        if replaces:
            item["replaces"] = replaces
        stored.append(item)
    return stored


# =====================================================================
# 2. RETRIEVAL
# =====================================================================
def retrieve(candidate_id, query, k=TOP_K):
    """
    Embed the query, score every active memory by
        cosine_similarity  x  decayed_importance  x  recency
    and return the top-k. Retrieving a memory also 'touches' it (resets its
    recency), so useful memories stay alive longer.
    """
    memories = db.get_active_memories(candidate_id)
    if not memories:
        return []

    query_vec = np.array(qwen.embed(query))
    now = time.time()

    scored = []
    for m in memories:
        mem_vec = np.array(m["embedding"])
        similarity = _cosine(query_vec, mem_vec)
        importance = _decayed_importance(m, now)   # importance faded by age
        recency = _recency_weight(m["last_accessed_at"], now)
        score = similarity * importance * recency
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [m for _, m in scored[:k]]

    db.touch_memories([m["id"] for m in top])  # they were just useful
    return top


# =====================================================================
# 3. DECAY + CONSOLIDATION
# =====================================================================
def decay_and_consolidate(candidate_id):
    """
    Run housekeeping for one candidate:
      (a) DECAY: recompute each memory's importance based on how long since it
          was last accessed; if it has faded below ARCHIVE_THRESHOLD, archive it.
      (b) CONSOLIDATION: if too many active memories remain, take the oldest /
          lowest-importance batch and ask Qwen to compress them into ONE summary
          memory, then archive the originals. This keeps token usage bounded.
    """
    now = time.time()

    # (a) DECAY -----------------------------------------------------
    archived = 0
    for m in db.get_active_memories(candidate_id):
        faded = _decayed_importance(m, now)
        db.update_importance(m["id"], round(faded, 3))
        if faded < ARCHIVE_THRESHOLD:
            db.archive_memory(m["id"])
            archived += 1

    # (b) CONSOLIDATION --------------------------------------------
    active = db.get_active_memories(candidate_id)
    if len(active) <= CONSOLIDATE_LIMIT:
        return {"consolidated": 0, "archived": archived}

    # pick the oldest + least important to merge (keep the fresh/important ones)
    active.sort(key=lambda m: (m["importance"], m["last_accessed_at"]))
    batch = active[:CONSOLIDATE_BATCH]

    joined = "\n".join(f"- {m['fact_text']}" for m in batch)
    prompt = [
        {
            "role": "system",
            "content": (
                "Summarize these candidate facts into ONE concise memory that "
                "preserves the important details. Return plain text, one paragraph."
            ),
        },
        {"role": "user", "content": joined},
    ]
    summary = qwen.chat(prompt, temperature=0).strip()

    # store the single summary (importance = the max of what it replaced)
    summary_importance = max(m["importance"] for m in batch)
    db.add_memory(
        candidate_id, summary, "summary", summary_importance, qwen.embed(summary)
    )
    for m in batch:
        db.archive_memory(m["id"])

    return {"consolidated": len(batch), "archived": archived}


def compare(candidate_ids, question):
    """
    Cross-candidate reasoning. For the same question, recall each candidate's
    relevant memories (reusing retrieve() — the same importance x recency ranking
    that powers chat) and ask Qwen to weigh them against each other and recommend.

    Returns {reply, candidates:[{id,name,role,recalled:[fact_text,...]}]}.
    """
    per, blocks = [], []
    for cid in candidate_ids:
        cand = db.get_candidate(cid)
        if not cand:
            continue
        facts = [m["fact_text"] for m in retrieve(cid, question)]
        per.append({"id": cid, "name": cand["name"], "role": cand["role"], "recalled": facts})
        joined = "\n".join(f"- {t}" for t in facts) or "(no relevant memories yet)"
        blocks.append(f"{cand['name']} ({cand['role'] or 'no role'}):\n{joined}")

    prompt = [
        {"role": "system", "content": (
            "You are a hiring assistant for Jabbar Jute Mills. Compare the candidates "
            "below using ONLY the remembered facts about each. Give a concise, decisive "
            "recommendation for the interviewer's question and name the key trade-offs. "
            "If a candidate has no relevant memories, say so plainly."
        )},
        {"role": "user", "content": "Candidates:\n\n" + "\n\n".join(blocks) +
                                    f"\n\nQuestion: {question}"},
    ]
    return {"reply": qwen.chat(prompt), "candidates": per}


def simulate_days(candidate_id, days):
    """
    Demo time-machine: pretend `days` days have passed with no access, then run
    the REAL decay + consolidation. Lets a judge watch importance fade, stale
    facts archive, and consolidation fire live instead of waiting two weeks.
    ponytail: demo-only; in production decay happens on its own via the wall clock.
    """
    db.age_memories(candidate_id, days * 86400.0)
    return decay_and_consolidate(candidate_id)


# =====================================================================
# helpers
# =====================================================================
def _cosine(a, b):
    """Cosine similarity between two vectors (1 = identical, 0 = unrelated)."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _most_similar(vec, rows):
    """Return (row, cosine) of the existing memory closest to `vec`, or (None, 0.0)."""
    best, best_sim = None, 0.0
    for r in rows:
        s = _cosine(vec, np.array(r["embedding"]))
        if s > best_sim:
            best, best_sim = r, s
    return best, best_sim


def _is_update(old_fact, new_fact):
    """One cheap yes/no LLM check: does `new_fact` correct/contradict `old_fact`
    (same attribute, changed value) rather than just relate to it? This is what
    turns two similar facts into a belief update instead of a duplicate."""
    ans = qwen.chat(
        [
            {"role": "system", "content": (
                "Two facts about the same job candidate. Does the NEW fact update, "
                "correct, or contradict the OLD one — i.e. the same attribute with a "
                "changed value? Answer only YES or NO."
            )},
            {"role": "user", "content": f"OLD: {old_fact}\nNEW: {new_fact}"},
        ],
        temperature=0,
    )
    return ans.strip().upper().startswith("Y")


def _decayed_importance(memory, now):
    """
    Exponential decay: importance halves every HALF_LIFE_DAYS since last access.
    A memory that keeps getting retrieved stays important; a forgotten one fades.
    """
    age_days = (now - memory["last_accessed_at"]) / 86400.0
    decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
    return memory["importance"] * decay


def _recency_weight(last_accessed_at, now):
    """Same decay curve, used as a 0..1 multiplier during retrieval ranking."""
    age_days = (now - last_accessed_at) / 86400.0
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def _parse_json_array(raw):
    """
    Be forgiving about how the model wraps its JSON (it may add ```json fences
    or prose). Extract the first [...] block and parse it; return [] on failure.
    """
    if not raw:
        return []
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


# ponytail: tiny self-check for the pure math (no network). Run: python memory.py
if __name__ == "__main__":
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    # a memory accessed one half-life ago should have ~half its importance
    fake = {"importance": 8, "last_accessed_at": time.time() - HALF_LIFE_DAYS * 86400}
    assert abs(_decayed_importance(fake, time.time()) - 4.0) < 0.05
    # _most_similar picks the closest row (belief-update path relies on this)
    _rows = [{"embedding": [1, 0]}, {"embedding": [0, 1]}]
    _m, _s = _most_similar(np.array([1, 0]), _rows)
    assert _m is _rows[0] and abs(_s - 1.0) < 1e-9
    assert _parse_json_array('```json\n[{"fact":"x"}]\n```') == [{"fact": "x"}]
    assert _parse_json_array("no json here") == []
    print("memory.py self-checks passed")
