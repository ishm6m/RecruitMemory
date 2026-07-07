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

    # Load existing memories once so we can skip duplicates. Without this, the
    # agent re-learns facts it already knows every time its reply restates them.
    existing = [np.array(m["embedding"]) for m in db.get_active_memories(candidate_id)]

    stored = []
    for f in facts:
        fact_text = str(f.get("fact", "")).strip()
        if not fact_text:
            continue
        category = f.get("category", "note")
        # clamp importance into 1..10 in case the model returns something odd
        importance = max(1, min(10, int(f.get("importance", 5))))
        embedding = qwen.embed(fact_text)

        # DEDUP: skip if this fact is near-identical to one we already stored.
        vec = np.array(embedding)
        if any(_cosine(vec, e) >= DUP_THRESHOLD for e in existing):
            continue

        db.add_memory(candidate_id, fact_text, category, importance, embedding)
        existing.append(vec)  # so duplicates *within* this batch are caught too
        stored.append({"fact": fact_text, "category": category, "importance": importance})
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
    for m in db.get_active_memories(candidate_id):
        faded = _decayed_importance(m, now)
        db.update_importance(m["id"], round(faded, 3))
        if faded < ARCHIVE_THRESHOLD:
            db.archive_memory(m["id"])

    # (b) CONSOLIDATION --------------------------------------------
    active = db.get_active_memories(candidate_id)
    if len(active) <= CONSOLIDATE_LIMIT:
        return {"consolidated": 0}

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

    return {"consolidated": len(batch)}


# =====================================================================
# helpers
# =====================================================================
def _cosine(a, b):
    """Cosine similarity between two vectors (1 = identical, 0 = unrelated)."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


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
    assert _parse_json_array('```json\n[{"fact":"x"}]\n```') == [{"fact": "x"}]
    assert _parse_json_array("no json here") == []
    print("memory.py self-checks passed")
