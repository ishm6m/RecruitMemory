"""
memory.py, the memory engine. This is the heart of the MemoryAgent track.

Four mechanisms, each its own function:

  1. EXTRACTION   -> extract_and_store()   : pull structured facts from a chat
                                              exchange and store each as a row
                                              with an embedding vector; retire a
                                              stale belief when a fact updates it.
  2. RETRIEVAL    -> retrieve()            : rank a candidate's memories by a
                                              weighted SUM of relevance,
                                              importance and recency; return top-k.
  3. DECAY +      -> decay_and_consolidate(): let importance fade over time,
     CONSOLIDATION                          archive faded memories, and merge
                                            old/low ones into one summary row.
  4. REFLECTION   -> reflect()             : synthesise higher-order INSIGHTS
                                              from the raw facts so understanding
                                              deepens, not just accumulates.
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

# Retrieval ranking = WEIGHTED SUM of three normalised 0..1 signals (Generative
# Agents, Park et al. 2023). A sum, not a product: relevance leads, so a highly
# relevant memory is never buried just for being low-importance, while importance
# and recency still break ties and surface what matters. (Weights tuned on eval.py.)
W_RELEVANCE = 1.0
W_IMPORTANCE = 0.3
W_RECENCY = 0.3


# =====================================================================
# 1. EXTRACTION
# =====================================================================
def extract_and_store(candidate_id, interviewer_msg, source="manual_note"):
    """
    Ask Qwen to read the interviewer's latest message and return NEW candidate
    facts as structured JSON: [{fact, category, importance}]. Each fact is
    embedded and stored as its own memory row, tagged with `source` so the
    record always shows where a fact came from (typed note, live interview,
    or an uploaded resume).

    We extract from the interviewer's message only (not the agent's reply): the
    reply just parrots back recalled facts, which would create duplicates.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "You extract durable facts about a job candidate from an "
                "interview conversation or a candidate document such as a "
                "resume. Return ONLY a JSON array. Each item: "
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
        importance = _clamp_importance(f.get("importance", 5), 5)
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
                continue  # genuine duplicate, already known, skip

        emb_list = vec.tolist()
        db.add_memory(candidate_id, fact_text, category, importance, emb_list, source)
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
def _rank(memories, query, k):
    """
    Score memories by the weighted SUM
        W_RELEVANCE*relevance + W_IMPORTANCE*importance + W_RECENCY*recency
    and return the top-k. Retrieving a memory also 'touches' it (resets its
    recency), so useful memories stay alive longer.
    """
    if not memories:
        return []

    query_vec = np.array(qwen.embed(query))
    now = time.time()

    scored = []
    for m in memories:
        mem_vec = np.array(m["embedding"])
        relevance = max(0.0, _cosine(query_vec, mem_vec))  # 0..1
        importance = m["importance"] / 10.0                # static baseline, 0..1
        recency = _recency_weight(m["last_accessed_at"], now)  # 0..1, decays on age
        # Weighted SUM. Age enters exactly once (recency); importance is the static
        # baseline. (Earlier this was a product that both squared the age term and
        # let importance drown out relevance; eval.py recall improved after the fix.)
        score = W_RELEVANCE * relevance + W_IMPORTANCE * importance + W_RECENCY * recency
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [m for _, m in scored[:k]]

    db.touch_memories([m["id"] for m in top])  # they were just useful
    return top


def retrieve(candidate_id, query, k=TOP_K):
    """Top-k memories for ONE candidate, ranked by relevance+importance+recency."""
    return _rank(db.get_active_memories(candidate_id), query, k)


def retrieve_global(query, k=8):
    """Same ranking over EVERY candidate's active memories in one pool. Each
    result carries the candidate's name so answers can say who a fact belongs to."""
    return _rank(db.get_all_active_memories(), query, k)


def ask_all(question):
    """
    Answer a question across ALL candidates ("who scored best on safety?").
    Recall the globally top-ranked memories, then have Qwen answer citing each
    fact's candidate by name. Query-only: nothing here stores a memory, because
    a question about candidates is not a fact about any one of them.
    """
    recalled = retrieve_global(question)
    lines = "\n".join(f"- {m['candidate_name']}: {m['fact_text']}" for m in recalled) \
            or "(no memories stored yet)"
    prompt = [
        {"role": "system", "content": (
            "You are a hiring assistant for Jabbar Jute Mills. Answer the "
            "interviewer's question using ONLY these remembered facts, gathered "
            "across all candidates. Always name which candidate each point is "
            "about. If the facts don't answer the question, say so plainly.\n"
            f"{lines}"
        )},
        {"role": "user", "content": question},
    ]
    return {
        "reply": qwen.chat(prompt),
        "recalled": [{"candidate": m["candidate_name"], "fact": m["fact_text"]}
                     for m in recalled],
    }


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

    # (a) DECAY / FORGETTING ---------------------------------------
    # Importance stays a STATIC baseline in storage; forgetting is a live function
    # of time-since-last-access. We only ARCHIVE here (an irreversible commitment).
    # We never write the faded value back, so decay can't compound across the many
    # housekeeping runs that happen in a single session (which would wrongly wipe
    # untouched memories after a few messages).
    archived = 0
    for m in db.get_active_memories(candidate_id):
        if _decayed_importance(m, now) < ARCHIVE_THRESHOLD:
            db.archive_memory(m["id"])
            archived += 1

    # (b) CONSOLIDATION --------------------------------------------
    active = db.get_active_memories(candidate_id)
    if len(active) <= CONSOLIDATE_LIMIT:
        return {"consolidated": 0, "archived": archived}

    # pick the oldest + least important to merge (keep the fresh/important ones)
    active.sort(key=lambda m: (m["importance"], m["last_accessed_at"]))
    batch = active[:CONSOLIDATE_BATCH]

    joined = _bullets(batch)
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
    # `content` can come back None (length cutoff, refusal); `or ""` keeps .strip()
    # from crashing. An empty summary must NOT archive the batch behind it (that
    # would silently drop those facts), so bail out and leave them active.
    summary = (qwen.chat(prompt, temperature=0) or "").strip()
    if not summary:
        return {"consolidated": 0, "archived": archived}

    # store the single summary (importance = the max of what it replaced)
    summary_importance = max(m["importance"] for m in batch)
    db.add_memory(
        candidate_id, summary, "summary", summary_importance, qwen.embed(summary),
        source="consolidation",
    )
    for m in batch:
        db.archive_memory(m["id"])

    return {"consolidated": len(batch), "archived": archived}


# =====================================================================
# 4. REFLECTION  (higher-order memory)
# =====================================================================
REFLECT_MIN_FACTS = 3      # need at least this many raw facts to generalise from
INSIGHT_MIN_IMPORTANCE = 7 # insights outrank raw facts so they steer recommendations
REFLECT_EVERY = 4          # auto-reflect once this many *unreflected* raw facts pile up


def _should_reflect(raw_count, insight_count):
    """Gate for auto-reflection: fire only when a fresh batch of raw facts has
    accumulated beyond what the existing insights already cover. Pure -> testable."""
    return raw_count >= REFLECT_MIN_FACTS and raw_count >= (insight_count + 1) * REFLECT_EVERY


def maybe_reflect(candidate_id):
    """
    Reflection that happens *on its own* as a conversation grows, so the agent's
    understanding deepens over time, not only when someone presses the button. Fired
    from the chat loop; reflect() dedups, so an eager trigger can't spam near-identical
    insights.
    """
    active = db.get_active_memories(candidate_id)
    raw = sum(1 for m in active if m["category"] != "insight")
    insights = sum(1 for m in active if m["category"] == "insight")
    if _should_reflect(raw, insights):
        return reflect(candidate_id)
    return {"insights": []}


def reflect(candidate_id):
    """
    Turn many low-level facts into a few higher-order INSIGHTS, the thing that
    makes understanding *deepen* over time instead of just accumulate.

    Reads the candidate's raw memories, asks Qwen to synthesise 1-2 grounded
    trait-level judgements ("consistently safety-conscious", "reliability is her
    standout"), and stores each as a high-importance `insight` memory. Because
    insights are embedded and stored like any memory, they then compete in
    retrieve() and influence future recommendations, so the agent forms opinions it
    was never explicitly told.

    Idempotent-ish: an insight too similar to one already on file is skipped, so
    re-running doesn't pile up duplicates.
    """
    facts = [m for m in db.get_active_memories(candidate_id)
             if m["category"] != "insight"]
    if len(facts) < REFLECT_MIN_FACTS:
        return {"insights": [], "reason": "not enough facts to reflect on yet"}

    joined = _bullets(facts)
    prompt = [
        {"role": "system", "content": (
            "You are a hiring assistant building a deeper understanding of a candidate. "
            "From the raw facts below, infer 1-2 HIGHER-ORDER insights: durable traits "
            "or patterns that no single fact states outright but several together imply. "
            "Each must be grounded in the facts (no speculation about protected "
            "attributes). Return ONLY a JSON array; each item "
            '{"insight": str, "importance": integer 7-10}.'
        )},
        {"role": "user", "content": joined},
    ]
    proposed = _parse_json_array(qwen.chat(prompt, temperature=0.2))

    existing_insights = [m for m in db.get_active_memories(candidate_id)
                         if m["category"] == "insight"]
    stored = []
    for p in proposed:
        text = str(p.get("insight", "")).strip()
        if not text:
            continue
        importance = _clamp_importance(p.get("importance", 8), 8, INSIGHT_MIN_IMPORTANCE)
        vec = np.array(qwen.embed(text))
        match, sim = _most_similar(vec, existing_insights)
        if match and sim >= DUP_THRESHOLD:
            continue  # already inferred this, don't duplicate
        db.add_memory(candidate_id, text, "insight", importance, vec.tolist(),
                      source="reflection")
        existing_insights.append({"id": None, "fact_text": text, "embedding": vec.tolist()})
        stored.append({"insight": text, "importance": importance})
    return {"insights": stored, "from_facts": len(facts)}


def suggest_questions(candidate_id):
    """
    5-8 interview questions (or one small practical task) tailored to what
    memory holds about this candidate. Runs after a resume upload, so each
    question can probe depth on something the document actually claims,
    instead of generic "tell me about yourself" filler.
    """
    facts = db.get_active_memories(candidate_id)
    if not facts:
        return []
    prompt = [
        {"role": "system", "content": (
            "You prepare an interviewer at Jabbar Jute Mills. From the facts "
            "below about one candidate, write 5-8 interview questions (one may "
            "be a small practical task) that probe DEPTH on this candidate's "
            "specific background. Every question must anchor to something "
            "concrete in the facts; no generic questions. Return ONLY a JSON "
            "array of strings."
        )},
        {"role": "user", "content": _bullets(facts)},
    ]
    proposed = _parse_json_array(qwen.chat(prompt, temperature=0.4))
    return [str(q).strip() for q in proposed if str(q).strip()][:8]


def compare(candidate_ids, question):
    """
    Cross-candidate reasoning. For the same question, recall each candidate's
    relevant memories (reusing retrieve(), the same relevance/importance/recency
    ranking that powers chat) and ask Qwen to weigh them against each other and recommend.

    Returns {reply, candidates:[{id,name,role,recalled:[fact_text,...]}]}.
    """
    per, blocks = [], []
    for cid in candidate_ids:
        cand = db.get_candidate(cid)
        if not cand:
            continue
        facts = [m["fact_text"] for m in retrieve(cid, question)]
        per.append({"id": cid, "name": cand["name"], "role": cand["role"], "recalled": facts})
        joined = _bullets(facts, "(no relevant memories yet)")
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
def _clamp_importance(value, default, lo=1, hi=10):
    """Coerce a model-supplied importance into an int in [lo, hi], tolerating junk
    like "high", 7.5, or null without crashing the request that triggered it."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _bullets(items, empty=""):
    """A '- ' bullet list of fact texts. `items` may be memory rows (dicts with
    fact_text) or plain strings; empty input renders as `empty`."""
    texts = [i["fact_text"] if isinstance(i, dict) else i for i in items]
    return "\n".join(f"- {t}" for t in texts) or empty


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
                "correct, or contradict the OLD one, i.e. the same attribute with a "
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
    # recency is the ONLY time term in retrieval ranking now (halves per half-life)
    assert abs(_recency_weight(time.time() - HALF_LIFE_DAYS * 86400, time.time()) - 0.5) < 0.01
    # _most_similar picks the closest row (belief-update path relies on this)
    _rows = [{"embedding": [1, 0]}, {"embedding": [0, 1]}]
    _m, _s = _most_similar(np.array([1, 0]), _rows)
    assert _m is _rows[0] and abs(_s - 1.0) < 1e-9
    assert _parse_json_array('```json\n[{"fact":"x"}]\n```') == [{"fact": "x"}]
    assert _parse_json_array("no json here") == []
    # auto-reflect gate: fires on a full batch of unreflected facts, not before
    assert _should_reflect(4, 0) and not _should_reflect(3, 0)
    assert not _should_reflect(4, 1) and _should_reflect(8, 1)
    # importance parse survives junk the model might return, and stays in range
    assert _clamp_importance("high", 5) == 5 and _clamp_importance(None, 5) == 5
    assert _clamp_importance("11", 5) == 10 and _clamp_importance(0, 5) == 1
    assert _clamp_importance(7.9, 5) == 7 and _clamp_importance("3", 8, 7) == 7
    # bullets renders rows or plain strings, and falls back when empty
    assert _bullets([{"fact_text": "a"}, "b"]) == "- a\n- b"
    assert _bullets([], "(none)") == "(none)"
    print("memory.py self-checks passed")
