"""
app.py, the FastAPI web server. Wires the memory engine to HTTP endpoints and
serves the single-page frontend.

Endpoints:
  POST   /candidates                 create a candidate
  GET    /candidates                 list candidates
  DELETE /candidates/{id}            delete a candidate and all their memories
  POST   /candidates/{id}/chat       chat about a candidate (recall + reply + extract)
  POST   /candidates/{id}/reflect    synthesize higher-order insights (mechanism 4)
  POST   /candidates/{id}/simulate   demo time-machine: fast-forward decay
  GET    /candidates/{id}/memories   view stored memories (live decayed strength)
  POST   /seed-demo                  one-click demo candidate
  POST   /compare                    cross-candidate reasoning
"""

import time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import memory
import qwen

app = FastAPI(title="RecruitMemory")

db.init_db()  # create tables on startup


# ---- request body shapes ----
class NewCandidate(BaseModel):
    name: str
    role: str = ""


class ChatIn(BaseModel):
    message: str


class SimulateIn(BaseModel):
    days: int = 14


class CompareIn(BaseModel):
    candidate_ids: list[int]
    question: str


@app.post("/candidates")
def create_candidate(body: NewCandidate):
    return db.create_candidate(body.name, body.role)


@app.get("/candidates")
def get_candidates():
    return db.list_candidates()


@app.delete("/candidates/{candidate_id}")
def remove_candidate(candidate_id: int):
    """Delete a candidate and all their memories (keeps the demo/rehearsals clean)."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    db.delete_candidate(candidate_id)
    return {"ok": True}


@app.post("/candidates/{candidate_id}/chat")
def chat(candidate_id: int, body: ChatIn):
    candidate = db.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(404, "candidate not found")

    # Size of the memory pool retrieval will search, so the UI can show
    # "recalled 5 of N", making the "cost stays flat as history grows" claim visible.
    # COUNT only: no need to load + JSON-decode every embedding just to size the pool.
    total_active = db.count_active_memories(candidate_id)

    # 1. RETRIEVAL: pull only the memories relevant to this message.
    recalled = memory.retrieve(candidate_id, body.message)
    memory_block = memory._bullets(recalled, "(none yet)")

    # 2. Build the prompt with ONLY the relevant memories injected.
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a hiring assistant for Jabbar Jute Mills, discussing the "
                f"candidate {candidate['name']} ({candidate['role']}). "
                f"Use these remembered facts about them:\n{memory_block}\n"
                "Answer the interviewer concisely."
            ),
        },
        {"role": "user", "content": body.message},
    ]
    reply = qwen.chat(messages)

    # 3. EXTRACTION: learn new facts from this exchange.
    new_facts = memory.extract_and_store(candidate_id, body.message)

    # 4. DECAY + CONSOLIDATION: housekeeping so memory stays bounded.
    housekeeping = memory.decay_and_consolidate(candidate_id)

    # 5. REFLECTION (automatic): once enough raw facts accrue, the agent deepens its
    #    understanding on its own, forming higher-order insights mid-conversation,
    #    no button press required.
    reflected = memory.maybe_reflect(candidate_id).get("insights", [])

    return {
        "reply": reply,
        "recalled": [m["fact_text"] for m in recalled],
        "total_active": total_active,   # pool size retrieval ranked (for the "5 of N" proof)
        "new_facts": new_facts,
        "reflected": reflected,         # insights auto-synthesised this turn (if any)
        "housekeeping": housekeeping,
    }


# A ready-made candidate so presenters don't hand-type facts on stage. She's the
# Compare partner in DEMO.md; facts go through the real extraction pipeline.
DEMO_CANDIDATE = {
    "name": "Salma (demo)",
    "role": "Loom Operator",
    "facts": [
        "Salma has 2 years of experience on jute looms but scored 10 out of 10 on quality control.",
        "Salma is extremely reliable and has never missed a shift.",
    ],
}


@app.post("/seed-demo")
def seed_demo():
    """One-click demo candidate. Idempotent: reuse the existing one if already seeded."""
    for c in db.list_candidates():
        if c["name"] == DEMO_CANDIDATE["name"]:
            return c
    cand = db.create_candidate(DEMO_CANDIDATE["name"], DEMO_CANDIDATE["role"])
    for msg in DEMO_CANDIDATE["facts"]:
        memory.extract_and_store(cand["id"], msg)
    return cand


@app.post("/compare")
def compare(body: CompareIn):
    """Cross-candidate reasoning: recall each candidate's memories and weigh them."""
    ids = [i for i in body.candidate_ids if db.get_candidate(i)]
    if len(ids) < 2:
        raise HTTPException(400, "pick at least two existing candidates")
    if not body.question.strip():
        raise HTTPException(400, "a question is required")
    return memory.compare(ids, body.question)


@app.post("/candidates/{candidate_id}/reflect")
def reflect(candidate_id: int):
    """Synthesize higher-order insights from this candidate's raw memories."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    return memory.reflect(candidate_id)


@app.post("/candidates/{candidate_id}/simulate")
def simulate(candidate_id: int, body: SimulateIn):
    """Demo control: fast-forward this candidate's memory clock and run decay."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    result = memory.simulate_days(candidate_id, max(1, body.days))
    result["days"] = max(1, body.days)
    return result


@app.get("/candidates/{candidate_id}/memories")
def get_memories(candidate_id: int):
    # includes archived=0 only; strip the big embedding vector from the response.
    # `importance` here is the LIVE decayed strength (baseline faded by time since
    # last access), so the UI bars reflect true current strength and shrink when a
    # judge advances the clock. `base_importance` keeps the static value for reference.
    now = time.time()
    mems = db.get_active_memories(candidate_id)
    for m in mems:
        m.pop("embedding", None)
        m["base_importance"] = m["importance"]
        m["importance"] = round(memory._decayed_importance(m, now), 3)
    return mems


# Serve the frontend (static/index.html) at the root URL "/".
app.mount("/", StaticFiles(directory="static", html=True), name="static")
