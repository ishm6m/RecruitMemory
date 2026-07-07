"""
app.py — the FastAPI web server. Wires the memory engine to HTTP endpoints and
serves the single-page frontend.

Endpoints:
  POST /candidates                 create a candidate
  GET  /candidates                 list candidates
  POST /candidates/{id}/chat       chat about a candidate (recall + reply + extract)
  GET  /candidates/{id}/memories   view stored memories (for the demo/debugging)
"""

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


@app.post("/candidates")
def create_candidate(body: NewCandidate):
    return db.create_candidate(body.name, body.role)


@app.get("/candidates")
def get_candidates():
    return db.list_candidates()


@app.post("/candidates/{candidate_id}/chat")
def chat(candidate_id: int, body: ChatIn):
    candidate = db.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(404, "candidate not found")

    # 1. RETRIEVAL: pull only the memories relevant to this message.
    recalled = memory.retrieve(candidate_id, body.message)
    memory_block = "\n".join(f"- {m['fact_text']}" for m in recalled) or "(none yet)"

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

    return {
        "reply": reply,
        "recalled": [m["fact_text"] for m in recalled],
        "new_facts": new_facts,
        "housekeeping": housekeeping,
    }


@app.get("/candidates/{candidate_id}/memories")
def get_memories(candidate_id: int):
    # includes archived=0 only; strip the big embedding vector from the response
    mems = db.get_active_memories(candidate_id)
    for m in mems:
        m.pop("embedding", None)
    return mems


# Serve the frontend (static/index.html) at the root URL "/".
app.mount("/", StaticFiles(directory="static", html=True), name="static")
