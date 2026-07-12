"""
app.py, the FastAPI web server. Wires the memory engine to HTTP endpoints and
serves the single-page frontend.

Endpoints:
  POST   /candidates                 create a candidate
  GET    /candidates                 list candidates
  DELETE /candidates/{id}            delete a candidate and all their memories
  POST   /candidates/{id}/chat       ask (read-only) or note (recall + reply + extract)
  GET    /candidates/{id}/interviews list past interview sessions with transcripts
  POST   /candidates/{id}/reflect    synthesize higher-order insights (mechanism 4)
  POST   /candidates/{id}/simulate   demo time-machine: fast-forward decay
  GET    /candidates/{id}/memories   view stored memories (live decayed strength)
  POST   /seed-demo                  one-click demo candidate
  POST   /compare                    cross-candidate reasoning
  POST   /candidates/{id}/interviews start a consent-gated live interview session
  POST   /interviews/{id}/audio      transcribe a ~6s audio chunk + extract facts
  POST   /interviews/{id}/end        finish the session (flush, decay, reflect)
  POST   /candidates/{id}/notes      typed note (works mid-recording), extraction only
  POST   /candidates/{id}/resume     pre-load memory from a resume/portfolio (PDF/.txt)
  POST   /ask                        query across ALL candidates' memories
"""

import base64
import io
import re
import time

import pypdf
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
    # "note" = learn from this text AND reply (the original behavior).
    # "ask"  = answer from memory only; asking never writes anything.
    mode: str = "note"


class SimulateIn(BaseModel):
    days: int = 14


class CompareIn(BaseModel):
    candidate_ids: list[int]
    question: str


class NewInterview(BaseModel):
    consent_confirmed: bool = False


class NoteIn(BaseModel):
    text: str


class ResumeIn(BaseModel):
    file_b64: str      # the whole file, base64 (same transport style as audio)
    filename: str = ""


class AudioIn(BaseModel):
    audio_b64: str


class AskIn(BaseModel):
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

    # 1. RETRIEVAL: pull only the memories relevant to this message. Each fact
    # carries its source so "where did this come from?" is answerable in chat.
    recalled = memory.retrieve(candidate_id, body.message)
    memory_block = "\n".join(
        f"- [from {m['source']}] {m['fact_text']}" for m in recalled
    ) or "(none yet)"

    # 2. Build the prompt with ONLY the relevant memories injected.
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a hiring assistant for Jabbar Jute Mills, discussing the "
                f"candidate {candidate['name']} ({candidate['role']}). "
                f"Use these remembered facts about them:\n{memory_block}\n"
                "Each fact is prefixed with its source (manual_note = typed by "
                "the recruiter, live_transcript = spoken in a recorded interview, "
                "resume = the candidate's uploaded resume). If asked where "
                "something came from, say so. Answer the interviewer concisely."
            ),
        },
        {"role": "user", "content": body.message},
    ]
    reply = qwen.chat(messages)

    # Ask mode is read-only: recall + answer, nothing stored, no housekeeping.
    if body.mode == "ask":
        return {
            "reply": reply,
            "recalled": [m["fact_text"] for m in recalled],
            "total_active": total_active,
            "new_facts": [],
            "reflected": [],
            "housekeeping": {},
        }

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


@app.post("/candidates/{candidate_id}/notes")
def add_note(candidate_id: int, body: NoteIn):
    """A quick typed note, usable while an interview is recording: the same
    extraction pipeline as everything else, tagged source=manual_note, but no
    chat reply (mid-interview, the recruiter just wants the fact captured)."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    if not body.text.strip():
        raise HTTPException(400, "note text is required")
    return {"new_facts": memory.extract_and_store(candidate_id, body.text)}


# Resume upload limits: a real resume is a few pages; these caps keep one
# request from swallowing a book while never touching a legitimate file.
RESUME_MAX_BYTES = 10 * 1024 * 1024
RESUME_TEXT_CAP = 15000  # chars fed to extraction (several pages of text)


@app.post("/candidates/{candidate_id}/resume")
def upload_resume(candidate_id: int, body: ResumeIn):
    """Pre-load memory from a resume/portfolio (PDF or .txt): extract text,
    feed it through the SAME extraction pipeline as notes and interviews
    (tagged source=resume), then draft tailored interview questions."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    try:
        raw = base64.b64decode(body.file_b64, validate=True)
    except Exception:
        raise HTTPException(400, "file_b64 is not valid base64")
    if len(raw) > RESUME_MAX_BYTES:
        raise HTTPException(400, "file too large (10 MB max)")

    name = body.filename.lower()
    if name.endswith(".pdf"):
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            raise HTTPException(400, "couldn't read that PDF; is the file intact?")
    elif name.endswith(".txt"):
        text = raw.decode("utf-8", errors="replace")
    else:
        raise HTTPException(400, "only .pdf or .txt files are supported")

    text = text.strip()[:RESUME_TEXT_CAP]
    if len(text) < 40:
        raise HTTPException(
            422, "no readable text in that file; a scanned or photographed "
                 "resume needs OCR, which isn't supported, use a text PDF or .txt")

    new_facts = memory.extract_and_store(candidate_id, text, source="resume")
    questions = memory.suggest_questions(candidate_id)
    if questions:
        db.set_questions(candidate_id, questions)
    return {"new_facts": new_facts, "questions": questions}


@app.post("/ask")
def ask(body: AskIn):
    """Query across ALL candidates' memories. Read-only: stores nothing."""
    if not body.question.strip():
        raise HTTPException(400, "a question is required")
    return memory.ask_all(body.question)


# ---- live interview capture ----

# Trailing sentence fragments per interview: an audio chunk often ends
# mid-sentence, and extracting from half a sentence loses the fact. We hold the
# unfinished tail and prepend it to the next chunk, so extraction only ever sees
# whole sentences. The transcript itself always stores every chunk verbatim.
# ponytail: in-process dict; a restart mid-interview drops at most one fragment.
_carry = {}
_CARRY_MAX = 600  # safety valve: extract anyway if no punctuation ever arrives
_SENTENCE_END = (".", "!", "?", "。")

# Each audio chunk begins with ~1.5s of the previous one (so words cut at the
# chunk boundary are transcribed whole in the next chunk). That means a chunk's
# transcript can start by repeating the previous chunk's last words; we remember
# those words per interview and drop the repeats. Word-set matching (not exact
# sequence) because the ASR can render the same overlap audio slightly
# differently in each chunk ("at Adamjee" vs "@Adamjee").
_tail_words = {}     # interview_id -> normalized last words of the previous chunk
_WORD = re.compile(r"[A-Za-z0-9']+")
_OVERLAP_WORDS = 8   # never drop more than ~1.5s of speech


def _strip_overlap(interview_id, text):
    spans = [(m.group().lower(), m.end()) for m in _WORD.finditer(text)]
    prev = set(_tail_words.get(interview_id, []))
    _tail_words[interview_id] = [w for w, _ in spans][-12:]
    drop = 0
    for w, _ in spans[:_OVERLAP_WORDS]:
        if w not in prev:
            break
        drop += 1
    if not drop:
        return text
    return text[spans[drop - 1][1]:].lstrip(" \t,.;:!?、。")


def _split_complete(text):
    """Split text into (complete sentences, trailing fragment)."""
    last = max(text.rfind(p) for p in _SENTENCE_END)
    if last == -1:
        return "", text
    return text[: last + 1], text[last + 1:]


@app.get("/candidates/{candidate_id}/interviews")
def get_interviews(candidate_id: int):
    """Past interview sessions for this candidate, transcripts included, so the
    UI can show the record (and its consent timestamp) after the fact."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    return db.list_interviews(candidate_id)


@app.post("/candidates/{candidate_id}/interviews")
def start_interview(candidate_id: int, body: NewInterview):
    """Consent gate: recording may only begin after the recruiter confirms the
    candidate has been informed. The confirmation timestamp is stored on the
    interview record; no interview row (and hence no audio) exists without it."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    if body.consent_confirmed is not True:
        raise HTTPException(400, "consent confirmation required before recording")
    return db.create_interview(candidate_id)


@app.post("/interviews/{interview_id}/audio")
def interview_audio(interview_id: int, body: AudioIn):
    """Transcribe one ~6s WAV chunk and feed the words into the SAME extraction
    pipeline typed notes use. Returns the chunk's text plus any new facts."""
    iv = db.get_interview(interview_id)
    if not iv:
        raise HTTPException(404, "interview not found")
    if iv["ended_at"] is not None:
        raise HTTPException(409, "interview already ended")

    text = qwen.transcribe(body.audio_b64).strip()
    if text:
        text = _strip_overlap(interview_id, text)
    if not text:
        return {"text": "", "new_facts": []}

    offset = int(time.time() - iv["started_at"])
    db.append_transcript(interview_id, f"[{offset // 60:02d}:{offset % 60:02d}] {text}\n")

    pending = (_carry.pop(interview_id, "") + " " + text).strip()
    complete, fragment = _split_complete(pending)
    if len(fragment) > _CARRY_MAX:
        complete, fragment = pending, ""
    _carry[interview_id] = fragment

    new_facts = memory.extract_and_store(
        iv["candidate_id"], complete, source="live_transcript") if complete else []
    return {"text": text, "new_facts": new_facts}


@app.post("/interviews/{interview_id}/end")
def end_interview(interview_id: int):
    """Finish the session: extract from any held fragment, close the record,
    then run the usual housekeeping (decay + reflection) once."""
    iv = db.get_interview(interview_id)
    if not iv:
        raise HTTPException(404, "interview not found")

    _tail_words.pop(interview_id, None)
    leftover = _carry.pop(interview_id, "").strip()
    new_facts = memory.extract_and_store(
        iv["candidate_id"], leftover, source="live_transcript") if leftover else []

    db.end_interview(interview_id)
    housekeeping = memory.decay_and_consolidate(iv["candidate_id"])
    insights = memory.reflect(iv["candidate_id"]).get("insights", [])

    iv = db.get_interview(interview_id)
    return {
        "id": iv["id"],
        "ended_at": iv["ended_at"],
        "transcript": iv["transcript"],
        "new_facts": new_facts,
        "housekeeping": housekeeping,
        "insights": insights,
    }


# Serve the frontend (static/index.html) at the root URL "/".
app.mount("/", StaticFiles(directory="static", html=True), name="static")
