"""
app.py, the FastAPI web server. Wires the memory engine to HTTP endpoints and
serves the single-page frontend.

Endpoints:
  POST   /candidates                 create a candidate
  GET    /candidates                 list candidates
  PUT    /candidates/{id}/stage      move a candidate along the hiring pipeline
  DELETE /candidates/{id}            delete a candidate and all their memories
  POST   /candidates/{id}/chat       ask (read-only) or note (recall + reply + extract)
  GET    /candidates/{id}/interviews list past interview sessions with transcripts
  POST   /candidates/{id}/reflect    synthesize higher-order insights (mechanism 4)
  POST   /candidates/{id}/scorecard/draft  AI-draft a competency scorecard from memory
  PUT    /candidates/{id}/scorecard  save the recruiter-edited scorecard
  PUT    /candidates/{id}/questions  save the recruiter-edited interview questions
  POST   /candidates/{id}/simulate   demo time-machine: fast-forward decay
  GET    /candidates/{id}/memories   view stored memories (live decayed strength)
  POST   /seed-demo                  one-click demo candidate
  POST   /compare                    cross-candidate reasoning
  POST   /candidates/{id}/interviews start a consent-gated live interview session
  POST   /interviews/{id}/audio      transcribe a ~6s audio chunk + extract facts
  POST   /interviews/{id}/end        finish the session (flush, decay, reflect)
  POST   /candidates/{id}/notes      typed note (works mid-recording), extraction only
  POST   /candidates/{id}/resume     pre-load memory from a resume/portfolio (PDF/.txt)
  POST   /bulk/parse                 preview ONE file of a bulk CV batch (stores nothing)
  POST   /bulk/spreadsheet           parse an .xlsx/.csv into candidate rows (stores nothing)
  POST   /bulk/row                   fetch ONE spreadsheet row's resume+portfolio links (stores nothing)
  POST   /bulk/confirm               recruiter-approved row -> real candidate + resume pipeline
  POST   /ask                        query across ALL candidates' memories
"""

import base64
import csv
import difflib
import io
import json
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urlparse, parse_qs

import httpx
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
# The hiring pipeline. A candidate moves left to right; `rejected` is a terminal
# status reachable from anywhere. New candidates start at `applied`.
STAGES = ["applied", "screening", "interview", "offer", "hired", "rejected"]


class NewCandidate(BaseModel):
    name: str
    role: str = ""


class StageIn(BaseModel):
    stage: str


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


class ScorecardIn(BaseModel):
    scorecard: list


class QuestionsIn(BaseModel):
    questions: list[str]


@app.post("/candidates")
def create_candidate(body: NewCandidate):
    return db.create_candidate(body.name, body.role)


@app.get("/candidates")
def get_candidates():
    return db.list_candidates()


@app.put("/candidates/{candidate_id}/stage")
def set_stage(candidate_id: int, body: StageIn):
    """Move a candidate along the hiring pipeline (applied -> ... -> hired/rejected)."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    if body.stage not in STAGES:
        raise HTTPException(400, f"stage must be one of {STAGES}")
    db.set_stage(candidate_id, body.stage)
    return {"ok": True, "stage": body.stage}


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
                "resume = the candidate's uploaded resume, portfolio = their "
                "portfolio site, spreadsheet = a bulk import row). If asked where "
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


def _clean_scorecard(rows):
    """Coerce a client-supplied scorecard into well-formed rows: a named
    competency, a rating clamped to 0-5, notes, and its cited facts. Shared by
    the save endpoint so junk from the browser can never reach the database."""
    clean = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        comp = memory._no_dashes(str(item.get("competency", "")).strip())
        if not comp:
            continue
        clean.append({
            "competency": comp,
            "rating": memory._clamp_importance(item.get("rating", 0), 0, lo=0, hi=5),
            "rationale": memory._no_dashes(str(item.get("rationale", "")).strip()),
            "cited": [str(c) for c in (item.get("cited") or []) if str(c).strip()],
            "ai_suggested": bool(item.get("ai_suggested", False)),
        })
    return clean


@app.post("/candidates/{candidate_id}/scorecard/draft")
def draft_scorecard(candidate_id: int):
    """Draft a competency scorecard from memory (the AI proposes ratings + cited
    facts). Persisted so a reload keeps it; the recruiter then confirms/overrides."""
    cand = db.get_candidate(candidate_id)
    if not cand:
        raise HTTPException(404, "candidate not found")
    sc = memory.draft_scorecard(candidate_id, cand["role"])
    db.set_scorecard(candidate_id, sc)
    return {"scorecard": sc}


@app.put("/candidates/{candidate_id}/scorecard")
def save_scorecard(candidate_id: int, body: ScorecardIn):
    """Save the recruiter-edited scorecard (ratings, notes, added/removed rows)."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    clean = _clean_scorecard(body.scorecard)
    db.set_scorecard(candidate_id, clean)
    return {"scorecard": clean}


@app.post("/candidates/{candidate_id}/simulate")
def simulate(candidate_id: int, body: SimulateIn):
    """Demo control: fast-forward this candidate's memory clock and run decay."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    result = memory.simulate_days(candidate_id, max(1, body.days))
    result["days"] = max(1, body.days)
    return result


@app.put("/candidates/{candidate_id}/questions")
def save_questions(candidate_id: int, body: QuestionsIn):
    """Save the recruiter-edited interview questions (add, remove, reorder, reword).
    The AI drafts them from a resume; this lets the recruiter make them their own."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    qs = [memory._no_dashes(str(q).strip()) for q in body.questions if str(q).strip()][:20]
    db.set_questions(candidate_id, qs)
    return {"questions": qs}


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


# A .docx is a zip of XML; paragraph text lives in <w:t> runs under <w:p>.
# stdlib zipfile + ElementTree read it with no new dependency.
_DOCX_T = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
_DOCX_P = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"


def _docx_text(raw):
    """Extract visible text from a .docx: one line per paragraph, runs joined."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            xml = z.read("word/document.xml")
        root = ET.fromstring(xml)
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        raise HTTPException(400, "couldn't read that .docx; is the file intact?")
    paras = ("".join(t.text or "" for t in p.iter(_DOCX_T)) for p in root.iter(_DOCX_P))
    return "\n".join(p for p in paras if p)


def _pdf_text(raw):
    """Extract text from PDF bytes (shared by file upload and fetched resumes)."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        raise HTTPException(400, "couldn't read that PDF; is the file intact?")


def _file_text(file_b64, filename):
    """Decode an uploaded resume file (PDF, .docx, or .txt) to plain text. One
    parser shared by the single upload and every file of a bulk batch, so the two
    flows can never drift apart. Raises HTTPException with a human reason."""
    try:
        raw = base64.b64decode(file_b64, validate=True)
    except Exception:
        raise HTTPException(400, "file_b64 is not valid base64")
    if len(raw) > RESUME_MAX_BYTES:
        raise HTTPException(400, "file too large (10 MB max)")

    name = filename.lower()
    if name.endswith(".pdf"):
        text = _pdf_text(raw)
    elif name.endswith(".docx"):
        text = _docx_text(raw)
    elif name.endswith(".txt"):
        text = raw.decode("utf-8", errors="replace")
    else:
        raise HTTPException(400, "only .pdf, .docx, or .txt files are supported")

    text = text.strip()[:RESUME_TEXT_CAP]
    if len(text) < 40:
        raise HTTPException(
            422, "no readable text in that file; a scanned or photographed "
                 "resume needs OCR, which isn't supported, use a text PDF or .txt")
    return text


def _preload_sources(candidate_id, resume_text="", portfolio_text="",
                     email="", phone="", resume_failed=False):
    """The pipeline tail shared by every intake path. Runs the SAME extraction
    that notes and interviews use over whatever content we actually have, tagging
    each fact by where it came from, then drafts tailored questions if any real
    facts landed. Contact info and a resume-access-failure note are stored
    verbatim (not guessed at), so a row with a broken Drive link still yields a
    usable profile (partial success is normal, not an error)."""
    new_facts = []
    if len(resume_text.strip()) >= 40:
        new_facts += memory.extract_and_store(candidate_id, resume_text, source="resume")
    if len(portfolio_text.strip()) >= 40:
        new_facts += memory.extract_and_store(candidate_id, portfolio_text, source="portfolio")
    if email.strip():
        memory.store_fact(candidate_id, f"Email: {email.strip()}", "note", 6, "spreadsheet")
    if phone.strip():
        memory.store_fact(candidate_id, f"Phone: {phone.strip()}", "note", 6, "spreadsheet")
    if resume_failed:
        memory.store_fact(candidate_id,
                          "Resume link could not be accessed; check the file's sharing permissions.",
                          "red_flag", 4, "spreadsheet")
    # Questions only make sense once we have substance to ask about; skip drafting
    # them from contact info alone.
    questions = memory.suggest_questions(candidate_id) if new_facts else []
    if questions:
        db.set_questions(candidate_id, questions)
    return {"new_facts": new_facts, "questions": questions}


def _preload_resume(candidate_id, text):
    """Single-file resume upload: the resume-only case of _preload_sources."""
    return _preload_sources(candidate_id, resume_text=text)


@app.post("/candidates/{candidate_id}/resume")
def upload_resume(candidate_id: int, body: ResumeIn):
    """Pre-load memory from a resume/portfolio (PDF or .txt) for ONE existing
    candidate: extract text, run the shared resume pipeline."""
    if not db.get_candidate(candidate_id):
        raise HTTPException(404, "candidate not found")
    return _preload_resume(candidate_id, _file_text(body.file_b64, body.filename))


# ---- bulk CV upload ----
# The browser sends each file of a batch here one at a time; nothing is stored
# until the recruiter confirms a row on the review screen (human-in-the-loop,
# same checkpoint principle as the consent gate). The parsed text rides back to
# the browser and returns on confirm, so the server holds no batch state.

BULK_PREVIEW_SYSTEM = (
    "You are screening one resume for a hiring team. Return ONLY a JSON "
    'object: {"name": str, "role": str, "facts": [str, ...]}. '
    "name = the candidate's full name as written on the resume, or \"\" if "
    "none appears. role = the position they hold or seek, or \"\". "
    "facts = up to 4 short, concrete, decision-relevant facts (experience, "
    "skills, scores). Never invent details the text does not state."
)


def _parse_json_object(raw):
    """Extract the first {...} block from a model reply; {} on failure."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _find_duplicate(name, batch_names=()):
    """Closest match for `name` among existing candidates AND the other names
    in the current batch (id None), so the same fuzzy rule catches spelling
    variations like Rahim/Raheem everywhere. difflib is stdlib.
    ponytail: name similarity only; it will not catch 'R. Uddin' vs the
    full name, compare extracted facts too if that starts to matter."""
    pool = [(c["name"], c["id"]) for c in db.list_candidates()]
    pool += [(n, None) for n in batch_names]
    target = " ".join(name.lower().split())
    best = None
    for cand_name, cand_id in pool:
        ratio = difflib.SequenceMatcher(
            None, target, " ".join(cand_name.lower().split())).ratio()
        if ratio >= 0.85 and (best is None or ratio > best[0]):
            best = (ratio, cand_name, cand_id)
    return {"name": best[1], "id": best[2]} if best else None


class BulkParseIn(BaseModel):
    file_b64: str
    filename: str = ""
    batch_names: list[str] = []   # names already detected in this batch


class BulkConfirmIn(BaseModel):
    name: str
    role: str = ""
    text: str = ""             # resume text (may be empty for a spreadsheet row)
    email: str = ""            # spreadsheet-only: contact fields + fetch outcomes
    phone: str = ""
    portfolio_text: str = ""
    resume_status: str = ""    # "no_access"/"invalid" -> store a follow-up note


class SpreadsheetIn(BaseModel):
    file_b64: str
    filename: str = ""


class BulkRowIn(BaseModel):
    name: str = ""
    phone: str = ""
    email: str = ""
    resume_url: str = ""
    portfolio_url: str = ""
    batch_names: list[str] = []   # names already detected in this batch


@app.post("/bulk/parse")
def bulk_parse(body: BulkParseIn):
    """Preview ONE file of a bulk batch: extract its text with the shared
    parser, ask the model who the candidate is, and flag likely duplicates.
    Stores NOTHING; only /bulk/confirm writes."""
    text = _file_text(body.file_b64, body.filename)
    d = _parse_json_object(qwen.chat(
        [{"role": "system", "content": BULK_PREVIEW_SYSTEM},
         {"role": "user", "content": text}], temperature=0))

    name = memory._no_dashes(str(d.get("name", "")).strip())
    name_guessed = False
    if not name:
        # fall back to the filename so the row is still reviewable
        stem = os.path.splitext(os.path.basename(body.filename))[0]
        name = re.sub(r"[_.-]+", " ", stem).strip().title() or "Unknown"
        name_guessed = True

    facts = [memory._no_dashes(str(f).strip()) for f in d.get("facts", [])
             if str(f).strip()][:4]
    return {
        "name": name,
        "name_guessed": name_guessed,
        "role": memory._no_dashes(str(d.get("role", "")).strip()),
        "facts": facts,
        # id is None when the match is another CV in this batch, not a stored candidate
        "duplicate_of": _find_duplicate(name, body.batch_names),
        "text": text,
    }


@app.post("/bulk/confirm")
def bulk_confirm(body: BulkConfirmIn):
    """The recruiter approved one reviewed row: create the candidate and run the
    SAME resume pipeline the single upload uses. Handles both a CV file (resume
    text) and a spreadsheet row (contact info + fetched resume/portfolio, any of
    which may have failed). A row needs at least ONE of: resume text, portfolio
    text, or contact info, so a candidate with a broken Drive link is still
    confirmable (partial success is normal), just with a follow-up note."""
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "a candidate name is required")
    resume_text = body.text.strip()
    portfolio_text = body.portfolio_text.strip()
    has_resume = len(resume_text) >= 40
    has_portfolio = len(portfolio_text) >= 40
    has_contact = bool(body.email.strip() or body.phone.strip())
    if not (has_resume or has_portfolio or has_contact):
        raise HTTPException(422, "nothing to store: no resume text, portfolio text, or contact info")
    cand = db.create_candidate(name, body.role.strip())
    result = _preload_sources(
        cand["id"],
        resume_text=resume_text, portfolio_text=portfolio_text,
        email=body.email, phone=body.phone,
        resume_failed=body.resume_status in ("no_access", "invalid"),
    )
    return {**cand, **result}


# ---- spreadsheet import (name/contact + linked resume/portfolio fetching) ----
# A recruiter uploads an .xlsx/.csv of applicants with links to their resume
# (usually Google Drive) and portfolio. We parse the rows, detect the columns
# without demanding exact headers, and best-effort fetch each link. Every fetch
# CAN fail, so each returns an explicit status the review screen surfaces; we
# never present a failed fetch as if it succeeded, and never extract facts from a
# login wall. Nothing is stored until the recruiter confirms a row (same
# human-in-the-loop checkpoint as the CV batch).

# Column detection: match normalized headers against keyword sets. Fields are
# resolved in this order, and a column is claimed by the first field that wants
# it, so "resume drive link" goes to the resume field before portfolio can grab
# it. A header matching nothing (a bare "link"/"url") is left unmapped rather
# than guessed. ponytail: substring keyword match; good enough for real headers.
_COL_KEYS = {
    "email": ("email", "mail"),
    "phone": ("phone", "mobile", "cell", "whatsapp", "contactnumber", "tel"),
    "name": ("name", "candidate", "applicant"),
    "resume_url": ("resume", "cv", "curriculum", "drive"),
    "portfolio_url": ("portfolio", "website", "personalsite", "personalweb", "behance", "dribbble"),
}


def _norm_header(h):
    return re.sub(r"[^a-z0-9]", "", str(h).lower())


def _detect_columns(header):
    """Map each field to a column index (or None) by keyword. Pure -> testable."""
    norm = [_norm_header(h) for h in header]
    used, cols = set(), {}
    for field, keys in _COL_KEYS.items():
        cols[field] = None
        for i, h in enumerate(norm):
            if i not in used and any(k in h for k in keys):
                cols[field] = i
                used.add(i)
                break
    return cols


# --- .xlsx / .csv parsing (stdlib only: an .xlsx is a zip of XML) ---
_XLSX = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col_index(ref):
    """Cell ref 'B3' -> zero-based column index 1 (needed because xlsx omits
    empty cells, so we place each cell by its letter to keep columns aligned)."""
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - 64)
    return idx - 1


def _xlsx_rows(raw):
    """Read the first worksheet of an .xlsx into a list of string rows."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            shared = []
            if "xl/sharedStrings.xml" in names:
                sroot = ET.fromstring(z.read("xl/sharedStrings.xml"))
                shared = ["".join(t.text or "" for t in si.iter(f"{_XLSX}t"))
                          for si in sroot.findall(f"{_XLSX}si")]
            sheets = sorted(n for n in names
                            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
            wroot = ET.fromstring(z.read(sheets[0]))
    except (zipfile.BadZipFile, KeyError, ET.ParseError, IndexError, ValueError):
        raise HTTPException(400, "couldn't read that spreadsheet; is the .xlsx file intact?")

    rows = []
    for row in wroot.iter(f"{_XLSX}row"):
        cells, maxc = {}, -1
        for i, c in enumerate(row.findall(f"{_XLSX}c")):
            ci = _col_index(c.get("r")) if c.get("r") else i
            t = c.get("t")
            if t == "inlineStr":
                is_ = c.find(f"{_XLSX}is")
                val = "".join(x.text or "" for x in is_.iter(f"{_XLSX}t")) if is_ is not None else ""
            else:
                v = c.find(f"{_XLSX}v")
                text = v.text if v is not None else None
                val = (shared[int(text)] if t == "s" and text is not None else (text or ""))
            cells[ci] = val
            maxc = max(maxc, ci)
        rows.append([str(cells.get(i, "")).strip() for i in range(maxc + 1)])
    return rows


def _csv_rows(raw):
    """Read CSV bytes into a list of string rows (BOM-tolerant)."""
    text = raw.decode("utf-8-sig", errors="replace")
    return [[(c or "").strip() for c in r] for r in csv.reader(io.StringIO(text))]


def _spreadsheet_rows(raw, filename):
    name = filename.lower()
    if name.endswith(".xlsx"):
        rows = _xlsx_rows(raw)
    elif name.endswith(".csv"):
        rows = _csv_rows(raw)
    else:
        raise HTTPException(400, "only .xlsx or .csv spreadsheets are supported")
    return [r for r in rows if any(c for c in r)]   # drop fully blank rows


# --- link fetching, with explicit success/failure states ---
# Every fetch returns (text, status). status is one of:
#   "ok"        got real content we could read
#   "no_access" reached something, but it's a login/permission/preview page or an
#               unreadable/empty response (e.g. a private Drive link) - NOT stored
#   "invalid"   the URL is malformed or unreachable
#   "none"      no URL was provided for this row
FETCH_TIMEOUT = 15.0
FETCH_MAX_BYTES = RESUME_MAX_BYTES
_UA = "Mozilla/5.0 (compatible; RecruitMemory/1.0)"
_DRIVE_ID = re.compile(r"/d/([A-Za-z0-9_-]{10,})")   # .../file/d/<id>/view


def _http_get(url):
    """One GET, following redirects. The single network seam (tests stub this)."""
    with httpx.Client(follow_redirects=True, timeout=FETCH_TIMEOUT,
                      headers={"User-Agent": _UA}) as c:
        resp = c.get(url)
    resp.raise_for_status()
    return resp


def _drive_file_id(url):
    """Extract a Google Drive/Docs file id from a share URL, or None."""
    try:
        u = urlparse(url)
    except ValueError:
        return None
    if "drive.google.com" not in (u.hostname or "") and "docs.google.com" not in (u.hostname or ""):
        return None
    m = _DRIVE_ID.search(u.path)
    if m:
        return m.group(1)
    return parse_qs(u.query).get("id", [None])[0]


def _sniff(data):
    """Classify fetched bytes by their signature: 'pdf', 'zip' (.docx), 'html',
    or 'other'. This is how we tell a real document from a login page that a
    private Drive link returns with HTTP 200."""
    if data[:4] == b"%PDF":
        return "pdf"
    if data[:2] == b"PK":
        return "zip"
    low = data[:512].lstrip().lower()
    if low.startswith(b"<!doctype html") or b"<html" in low or b"<head" in low:
        return "html"
    return "other"


def _fetch_resume(url):
    """Fetch a resume link (usually Google Drive) and return (text, status).
    A Drive link is converted to its direct-download form first. HTML back means
    a permission wall or preview page, not the file, so we never extract from it.
    ponytail: handles the common small-public-file case; a very large public file
    that returns Drive's virus-scan interstitial reads as no_access."""
    url = (url or "").strip()
    if not url:
        return "", "none"
    fid = _drive_file_id(url)
    target = (f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
              if fid else url)
    try:
        resp = _http_get(target)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        return "", ("no_access" if code in (401, 403) else "invalid")
    except Exception:
        return "", "invalid"

    data = resp.content[:FETCH_MAX_BYTES]
    kind = _sniff(data)
    if kind == "html" or "text/html" in resp.headers.get("content-type", "").lower():
        return "", "no_access"
    try:
        if kind == "pdf":
            text = _pdf_text(data)
        elif kind == "zip":
            text = _docx_text(data)
        else:
            text = data.decode("utf-8", errors="replace")
    except HTTPException:
        return "", "no_access"   # bytes came back but weren't a readable document
    text = text.strip()[:RESUME_TEXT_CAP]
    return (text, "ok") if len(text) >= 40 else ("", "no_access")


class _TextExtractor(HTMLParser):
    """Collect visible text, skipping script/style/nav/boilerplate."""
    _SKIP = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            t = data.strip()
            if t:
                self.parts.append(t)


def _visible_text(html):
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    return re.sub(r"[ \t]+", " ", " ".join(p.parts)).strip()


def _fetch_portfolio(url):
    """Fetch a portfolio page and return (visible_text, status). Anything that
    isn't a reachable page with real text (broken link, 'site not found', an
    empty page) is reported as no_access rather than extracted as garbage."""
    url = (url or "").strip()
    if not url:
        return "", "none"
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url          # tolerate a bare "jane.dev"
    try:
        resp = _http_get(url)
    except Exception:
        return "", "no_access"          # unreachable, 404 "site not found", timeout
    data = resp.content[:FETCH_MAX_BYTES]
    text = _visible_text(data.decode("utf-8", errors="replace"))[:RESUME_TEXT_CAP]
    return (text, "ok") if len(text) >= 50 else ("", "no_access")


@app.post("/bulk/spreadsheet")
def bulk_spreadsheet(body: SpreadsheetIn):
    """Parse an uploaded .xlsx/.csv into candidate rows with detected columns.
    Stores nothing and fetches nothing; the browser then processes each row."""
    try:
        raw = base64.b64decode(body.file_b64, validate=True)
    except Exception:
        raise HTTPException(400, "file_b64 is not valid base64")
    if len(raw) > RESUME_MAX_BYTES:
        raise HTTPException(400, "file too large (10 MB max)")
    rows = _spreadsheet_rows(raw, body.filename)
    if len(rows) < 2:
        raise HTTPException(422, "no data rows found under the header row")
    header, cols = rows[0], _detect_columns(rows[0])
    out = []
    for r in rows[1:]:
        rec = {field: (r[idx].strip() if idx is not None and idx < len(r) else "")
               for field, idx in cols.items()}
        if any(rec.values()):
            out.append(rec)
    return {"rows": out, "unmapped": [f for f, idx in cols.items() if idx is None]}


@app.post("/bulk/row")
def bulk_row(body: BulkRowIn):
    """Process ONE spreadsheet row: fetch its resume + portfolio links, extract a
    preview, and report exactly what each fetch did. Stores nothing (like
    /bulk/parse); the real extraction runs on /bulk/confirm."""
    resume_text, resume_status = _fetch_resume(body.resume_url)
    portfolio_text, portfolio_status = _fetch_portfolio(body.portfolio_url)

    name = memory._no_dashes(body.name.strip())
    role, facts = "", []
    combined = "\n\n".join(t for t in (resume_text, portfolio_text) if t).strip()
    if combined:
        d = _parse_json_object(qwen.chat(
            [{"role": "system", "content": BULK_PREVIEW_SYSTEM},
             {"role": "user", "content": combined}], temperature=0))
        role = memory._no_dashes(str(d.get("role", "")).strip())
        facts = [memory._no_dashes(str(f).strip()) for f in d.get("facts", [])
                 if str(f).strip()][:4]
        if not name:
            name = memory._no_dashes(str(d.get("name", "")).strip())

    return {
        "name": name,
        "role": role,
        "facts": facts,
        "email": body.email.strip(),
        "phone": body.phone.strip(),
        "resume_status": resume_status,
        "portfolio_status": portfolio_status,
        "resume_text": resume_text,          # ride back for /bulk/confirm
        "portfolio_text": portfolio_text,
        "duplicate_of": _find_duplicate(name, body.batch_names) if name else None,
    }


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
