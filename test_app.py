"""
test_app.py, a fast OFFLINE test for the HTTP + storage wiring.

No network: qwen.chat / qwen.embed are stubbed with deterministic fakes, and the
whole stack runs against a throwaway SQLite file (your real recruitmemory.db is
never touched). This covers the endpoint plumbing and the db layer that eval.py
(which needs a live QWEN_API_KEY and ~60 API calls) does not.

Run:  .venv/bin/python test_app.py      (or: pytest test_app.py, if installed)
"""

import os
import json
import hashlib
import tempfile

# Point the stack at a throwaway DB and dummy creds BEFORE importing anything that
# opens a connection or builds the OpenAI client (both happen at import time).
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ.setdefault("QWEN_API_KEY", "test-key")
os.environ.setdefault("QWEN_BASE_URL", "http://localhost:9/none")

from fastapi.testclient import TestClient

import qwen
import db
from app import app


# --- deterministic stubs (no network) --------------------------------
def _fake_embed(text):
    """Signed pseudo-embedding: identical text -> cosine 1.0, distinct text -> ~0,
    so the dedup/supersede thresholds behave predictably without a real model."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [(b - 128) / 128.0 for b in h[:16]]


def _fake_chat(messages, temperature=0.3):
    """Route by the system prompt so each mechanism gets a plausible fake reply."""
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if "extract durable facts" in system:
        # one fact = the interviewer message verbatim, so a repeat dedups
        return json.dumps([{"fact": user, "category": "skill", "importance": 7}])
    if "Summarize these candidate facts" in system:
        return "Consolidated summary of older facts."
    if "HIGHER-ORDER insights" in system:
        return json.dumps([{"insight": "Reliable and safety-conscious.", "importance": 8}])
    if "Does the NEW fact update" in system:
        return "NO"
    if "interview questions" in system:
        return json.dumps(["How did you maintain the looms?", "Describe a quality check you ran."])
    if "structured interview scorecard" in system:
        # cite the first real fact verbatim plus one invented fact that the
        # citation-integrity filter must drop; a 0-rated row cites nothing
        facts = [l[2:] for l in user.splitlines() if l.startswith("- ")]
        real = facts[0] if facts else ""
        return json.dumps([
            {"competency": "Loom operation", "rating": 4, "rationale": "hands-on depth",
             "cited": [real, "A fact that was never stored"]},
            {"competency": "Safety", "rating": 0, "rationale": "no evidence", "cited": []},
        ])
    if "screening one resume" in system:
        # name = the resume's first line if it looks like a header (short),
        # so tests steer the detected name; a long first line means no name
        first = user.strip().splitlines()[0].strip()
        return json.dumps({"name": first if len(first) <= 40 else "",
                           "role": "Loom Operator",
                           "facts": ["Worked 5 years on jute looms."]})
    if "Compare the candidates" in system:
        return "Candidate comparison verdict."
    if "gathered across all candidates" in system:
        return "Cross-candidate answer naming names."
    return "Noted."


# Tests preload this list; each audio POST pops the next "transcript".
_fake_transcripts = []


def _fake_transcribe(audio_b64):
    return _fake_transcripts.pop(0) if _fake_transcripts else ""


qwen.chat = _fake_chat
qwen.embed = _fake_embed
qwen.transcribe = _fake_transcribe


# --- fake network for spreadsheet link fetching (no real HTTP) ---------
# app._http_get is the single seam every fetch goes through; we route by the id
# baked into the URL so each Drive/portfolio case is deterministic and offline.
import httpx
import app as _app


class _FakeResp:
    def __init__(self, content, ctype="application/octet-stream", status=200):
        self.content, self.status_code = content, status
        self.headers = {"content-type": ctype}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)


def _fake_http_get(url):
    if "PUBLICFILEID" in url:       # a readable public resume (plain-text branch)
        return _FakeResp(b"Rahim has five years of loom maintenance and quality control experience.")
    if "PRIVATEFILEID" in url:      # a private Drive link returns a login HTML page
        return _FakeResp(b"<!DOCTYPE html><html><body>Sign in to continue</body></html>", "text/html")
    if "FORBIDFILEID" in url:       # permission denied by status code
        return _FakeResp(b"", "text/html", status=403)
    if "good-folio" in url:         # a real portfolio page
        return _FakeResp(b"<html><body><nav>Home About</nav><h1>Nadia</h1>"
                         b"<p>Product designer, six years, brand systems and design ops.</p>"
                         b"<script>track()</script></body></html>", "text/html")
    raise httpx.ConnectError("unreachable")   # malformed/dead link


_app._http_get = _fake_http_get

client = TestClient(app)


# --- endpoint tests ---------------------------------------------------
def test_create_and_list():
    r = client.post("/candidates", json={"name": "Karim", "role": "Loom Operator"})
    assert r.status_code == 200, r.text
    cand = r.json()
    assert cand["id"] and cand["name"] == "Karim"
    assert "Karim" in [c["name"] for c in client.get("/candidates").json()]


def test_chat_stores_then_dedups():
    cid = client.post("/candidates", json={"name": "Rina"}).json()["id"]
    msg = "Rina has 5 years on jute looms and scored 9 on safety."
    body = client.post(f"/candidates/{cid}/chat", json={"message": msg}).json()
    for key in ("reply", "recalled", "total_active", "new_facts", "reflected", "housekeeping"):
        assert key in body, f"missing {key}"
    assert len(body["new_facts"]) == 1              # stub extracted one fact
    again = client.post(f"/candidates/{cid}/chat", json={"message": msg}).json()
    assert again["new_facts"] == []                 # identical fact deduped
    mems = client.get(f"/candidates/{cid}/memories").json()
    assert len(mems) == 1
    m = mems[0]
    assert "embedding" not in m and "base_importance" in m   # embedding stripped
    assert m["importance"] <= m["base_importance"]           # live decayed <= baseline


def test_chat_404_for_unknown_candidate():
    assert client.post("/candidates/999999/chat", json={"message": "hi"}).status_code == 404


def test_pipeline_stage_defaults_and_moves():
    cand = client.post("/candidates", json={"name": "Pipeline"}).json()
    assert cand["stage"] == "applied"               # new candidates start at applied
    cid = cand["id"]
    # move it along the pipeline
    r = client.put(f"/candidates/{cid}/stage", json={"stage": "interview"})
    assert r.status_code == 200 and r.json()["stage"] == "interview"
    listed = [c for c in client.get("/candidates").json() if c["id"] == cid][0]
    assert listed["stage"] == "interview"           # persisted, visible in the list
    # bad stage rejected, unknown candidate 404
    assert client.put(f"/candidates/{cid}/stage", json={"stage": "hired-ish"}).status_code == 400
    assert client.put("/candidates/999999/stage", json={"stage": "hired"}).status_code == 404


def test_compare_needs_two_candidates():
    a = client.post("/candidates", json={"name": "CompareA"}).json()["id"]
    assert client.post("/compare", json={"candidate_ids": [a], "question": "best?"}).status_code == 400
    b = client.post("/candidates", json={"name": "CompareB"}).json()["id"]
    ok = client.post("/compare", json={"candidate_ids": [a, b], "question": "best?"})
    assert ok.status_code == 200 and ok.json()["reply"]
    assert len(ok.json()["candidates"]) == 2


def test_seed_demo_is_idempotent():
    first = client.post("/seed-demo").json()
    second = client.post("/seed-demo").json()
    assert first["id"] == second["id"]              # reused, not duplicated
    dupes = sum(c["name"] == first["name"] for c in client.get("/candidates").json())
    assert dupes == 1


def test_simulate_ages_and_decays():
    cid = client.post("/candidates", json={"name": "Aged"}).json()["id"]
    client.post(f"/candidates/{cid}/chat", json={"message": "Aged is very reliable."})
    before = client.get(f"/candidates/{cid}/memories").json()[0]["importance"]
    r = client.post(f"/candidates/{cid}/simulate", json={"days": 28})
    assert r.status_code == 200 and r.json()["days"] == 28
    after = client.get(f"/candidates/{cid}/memories").json()
    assert after and after[0]["importance"] < before   # decayed with the clock


def test_delete_cascades_memories():
    cid = client.post("/candidates", json={"name": "Gone"}).json()["id"]
    client.post(f"/candidates/{cid}/chat", json={"message": "Gone left a fact."})
    assert db.count_active_memories(cid) == 1
    assert client.delete(f"/candidates/{cid}").status_code == 200
    assert db.get_candidate(cid) is None
    assert db.count_active_memories(cid) == 0       # memories cascaded away


# --- live interview tests ----------------------------------------------
def test_interview_requires_consent():
    cid = client.post("/candidates", json={"name": "Consent"}).json()["id"]
    r = client.post(f"/candidates/{cid}/interviews", json={"consent_confirmed": False})
    assert r.status_code == 400                     # no consent, no recording
    r = client.post(f"/candidates/{cid}/interviews", json={"consent_confirmed": True})
    assert r.status_code == 200 and r.json()["consent_confirmed_at"]
    assert client.post("/candidates/999999/interviews",
                       json={"consent_confirmed": True}).status_code == 404
    # audio is impossible without a consented session
    assert client.post("/interviews/999999/audio", json={"audio_b64": "AAAA"}).status_code == 404


def test_audio_chunk_transcribes_and_extracts():
    cid = client.post("/candidates", json={"name": "Spoken"}).json()["id"]
    iid = client.post(f"/candidates/{cid}/interviews",
                      json={"consent_confirmed": True}).json()["id"]
    _fake_transcripts.append("Spoken scored 9 on safety.")
    body = client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"}).json()
    assert body["text"] == "Spoken scored 9 on safety."
    assert len(body["new_facts"]) == 1              # fed through extract_and_store
    assert db.count_active_memories(cid) == 1
    done = client.post(f"/interviews/{iid}/end").json()
    assert "Spoken scored 9 on safety." in done["transcript"]
    assert done["transcript"].startswith("[00:")    # timestamped line
    assert done["ended_at"]
    # no more audio after the interview ended
    assert client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"}).status_code == 409


def test_chunk_carry_over_joins_split_sentence():
    cid = client.post("/candidates", json={"name": "Split"}).json()["id"]
    iid = client.post(f"/candidates/{cid}/interviews",
                      json={"consent_confirmed": True}).json()["id"]
    _fake_transcripts.extend(["Split scored", "9 on communication."])
    first = client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"}).json()
    assert first["new_facts"] == []                 # fragment held, not extracted
    second = client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"}).json()
    assert len(second["new_facts"]) == 1
    assert "Split scored 9 on communication." in second["new_facts"][0]["fact"]
    client.post(f"/interviews/{iid}/end")


def test_overlap_repeat_words_are_dropped():
    # chunks are sent with ~1.5s of overlapping audio, so a chunk's transcript
    # can start by repeating the previous chunk's last words; the server drops them
    cid = client.post("/candidates", json={"name": "Overlap"}).json()["id"]
    iid = client.post(f"/candidates/{cid}/interviews",
                      json={"consent_confirmed": True}).json()["id"]
    _fake_transcripts.extend(["Overlap scored nine on safety.",
                              "on safety. He trained four operators."])
    first = client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"}).json()
    assert first["text"] == "Overlap scored nine on safety."
    second = client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"}).json()
    assert second["text"] == "He trained four operators."   # repeated words stripped
    done = client.post(f"/interviews/{iid}/end").json()
    assert done["transcript"].count("on safety") == 1       # no duplicate in the record


def test_silent_chunk_is_a_noop():
    cid = client.post("/candidates", json={"name": "Quiet"}).json()["id"]
    iid = client.post(f"/candidates/{cid}/interviews",
                      json={"consent_confirmed": True}).json()["id"]
    body = client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"}).json()
    assert body == {"text": "", "new_facts": []}
    assert db.count_active_memories(cid) == 0


def test_ask_mode_is_read_only():
    cid = client.post("/candidates", json={"name": "ReadOnly"}).json()["id"]
    client.post(f"/candidates/{cid}/chat", json={"message": "ReadOnly scored 8 on safety."})
    before = db.count_active_memories(cid)
    body = client.post(f"/candidates/{cid}/chat",
                       json={"message": "How safe are they?", "mode": "ask"}).json()
    assert body["reply"] and body["new_facts"] == []
    assert db.count_active_memories(cid) == before   # asking stored nothing


def test_list_interviews_returns_transcript():
    cid = client.post("/candidates", json={"name": "History"}).json()["id"]
    iid = client.post(f"/candidates/{cid}/interviews",
                      json={"consent_confirmed": True}).json()["id"]
    _fake_transcripts.append("History mentioned a forklift license.")
    client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"})
    client.post(f"/interviews/{iid}/end")
    ivs = client.get(f"/candidates/{cid}/interviews").json()
    assert len(ivs) == 1
    assert "forklift license" in ivs[0]["transcript"]
    assert ivs[0]["consent_confirmed_at"] and ivs[0]["ended_at"]
    assert client.get("/candidates/999999/interviews").status_code == 404


def test_note_during_interview_tags_manual_note():
    cid = client.post("/candidates", json={"name": "Noted"}).json()["id"]
    assert client.post(f"/candidates/{cid}/notes", json={"text": "  "}).status_code == 400
    assert client.post("/candidates/999999/notes", json={"text": "x"}).status_code == 404
    body = client.post(f"/candidates/{cid}/notes",
                       json={"text": "Noted showed a forklift certificate."}).json()
    assert len(body["new_facts"]) == 1              # same extraction pipeline
    mems = db.get_active_memories(cid)
    assert mems[0]["source"] == "manual_note"


def test_sources_distinguish_spoken_from_typed():
    cid = client.post("/candidates", json={"name": "Sourced"}).json()["id"]
    iid = client.post(f"/candidates/{cid}/interviews",
                      json={"consent_confirmed": True}).json()["id"]
    _fake_transcripts.append("Sourced repaired looms for 3 years.")
    client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"})
    client.post(f"/interviews/{iid}/end")
    client.post(f"/candidates/{cid}/notes", json={"text": "Sourced arrived 20 minutes late."})
    sources = {m["fact_text"]: m["source"] for m in db.get_active_memories(cid)}
    assert sources["Sourced repaired looms for 3 years."] == "live_transcript"
    assert sources["Sourced arrived 20 minutes late."] == "manual_note"
    # the memories endpoint exposes the source so the UI can label each fact
    assert all("source" in m for m in client.get(f"/candidates/{cid}/memories").json())


# --- resume pre-load ------------------------------------------------------
def _b64(data):
    import base64
    return base64.b64encode(data).decode()


def test_resume_txt_preloads_memory_and_questions():
    cid = client.post("/candidates", json={"name": "ResumeKarim"}).json()["id"]
    text = "ResumeKarim maintained 40 power looms for 5 years at Adamjee and holds a safety certification."
    r = client.post(f"/candidates/{cid}/resume",
                    json={"file_b64": _b64(text.encode()), "filename": "karim.txt"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["new_facts"]) == 1              # same extraction pipeline
    assert body["questions"]                        # tailored questions drafted
    mems = db.get_active_memories(cid)
    assert mems[0]["source"] == "resume"            # retrievable like any other memory
    # questions persist on the candidate row so the UI has them after a reload
    me = [c for c in client.get("/candidates").json() if c["id"] == cid][0]
    assert me["questions"] == body["questions"]
    # resume facts and interview facts coexist in the same store
    iid = client.post(f"/candidates/{cid}/interviews",
                      json={"consent_confirmed": True}).json()["id"]
    _fake_transcripts.append("ResumeKarim answered the loom question well.")
    client.post(f"/interviews/{iid}/audio", json={"audio_b64": "AAAA"})
    client.post(f"/interviews/{iid}/end")
    sources = {m["source"] for m in db.get_active_memories(cid)}
    assert {"resume", "live_transcript"} <= sources


def _docx_bytes(text):
    """A minimal .docx (zip with word/document.xml) carrying `text`, one line per
    paragraph. Enough for the stdlib reader, which only opens word/document.xml."""
    import io as _io, zipfile as _zip
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in text.split("\n"))
    doc = f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", doc)
    return buf.getvalue()


def test_resume_accepts_docx():
    cid = client.post("/candidates", json={"name": "DocxKarim"}).json()["id"]
    docx = _docx_bytes("DocxKarim maintained 40 power looms for 5 years at Adamjee mills.")
    r = client.post(f"/candidates/{cid}/resume",
                    json={"file_b64": _b64(docx), "filename": "karim.docx"})
    assert r.status_code == 200, r.text
    assert r.json()["new_facts"]                    # same extraction pipeline
    assert db.get_active_memories(cid)[0]["source"] == "resume"
    # a .docx that is not a real zip is rejected with a reason, not stored silently
    bad = client.post(f"/candidates/{cid}/resume",
                      json={"file_b64": _b64(b"this is plainly not a docx zip archive"),
                            "filename": "broken.docx"})
    assert bad.status_code == 400


def test_edit_questions_saves():
    cid = client.post("/candidates", json={"name": "EditQ"}).json()["id"]
    r = client.put(f"/candidates/{cid}/questions",
                   json={"questions": ["  First question?  ", "", "Second question?"]})
    assert r.status_code == 200
    assert r.json()["questions"] == ["First question?", "Second question?"]   # trimmed, blanks dropped
    me = [c for c in client.get("/candidates").json() if c["id"] == cid][0]
    assert me["questions"] == ["First question?", "Second question?"]
    assert client.put("/candidates/999999/questions", json={"questions": ["x"]}).status_code == 404


def test_resume_rejects_bad_files():
    cid = client.post("/candidates", json={"name": "BadFile"}).json()["id"]
    ok = {"file_b64": _b64(b"some resume text long enough to pass the emptiness check")}
    assert client.post("/candidates/999999/resume",
                       json={**ok, "filename": "x.txt"}).status_code == 404
    assert client.post(f"/candidates/{cid}/resume",
                       json={**ok, "filename": "x.docx"}).status_code == 400
    assert client.post(f"/candidates/{cid}/resume",
                       json={"file_b64": "!!not-base64!!", "filename": "x.txt"}).status_code == 400
    # a PDF with no extractable text (e.g. a scan) is told apart, not stored silently
    import io as _io
    from pypdf import PdfWriter
    buf = _io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.write(buf)
    r = client.post(f"/candidates/{cid}/resume",
                    json={"file_b64": _b64(buf.getvalue()), "filename": "scan.pdf"})
    assert r.status_code == 422
    assert db.count_active_memories(cid) == 0       # nothing junk was stored


# --- bulk CV upload -------------------------------------------------------
def test_bulk_parse_previews_without_storing():
    text = "BulkParse Karim\nMaintained 40 power looms for 5 years at Adamjee mills."
    before = len(client.get("/candidates").json())
    r = client.post("/bulk/parse", json={"file_b64": _b64(text.encode()),
                                         "filename": "karim.txt"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "BulkParse Karim" and not body["name_guessed"]
    assert body["role"] and body["facts"]           # preview to show the recruiter
    assert body["text"].startswith("BulkParse")     # rides back for /bulk/confirm
    assert body["duplicate_of"] is None
    # preview is read-only: no candidate row appeared
    assert len(client.get("/candidates").json()) == before


def test_bulk_parse_flags_duplicates_even_misspelled():
    client.post("/candidates", json={"name": "Rahim Uddin Bulk"})
    text = "Rahim Uddine Bulk\nTen years weaving experience in Narayanganj."
    body = client.post("/bulk/parse", json={"file_b64": _b64(text.encode()),
                                            "filename": "rahim.txt"}).json()
    assert body["duplicate_of"] and body["duplicate_of"]["name"] == "Rahim Uddin Bulk"


def test_bulk_parse_flags_duplicates_within_a_batch():
    # nothing stored yet: the match comes from batch_names, so id is None
    text = "Karim Hosain Batch\nThree years weaving on jute looms in Demra."
    body = client.post("/bulk/parse", json={
        "file_b64": _b64(text.encode()), "filename": "karim2.txt",
        "batch_names": ["Karim Hossain Batch"]}).json()
    assert body["duplicate_of"] == {"name": "Karim Hossain Batch", "id": None}


def test_bulk_parse_falls_back_to_filename_when_no_name():
    text = "\nThis resume text names nobody but is long enough to parse fine."
    body = client.post("/bulk/parse", json={"file_b64": _b64(text.encode()),
                                            "filename": "mystery_cv-2.txt"}).json()
    assert body["name_guessed"] and body["name"] == "Mystery Cv 2"


def test_bulk_parse_rejects_unreadable_files():
    ok = _b64(b"long enough text to pass the emptiness check either way")
    # renamed non-PDF: extension says pdf, bytes are not
    r = client.post("/bulk/parse", json={"file_b64": ok, "filename": "fake.pdf"})
    assert r.status_code == 400 and "PDF" in r.json()["detail"]
    assert client.post("/bulk/parse", json={"file_b64": ok,
                                            "filename": "cv.docx"}).status_code == 400
    assert client.post("/bulk/parse", json={"file_b64": _b64(b"too short"),
                                            "filename": "cv.txt"}).status_code == 422


def test_bulk_confirm_runs_the_resume_pipeline():
    text = "BulkConfirm Rina\nQuality control lead, scored 10 out of 10 on inspection."
    assert client.post("/bulk/confirm",
                       json={"name": " ", "text": text}).status_code == 400
    assert client.post("/bulk/confirm",
                       json={"name": "X", "text": "short"}).status_code == 422
    body = client.post("/bulk/confirm", json={
        "name": "BulkConfirm Rina", "role": "QC Lead", "text": text}).json()
    assert body["id"] and body["new_facts"] and body["questions"]
    mems = db.get_active_memories(body["id"])
    assert mems and all(m["source"] == "resume" for m in mems)   # same tag as single upload
    me = [c for c in client.get("/candidates").json() if c["id"] == body["id"]][0]
    assert me["role"] == "QC Lead" and me["questions"] == body["questions"]


# --- scorecards ---------------------------------------------------------
def test_scorecard_draft_cites_only_real_facts():
    cid = client.post("/candidates", json={"name": "Scored", "role": "Loom Operator"}).json()["id"]
    client.post(f"/candidates/{cid}/chat", json={"message": "Scored ran 40 looms for 6 years."})
    r = client.post(f"/candidates/{cid}/scorecard/draft")
    assert r.status_code == 200, r.text
    sc = r.json()["scorecard"]
    assert len(sc) == 2
    loom = [row for row in sc if row["competency"] == "Loom operation"][0]
    assert loom["rating"] == 4 and loom["ai_suggested"] is True
    # the invented citation is dropped; only the real stored fact survives
    assert loom["cited"] == ["Scored ran 40 looms for 6 years."]
    safety = [row for row in sc if row["competency"] == "Safety"][0]
    assert safety["rating"] == 0 and safety["ai_suggested"] is False
    # persisted on the candidate row so a reload keeps it
    me = [c for c in client.get("/candidates").json() if c["id"] == cid][0]
    assert len(me["scorecard"]) == 2
    assert client.post("/candidates/999999/scorecard/draft").status_code == 404


def test_scorecard_save_sanitizes_and_persists():
    cid = client.post("/candidates", json={"name": "SaveCard"}).json()["id"]
    r = client.put(f"/candidates/{cid}/scorecard", json={"scorecard": [
        {"competency": "Reliability", "rating": 9, "rationale": "great", "cited": ["x"], "ai_suggested": True},
        {"competency": "   ", "rating": 3},   # blank competency -> dropped
        "not-a-dict",                         # junk -> dropped
    ]})
    assert r.status_code == 200
    sc = r.json()["scorecard"]
    assert len(sc) == 1 and sc[0]["competency"] == "Reliability"
    assert sc[0]["rating"] == 5             # 9 clamped into 0-5
    me = [c for c in client.get("/candidates").json() if c["id"] == cid][0]
    assert me["scorecard"][0]["rating"] == 5
    assert client.put("/candidates/999999/scorecard", json={"scorecard": []}).status_code == 404


# --- cross-candidate ask ------------------------------------------------
def test_ask_searches_across_all_candidates():
    a = client.post("/candidates", json={"name": "AskKarim"}).json()["id"]
    b = client.post("/candidates", json={"name": "AskRahim"}).json()["id"]
    client.post(f"/candidates/{a}/chat", json={"message": "AskKarim scored 9 on communication."})
    client.post(f"/candidates/{b}/chat", json={"message": "AskRahim scored 6 on communication."})
    before = db.count_active_memories(a) + db.count_active_memories(b)

    assert client.post("/ask", json={"question": "  "}).status_code == 400
    r = client.post("/ask", json={"question": "Who communicates best?"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"]
    names = {m["candidate"] for m in body["recalled"]}
    assert {"AskKarim", "AskRahim"} <= names      # facts came from BOTH candidates
    # query-only: asking must not have stored anything new
    assert db.count_active_memories(a) + db.count_active_memories(b) == before


# --- db-layer tests (the part eval.py and the endpoints don't isolate) ---
def test_db_count_and_archive():
    cid = db.create_candidate("DBLayer")["id"]
    db.add_memory(cid, "fact one", "note", 5, _fake_embed("fact one"))
    db.add_memory(cid, "fact two", "note", 5, _fake_embed("fact two"))
    assert db.count_active_memories(cid) == 2
    db.archive_memory(db.get_active_memories(cid)[0]["id"])
    assert db.count_active_memories(cid) == 1        # archived rows drop out of the count


# --- spreadsheet import ---------------------------------------------------
def _xlsx_bytes(rows):
    """A minimal real .xlsx (shared strings + one worksheet) from a list of
    string rows. Enough for the stdlib reader, which reads those two parts."""
    import io as _io, zipfile as _zip
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    strings, idx = [], {}
    def sid(s):
        if s not in idx:
            idx[s] = len(strings); strings.append(s)
        return idx[s]
    body = ""
    for ri, row in enumerate(rows, 1):
        cells = "".join(
            f'<c r="{chr(65 + ci)}{ri}" t="s"><v>{sid(v)}</v></c>'
            for ci, v in enumerate(row) if v != "")
        body += f'<row r="{ri}">{cells}</row>'
    sheet = f'<worksheet xmlns="{ns}"><sheetData>{body}</sheetData></worksheet>'
    ss = f'<sst xmlns="{ns}">' + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>"
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", ss)
    return buf.getvalue()


def test_detect_columns_is_flexible_and_wont_guess():
    from app import _detect_columns
    cols = _detect_columns(["Full Name", "Contact Email", "Mobile", "Resume Drive Link", "Portfolio"])
    assert cols == {"email": 1, "phone": 2, "name": 0, "resume_url": 3, "portfolio_url": 4}
    alt = _detect_columns(["email_address", "phone number", "candidate", "cv url", "personal website"])
    assert alt["email"] == 0 and alt["resume_url"] == 3 and alt["portfolio_url"] == 4
    # ambiguous headers are left unmapped rather than guessed wrong (step 2)
    amb = _detect_columns(["Name", "Link", "URL"])
    assert amb["name"] == 0 and amb["resume_url"] is None and amb["portfolio_url"] is None


def test_spreadsheet_parses_xlsx_and_csv():
    xb = _xlsx_bytes([["Full Name", "Email", "Phone", "Resume Link", "Portfolio"],
                      ["Rahim Uddin", "rahim@x.com", "0171", "https://drive.google.com/file/d/PUBLICFILEID1/view", ""],
                      ["Nadia Karim", "nadia@y.com", "", "", "https://good-folio.example"]])
    r = client.post("/bulk/spreadsheet", json={"file_b64": _b64(xb), "filename": "a.xlsx"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["unmapped"] == [] and len(d["rows"]) == 2
    assert d["rows"][0]["name"] == "Rahim Uddin" and d["rows"][0]["email"] == "rahim@x.com"
    assert d["rows"][1]["portfolio_url"] == "https://good-folio.example"
    # CSV, with a column that can't be mapped -> reported in unmapped, not guessed
    csv = "Name,Contact Email,Portfolio URL\nJoy,joy@z.com,good-folio.example\n"
    dc = client.post("/bulk/spreadsheet", json={"file_b64": _b64(csv.encode()), "filename": "a.csv"}).json()
    assert dc["rows"][0]["name"] == "Joy" and "phone" in dc["unmapped"] and "resume_url" in dc["unmapped"]
    # a spreadsheet with only a header is rejected, not silently empty
    only_header = _xlsx_bytes([["Name", "Email"]])
    assert client.post("/bulk/spreadsheet",
                       json={"file_b64": _b64(only_header), "filename": "h.xlsx"}).status_code == 422


def test_bulk_row_reports_each_fetch_outcome_honestly():
    # public Drive resume reads fine; a portfolio link is absent
    ok = client.post("/bulk/row", json={
        "name": "Rahim", "email": "rahim@x.com",
        "resume_url": "https://drive.google.com/file/d/PUBLICFILEID1/view"}).json()
    assert ok["resume_status"] == "ok" and ok["portfolio_status"] == "none"
    assert ok["resume_text"] and ok["facts"]        # extracted a preview
    # private Drive link -> couldn't access (login page), never treated as success
    priv = client.post("/bulk/row", json={
        "name": "Sara", "resume_url": "https://drive.google.com/file/d/PRIVATEFILEID9/view"}).json()
    assert priv["resume_status"] == "no_access" and priv["resume_text"] == ""
    # permission denied by HTTP status is also no_access, not a hard error
    forb = client.post("/bulk/row", json={
        "name": "Ali", "resume_url": "https://drive.google.com/file/d/FORBIDFILEID2/view"}).json()
    assert forb["resume_status"] == "no_access"
    # a real portfolio page yields text; a broken resume link is invalid
    port = client.post("/bulk/row", json={
        "name": "Nadia", "resume_url": "https://drive.google.com/file/d/DEADFILEID77/view",
        "portfolio_url": "https://good-folio.example"}).json()
    assert port["resume_status"] == "invalid" and port["portfolio_status"] == "ok"
    assert "Nadia" in port["portfolio_text"] and "track()" not in port["portfolio_text"]  # scripts stripped
    # a totally unreachable portfolio -> couldn't access
    dead = client.post("/bulk/row", json={
        "name": "Gone", "portfolio_url": "https://nope.invalid"}).json()
    assert dead["portfolio_status"] == "no_access"


def test_bulk_confirm_partial_success_still_creates_profile():
    # name + email parsed, but the resume link failed: the row is STILL confirmable
    # (partial success is normal), creating a profile with contact info + a note.
    r = client.post("/bulk/confirm", json={
        "name": "Partial Person", "email": "partial@x.com", "phone": "0199",
        "resume_status": "no_access"})
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    facts = [m["fact_text"] for m in db.get_active_memories(cid)]
    assert any("partial@x.com" in f for f in facts)                 # contact stored
    assert any("could not be accessed" in f for f in facts)         # honest follow-up note
    assert all(m["source"] == "spreadsheet" for m in db.get_active_memories(cid))
    # a row with a portfolio but no resume/contact still confirms via portfolio text
    p = client.post("/bulk/confirm", json={
        "name": "Folio Only",
        "portfolio_text": "Senior product designer with eight years across fintech and logistics teams."})
    assert p.status_code == 200 and p.json()["new_facts"]
    # a row with truly nothing to store is rejected, not turned into an empty profile
    assert client.post("/bulk/confirm", json={"name": "Empty"}).status_code == 422


if __name__ == "__main__":
    print("Running RecruitMemory offline tests (stubbed qwen, throwaway DB)…\n")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} offline tests passed (no network).")
