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


if __name__ == "__main__":
    print("Running RecruitMemory offline tests (stubbed qwen, throwaway DB)…\n")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} offline tests passed (no network).")
