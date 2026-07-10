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
    return "Noted."


qwen.chat = _fake_chat
qwen.embed = _fake_embed

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
