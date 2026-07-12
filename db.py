"""
db.py, SQLite schema plus a tiny data-access layer.

Plain language: this file owns the database file (recruitmemory.db). It creates
two tables the first time it runs, and provides small helper functions so the
rest of the app never has to write raw SQL. SQLite is just a single file on
disk, no server to install.
"""

import sqlite3
import json
import time
import os

# Where the SQLite file lives. Defaults to the project folder for local runs;
# in Docker we point DB_PATH at a mounted volume so data survives restarts.
DB_PATH = os.getenv("DB_PATH", "recruitmemory.db")


def _conn():
    # check_same_thread=False lets FastAPI's worker threads share the connection.
    # ponytail: single shared connection is fine for a demo; use a pool if scaling.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # rows behave like dicts (row["fact_text"])
    return conn


_c = _conn()


def init_db():
    """Create the tables if they don't exist yet. Safe to call every startup."""
    _c.executescript(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            role       TEXT,
            created_at REAL NOT NULL,
            questions  TEXT NOT NULL DEFAULT '[]'  -- JSON array of suggested interview questions
        );

        CREATE TABLE IF NOT EXISTS memories (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id     INTEGER NOT NULL,
            fact_text        TEXT NOT NULL,
            category         TEXT,
            importance       REAL NOT NULL,     -- 1..10, decays over time
            embedding        TEXT NOT NULL,     -- JSON array of floats
            created_at       REAL NOT NULL,
            last_accessed_at REAL NOT NULL,
            archived         INTEGER NOT NULL DEFAULT 0,  -- 0 = active, 1 = archived
            source           TEXT NOT NULL DEFAULT 'manual_note',
            FOREIGN KEY (candidate_id) REFERENCES candidates(id)
        );

        CREATE TABLE IF NOT EXISTS interviews (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id         INTEGER NOT NULL,
            consent_confirmed_at REAL NOT NULL,  -- no row without logged consent
            started_at           REAL NOT NULL,
            ended_at             REAL,           -- NULL = still in progress
            transcript           TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (candidate_id) REFERENCES candidates(id)
        );
        """
    )
    # Databases created before the source column existed get it added in place.
    # Old rows default to 'manual_note': their true origin was never recorded.
    cols = [r[1] for r in _c.execute("PRAGMA table_info(memories)")]
    if "source" not in cols:
        _c.execute("ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'manual_note'")
    cand_cols = [r[1] for r in _c.execute("PRAGMA table_info(candidates)")]
    if "questions" not in cand_cols:
        _c.execute("ALTER TABLE candidates ADD COLUMN questions TEXT NOT NULL DEFAULT '[]'")
    _c.commit()


# ---------- candidates ----------

def create_candidate(name, role=""):
    cur = _c.execute(
        "INSERT INTO candidates (name, role, created_at) VALUES (?, ?, ?)",
        (name, role, time.time()),
    )
    _c.commit()
    return {"id": cur.lastrowid, "name": name, "role": role, "questions": []}


def _cand(row):
    """Row -> dict, with the questions JSON decoded to a real list."""
    d = dict(row)
    d["questions"] = json.loads(d.get("questions") or "[]")
    return d


def list_candidates():
    rows = _c.execute("SELECT * FROM candidates ORDER BY created_at DESC").fetchall()
    return [_cand(r) for r in rows]


def get_candidate(candidate_id):
    row = _c.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    return _cand(row) if row else None


def set_questions(candidate_id, questions):
    """Store the suggested interview questions (a list of strings)."""
    _c.execute("UPDATE candidates SET questions = ? WHERE id = ?",
               (json.dumps(questions), candidate_id))
    _c.commit()


def delete_candidate(candidate_id):
    """Remove a candidate and all of their memories and interviews."""
    _c.execute("DELETE FROM memories WHERE candidate_id = ?", (candidate_id,))
    _c.execute("DELETE FROM interviews WHERE candidate_id = ?", (candidate_id,))
    _c.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    _c.commit()


# ---------- interviews ----------

def create_interview(candidate_id):
    """Start a consented interview session. Consent is logged by existing:
    this row is only ever created after the recruiter confirms the candidate
    was informed, so consent_confirmed_at doubles as the audit timestamp."""
    now = time.time()
    cur = _c.execute(
        "INSERT INTO interviews (candidate_id, consent_confirmed_at, started_at) "
        "VALUES (?, ?, ?)",
        (candidate_id, now, now),
    )
    _c.commit()
    return {"id": cur.lastrowid, "candidate_id": candidate_id,
            "consent_confirmed_at": now, "started_at": now}


def get_interview(interview_id):
    row = _c.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,)).fetchone()
    return dict(row) if row else None


def list_interviews(candidate_id):
    """All of a candidate's interview sessions, newest first, with transcripts."""
    rows = _c.execute(
        "SELECT * FROM interviews WHERE candidate_id = ? ORDER BY started_at DESC",
        (candidate_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def append_transcript(interview_id, line):
    _c.execute(
        "UPDATE interviews SET transcript = transcript || ? WHERE id = ?",
        (line, interview_id),
    )
    _c.commit()


def end_interview(interview_id):
    _c.execute(
        "UPDATE interviews SET ended_at = ? WHERE id = ?",
        (time.time(), interview_id),
    )
    _c.commit()


# ---------- memories ----------

def add_memory(candidate_id, fact_text, category, importance, embedding,
               source="manual_note"):
    """`source` records where the fact came from: manual_note (typed),
    live_transcript (spoken in an interview), resume (uploaded document),
    or the engine's own consolidation/reflection output."""
    now = time.time()
    _c.execute(
        """INSERT INTO memories
           (candidate_id, fact_text, category, importance, embedding,
            created_at, last_accessed_at, archived, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
        (candidate_id, fact_text, category, importance,
         json.dumps(embedding), now, now, source),
    )
    _c.commit()


def get_active_memories(candidate_id):
    """All non-archived memories for a candidate, embeddings decoded to lists."""
    rows = _c.execute(
        "SELECT * FROM memories WHERE candidate_id = ? AND archived = 0",
        (candidate_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["embedding"] = json.loads(d["embedding"])
        out.append(d)
    return out


def get_all_active_memories():
    """All non-archived memories across every candidate, with candidate names,
    for cross-candidate queries. Embeddings decoded to lists."""
    rows = _c.execute(
        "SELECT m.*, c.name AS candidate_name FROM memories m "
        "JOIN candidates c ON c.id = m.candidate_id WHERE m.archived = 0"
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["embedding"] = json.loads(d["embedding"])
        out.append(d)
    return out


def count_active_memories(candidate_id):
    """How many active memories a candidate has, without decoding embeddings.
    Cheap COUNT for callers (e.g. the "5 of N" UI proof) that only need the size."""
    row = _c.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE candidate_id = ? AND archived = 0",
        (candidate_id,),
    ).fetchone()
    return row["n"]


def touch_memories(memory_ids):
    """Mark memories as just-accessed (resets their recency for decay)."""
    if not memory_ids:
        return
    now = time.time()
    _c.executemany(
        "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
        [(now, mid) for mid in memory_ids],
    )
    _c.commit()


def age_memories(candidate_id, seconds):
    """Backdate every active memory's last-access time by `seconds`, i.e. make
    the decay curve behave as if that much time has passed. Demo-only helper."""
    _c.execute(
        "UPDATE memories SET last_accessed_at = last_accessed_at - ? "
        "WHERE candidate_id = ? AND archived = 0",
        (seconds, candidate_id),
    )
    _c.commit()


def archive_memory(memory_id):
    _c.execute("UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,))
    _c.commit()
