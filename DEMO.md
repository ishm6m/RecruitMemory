# RecruitMemory — 3-Minute Demo Script

A tight talk-track for judges. Goal: make all **three memory mechanisms**
visible on screen, not just described. Practice once so the timing feels natural.

**Before you start:** run `docker compose up --build` (or `uvicorn app:app`),
open **http://localhost:8000**, and make sure you have **no candidates** yet
(fresh start). Have the **"Stored memories"** panel button ready.

---

## 0:00 — 0:20 · The problem (hook)

> "Interviewers at a jute mill talk to dozens of candidates over multiple
> sessions. Normal AI chatbots forget everything the moment the chat ends.
> **RecruitMemory remembers each candidate across sessions** — like a colleague
> who actually recalls the people they've met. It's built for the MemoryAgent
> track on Qwen Cloud."

*(Show the clean UI, empty candidate list.)*

---

## 0:20 — 0:50 · Teach it facts → **Extraction**

1. Create a candidate: **Karim**, role **Loom Operator**.
2. Type: **"Karim has 5 years of experience on jute looms."** → send.
3. Type: **"He scored 8 out of 10 on the safety assessment."** → send.
4. Type: **"One concern — he was late to two shifts last month."** → send.

> "Watch under each message — as I talk, the assistant **extracts structured
> facts** and saves them. Each pill is a new memory: a skill, a score, a red
> flag. This is mechanism one — **extraction**."

*(Point at the accent-colored pills sliding in.)*

---

## 0:50 — 1:40 · Ask a question → **Retrieval**

1. Click **"Stored memories"** — briefly show the saved facts. Close it.
2. Ask: **"Would Karim be a good fit for a senior loom operator role?"** → send.

> "I never repeated any of those facts in my question. But the assistant
> **searched its memory**, pulled the relevant ones — his experience, his safety
> score, the lateness concern — and used them in the answer. It ranks every
> memory by **meaning, importance, and recency**, and injects only the top few.
> That's mechanism two — **retrieval**."

*(Point at the "recalled N memories" note above the reply, and how the reply
cites the specific facts.)*

---

## 1:40 — 2:30 · The forgetting → **Decay + Consolidation**

*(This is the differentiator — most "memory" demos skip it.)*

> "A memory system that only remembers is just a database. Real memory
> **forgets**. Every fact has an importance score that **halves every two weeks**
> if it's never brought up again — so stale trivia fades and gets archived
> automatically. And when a candidate piles up too many memories, the oldest,
> least-important ones are **summarized into a single memory** to keep it
> efficient. That's mechanism three — **decay and consolidation**. It's what
> makes this a memory *agent*, not a log file."

*(Open [memory.py](memory.py) if you want to show `decay_and_consolidate()`
and the tuning knobs at the top — half-life, thresholds — proving it's real.)*

---

## 2:30 — 3:00 · Close (durability + stack)

> "Everything runs on **Qwen Cloud** — qwen-plus for reasoning, text-embedding-v3
> for the memory search. It ships as **one Docker container**, and the whole
> memory store **backs up to Alibaba Cloud OSS** with one command, so a
> candidate's history survives even if the server dies.
> The code's on GitHub, MIT-licensed. **That's RecruitMemory — an agent that
> remembers, recalls, and forgets, just like we do.**"

*(Show the GitHub repo page with the architecture diagram as the final frame.)*

---

## If something breaks (backup plan)
- **Qwen slow/erroring live?** Have a candidate with facts already loaded from a
  practice run, and lead with the "Stored memories" panel + a retrieval question.
- **Wifi dies?** Walk through the README's architecture diagram and the three
  functions in `memory.py` — the story holds without a live server.
- **Running short on time?** Cut the consolidation *demo* but still *say* the one
  sentence about forgetting — it's the part that sets you apart.
