# RecruitMemory — 3-Minute Demo Script

A tight talk-track for judges. Goal: make all **three memory mechanisms**
*happen on screen*, not just get described. Practice once so the timing feels
natural.

**Before you start:**
- Run `docker compose up --build` (or `uvicorn app:app`) and open
  **http://localhost:8000**.
- **Pre-load one candidate** so the *Compare* finale is instant: create
  **Salma** (role *Loom Operator*) and give her two facts —
  *"Salma has 2 years on jute looms but scored 10/10 on quality control."* and
  *"Salma is extremely reliable — never missed a shift."*
- Leave **Karim uncreated** — you'll build him live. Have the **"Stored
  memories"** button and the sidebar **"Compare"** button in mind.

---

## 0:00 — 0:20 · The problem (hook)

> "Interviewers at a jute mill talk to dozens of candidates over multiple
> sessions. Normal AI chatbots forget everything the moment the chat ends.
> **RecruitMemory remembers each candidate across sessions** — like a colleague
> who actually recalls the people they've met. It's built for the MemoryAgent
> track on Qwen Cloud."

*(Show the clean UI. Salma already sits in the sidebar; Karim doesn't exist yet.)*

---

## 0:20 — 0:55 · Teach it facts, then correct it → **Extraction + belief update**

1. Create a candidate: **Karim**, role **Loom Operator**.
2. Type: **"Karim has 5 years of experience on jute looms."** → send.
3. Type: **"He scored 6 out of 10 on the safety assessment."** → send.
4. Type: **"Correction — Karim actually scored 9 out of 10 on the safety test."** → send.

> "As I talk, the assistant **extracts structured facts** — each pill is a new
> memory: a skill, a score. This is mechanism one, **extraction**. But watch the
> last one: I *corrected* his safety score. It doesn't just add a second fact and
> contradict itself — it sees this **updates an existing belief**, retires the old
> 6/10, and keeps the 9/10. The agent **changes its mind**."

*(Point at the **"Updated belief — ~~scored 6/10~~ → scored 9/10"** line and the
teal "updated" chip.)*

---

## 0:55 — 1:35 · Ask a question → **Retrieval**

1. Ask: **"Would Karim be a good fit for a senior loom operator role?"** → send.

> "I never repeated any of those facts in my question. The assistant **searched
> its memory**, pulled the relevant ones — his experience, the corrected safety
> score — and used them. It ranks every memory by **meaning, importance, and
> recency**, injecting only the top few. That's mechanism two — **retrieval**."

*(Point at the "recalled N memories" note above the reply.)*

---

## 1:35 — 2:15 · The forgetting, made visible → **Decay + Consolidation**

*(This is the differentiator — most "memory" demos only claim it. We show it.)*

1. Open **"Stored memories."** Point at the importance bars.
2. Set the **"Advance time"** control to **+3 months** and click it.

> "A memory system that only remembers is just a database. Real memory
> **forgets**. Importance **halves every two weeks** a fact goes untouched. Instead
> of asking you to imagine that, watch — I'll fast-forward the clock three months."
>
> *(bars shrink; low-value facts drop out)*
>
> "The stale facts just **faded below the threshold and archived themselves**, and
> when a candidate piles up too many memories the oldest get **summarized into one**.
> That's mechanism three — **decay and consolidation**. It's what makes this a
> memory *agent*, not a log file. And it's the real decay math, not an animation —
> [`memory.py`](memory.py) has the half-life and thresholds right at the top."

---

## 2:15 — 2:45 · Two candidates at once → **Compare**

1. Click **"Compare"** in the sidebar. Tick **Karim** and **Salma**.
2. Ask: **"Who is the better fit for a senior loom role?"** → Compare.

> "Memory isn't just per-person recall. Here it reasons across candidates: it
> pulls **each one's own remembered facts** and weighs them. Karim's experience
> versus Salma's quality record — and crucially, it **shows the exact memories
> behind the call**, so the recommendation is auditable, not a black box."

*(Point at the recommendation card, then the per-candidate evidence columns.)*

---

## 2:45 — 3:05 · Close (durability + stack)

> "Everything runs on **Qwen Cloud** — qwen-plus for reasoning, text-embedding-v3
> for the memory search. It ships as **one Docker container**, and the whole
> memory store **backs up to Alibaba Cloud OSS** with one command.
> **RecruitMemory — an agent that remembers, recalls, forgets, changes its mind,
> and reasons across people, just like we do.**"

*(Show the GitHub repo page with the architecture diagram as the final frame.)*

---

## If something breaks (backup plan)
- **Qwen slow/erroring live?** Salma is already loaded — lead with "Stored
  memories" + the *Advance time* decay demo + *Compare*, which lean less on fresh
  generation.
- **Wifi dies?** Walk through the README's architecture diagram and the functions
  in `memory.py` — the story holds without a live server.
- **Running short?** The two must-keep beats are **Advance time** (visible
  forgetting) and **Compare** — they're what judges haven't seen elsewhere. The
  belief-update correction can be dropped to a single sentence if needed.
