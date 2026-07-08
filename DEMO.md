# RecruitMemory, 3-Minute Demo Script

A tight talk-track for judges. Goal: make all **four memory mechanisms**
(extraction + belief-update, retrieval, reflection, decay + consolidation)
*happen on screen*, not just get described. Practice once so the timing feels
natural.

**Before you start:**
- Run `docker compose up --build` (or `uvicorn app:app`) and open
  **http://localhost:8000**.
- **Pre-load the demo candidate** so the *Compare* finale is instant: click
  **"Load demo candidate"** under the Add-candidate box in the sidebar. This seeds
  **Salma (demo)** with a few facts in one click (no typing needed).
- Leave **Karim uncreated**, you'll build him live. Have the **"Stored
  memories"** button and the sidebar **"Compare"** button in mind.

---

## 0:00 to 0:20 · The problem (hook)

> "Interviewers at a jute mill talk to dozens of candidates over multiple
> sessions. Normal AI chatbots forget everything the moment the chat ends.
> **RecruitMemory remembers each candidate across sessions**, like a colleague
> who actually recalls the people they've met. It's built for the MemoryAgent
> track on Qwen Cloud."

*(Show the clean UI. Salma already sits in the sidebar; Karim doesn't exist yet.)*

**Prove it's real memory, not just a live chat. Reload the page.** Click Salma,
show her stored facts, then **hard-refresh the browser (Cmd+Shift+R)**. She and
every fact are still there.

> "I talked to Salma in an earlier session, and I've since closed the app. Watch: I
> reload the whole page, and she's still here, with everything I learned about
> her. This isn't chat history in a browser tab, it's a memory store on disk
> that outlives the session. That's the whole point."

---

## 0:20 to 0:55 · Teach it facts, then correct it → **Extraction + belief update**

1. Create a candidate: **Karim**, role **Loom Operator**.
2. Type: **"Karim has 5 years of experience on jute looms."** → send.
3. Type: **"He scored 6 out of 10 on the safety assessment."** → send.
4. Type: **"Correction: Karim actually scored 9 out of 10 on the safety test."** → send.

> "As I talk, the assistant **extracts structured facts**, each pill is a new
> memory: a skill, a score. This is mechanism one, **extraction**. But watch the
> last one: I *corrected* his safety score. It doesn't just add a second fact and
> contradict itself. It sees this **updates an existing belief**, retires the old
> 6/10, and keeps the 9/10. The agent **changes its mind**."

*(Point at the **"Updated belief: ~~scored 6/10~~ → scored 9/10"** line and the
teal "updated" chip.)*

---

## 0:55 to 1:35 · Ask a question → **Retrieval**

1. Ask: **"Would Karim be a good fit for a senior loom operator role?"** → send.

> "I never repeated any of those facts in my question. The assistant **searched
> its memory**, pulled the relevant ones, his experience and the corrected safety
> score, and used them. It ranks every memory by **meaning, importance, and
> recency**, injecting only the top few. That's mechanism two, **retrieval**."

*(Point at the **"Recalled 5 of N memories"** note above the reply. Expand it: it
ranked *all* N by relevance, importance & recency and injected only the top 5. That
"of N" is the cost story made visible: as a candidate's history grows, the number
injected stays flat.)*

---

## 1:35 to 1:55 · It forms opinions it was never told → **Reflection**

1. Open **"Stored memories."** Click **"Synthesize insights."**

> "So far it's *storing* facts. But real understanding means noticing patterns no
> single fact states. Watch: I click **Synthesize insights**, and it reads
> everything it knows about Karim and writes a new, higher-order memory:
> *'consistent safety ownership and procedural mastery.'* Nobody told it that,
> it **inferred a trait** from the evidence. And because that insight is stored as
> a high-importance memory, it now **steers future recommendations**. The agent's
> understanding is *deepening*, not just growing. This also fires **on its own** as
> a conversation grows. I'm pressing the button here only to show it on cue."

*(Point at the amber **insight** rows that appear at the top of the store.)*

---

## 1:55 to 2:25 · The forgetting, made visible → **Decay + Consolidation**

*(This is the differentiator. Most "memory" demos only claim it. We show it.)*

1. Open **"Stored memories."** Point at the importance bars.
2. Set the **"Advance time"** control to **+3 months** and click it.

> "A memory system that only remembers is just a database. Real memory
> **forgets**. Importance **halves every two weeks** a fact goes untouched. Instead
> of asking you to imagine that, watch me fast-forward the clock three months."
>
> *(bars shrink; low-value facts drop out)*
>
> "The stale facts just **faded below the threshold and archived themselves**, and
> when a candidate piles up too many memories the oldest get **summarized into one**.
> That's mechanism three, **decay and consolidation**. It's what makes this a
> memory *agent*, not a log file. And it's the real decay math, not an animation.
> [`memory.py`](memory.py) has the half-life and thresholds right at the top."

---

## 2:25 to 2:50 · Two candidates at once → **Compare**

1. Click **"Compare"** in the sidebar. Tick **Karim** and **Salma**.
2. Ask: **"Who is the better fit for a senior loom role?"** → Compare.

> "Memory isn't just per-person recall. Here it reasons across candidates: it
> pulls **each one's own remembered facts** and weighs them. Karim's experience
> versus Salma's quality record, and crucially it **shows the exact memories
> behind the call**, so the recommendation is auditable, not a black box."

*(Point at the recommendation card, then the per-candidate evidence columns.)*

---

## 2:50 to 3:10 · Close (durability + stack)

> "Everything runs on **Qwen Cloud**: qwen-plus for reasoning, text-embedding-v3
> for the memory search. And notice what retrieval buys us: instead of pasting a
> candidate's entire history into every prompt, we inject only the **top 5
> relevant memories**, so the cost per question stays flat no matter how long
> their history grows. It ships as **one Docker container**, and the whole
> memory store **backs up to Alibaba Cloud OSS** with one command.
> **RecruitMemory, an agent that remembers, recalls, forgets, changes its mind,
> forms its own opinions, and reasons across people, just like we do.**"

*(Show the GitHub repo page with the architecture diagram as the final frame.)*

---

## If something breaks (backup plan)
- **Qwen slow/erroring live?** Salma is already loaded, lead with "Stored
  memories" + the *Advance time* decay demo + *Compare*, which lean less on fresh
  generation.
- **Wifi dies?** Walk through the README's architecture diagram and the functions
  in `memory.py`, the story holds without a live server.
- **Running short?** The three must-keep beats are **Synthesize insights**
  (understanding that deepens), **Advance time** (visible forgetting), and
  **Compare**, they're what judges haven't seen elsewhere. The belief-update
  correction can be dropped to a single sentence if needed.
- **Have the numbers ready.** If a judge asks "does the memory actually work?",
  the answer is in the README table: recall@5 = 100%, and the ranking recovers
  the current truth 5/5 vs 2/5 for plain vector search (`python eval.py`).
