# Job Scout + Resume ATS Agent (LangGraph + Gemini)

A LangGraph agent that scouts jobs and parses a resume **in parallel**, scores the
resume against a selected job (weighted ATS), and runs a **human-in-the-loop** loop:
show matched / missing / advice → let the user add info → **re-score** → generate a
cover letter, tailored resume, or interview questions.

## Graph

```
START ─┬─► scout   (coarse retrieval from the 57k index — no profile needed)
       └─► parser  (resume → structured profile — no jobs needed)
                 └──► match  (join: score scouted pool vs profile)
                        └──► select_job      [interrupt — pick a job]
                               └──► ats_score
                                      └──► human_review  [interrupt]
                                             ├─ add_info ─► ats_score   (re-score loop)
                                             ├─ ask / cover_letter /
                                             │  tailored_resume /
                                             │  interview_questions ─► extras ─► human_review
                                             └─ done ─► END
```

Parallelism is real: `scout` and `parser` have no data dependency, so they run in the
same LangGraph super-step; `match` fans them in. The two `interrupt()` points are the
HITL gates — the CLI resumes them with `Command(resume=...)`.

## Run the web app (front end + backend):
# 🚀 Live Deployment

| Service | Status | URL |
| :--- | :--- | :--- |
| **Main Web App** | 🟢 Live on Render | [aijobboard-o52o.onrender.com](https://aijobboard-o52o.onrender.com) |
| **AI Scoring Engine** | ⚡ HF Space (Gradio) | [rajuiscoding-resume-parser.hf.space](https://rajuiscoding-resume-parser.hf.space) |
| **Observability** | 📊 Tracing Enabled | LangSmith Dashboard |

---

# 🏗 Architecture & Deployment

The application is split into two specialized environments to optimize performance and stay within free-tier resource limits:

### Main Application (Render)
* **Runtime:** Native Python (Non-Docker) to minimize overhead.
* **Stack:** FastAPI + LangGraph + SQLite.
* **Memory:** Optimized to run under 200MB, fitting easily within Render's 512MB free limit.
* **Responsibility:** Handles user authentication (`bcrypt`), job scouting (57k index), and the LangGraph state machine.

### Semantic Scorer Sidecar (Hugging Face Spaces)
* **Runtime:** Gradio + ZeroGPU.
* **Stack:** SBERT (Sentence-Transformers) + PyTorch.
* **Responsibility:** Performs the heavy lifting of ResumeHQ scoring. It uses GPU-accelerated embeddings to compare the semantic meaning of a resume against a job description.

### How They Communicate
The Render app acts as the orchestrator. When a user selects a job, the app sends an internal HTTP request to the Hugging Face Scorer via the `RQ_SCORER_URL`. This **Sidecar architecture** prevents the main web app from crashing due to high memory demands of ML models (OOM errors).

---

# 🧠 Deep Dive: What is ResumeHQ?

Unlike traditional ATS systems that look for exact keyword matches, this agent uses **ResumeHQ Semantic Scoring**:

* **Vector Embeddings:** Converts the full résumé and job description into high-dimensional vectors using SBERT.
* **Contextual Matching:** Understands that *"Machine Learning"* and *"Statistical Modeling"* are related, even if the exact words differ.
* **Layered Analysis:** After the semantic check, it layers on keyword density, readability scores, and domain-specific validation.

## Run the CLI instead:
Which module you run
server:app → the app (web UI + LangGraph). It only scores by either importing ResumeHQ in-process or calling the sidecar.
app:app from inside scorer_service/ → the actual local scorer.
```## Run the CLI instead
pip install -r requirements.txt
export GEMINI_API_KEY=...                  # optional; stubs used if unset
export JOBS_INDEX=data/jobs_index.jsonl    # full 57k; omit for the 3k sample

uvicorn server:app --reload
```

Then open **http://127.0.0.1:8000** — paste or upload a resume, pick a domain,
click *Find matching jobs*, select a job, review the ATS score, add info to
re-score, and generate a cover letter / tailored resume / interview questions.

The browser UI (`web/index.html`) is a single static file with no build step; it
talks to three FastAPI endpoints that map onto the graph's two interrupts:
`/api/start` (→ select_job interrupt), `/api/select` (→ review interrupt),
`/api/action` (add_info re-score / ask / generate / done).

### Run the CLI 

```bash
python run.py --resume my_resume.pdf --domain "Data Science"
```

`--domain` is `Data Science` or `Web Development`. `--resume` accepts `.txt/.md/.pdf/.docx`.
Omit `--resume` to run with a built-in demo profile.

## Files

| File | Role |
|------|------|
| `agent/graph.py` | assembles the StateGraph (fork, interrupts, conditional loop) |
| `agent/nodes.py` | scout, parser, match, select_job, ats_score, human_review, extras |
| `agent/matcher.py` | deterministic skill overlap + **weighted ATS** (required vs nice-to-have) |
| `agent/llm.py` | Gemini wrapper with no-key fallback |
| `agent/state.py` | shared graph state |
| `run.py` | interactive CLI driving the HITL loop |
| `data/jobs_index.jsonl` | slim 57k index built from your source dump |
| `data/jobs_sample.jsonl` | 3k sample for fast startup |

## New features (auth, filters, market intel)

### Persistence on Render (SQLite writes are ephemeral)

Render's container filesystem is wiped on every deploy/restart, so a *writable*
SQLite file does not persist and is not shared across instances. The fix is to
split by mutability:

- **Jobs** — read-only SQLite baked into the image (`APP_DB`). Immutable and
  rebuilt each deploy, so ephemeral storage is fine.
- **User data** (users / profiles / saved_jobs) — `agent/userstore.py`. Set
  `DATABASE_URL` and it uses **Postgres** (durable, shared); unset, it falls back
  to a local SQLite file (`USER_DB`, dev only).

`render.yaml` provisions a managed Postgres (`jobscout-db`) and injects its
`connectionString` as `DATABASE_URL`. No app code changes between local and prod
— same queries, different backend (`?`→`%s`, `SERIAL` vs `AUTOINCREMENT`, and
`RETURNING id` are handled in userstore). The Postgres path is written to spec
but was validated here only on the SQLite fallback (no Postgres in the build
sandbox) — verify on first deploy. Do NOT use `:memory:`; it lives only in RAM.

### Database as the single store (offloading the big JSON)

Yes — the ~377 MB source JSON should not ship or load at runtime. `data/build_db.py`
ingests it once into SQLite (`jobs` + auth tables); the app then only touches the
DB. As of now **SQLite is the single source**: job ranking, browse/filters, JD,
market intel, recommendations, auth and saved jobs all read from it, and the old
20 MB `jobs_index.jsonl` is gone. Ship just `data/app.db` (or the 13 MB
`app_sample.db`). Full DB with capped descriptions is ~196 MB vs the 377 MB JSON;
drop `--desc-cap` lower to shrink further.

### Recommended jobs

`GET /api/recommended` ranks jobs against the saved profile (skill overlap +
preferred-title match, filtered by preferred location / remote pref). The
dashboard shows the count and the home page lists the top matches after login.

### LangSmith tracing

Fully wired — see **TRACING.md** for a table of every graph node and traced
function (inputs, outputs, span type, and why). Toggle with
`LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY`; off = zero overhead.



Backend is SQLite-backed (`agent/db.py`) and all endpoints below are tested.

- **Auth** — `POST /api/auth/register|login` (bcrypt hashing + JWT), `GET /api/auth/me`.
  OAuth was removed (Google/GitHub). Instead a built-in **admin** account is
  seeded on startup from `ADMIN_EMAIL`/`ADMIN_PASSWORD` (default `admin@local` /
  `admin12345` — change in production). Admin-only endpoints use `is_admin`, e.g.
  `GET /api/admin/summary`.
- **Profile** — `GET/PUT /api/profile` (name, location, exp, skills, preferred
  titles/locations, remote|hybrid|onsite, experience level).
- **Saved jobs + dashboard** — `POST/DELETE /api/jobs/{id}/save` (status =
  saved|applied|interview), `GET /api/saved`, `GET /api/dashboard` (welcome +
  counts). UI shows the chips after login.
- **Source dropdown + filters** — `GET /api/facets`, `GET /api/jobs?source=&skills=
  &location=&min_exp=&max_exp=&domain=&remote=`. Sources are the dataset's REAL
  `via` values (LinkedIn dominates; Naukri/Indeed are barely present).
- **JD-on-select** — `GET /api/jobs/{id}` returns the full description; the AI
  match now scores against the real JD, not a synthesised skill line.
- **AI match card + "Why?"** — the ATS report carries `apply_verdict`
  (YES/MAYBE/NO), `apply_confidence`, `strengths`, `concerns` (from ResumeHQ HR).
- **Market intelligence** — `GET /api/market`: top skills, top companies, top
  locations, remote-vs-onsite, month-over-month emerging skills. Salary is NOT in
  the dataset (12/56,769 rows) so it is omitted rather than faked.
- **Guardrails** — `agent/guardrails.py` caps size and strips prompt-injection
  before text reaches the LLM/scorer; short/empty resumes are rejected (400).
- **LangSmith** — set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`
  (+ `LANGCHAIN_PROJECT`); LangGraph auto-traces every node.

### Build the jobs DB

```bash
# full 57k (production): needs the original dump
python data/build_db.py --source /path/to/query_result.json --out data/app.db
export APP_DB=data/app.db
```
A 4,000-job `data/app_sample.db` ships for instant startup. Set a strong
`JWT_SECRET` in production (the default warns).

## Resume scoring: ResumeHQ (via strict tools)

Resume scoring is done by **ResumeHQ** (PyPI: `resumehq`), imported in-process —
no separate MCP server. IMPORTANT: the package installs FLAT modules, so the
real import is:

```python
from mcp_scorer import score_ats, score_hr, explain_score, generate_cover_letter
```

`import resumehq` / `resumehq.ResumeBuilder()` do **not** exist (that was a wrong
snippet). Console script: `resumehq-mcp`.

All ResumeHQ access goes through `agent/rq_tools.py`, which enforces the three
MCP disciplines while running as a plain import:

- **Enumerate strict tools** — `TOOLS` is a fixed registry (`score_ats`,
  `score_hr`, `explain_score`, `generate_cover_letter`, `extract_text`). Unknown
  names and missing required args are rejected before anything runs.
- **Force ground truth** — the scorer's raw dict is returned unchanged. The app
  never invents a score, keyword, or tip. If `resumehq` isn't installed (or a
  tool has no data) callers get an explicit `{"error": ...}` — never a fake number.
- **Audit trail** — every call is logged (id, UTC timestamp, tool, input SHA +
  length, returned keys, score, latency) to memory and an append-only JSONL
  (`RQ_AUDIT_LOG`). View it at `GET /api/audit` or the "view audit trail" link.

The old custom scorer is retired for scoring; the fast skill-overlap matcher is
kept only for **job ranking** (retrieval over 57k rows — not resume scoring).

`score_ats` gives `total_score`, `matched_keywords`, `missing_keywords`, `rating`;
`score_hr` gives `overall_score`, `recommendation`, `strengths`, `concerns`, and
`suggested_questions` (used verbatim for the Interview Questions output —
grounded, not model-invented).

### Memory & deployment profiles (the 512 MB / OOM question)

`import mcp_scorer` is light — measured at ~76 MB, and it does **not** load
torch: ResumeHQ imports its local scorers lazily, only when a score falls
through to the local engine. The heavy cost (sentence-transformers + torch +
SBERT model, well over 512 MB) is paid **only on the local path**. So:

| Profile | Install | Scoring | RAM | Fits 512 MB? |
|---------|---------|---------|-----|--------------|
| **Cloud (production)** | `pip install resumehq --no-deps` then `pip install fastmcp` | Hosted scorer (`resume-scorer.fly.dev`) | ~76 MB + app | **Yes** |
| **Local (dev/offline)** | `pip install resumehq` (pulls torch) | In-process SBERT | > 512 MB | No — needs its own process |

**Production recipe (single process, under budget):**

```bash
pip install -r requirements.txt          # app deps incl. fastmcp (no torch)
pip install resumehq --no-deps           # ResumeHQ modules only, torch NOT pulled
export SCORER_CLOUD_API_KEY=...          # from getresumehq.com (free tier = 5 scores)
export RQ_CLOUD_ONLY=1                    # hard-disable the local fallback
uvicorn server:app
```

With `RQ_CLOUD_ONLY=1`, if a cloud call misses, `rq_tools` blocks the local
scorer from importing torch and returns an explicit audited error instead of
OOM-killing the container. Verified: cloud-miss under this flag stays at ~99 MB
with torch never loaded.

**If you need offline/self-hosted scoring** you cannot keep it in the same
512 MB process — torch+SBERT alone exceed it. Run ResumeHQ's scorer as a
separate process with its own memory budget and call it over HTTP:

```bash
# scorer container (full deps, its own memory):  resumehq-mcp    (the bundled MCP server)
# or a thin HTTP wrapper exposing score_ats/score_hr; point the app at it.
```

This is the one place the memory limit forces a process split — everything else
stays in the single app runtime.

## Self-hosted scorer sidecar (unlimited, no free-tier cap)

`scorer_service/` is a ready-to-run FastAPI wrapper around ResumeHQ's LOCAL
engines. Because it scores locally, it has **no 5-score cap** (that limit lives
in the ResumeHQ cloud) and needs no `SCORER_CLOUD_API_KEY`.

Topology: two containers, two memory budgets.

```
  ┌────────────┐   RQ_SCORER_URL   ┌──────────────────┐
  │  app        │ ───HTTP─────────► │  scorer sidecar   │
  │  ~512 MB    │  /score/ats …     │  full torch/SBERT │
  │  torch-free │ ◄──JSON────────── │  own (larger) RAM │
  └────────────┘                    └──────────────────┘
```

Bring both up:

```bash
docker compose up --build      # app on :8000, scorer internal on :8100
```

`docker-compose.yml` sets `RQ_SCORER_URL=http://scorer:8100` on the app and caps
the app at 512 MB / the scorer at 2 GB. The app never imports torch (verified
~174 MB); every score is an HTTP call, audited with `"via": "remote"`.

Endpoints: `POST /score/ats`, `/score/hr`, `/explain`, `/cover-letter`, `GET /health`.
Cover letters need an LLM: ResumeHQ's local path can't generate them (the wheel
omits `llm_scorer`), so the app falls back to Gemini for the letter, grounded on
the final resume.

## Deploy on Render

Render doesn't read `docker-compose.yml` — it deploys each service from its own
Dockerfile, wired by `render.yaml` (included). The compose file is kept for
LOCAL dev only.

`render.yaml` defines two services that match the memory split:

| Service | Render type | Plan | torch? | Public? |
|---------|-------------|------|--------|---------|
| `job-scout-app` | `web` | free (512 MB) | no | yes |
| `resumehq-scorer` | `pserv` (private) | paid (~2 GB) | yes | no (internal only) |

Deploy: push this repo to GitHub → Render → **New → Blueprint** → pick the repo.
Render builds both, and injects the scorer's internal `host:port` into the app's
`RQ_SCORER_URL` (rq_tools adds the `http://`). Set `GEMINI_API_KEY` (app) and
optional `ANTHROPIC_API_KEY` (scorer, for cover letters) in the dashboard.

**Feasibility, stated plainly:** you cannot run everything on one free 512 MB
Render service — torch+SBERT alone exceed it and Render will OOM-kill the
instance. Your options:

1. **App free + scorer paid private service** (this `render.yaml`) — unlimited
   local scoring, no cloud cap. Costs the scorer plan.
2. **App free + ResumeHQ cloud** (`SCORER_CLOUD_API_KEY`, `RQ_CLOUD_ONLY=1`, no
   scorer service) — $0 infra but capped at the cloud free tier (5 scores) until
   you buy ResumeHQ Pro.
3. **Scorer off-Render** (any host with ≥2 GB) + app free on Render pointing
   `RQ_SCORER_URL` at it.

Plan names and free-tier terms change — confirm current Render docs before you
rely on the numbers above.

Run the sidecar without Docker:

```bash
cd scorer_service && pip install resumehq fastapi "uvicorn[standard]"
uvicorn app:app --host 0.0.0.0 --port 8100
# then, for the app:  export RQ_SCORER_URL=http://localhost:8100
```


## The revise loop

After scoring, the human-in-the-loop review lets you, repeatedly:
- **add_info** — describe experience to add; the agent folds it into the working
  resume, **re-scores**, and shows *how to add it* as a resume bullet.
- **upload_resume** — upload a fully revised resume; the agent re-scores it.

Loop until satisfied, then pick **cover letter / interview questions / tailored
resume** — all computed against the *final* resume text.

## ATS scoring note

Skill *counts* are deterministic and reproducible — the LLM only writes the
explanations and (when a key is set) splits each job's skills into **required vs
nice-to-have** so short postings don't inflate to 100%. Postings with <3 listed
skills are marked `low` confidence and discounted. Verified: a 4-required-skill job
scores 50 with 2 hits, then 100 after the user adds the 4 missing skills via HITL.
