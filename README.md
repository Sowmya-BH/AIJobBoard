# Job Scout — AI Job-Matching & ATS Agent

### **An AI-powered job board and résumé-scoring agent that is a AI heavy system (57,000 jobs + deep scoring) is managed on a tiny 512MB RAM budget by offloading math to the cloud.
BusineUpload a résumé, get semantically-ranked matches, score it against any job with a real ATS/HR model, and generate cover letters, interview questions and tailored résumés — using **your own** LLM key** that uses Python and Fast API in the backend.

An AI-powered job board and résumé-scoring agent on ~57,000 real job listings.


Built to run **production-grade on free tiers**: the web app fits inside a
**512 MB Render instance** .




> **Production-Grade on Free Tiers:** Built to run efficiently — the entire web application fits comfortably inside a **512 MB Render free instance**.

---


### 🌐 Live Deployments
> 🚀 **Web Application (Render):**  
> **[https://aijobboard-o52o.onrender.com](https://aijobboard-o52o.onrender.com)**  
>
> ⚡ **ATS Resume Scorer (Hugging Face Spaces):**  
> **[https://rajuiscoding-resume-parser.hf.space](https://rajuiscoding-resume-parser.hf.space)**
> 
> 📊 **Observability Tracing Enabled	LangSmith Dashboard.**

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | **LangGraph** (stateful graph, human-in-the-loop interrupts, checkpointer) |
| API / web server | **FastAPI** + **Uvicorn** |
| Jobs + user data | **MongoDB Atlas** (single database, indexed) |
| Semantic search | **Qdrant Cloud** (vector DB) + **Sentence-BERT** `all-mpnet-base-v2` (768-dim) |
| Résumé ATS/HR scoring | **ResumeHQ** (SBERT/torch) on a **Hugging Face Gradio + ZeroGPU** Space |
| Auth | **JWT** (PyJWT) + **bcrypt** password hashing |
| Secret storage | **Fernet** application-level encryption (`cryptography`) |
| Bring-your-own LLM | OpenAI · Anthropic · Groq · any OpenAI-compatible endpoint |
| Server-side LLM | **Google Gemini** (résumé parsing, chat fallback) |
| Observability | **LangSmith** tracing (optional, no-op unless enabled) |
| Frontend | Vanilla HTML/JS (auth-gated single-page app) |

---

## Architecture

<img width="2014" height="1816" alt="1E19A70B-6385-458F-B7A2-91C899F62213" src="https://github.com/user-attachments/assets/80c020db-4a07-4432-b3b4-0e30cf4b19b5" />





```
                      Browser (SPA)
                           |  JWT
                           v
                 +-------------------+
                 |  FastAPI (Render) |  ~150 MB, torch-free, 512 MB tier
                 |  LangGraph agent  |
                 +----+----+----+----+
                      |    |    |
      +---------------+    |    +---------------+
      v                    v                    v
MongoDB Atlas         Qdrant Cloud       HF Gradio Space
 jobs + users       vectors (SBERT)   ResumeHQ ATS/HR Assistant
  (indexed)        semantic search    (torch/SBERT, ZeroGPU)
                          ^
                          | (REST via HF Inference Client)
                          +---------------------+
```

**Agent graph (LangGraph):**
`scout || parser -> match -> select_job (interrupt) -> ats_score -> human_review
(interrupt) -> {re-score | cover letter | interview Qs | tailored résumé | Q&A}`

Two `interrupt()` points make it genuinely human-in-the-loop: the user picks a
job, sees the ATS/HR result, then chooses the next action — all on one
checkpointed thread.

---

## Optimizations (the interesting part)

Every choice below exists to keep the app **small, cheap and fast** while still
running heavyweight NLP.

### 1. Everything heavy is offloaded — the app stays under 512 MB
- **No data in the image.** The 395 MB source JSON is ingested **once** into
  MongoDB Atlas; the app connects by URI. No SQLite/JSON is shipped, so cold
  starts are fast and there's no ephemeral-storage risk.
- **No torch in the app.** ResumeHQ (SBERT + torch, ~2 GB) runs on a separate
  **Hugging Face Gradio/ZeroGPU** Space and is called over HTTP. The app installs
  ~13 light packages and never imports torch, `sentence-transformers`, or
  `onnxruntime`.
- **`runtime: python` on Render** installs only `requirements.txt` and ignores
  every Dockerfile in the repo, so the heavy scorer image can't accidentally be
  built into the app.

### 2. Semantic search that fits a tiny box:
- Is semantic search required?  If I used standard keyword search, users would miss 60% of relevant jobs   due to different naming conventions. By using LangGraph, I created a system that can pause and wait      for human judgment, and by using Qdrant, I ensured that the retrieval of those 57,000 jobs remains       sub-second and conceptually accurate."
- Jobs are embedded **offline on a dev machine** with local Sentence-BERT and
  upserted to **Qdrant Cloud** (~225 MB for 57k × 768-dim — under the 1 GB free
  tier).
- At query time the app embeds the résumé via the **HF Inference API** (one REST
  call, `<10 MB` RAM) — the same model on both sides, so vectors are comparable.
  No model is ever loaded in the app.
- **Automatic fallback:** if Qdrant/HF are unset or unreachable, ranking silently
  falls back to a fast local **skill-overlap** matcher. This also lets you **A/B
  semantic vs keyword** accuracy just by toggling env vars.

### 3. One database, indexed, schemaless
- MongoDB holds **both** jobs and users. Filters (source/location/experience/
  remote/skills) run as **indexed queries and aggregations** server-side instead
  of scanning rows in Python. A lowercased `skills_lc` array powers fast `$all`
  skill filtering and skill-frequency market analytics.

### 4. Bring-your-own-key LLM (cost control)
- Users supply their own OpenAI/Anthropic/Groq key; it's **validated, then
  Fernet-encrypted at rest** (never stored in plaintext, never returned to the
  client). The app spends **zero** LLM budget on user generations.
- Keys are injected into the agent graph and honored **mid-session** (add a key
  after scoring and the next action uses it), with a graceful fallback to
  server-side Gemini when no key is set.

### 5. Cheap, free embeddings for indexing
- Sentence-BERT indexing runs locally (no API cost). The query-time HF Inference
  call is free-tier friendly, and the whole 57k index builds in minutes.

### 6. Offline NLP enrichment (spaCy + Gensim)
- A one-time ingestion script adds a `canonical_skills` field using **spaCy**
  (lemmatization + noun/tool extraction) and **Gensim Phrases** (so
  "machine learning" is one token and won't match "washing machine"). Runs
  offline; the app never imports spaCy.

### 7. Grounded, safe scoring
- ResumeHQ, a conversational AI resume builder with insustry standard ATS optimizated score generation     and skills listing,is behind a **strict tool boundary** (`rq_tools`) with an audit trail;on any scorer   error the app surfaces it instead of fabricating a score.
- Résumé text is guardrail-sanitized before reaching any LLM/scorer, and logs are
  secret-redacted.

---

## Repository layout

```
server.py                     FastAPI app + all routes
agent/
  graph.py                    LangGraph wiring
  nodes.py                    graph nodes (scout, match, ATS, review, extras)
  state.py                    graph state (incl. per-session BYO creds)
  db.py                       MongoDB jobs access (queries, facets, market intel)
  userstore.py                MongoDB users / profiles / saved jobs
  auth.py                     JWT + bcrypt
  crypto.py                   Fernet encryption for stored API keys
  llm.py                      server-side Gemini (parsing, chat fallback)
  llm_user.py                 BYO-key LLM adapter (OpenAI/Anthropic/Groq/custom)
  vectors.py                  Qdrant + HF-Inference SBERT embeddings (query side)
  matcher.py                  skill-overlap ranking (fallback)
  rq_tools.py                 strict ResumeHQ tool boundary (REST or Gradio)
  guardrails.py, trace.py, config.py
data/
  build_mongo.py              one-time: raw JSON -> MongoDB Atlas
  build_embeddings.py         one-time: jobs -> SBERT -> Qdrant (local encode)
  build_canonical_skills.py   one-time: spaCy/Gensim skill enrichment
web/                          landing / login / app (HTML/JS)
scorer_service/               ResumeHQ scorer for Hugging Face (Gradio/ZeroGPU)
render.yaml                   Render Blueprint (python runtime, app-only)
```

---

## Setup

### 1. One-time data load (run locally, not on Render)
```bash
pip install ijson pymongo
export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority"
export MONGODB_DB=jobscout
python data/build_mongo.py --source /path/to/query_result.json --desc-cap 1500 --drop
```

### 2. Optional: build the vector index (semantic search)
```bash
pip install sentence-transformers
export QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
export QDRANT_API_KEY=...
python data/build_embeddings.py --limit 2 --recreate --batch 2   # dry run
python data/build_embeddings.py --recreate                        # full 57k
```

### 3. Optional: skill enrichment
```bash
pip install spacy gensim && python -m spacy download en_core_web_sm
python data/build_canonical_skills.py
```

### 4. Deployment architecture the scorer (Hugging Face Space):

it is available on :

https://aijobboard-o52o.onrender.com 

https://huggingface.co/spaces/RajuisCODING/resume-parser
Create a **Docker** or **Gradio** Space from `scorer_service/`, select **ZeroGPU**
hardware. Note the Space URL.

### 5. Run the app
```bash
pip install -r requirements.txt
export MONGODB_URI=...            # required
export RQ_SCORER_KIND=gradio RQ_SCORER_URL=https://<user>-<space>.hf.space
export GEMINI_API_KEY=...  GEMINI_MODEL=gemini-2.0-flash
# optional semantic search:
export QDRANT_URL=... QDRANT_API_KEY=... HF_TOKEN=... \
       HF_EMBED_URL=https://router.huggingface.co/hf-inference/models/sentence-transformers/all-mpnet-base-v2/pipeline/feature-extraction
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Deploy to Render
Push to GitHub -> **New -> Blueprint** (uses `render.yaml`, `runtime: python`).
Set the `sync: false` env vars in the dashboard. Start command:
`uvicorn server:app --host 0.0.0.0 --port $PORT`.

---

## Environment variables

| Variable | Purpose | Required |
|---|---|---|
| `MONGODB_URI` / `MONGODB_DB` | jobs + users database | yes |
| `RQ_SCORER_KIND` / `RQ_SCORER_URL` | ResumeHQ scorer (`gradio` + Space URL) | for scoring |
| `RQ_SCORER_HF_TOKEN` | token for a private scorer Space | optional |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | résumé parsing + chat fallback | recommended |
| `QDRANT_URL` / `QDRANT_API_KEY` | vector search | optional |
| `HF_TOKEN` / `HF_EMBED_URL` | résumé embeddings (HF Inference) | optional |
| `JWT_SECRET` / `APP_ENCRYPTION_KEY` | auth + key encryption (auto-generated on Render) | yes |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | seeded admin account | optional |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` | LangSmith tracing | optional |

Semantic search activates only when `QDRANT_URL`, `QDRANT_API_KEY` and `HF_TOKEN`
are all set; otherwise the app uses skill-overlap ranking.

---

## Notes & limitations:
- **Free-tier sizing:** 57k jobs *with full descriptions* can exceed Atlas M0
  (512 MB) — cap descriptions on ingestion.
- **Scorer cold start:** a sleeping HF Space adds a few seconds on the first
  score; the app degrades gracefully if it's unreachable.
- **This dataset has no salary field** — market analytics omit salary by design.

## Project Deliverables:
Watch the complete breakdown of the system architecture, database design, vector pipeline, and live ATS scoring workflow:
- https://drive.google.com/file/d/1JYQWHpjIQTB_dCtAbW0cuwc_N6rdGeQa/view?usp=sharing 

- My email id : bhupatirajusowmya@gmail.com
