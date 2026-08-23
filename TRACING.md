# LangSmith Tracing — what is traced and why

Enable:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_PROJECT=job-scout-agent   # optional; groups runs
```

When these are set, `agent/trace.py` swaps in LangSmith's real `@traceable`; when
unset it's a zero-overhead identity decorator, so behaviour is identical either
way. Two layers of spans are produced:

1. **Graph nodes** — LangGraph auto-instruments every node it runs. Each appears
   as a child run under the graph invocation, in execution order, with the state
   delta it returned.
2. **Function spans** — hot functions are additionally wrapped with `@traceable`
   so the LLM calls, the ResumeHQ tool boundary, and the ranking/recommendation
   logic show up as their own typed spans (llm / tool / retriever / chain).

## Graph nodes (auto-traced by LangGraph)

| Node | File | Input (state keys read) | Output (state keys written) | Purpose |
|------|------|-------------------------|-----------------------------|---------|
| `scout` | nodes.scout_node | `domain`, `location` | `candidate_pool_ids` | Coarse retrieval: pull candidate job IDs from SQLite by domain/location. Runs in parallel with `parser`. |
| `parser` | nodes.parser_node | `resume_text` | `profile`, `resume_text` | Extract a lightweight skills/exp profile from the résumé (for ranking). Parallel with `scout`. |
| `match` | nodes.match_node | `candidate_pool_ids`, `profile` | `matches` | Join point: score the scouted pool against the profile (skill overlap) → ranked matches. |
| `select_job` | nodes.select_job_node | `matches` (+ resumed `job_id`) | `selected_job_id` | **interrupt()** — waits for the user to pick a job. |
| `ats_score` | nodes.ats_node | `selected_job_id`, `resume_text` | `ats`, `jd_text` | Score the résumé vs the selected job's real JD via ResumeHQ (ATS+HR+explain). |
| `human_review` | nodes.human_review_node | `ats` (+ resumed action) | `next_action`, `resume_text`, `how_to_add`, … | **interrupt()** — user adds info / re-uploads / asks / picks an output. |
| `extras` | nodes.extras_node | `next_action`, `resume_text`, `jd_text` | `artifacts`, `answer` | Cover letter / interview Qs / tailored resume / Q&A. |

Conditional edge after `human_review`: `add_info`/`upload_resume` → back to
`ats_score` (re-score loop); `done` → END; else → `extras`.

## Function spans (explicit `@traceable`)

| Span name | Type | File | Inputs | Output | Why traced |
|-----------|------|------|--------|--------|-----------|
| `gemini_generate` | llm | llm.generate | prompt, system, as_json | text or JSON | Every LLM call (resume parse, advice, how-to-add, cover letter, Q&A, tailored resume). Lets you inspect prompts/latency/cost per call. |
| `resumehq_tool` | tool | rq_tools.call | tool name + args | ResumeHQ raw dict | The strict-tool boundary to ResumeHQ (score_ats/hr/explain/cover-letter). Pairs with the built-in audit log; the span shows exactly what the scorer saw and returned. |
| `rank_jobs` | tool | matcher.score_pool | profile, candidate pool | ranked matches | The skill-overlap ranking used by `match`. Trace shows pool size and top scores. |
| `recommend_jobs` | retriever | db.recommend_jobs | user_id (→ profile) | scored recommendations | Personalised dashboard recommendations. Trace shows which profile signals drove the list. |
| `parse_resume` | chain | nodes._parse_resume | resume text | profile dict | Résumé→profile extraction (calls `gemini_generate` inside, so you see the nesting). |
| `ats_score_node` | chain | nodes.ats_node | state | ats report | Groups the three ResumeHQ tool calls + reshaping into one parent span per scoring pass. |

## Reading a trace

A full scoring session looks like:

```
graph.invoke
├─ scout
├─ parser
│  └─ parse_resume
│     └─ gemini_generate
├─ match
│  └─ rank_jobs
├─ select_job                 (interrupt)
├─ ats_score_node
│  ├─ resumehq_tool (score_ats)
│  ├─ resumehq_tool (score_hr)
│  └─ resumehq_tool (explain_score)
└─ human_review               (interrupt)
   └─ (loop back to ats_score_node, or → extras → gemini_generate)
```

Guardrails (`agent/guardrails.py`) run in the HTTP layer before the graph, so
sanitised input is what every span above receives.
