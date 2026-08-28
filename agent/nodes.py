"""LangGraph nodes.

Job *ranking* (scout/match) still uses the fast local skill-overlap matcher —
that is retrieval over 57k rows, not resume scoring. Resume *scoring* now goes
entirely through ResumeHQ via the strict-tool boundary in rq_tools (ATS + HR +
explain). The old custom ATS scorer is no longer used for scoring.
"""
from langgraph.types import interrupt
from . import matcher, llm, rq_tools, db
from .trace import traceable

import asyncio

# ATS scoring via ResumeHQ strict tools
from concurrent.futures import ThreadPoolExecutor

# parallel branch 1: SCOUT (retrieval, no profile needed)
def scout_node(state):
    pool = matcher.filter_pool(domain=state.get("domain"), location=state.get("location"))
    return {"candidate_pool_ids": [j["id"] for j in pool]}


# parallel branch 2: PARSER (lightweight profile for RANKING only)
def parser_node(state):
    text = state.get("resume_text", "") or ""
    return {"profile": _parse_resume(text), "resume_text": text}


@traceable(run_type="chain", name="parse_resume")
def _parse_resume(text: str) -> dict:
    if llm.available() and text.strip():
        data = llm.generate(
            f"Extract this resume into JSON: skills (short tech tokens), "
            f"exp_years (int|null), seniority, summary.\n\nRESUME:\n{text[:6000]}",
            system="Precise resume parser. Return ONLY JSON.", as_json=True)
        if isinstance(data, dict) and data.get("skills"):
            data.setdefault("exp_years", None)
            return data
    known = {"python", "sql", "pytorch", "tensorflow", "pandas", "numpy",
             "scikit-learn", "docker", "kubernetes", "react", "node", "aws",
             "gcp", "azure", "nlp", "machine learning", "fastapi", "langchain"}
    low = text.lower()
    return {"skills": sorted({k for k in known if k in low}),
            "exp_years": None, "summary": ""}


# JOIN: rank the scouted pool against the parsed profile
def match_node(state):
    ids = state.get("candidate_pool_ids", [])
    pool = db.get_jobs_by_ids(ids) if ids else matcher.filter_pool(
        state.get("domain"), state.get("location"))
    return {"matches": matcher.score_pool(state["profile"], pool)}


def select_job_node(state):
    payload = interrupt({"type": "select_job", "matches": state["matches"],
                         "prompt": "Select a job_id to score your resume against."})
    job_id = payload if isinstance(payload, str) else payload.get("job_id")
    return {"selected_job_id": job_id}


def build_jd_text(job) -> str:
    """Grounded JD text assembled from the indexed job fields (not invented)."""
    return (f"Job Title: {job.get('title','')}\n"
            f"Company: {job.get('company','')}\n"
            f"Location: {job.get('location','')}\n"
            f"Domain: {job.get('domain','')}\n"
            f"Employment: {job.get('emp','')}\n"
            f"Required skills: {', '.join(job.get('skills', []))}\n")




@traceable(run_type="chain", name="ats_score_node")
def ats_node(state):
    job = matcher.get_job(state["selected_job_id"])
    full = None
    try:
        full = db.get_job(state["selected_job_id"])
    except Exception:
        pass
    jd_text = (full or {}).get("description") or build_jd_text(job)
    resume_text = state.get("resume_text", "")

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_ats  = ex.submit(rq_tools.call, "score_ats",     resume_text=resume_text, jd_text=jd_text)
        f_hr   = ex.submit(rq_tools.call, "score_hr",      resume_text=resume_text, jd_text=jd_text)
        f_exp  = ex.submit(rq_tools.call, "explain_score", resume_text=resume_text, jd_text=jd_text)
        ats, hr, expl = f_ats.result(), f_hr.result(), f_exp.result()

    return {"ats": _reshape(full or job, jd_text, ats, hr, expl, resume_text),
            "jd_text": jd_text}

_SKILL_ALIASES = {
    "machine learning": ["ml"], "deep learning": ["dl"],
    "natural language processing": ["nlp"], "large language models": ["llm", "llms"],
    "javascript": ["js"], "typescript": ["ts"], "postgresql": ["postgres"],
    "kubernetes": ["k8s"], "scikit-learn": ["sklearn", "sci-kit"],
    "amazon web services": ["aws"], "google cloud": ["gcp"],
}


def _present(skill: str, rlow: str) -> bool:
    s = skill.lower().strip()
    if not s:
        return False
    if s in rlow:
        return True
    for full, als in _SKILL_ALIASES.items():
        if s == full and any(a in rlow for a in als):
            return True
        if s in als and full in rlow:
            return True
    return False


def _reshape(job, jd_text, ats, hr, expl, resume_text=""):
    """matched / missing come from the JOB's clean skill tags vs the resume
    (real skills, ✓/✗). ResumeHQ still supplies the % score, HR verdict, advice."""
    err = ats.get("error") or hr.get("error")
    rlow = (resume_text or "").lower()

    job_skills = [s for s in (job.get("skills") or []) if s and len(s) < 40]
    # de-dupe case-insensitively, keep original casing, cap for display
    seen, clean = set(), []
    for s in job_skills:
        k = s.lower()
        if k not in seen:
            seen.add(k); clean.append(s)
    matched_skills = [s for s in clean if _present(s, rlow)][:15]
    missing_skills = [s for s in clean if not _present(s, rlow)][:15]

    matched = [f"Both resume and job include {s}." for s in matched_skills[:8]]
    missing = [f"The job requires {s}, absent from the resume." for s in missing_skills[:8]]

    advice = []
    explanation = expl.get("explanation") if isinstance(expl, dict) else None
    if isinstance(explanation, dict):
        for key in ("quick_wins", "suggestions", "section_tips"):
            v = explanation.get(key)
            if isinstance(v, list):
                advice.extend(str(x) for x in v[:3])
            elif isinstance(v, str):
                advice.append(v)
    if missing_skills:
        advice.insert(0, f"Add these skills where you can honestly claim them: "
                         f"{', '.join(missing_skills[:6])}.")
    for s in (hr.get("strengths") or [])[:2]:
        advice.append(f"Keep emphasising: {s}")

    return {
        "job_id": job["id"], "job_title": job["title"], "company": job["company"],
        "ats_score": ats.get("total_score"), "hr_score": hr.get("overall_score"),
        "rating": ats.get("rating"), "likelihood": ats.get("likelihood"),
        "recommendation": hr.get("recommendation"),
        "apply_verdict": _apply_verdict(hr.get("recommendation")),
        "apply_confidence": hr.get("overall_score"),
        "strengths": hr.get("strengths", []), "concerns": hr.get("concerns", []),
        "matched": matched, "missing": missing, "advice": advice,
        "suggested_questions": hr.get("suggested_questions", []),
        "error": err,
        "_raw": {"matched_keywords": matched_skills, "missing_keywords": missing_skills},
    }


def _apply_verdict(rec):
    return {"INTERVIEW": "YES", "MAYBE": "MAYBE", "PASS": "NO"}.get(
        (rec or "").upper(), "MAYBE")


# HITL review: re-score on add_info / re-upload; else pick an output
def human_review_node(state):
    payload = interrupt({
        "type": "review", "ats": state["ats"],
        "options": ["add_info", "upload_resume", "ask", "cover_letter",
                    "tailored_resume", "interview_questions", "done"],
        "prompt": "Add info, upload a revised resume to re-score, ask, or pick an output.",
    })
    action = payload.get("action", "done") if isinstance(payload, dict) else str(payload)
    update = {"next_action": action}
    # a BYO key saved mid-session arrives on the payload — persist it to state so
    # extras_node/_how_to_add use it instead of falling back to server Gemini
    if isinstance(payload, dict) and payload.get("llm_creds"):
        update["llm_creds"] = payload["llm_creds"]

    if action == "add_info":
        info = payload.get("text", "")
        update["resume_text"] = (state.get("resume_text", "") +
                                 "\n\nADDITIONAL EXPERIENCE:\n" + info)
        update["user_edit"] = info
        update["how_to_add"] = _how_to_add(info, state.get("ats", {}), state)
    elif action == "upload_resume":
        update["resume_text"] = payload.get("resume_text", state.get("resume_text", ""))
    elif action == "ask":
        update["user_question"] = payload.get("text", "")
    return update


def _how_to_add(info: str, ats: dict, state: dict = None) -> str:
    missing = ats.get("_raw", {}).get("missing_keywords", [])
    covered = [k for k in missing if k.lower() in info.lower()]
    creds = (state or {}).get("llm_creds") or {}
    if creds.get("api_key") or llm.available():
        return _gen(state,
            f"Candidate wants to add: '{info}'.\nMissing job keywords: {missing}.\n"
            f"Write ONE STAR-style resume bullet (with a metric placeholder) that "
            f"incorporates this and surfaces the relevant missing keywords. Bullet only.")
    kws = f" (surfaces: {', '.join(covered)})" if covered else ""
    return f"Add under Experience/Projects: {info.strip()}{kws}"


# extras: cover letter / interview Qs / tailored resume / Q&A
def _gen(state, prompt, system=""):
    """Prefer the user's uploaded key/model (llm_user); fall back to Gemini (llm.py)
    only if no key is set."""
    creds = (state or {}).get("llm_creds") or {}
    if creds.get("api_key"):
        try:
            from . import llm_user
            return llm_user.chat(creds.get("provider", "openai"), creds["api_key"],
                                 creds.get("model", ""),
                                 system or "You are a helpful career assistant.",
                                 prompt, base_url=creds.get("base_url", ""))
        except Exception as e:
            return f"(Assistant error: {e})"
    if llm.available():
        return llm.generate(prompt, system=system)
    return "(No assistant key configured — add your API key in Settings to generate this.)"


def extras_node(state):
    action = state.get("next_action")
    job = matcher.get_job(state["selected_job_id"])
    jd_text = state.get("jd_text", build_jd_text(job))
    resume_text = state.get("resume_text", "")
    artifacts = dict(state.get("artifacts", {}))
    answer = None

    if action == "ask":
        answer = _gen(state,
            f"Resume:\n{resume_text[:4000]}\nQuestion: {state.get('user_question')}\nAnswer concisely.",
            system="Career coach. Ground answers in the resume text only.")
    elif action == "cover_letter":
        res = rq_tools.call("generate_cover_letter", resume_text=resume_text,
                            jd_text=jd_text, company_name=job.get("company", ""),
                            job_title=job.get("title", ""))
        letter = res.get("full_text")
        if not letter:   # ResumeHQ letters need cloud/Pro; draft with the user's LLM
            letter = _gen(state,
                f"Write a concise, specific cover letter for {job['title']} at "
                f"{job['company']}. Ground it ONLY in this resume:\n{resume_text[:4000]}\n"
                f"Job:\n{jd_text}\nDo not invent experience.")
        artifacts["cover_letter"] = letter
    elif action == "interview_questions":
        qs = state.get("ats", {}).get("suggested_questions") or []
        if not qs:
            hr = rq_tools.call("score_hr", resume_text=resume_text, jd_text=jd_text)
            qs = hr.get("suggested_questions", [])
        if not qs:   # last resort: generate with the user's LLM
            txt = _gen(state,
                f"List 6 likely interview questions for a {job['title']} role at "
                f"{job['company']} given this resume:\n{resume_text[:3000]}\n"
                f"Return one question per line, no numbering.")
            qs = [l.strip(" -*0123456789.") for l in txt.splitlines() if l.strip()][:8]
        artifacts["interview_questions"] = qs
    elif action == "tailored_resume":
        missing = state.get("ats", {}).get("_raw", {}).get("missing_keywords", [])
        artifacts["tailored_resume"] = _gen(state,
            f"Rewrite a resume summary + 5 bullets for {job['title']} at "
            f"{job['company']}. Base ONLY on this resume:\n{resume_text[:4000]}\n"
            f"Incorporate any of these missing keywords the candidate can honestly "
            f"claim: {missing}. Do not invent experience.")

    out = {"artifacts": artifacts}
    if answer is not None:
        out["answer"] = answer
    return out
