"""FastAPI app: LangGraph agent + auth + jobs/filters + market intel + saved jobs.

Scoring maps onto the graph's interrupts:
  /api/start -> select_job interrupt ; /api/select -> review ; /api/action -> loop
Auth is JWT (bcrypt hashing). Jobs/filters/market/saved/dashboard use SQLite.
Guardrails sanitise resume text before it reaches the LLM/scorer.
"""
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph.types import Command

from agent.graph import build_graph
from agent import db, auth, guardrails, rq_tools, userstore, crypto, llm_user

import logging as _logging
import re as _re
class _RedactSecrets(_logging.Filter):
    _pat = _re.compile(r'(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|"api_key"\s*:\s*"[^"]*")')
    def filter(self, rec):
        try:
            if isinstance(rec.msg, str):
                rec.msg = self._pat.sub("[REDACTED]", rec.msg)
            if rec.args:
                rec.args = tuple(self._pat.sub("[REDACTED]", str(a)) if isinstance(a, str) else a
                                 for a in rec.args)
        except Exception:
            pass
        return True
for _n in ("", "uvicorn", "uvicorn.access", "uvicorn.error"):
    _logging.getLogger(_n).addFilter(_RedactSecrets())

app = FastAPI(title="Job Scout ATS Agent")
GRAPH = build_graph()
userstore.init()  # create user tables (sqlite fallback or Postgres)

# Seed a built-in admin account (replaces OAuth). Configure via env.
_ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@local")
_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin12345")
if not userstore.get_user_by_email(_ADMIN_EMAIL):
    userstore.create_user(_ADMIN_EMAIL, auth.hash_password(_ADMIN_PASSWORD),
                          name="Admin", is_admin=True)
HERE = os.path.dirname(__file__)


def _cfg(tid): return {"configurable": {"thread_id": tid}}
def _interrupt(r): return r["__interrupt__"][0].value if "__interrupt__" in r else None


# ── schemas ──
class StartReq(BaseModel):
    resume_text: str = ""
    domain: str | None = None
    location: str | None = None


class SelectReq(BaseModel):
    thread_id: str
    job_id: str


class ActionReq(BaseModel):
    thread_id: str
    action: str
    text: str = ""
    resume_text: str = ""


class RegisterReq(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginReq(BaseModel):
    email: str
    password: str


class AnalyzeReq(BaseModel):
    resume_text: str
    source: str | None = None
    location: str | None = None
    domain: str | None = None
    remote: bool | None = None
    min_exp: int | None = None
    max_exp: int | None = None
    skills: str = ""          # comma-separated, from the filter box


class ApiKeyReq(BaseModel):
    provider: str          # openai | anthropic | gemini | custom | test
    model: str = ""
    api_key: str
    base_url: str = ""     # required when provider == custom (OpenAI-compatible)


class ChatReq(BaseModel):
    message: str
    job_id: str | None = None
    history: list = []


class ProfileReq(BaseModel):
    location: str = ""
    exp_years: int | None = None
    skills: list[str] = []
    preferred_titles: list[str] = []
    preferred_locations: list[str] = []
    work_pref: str = ""          # remote | hybrid | onsite
    experience_level: str = ""   # entry | mid | senior


# ── auth ──
@app.post("/api/auth/register")
def register(req: RegisterReq):
    if userstore.get_user_by_email(req.email):
        raise HTTPException(409, "Email already registered")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    uid = userstore.create_user(req.email, auth.hash_password(req.password), req.name)
    return {"token": auth.make_token(uid), "user": userstore.get_user(uid)}


@app.post("/api/auth/login")
def login(req: LoginReq):
    u = userstore.get_user_by_email(req.email)
    if not u or not auth.verify_password(req.password, u["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    return {"token": auth.make_token(u["id"]), "user": userstore.get_user(u["id"])}


@app.get("/api/auth/me")
def me(user=Depends(auth.current_user)):
    return user



def require_admin(user=Depends(auth.current_user)):
    if not user.get("is_admin"):
        raise HTTPException(403, "Admin only")
    return user


@app.get("/api/admin/summary")
def admin_summary(user=Depends(require_admin)):
    import sqlite3
    total_jobs = db.market_intel()["total_jobs"]
    return {"admin": user["email"], "total_jobs": total_jobs}




# ── profile ──
@app.get("/api/profile")
def get_profile(user=Depends(auth.current_user)):
    p = userstore.get_profile(user["id"])
    p.pop("encrypted_api_key", None)   # never expose the stored ciphertext
    return p


@app.put("/api/profile")
def put_profile(req: ProfileReq, user=Depends(auth.current_user)):
    return userstore.upsert_profile(user["id"], **req.model_dump())


# ── jobs / filters / facets / market ──
@app.get("/api/facets")
def facets():
    return db.facets()


@app.get("/api/jobs")
def jobs(source: str = None, location: str = None, domain: str = None,
         remote: bool = None, min_exp: int = None, max_exp: int = None,
         skills: str = None, q: str = None, limit: int = 50, offset: int = 0):
    sk = [s.strip() for s in (skills or "").split(",") if s.strip()]
    return {"jobs": db.query_jobs(source=source, location=location, domain=domain,
                                  remote=remote, min_exp=min_exp, max_exp=max_exp,
                                  skills=sk, q=q, limit=limit, offset=offset)}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j          # includes full description (JD-on-select)


@app.get("/api/market")
def market(domain: str = None):
    return db.market_intel(domain=domain)


@app.get("/api/market/position/{job_id}")
def market_position(job_id: str):
    return db.market_for_position(job_id)


# ── saved jobs / dashboard ──
@app.post("/api/jobs/{job_id}/save")
def save(job_id: str, status: str = "saved", user=Depends(auth.current_user)):
    userstore.save_job(user["id"], job_id, status)
    return {"ok": True}


@app.delete("/api/jobs/{job_id}/save")
def unsave(job_id: str, user=Depends(auth.current_user)):
    userstore.unsave_job(user["id"], job_id)
    return {"ok": True}


@app.get("/api/saved")
def saved(user=Depends(auth.current_user)):
    return {"saved": db.saved_jobs(user["id"])}


@app.get("/api/dashboard")
def dashboard(user=Depends(auth.current_user)):
    d = db.dashboard(user["id"])
    d["name"] = user.get("name") or user["email"].split("@")[0]
    return d


@app.get("/api/recommended")
def recommended(user=Depends(auth.current_user)):
    return {"jobs": db.recommend_jobs(user["id"], limit=40)}


@app.post("/api/analyze")
def analyze(req: AnalyzeReq, user=Depends(auth.current_user)):
    """Parse the uploaded resume, store the detected skills on the profile, and
    return recommendations. Recommendations are only produced AFTER this runs."""
    g = guardrails.check_resume(req.resume_text)
    if not g["ok"]:
        raise HTTPException(400, g["reason"])
    from agent.nodes import _parse_resume
    parsed = _parse_resume(g["text"])
    skills = parsed.get("skills", [])
    prof = userstore.get_profile(user["id"])
    merged = sorted(set((prof.get("skills") or []) + skills), key=str.lower)
    userstore.upsert_profile(
        user["id"], location=prof.get("location", ""),
        exp_years=parsed.get("exp_years") or prof.get("exp_years"),
        skills=merged, preferred_titles=prof.get("preferred_titles", []),
        preferred_locations=prof.get("preferred_locations", []),
        work_pref=prof.get("work_pref", ""),
        experience_level=parsed.get("seniority") or prof.get("experience_level", ""))
    return {"skills": merged, "summary": parsed.get("summary", ""),
            "recommended": db.recommend_jobs(
                user["id"], limit=40, source=req.source, location=req.location,
                domain=req.domain, remote=req.remote, min_exp=req.min_exp,
                max_exp=req.max_exp,
                extra_skills=[s.strip() for s in req.skills.split(",") if s.strip()])}


# ── BYO LLM key: settings (validate -> encrypt -> store), masked read, delete ──
@app.get("/api/settings/apikey")
def get_apikey(user=Depends(auth.current_user)):
    creds = userstore.get_api_credentials(user["id"])
    if not creds or not creds.get("encrypted_api_key"):
        return {"has_key": False, "provider": None, "model": None, "masked": None}
    plain = crypto.decrypt(creds["encrypted_api_key"])
    return {"has_key": True, "provider": creds["provider"], "model": creds["model"],
            "base_url": creds.get("base_url"), "masked": crypto.mask(plain or "")}


@app.put("/api/settings/apikey")
def put_apikey(req: ApiKeyReq, user=Depends(auth.current_user)):
    if req.provider not in llm_user.PROVIDERS:
        raise HTTPException(400, f"provider must be one of {llm_user.PROVIDERS}")
    req.api_key = req.api_key.strip()
    ok, msg = llm_user.validate_key(req.provider, req.api_key, req.model, req.base_url)
    if not ok:
        # print the exact provider reason to the terminal (no key is included in msg)
        import logging
        k = req.api_key
        logging.getLogger("uvicorn.error").warning(
            "API key validation FAILED [provider=%s model=%r base_url=%r]: %s "
            "(key_len=%d head=%r tail=%r)",
            req.provider, req.model, req.base_url, msg,
            len(k), k[:4], k[-3:] if len(k) >= 3 else k)
        raise HTTPException(400, f"Key validation failed: {msg}")
    userstore.set_api_key(user["id"], req.provider, req.model,
                          crypto.encrypt(req.api_key), req.base_url)
    return {"ok": True, "provider": req.provider, "model": req.model,
            "masked": crypto.mask(req.api_key), "message": msg}


@app.delete("/api/settings/apikey")
def delete_apikey(user=Depends(auth.current_user)):
    userstore.clear_api_key(user["id"])
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatReq, user=Depends(auth.current_user)):
    creds = userstore.get_api_credentials(user["id"])
    if not creds or not creds.get("encrypted_api_key"):
        raise HTTPException(400, "No API key set. Add one in Settings.")
    key = crypto.decrypt(creds["encrypted_api_key"])
    if not key:
        raise HTTPException(400, "Stored key could not be read; please re-enter it.")
    msg, _ = guardrails.sanitize(req.message)
    prof = userstore.get_profile(user["id"])
    ctx = f"Candidate skills: {prof.get('skills')}. "
    if req.job_id:
        j = db.get_job(req.job_id)
        if j:
            ctx += (f"Job of interest: {j['title']} at {j['company']} "
                    f"(skills: {j.get('skills')}).")
    system = ("You are a career assistant. Help the user with available jobs, "
              "their profile, and interview/application preparation. Be concise "
              "and specific. Context: " + ctx)
    try:
        reply = llm_user.chat(creds["provider"], key, creds["model"], system,
                              msg, req.history, base_url=creds.get("base_url", ""))
        return {"reply": reply}
    except llm_user.LLMError as e:
        status = {"invalid_key": 401, "quota_exceeded": 429}.get(e.kind, 502)
        raise HTTPException(status, e.message)


# ── resume parsing helper ──
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        import io
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    elif name.endswith(".docx"):
        import io, docx
        text = "\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)
    else:
        text = raw.decode("utf-8", errors="ignore")
    return {"resume_text": text}


# ── scoring graph (with guardrails) ──
@app.post("/api/start")
def start(req: StartReq, user=Depends(auth.current_user)):
    g = guardrails.check_resume(req.resume_text)
    if not g["ok"]:
        raise HTTPException(400, g["reason"])
    tid = str(uuid.uuid4())
    state = {"resume_text": g["text"], "domain": req.domain, "location": req.location}
    # attach the user's BYO key so cover letter / interview Qs / tailored resume
    # (extras_node) use their model instead of the server's Gemini key.
    creds = userstore.get_api_credentials(user["id"]) or {}
    if creds.get("encrypted_api_key"):
        key = crypto.decrypt(creds["encrypted_api_key"])
        if key:
            state["llm_creds"] = {"provider": creds["provider"], "model": creds["model"],
                                  "api_key": key, "base_url": creds.get("base_url", "")}
    r = GRAPH.invoke(state, _cfg(tid))
    itr = _interrupt(r)
    return {"thread_id": tid, "matches": itr["matches"] if itr else [],
            "guardrail_flags": g["flags"]}


@app.post("/api/select")
def select(req: SelectReq):
    r = GRAPH.invoke(Command(resume=req.job_id), _cfg(req.thread_id))
    itr = _interrupt(r)
    return {"ats": itr["ats"] if itr else None}


@app.post("/api/action")
def action(req: ActionReq):
    cmd = {"action": req.action, "text": req.text}
    if req.action == "upload_resume":
        g = guardrails.check_resume(req.resume_text)
        if not g["ok"]:
            raise HTTPException(400, g["reason"])
        cmd["resume_text"] = g["text"]
    r = GRAPH.invoke(Command(resume=cmd), _cfg(req.thread_id))
    itr = _interrupt(r)
    return {"ats": itr["ats"] if itr else None, "how_to_add": r.get("how_to_add"),
            "answer": r.get("answer"), "artifacts": r.get("artifacts", {}),
            "done": itr is None}


@app.get("/api/audit")
def audit():
    return {"available": rq_tools.available(), "calls": rq_tools.audit_log()}


# ── pages ──
@app.get("/")
def landing():
    return FileResponse(os.path.join(HERE, "web", "landing.html"))


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(HERE, "web", "login.html"))


@app.get("/app")
def app_page():
    return FileResponse(os.path.join(HERE, "web", "index.html"))

# """FastAPI app: LangGraph agent + auth + jobs/filters + market intel + saved jobs.

# Scoring maps onto the graph's interrupts:
#   /api/start -> select_job interrupt ; /api/select -> review ; /api/action -> loop
# Auth is JWT (bcrypt hashing). Jobs/filters/market/saved/dashboard use SQLite.
# Guardrails sanitise resume text before it reaches the LLM/scorer.
# """
# import os
# import uuid
# from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from langgraph.types import Command

# from agent.graph import build_graph
# from agent import db, auth, guardrails, rq_tools, userstore, crypto, llm_user

# import logging as _logging
# import re as _re
# class _RedactSecrets(_logging.Filter):
#     _pat = _re.compile(r'(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|"api_key"\s*:\s*"[^"]*")')
#     def filter(self, rec):
#         try:
#             if isinstance(rec.msg, str):
#                 rec.msg = self._pat.sub("[REDACTED]", rec.msg)
#             if rec.args:
#                 rec.args = tuple(self._pat.sub("[REDACTED]", str(a)) if isinstance(a, str) else a
#                                  for a in rec.args)
#         except Exception:
#             pass
#         return True
# for _n in ("", "uvicorn", "uvicorn.access", "uvicorn.error"):
#     _logging.getLogger(_n).addFilter(_RedactSecrets())

# app = FastAPI(title="Job Scout ATS Agent")
# GRAPH = build_graph()
# userstore.init()  # create user tables (sqlite fallback or Postgres)

# # Seed a built-in admin account (replaces OAuth). Configure via env.
# _ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@local")
# _ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin12345")
# if not userstore.get_user_by_email(_ADMIN_EMAIL):
#     userstore.create_user(_ADMIN_EMAIL, auth.hash_password(_ADMIN_PASSWORD),
#                           name="Admin", is_admin=True)
# HERE = os.path.dirname(__file__)


# def _cfg(tid): return {"configurable": {"thread_id": tid}}
# def _interrupt(r): return r["__interrupt__"][0].value if "__interrupt__" in r else None


# # ── schemas ──
# class StartReq(BaseModel):
#     resume_text: str = ""
#     domain: str | None = None
#     location: str | None = None


# class SelectReq(BaseModel):
#     thread_id: str
#     job_id: str


# class ActionReq(BaseModel):
#     thread_id: str
#     action: str
#     text: str = ""
#     resume_text: str = ""


# class RegisterReq(BaseModel):
#     email: str
#     password: str
#     name: str = ""


# class LoginReq(BaseModel):
#     email: str
#     password: str


# class AnalyzeReq(BaseModel):
#     resume_text: str
#     source: str | None = None
#     location: str | None = None
#     domain: str | None = None
#     remote: bool | None = None
#     min_exp: int | None = None
#     max_exp: int | None = None
#     skills: str = ""          # comma-separated, from the filter box


# class ApiKeyReq(BaseModel):
#     provider: str          # openai | anthropic | gemini | custom | test
#     model: str = ""
#     api_key: str
#     base_url: str = ""     # required when provider == custom (OpenAI-compatible)


# class ChatReq(BaseModel):
#     message: str
#     job_id: str | None = None
#     history: list = []


# class ProfileReq(BaseModel):
#     location: str = ""
#     exp_years: int | None = None
#     skills: list[str] = []
#     preferred_titles: list[str] = []
#     preferred_locations: list[str] = []
#     work_pref: str = ""          # remote | hybrid | onsite
#     experience_level: str = ""   # entry | mid | senior


# # ── auth ──
# @app.post("/api/auth/register")
# def register(req: RegisterReq):
#     if userstore.get_user_by_email(req.email):
#         raise HTTPException(409, "Email already registered")
#     if len(req.password) < 8:
#         raise HTTPException(400, "Password must be at least 8 characters")
#     uid = userstore.create_user(req.email, auth.hash_password(req.password), req.name)
#     return {"token": auth.make_token(uid), "user": userstore.get_user(uid)}


# @app.post("/api/auth/login")
# def login(req: LoginReq):
#     u = userstore.get_user_by_email(req.email)
#     if not u or not auth.verify_password(req.password, u["password_hash"]):
#         raise HTTPException(401, "Invalid credentials")
#     return {"token": auth.make_token(u["id"]), "user": userstore.get_user(u["id"])}


# @app.get("/api/auth/me")
# def me(user=Depends(auth.current_user)):
#     return user



# def require_admin(user=Depends(auth.current_user)):
#     if not user.get("is_admin"):
#         raise HTTPException(403, "Admin only")
#     return user


# @app.get("/api/admin/summary")
# def admin_summary(user=Depends(require_admin)):
#     import sqlite3
#     total_jobs = db.market_intel()["total_jobs"]
#     return {"admin": user["email"], "total_jobs": total_jobs}




# # ── profile ──
# @app.get("/api/profile")
# def get_profile(user=Depends(auth.current_user)):
#     p = userstore.get_profile(user["id"])
#     p.pop("encrypted_api_key", None)   # never expose the stored ciphertext
#     return p


# @app.put("/api/profile")
# def put_profile(req: ProfileReq, user=Depends(auth.current_user)):
#     return userstore.upsert_profile(user["id"], **req.model_dump())


# # ── jobs / filters / facets / market ──
# @app.get("/api/facets")
# def facets():
#     return db.facets()


# @app.get("/api/jobs")
# def jobs(source: str = None, location: str = None, domain: str = None,
#          remote: bool = None, min_exp: int = None, max_exp: int = None,
#          skills: str = None, q: str = None, limit: int = 50, offset: int = 0):
#     sk = [s.strip() for s in (skills or "").split(",") if s.strip()]
#     return {"jobs": db.query_jobs(source=source, location=location, domain=domain,
#                                   remote=remote, min_exp=min_exp, max_exp=max_exp,
#                                   skills=sk, q=q, limit=limit, offset=offset)}


# @app.get("/api/jobs/{job_id}")
# def job_detail(job_id: str):
#     j = db.get_job(job_id)
#     if not j:
#         raise HTTPException(404, "Job not found")
#     return j          # includes full description (JD-on-select)


# @app.get("/api/market")
# def market(domain: str = None):
#     return db.market_intel(domain=domain)


# @app.get("/api/market/position/{job_id}")
# def market_position(job_id: str):
#     return db.market_for_position(job_id)


# # ── saved jobs / dashboard ──
# @app.post("/api/jobs/{job_id}/save")
# def save(job_id: str, status: str = "saved", user=Depends(auth.current_user)):
#     userstore.save_job(user["id"], job_id, status)
#     return {"ok": True}


# @app.delete("/api/jobs/{job_id}/save")
# def unsave(job_id: str, user=Depends(auth.current_user)):
#     userstore.unsave_job(user["id"], job_id)
#     return {"ok": True}


# @app.get("/api/saved")
# def saved(user=Depends(auth.current_user)):
#     return {"saved": db.saved_jobs(user["id"])}


# @app.get("/api/dashboard")
# def dashboard(user=Depends(auth.current_user)):
#     d = db.dashboard(user["id"])
#     d["name"] = user.get("name") or user["email"].split("@")[0]
#     return d


# @app.get("/api/recommended")
# def recommended(user=Depends(auth.current_user)):
#     return {"jobs": db.recommend_jobs(user["id"], limit=40)}


# @app.post("/api/analyze")
# def analyze(req: AnalyzeReq, user=Depends(auth.current_user)):
#     """Parse the uploaded resume, store the detected skills on the profile, and
#     return recommendations. Recommendations are only produced AFTER this runs."""
#     g = guardrails.check_resume(req.resume_text)
#     if not g["ok"]:
#         raise HTTPException(400, g["reason"])
#     from agent.nodes import _parse_resume
#     parsed = _parse_resume(g["text"])
#     skills = parsed.get("skills", [])
#     prof = userstore.get_profile(user["id"])
#     merged = sorted(set((prof.get("skills") or []) + skills), key=str.lower)
#     userstore.upsert_profile(
#         user["id"], location=prof.get("location", ""),
#         exp_years=parsed.get("exp_years") or prof.get("exp_years"),
#         skills=merged, preferred_titles=prof.get("preferred_titles", []),
#         preferred_locations=prof.get("preferred_locations", []),
#         work_pref=prof.get("work_pref", ""),
#         experience_level=parsed.get("seniority") or prof.get("experience_level", ""))
#     return {"skills": merged, "summary": parsed.get("summary", ""),
#             "recommended": db.recommend_jobs(
#                 user["id"], limit=40, source=req.source, location=req.location,
#                 domain=req.domain, remote=req.remote, min_exp=req.min_exp,
#                 max_exp=req.max_exp,
#                 extra_skills=[s.strip() for s in req.skills.split(",") if s.strip()])}


# # ── BYO LLM key: settings (validate -> encrypt -> store), masked read, delete ──
# @app.get("/api/settings/apikey")
# def get_apikey(user=Depends(auth.current_user)):
#     creds = userstore.get_api_credentials(user["id"])
#     if not creds or not creds.get("encrypted_api_key"):
#         return {"has_key": False, "provider": None, "model": None, "masked": None}
#     plain = crypto.decrypt(creds["encrypted_api_key"])
#     return {"has_key": True, "provider": creds["provider"], "model": creds["model"],
#             "base_url": creds.get("base_url"), "masked": crypto.mask(plain or "")}


# @app.put("/api/settings/apikey")
# def put_apikey(req: ApiKeyReq, user=Depends(auth.current_user)):
#     if req.provider not in llm_user.PROVIDERS:
#         raise HTTPException(400, f"provider must be one of {llm_user.PROVIDERS}")
#     req.api_key = req.api_key.strip()
#     ok, msg = llm_user.validate_key(req.provider, req.api_key, req.model, req.base_url)
#     if not ok:
#         # print the exact provider reason to the terminal (no key is included in msg)
#         import logging
#         k = req.api_key
#         logging.getLogger("uvicorn.error").warning(
#             "API key validation FAILED [provider=%s model=%r base_url=%r]: %s "
#             "(key_len=%d head=%r tail=%r)",
#             req.provider, req.model, req.base_url, msg,
#             len(k), k[:4], k[-3:] if len(k) >= 3 else k)
#         raise HTTPException(400, f"Key validation failed: {msg}")
#     userstore.set_api_key(user["id"], req.provider, req.model,
#                           crypto.encrypt(req.api_key), req.base_url)
#     return {"ok": True, "provider": req.provider, "model": req.model,
#             "masked": crypto.mask(req.api_key), "message": msg}


# @app.delete("/api/settings/apikey")
# def delete_apikey(user=Depends(auth.current_user)):
#     userstore.clear_api_key(user["id"])
#     return {"ok": True}


# @app.post("/api/chat")
# def chat(req: ChatReq, user=Depends(auth.current_user)):
#     creds = userstore.get_api_credentials(user["id"])
#     if not creds or not creds.get("encrypted_api_key"):
#         raise HTTPException(400, "No API key set. Add one in Settings.")
#     key = crypto.decrypt(creds["encrypted_api_key"])
#     if not key:
#         raise HTTPException(400, "Stored key could not be read; please re-enter it.")
#     msg, _ = guardrails.sanitize(req.message)
#     prof = userstore.get_profile(user["id"])
#     ctx = f"Candidate skills: {prof.get('skills')}. "
#     if req.job_id:
#         j = db.get_job(req.job_id)
#         if j:
#             ctx += (f"Job of interest: {j['title']} at {j['company']} "
#                     f"(skills: {j.get('skills')}).")
#     system = ("You are a career assistant. Help the user with available jobs, "
#               "their profile, and interview/application preparation. Be concise "
#               "and specific. Context: " + ctx)
#     try:
#         reply = llm_user.chat(creds["provider"], key, creds["model"], system,
#                               msg, req.history, base_url=creds.get("base_url", ""))
#         return {"reply": reply}
#     except llm_user.LLMError as e:
#         status = {"invalid_key": 401, "quota_exceeded": 429}.get(e.kind, 502)
#         raise HTTPException(status, e.message)


# # ── resume parsing helper ──
# @app.post("/api/upload")
# async def upload(file: UploadFile = File(...)):
#     raw = await file.read()
#     name = (file.filename or "").lower()
#     if name.endswith(".pdf"):
#         import io
#         from pypdf import PdfReader
#         text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
#     elif name.endswith(".docx"):
#         import io, docx
#         text = "\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)
#     else:
#         text = raw.decode("utf-8", errors="ignore")
#     return {"resume_text": text}


# # ── scoring graph (with guardrails) ──
# @app.post("/api/start")
# def start(req: StartReq):
#     g = guardrails.check_resume(req.resume_text)
#     if not g["ok"]:
#         raise HTTPException(400, g["reason"])
#     tid = str(uuid.uuid4())
#     r = GRAPH.invoke({"resume_text": g["text"], "domain": req.domain,
#                       "location": req.location}, _cfg(tid))
#     itr = _interrupt(r)
#     return {"thread_id": tid, "matches": itr["matches"] if itr else [],
#             "guardrail_flags": g["flags"]}


# @app.post("/api/select")
# def select(req: SelectReq):
#     r = GRAPH.invoke(Command(resume=req.job_id), _cfg(req.thread_id))
#     itr = _interrupt(r)
#     return {"ats": itr["ats"] if itr else None}


# @app.post("/api/action")
# def action(req: ActionReq):
#     cmd = {"action": req.action, "text": req.text}
#     if req.action == "upload_resume":
#         g = guardrails.check_resume(req.resume_text)
#         if not g["ok"]:
#             raise HTTPException(400, g["reason"])
#         cmd["resume_text"] = g["text"]
#     r = GRAPH.invoke(Command(resume=cmd), _cfg(req.thread_id))
#     itr = _interrupt(r)
#     return {"ats": itr["ats"] if itr else None, "how_to_add": r.get("how_to_add"),
#             "answer": r.get("answer"), "artifacts": r.get("artifacts", {}),
#             "done": itr is None}


# @app.get("/api/audit")
# def audit():
#     return {"available": rq_tools.available(), "calls": rq_tools.audit_log()}


# # ── pages ──
# @app.get("/")
# def landing():
#     return FileResponse(os.path.join(HERE, "web", "landing.html"))


# @app.get("/login")
# def login_page():
#     return FileResponse(os.path.join(HERE, "web", "login.html"))


# @app.get("/app")
# def app_page():
#     return FileResponse(os.path.join(HERE, "web", "index.html"))


# # """FastAPI app: LangGraph agent + auth + jobs/filters + market intel + saved jobs.

# # Scoring maps onto the graph's interrupts:
# #   /api/start -> select_job interrupt ; /api/select -> review ; /api/action -> loop
# # Auth is JWT (bcrypt hashing). Jobs/filters/market/saved/dashboard use SQLite.
# # Guardrails sanitise resume text before it reaches the LLM/scorer.
# # """
# # import os
# # import uuid
# # from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# # from fastapi.responses import FileResponse
# # from pydantic import BaseModel
# # from langgraph.types import Command

# # from agent.graph import build_graph
# # from agent import db, auth, guardrails, rq_tools, userstore, crypto, llm_user

# # import logging as _logging
# # import re as _re
# # class _RedactSecrets(_logging.Filter):
# #     _pat = _re.compile(r'(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|"api_key"\s*:\s*"[^"]*")')
# #     def filter(self, rec):
# #         try:
# #             if isinstance(rec.msg, str):
# #                 rec.msg = self._pat.sub("[REDACTED]", rec.msg)
# #             if rec.args:
# #                 rec.args = tuple(self._pat.sub("[REDACTED]", str(a)) if isinstance(a, str) else a
# #                                  for a in rec.args)
# #         except Exception:
# #             pass
# #         return True
# # for _n in ("", "uvicorn", "uvicorn.access", "uvicorn.error"):
# #     _logging.getLogger(_n).addFilter(_RedactSecrets())

# # app = FastAPI(title="Job Scout ATS Agent")
# # GRAPH = build_graph()
# # userstore.init()  # create user tables (sqlite fallback or Postgres)

# # # Seed a built-in admin account (replaces OAuth). Configure via env.
# # _ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@local")
# # _ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin12345")
# # if not userstore.get_user_by_email(_ADMIN_EMAIL):
# #     userstore.create_user(_ADMIN_EMAIL, auth.hash_password(_ADMIN_PASSWORD),
# #                           name="Admin", is_admin=True)
# # HERE = os.path.dirname(__file__)


# # def _cfg(tid): return {"configurable": {"thread_id": tid}}
# # def _interrupt(r): return r["__interrupt__"][0].value if "__interrupt__" in r else None


# # # ── schemas ──
# # class StartReq(BaseModel):
# #     resume_text: str = ""
# #     domain: str | None = None
# #     location: str | None = None


# # class SelectReq(BaseModel):
# #     thread_id: str
# #     job_id: str


# # class ActionReq(BaseModel):
# #     thread_id: str
# #     action: str
# #     text: str = ""
# #     resume_text: str = ""


# # class RegisterReq(BaseModel):
# #     email: str
# #     password: str
# #     name: str = ""


# # class LoginReq(BaseModel):
# #     email: str
# #     password: str


# # class AnalyzeReq(BaseModel):
# #     resume_text: str
# #     source: str | None = None
# #     location: str | None = None
# #     domain: str | None = None
# #     remote: bool | None = None
# #     min_exp: int | None = None
# #     max_exp: int | None = None
# #     skills: str = ""          # comma-separated, from the filter box


# # class ApiKeyReq(BaseModel):
# #     provider: str          # openai | anthropic | gemini | custom | test
# #     model: str = ""
# #     api_key: str
# #     base_url: str = ""     # required when provider == custom (OpenAI-compatible)


# # class ChatReq(BaseModel):
# #     message: str
# #     job_id: str | None = None
# #     history: list = []


# # class ProfileReq(BaseModel):
# #     location: str = ""
# #     exp_years: int | None = None
# #     skills: list[str] = []
# #     preferred_titles: list[str] = []
# #     preferred_locations: list[str] = []
# #     work_pref: str = ""          # remote | hybrid | onsite
# #     experience_level: str = ""   # entry | mid | senior


# # # ── auth ──
# # @app.post("/api/auth/register")
# # def register(req: RegisterReq):
# #     if userstore.get_user_by_email(req.email):
# #         raise HTTPException(409, "Email already registered")
# #     if len(req.password) < 8:
# #         raise HTTPException(400, "Password must be at least 8 characters")
# #     uid = userstore.create_user(req.email, auth.hash_password(req.password), req.name)
# #     return {"token": auth.make_token(uid), "user": userstore.get_user(uid)}


# # @app.post("/api/auth/login")
# # def login(req: LoginReq):
# #     u = userstore.get_user_by_email(req.email)
# #     if not u or not auth.verify_password(req.password, u["password_hash"]):
# #         raise HTTPException(401, "Invalid credentials")
# #     return {"token": auth.make_token(u["id"]), "user": userstore.get_user(u["id"])}


# # @app.get("/api/auth/me")
# # def me(user=Depends(auth.current_user)):
# #     return user



# # def require_admin(user=Depends(auth.current_user)):
# #     if not user.get("is_admin"):
# #         raise HTTPException(403, "Admin only")
# #     return user


# # @app.get("/api/admin/summary")
# # def admin_summary(user=Depends(require_admin)):
# #     import sqlite3
# #     total_jobs = db.market_intel()["total_jobs"]
# #     return {"admin": user["email"], "total_jobs": total_jobs}




# # # ── profile ──
# # @app.get("/api/profile")
# # def get_profile(user=Depends(auth.current_user)):
# #     p = userstore.get_profile(user["id"])
# #     p.pop("encrypted_api_key", None)   # never expose the stored ciphertext
# #     return p


# # @app.put("/api/profile")
# # def put_profile(req: ProfileReq, user=Depends(auth.current_user)):
# #     return userstore.upsert_profile(user["id"], **req.model_dump())


# # # ── jobs / filters / facets / market ──
# # @app.get("/api/facets")
# # def facets():
# #     return db.facets()


# # @app.get("/api/jobs")
# # def jobs(source: str = None, location: str = None, domain: str = None,
# #          remote: bool = None, min_exp: int = None, max_exp: int = None,
# #          skills: str = None, q: str = None, limit: int = 50, offset: int = 0):
# #     sk = [s.strip() for s in (skills or "").split(",") if s.strip()]
# #     return {"jobs": db.query_jobs(source=source, location=location, domain=domain,
# #                                   remote=remote, min_exp=min_exp, max_exp=max_exp,
# #                                   skills=sk, q=q, limit=limit, offset=offset)}


# # @app.get("/api/jobs/{job_id}")
# # def job_detail(job_id: str):
# #     j = db.get_job(job_id)
# #     if not j:
# #         raise HTTPException(404, "Job not found")
# #     return j          # includes full description (JD-on-select)


# # @app.get("/api/market")
# # def market(domain: str = None):
# #     return db.market_intel(domain=domain)


# # @app.get("/api/market/position/{job_id}")
# # def market_position(job_id: str):
# #     return db.market_for_position(job_id)


# # # ── saved jobs / dashboard ──
# # @app.post("/api/jobs/{job_id}/save")
# # def save(job_id: str, status: str = "saved", user=Depends(auth.current_user)):
# #     userstore.save_job(user["id"], job_id, status)
# #     return {"ok": True}


# # @app.delete("/api/jobs/{job_id}/save")
# # def unsave(job_id: str, user=Depends(auth.current_user)):
# #     userstore.unsave_job(user["id"], job_id)
# #     return {"ok": True}


# # @app.get("/api/saved")
# # def saved(user=Depends(auth.current_user)):
# #     return {"saved": db.saved_jobs(user["id"])}


# # @app.get("/api/dashboard")
# # def dashboard(user=Depends(auth.current_user)):
# #     d = db.dashboard(user["id"])
# #     d["name"] = user.get("name") or user["email"].split("@")[0]
# #     return d


# # @app.get("/api/recommended")
# # def recommended(user=Depends(auth.current_user)):
# #     return {"jobs": db.recommend_jobs(user["id"], limit=40)}


# # @app.post("/api/analyze")
# # def analyze(req: AnalyzeReq, user=Depends(auth.current_user)):
# #     """Parse the uploaded resume, store the detected skills on the profile, and
# #     return recommendations. Recommendations are only produced AFTER this runs."""
# #     g = guardrails.check_resume(req.resume_text)
# #     if not g["ok"]:
# #         raise HTTPException(400, g["reason"])
# #     from agent.nodes import _parse_resume
# #     parsed = _parse_resume(g["text"])
# #     skills = parsed.get("skills", [])
# #     prof = userstore.get_profile(user["id"])
# #     merged = sorted(set((prof.get("skills") or []) + skills), key=str.lower)
# #     userstore.upsert_profile(
# #         user["id"], location=prof.get("location", ""),
# #         exp_years=parsed.get("exp_years") or prof.get("exp_years"),
# #         skills=merged, preferred_titles=prof.get("preferred_titles", []),
# #         preferred_locations=prof.get("preferred_locations", []),
# #         work_pref=prof.get("work_pref", ""),
# #         experience_level=parsed.get("seniority") or prof.get("experience_level", ""))
# #     return {"skills": merged, "summary": parsed.get("summary", ""),
# #             "recommended": db.recommend_jobs(
# #                 user["id"], limit=40, source=req.source, location=req.location,
# #                 domain=req.domain, remote=req.remote, min_exp=req.min_exp,
# #                 max_exp=req.max_exp,
# #                 extra_skills=[s.strip() for s in req.skills.split(",") if s.strip()])}


# # # ── BYO LLM key: settings (validate -> encrypt -> store), masked read, delete ──
# # @app.get("/api/settings/apikey")
# # def get_apikey(user=Depends(auth.current_user)):
# #     creds = userstore.get_api_credentials(user["id"])
# #     if not creds or not creds.get("encrypted_api_key"):
# #         return {"has_key": False, "provider": None, "model": None, "masked": None}
# #     plain = crypto.decrypt(creds["encrypted_api_key"])
# #     return {"has_key": True, "provider": creds["provider"], "model": creds["model"],
# #             "base_url": creds.get("base_url"), "masked": crypto.mask(plain or "")}


# # @app.put("/api/settings/apikey")
# # def put_apikey(req: ApiKeyReq, user=Depends(auth.current_user)):
# #     if req.provider not in llm_user.PROVIDERS:
# #         raise HTTPException(400, f"provider must be one of {llm_user.PROVIDERS}")
# #     req.api_key = req.api_key.strip()
# #     ok, msg = llm_user.validate_key(req.provider, req.api_key, req.model, req.base_url)
# #     if not ok:
# #         # print the exact provider reason to the terminal (no key is included in msg)
# #         import logging
# #         k = req.api_key
# #         logging.getLogger("uvicorn.error").warning(
# #             "API key validation FAILED [provider=%s model=%r base_url=%r]: %s "
# #             "(key_len=%d head=%r tail=%r)",
# #             req.provider, req.model, req.base_url, msg,
# #             len(k), k[:4], k[-3:] if len(k) >= 3 else k)
# #         raise HTTPException(400, f"Key validation failed: {msg}")
# #     userstore.set_api_key(user["id"], req.provider, req.model,
# #                           crypto.encrypt(req.api_key), req.base_url)
# #     return {"ok": True, "provider": req.provider, "model": req.model,
# #             "masked": crypto.mask(req.api_key), "message": msg}


# # @app.delete("/api/settings/apikey")
# # def delete_apikey(user=Depends(auth.current_user)):
# #     userstore.clear_api_key(user["id"])
# #     return {"ok": True}


# # @app.post("/api/chat")
# # def chat(req: ChatReq, user=Depends(auth.current_user)):
# #     creds = userstore.get_api_credentials(user["id"])
# #     if not creds or not creds.get("encrypted_api_key"):
# #         raise HTTPException(400, "No API key set. Add one in Settings.")
# #     key = crypto.decrypt(creds["encrypted_api_key"])
# #     if not key:
# #         raise HTTPException(400, "Stored key could not be read; please re-enter it.")
# #     msg, _ = guardrails.sanitize(req.message)
# #     prof = userstore.get_profile(user["id"])
# #     ctx = f"Candidate skills: {prof.get('skills')}. "
# #     if req.job_id:
# #         j = db.get_job(req.job_id)
# #         if j:
# #             ctx += (f"Job of interest: {j['title']} at {j['company']} "
# #                     f"(skills: {j.get('skills')}).")
# #     system = ("You are a career assistant. Help the user with available jobs, "
# #               "their profile, and interview/application preparation. Be concise "
# #               "and specific. Context: " + ctx)
# #     try:
# #         reply = llm_user.chat(creds["provider"], key, creds["model"], system,
# #                               msg, req.history, base_url=creds.get("base_url", ""))
# #         return {"reply": reply}
# #     except llm_user.LLMError as e:
# #         status = {"invalid_key": 401, "quota_exceeded": 429}.get(e.kind, 502)
# #         raise HTTPException(status, e.message)


# # # ── resume parsing helper ──
# # @app.post("/api/upload")
# # async def upload(file: UploadFile = File(...)):
# #     raw = await file.read()
# #     name = (file.filename or "").lower()
# #     if name.endswith(".pdf"):
# #         import io
# #         from pypdf import PdfReader
# #         text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
# #     elif name.endswith(".docx"):
# #         import io, docx
# #         text = "\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)
# #     else:
# #         text = raw.decode("utf-8", errors="ignore")
# #     return {"resume_text": text}


# # # ── scoring graph (with guardrails) ──
# # @app.post("/api/start")
# # def start(req: StartReq):
# #     g = guardrails.check_resume(req.resume_text)
# #     if not g["ok"]:
# #         raise HTTPException(400, g["reason"])
# #     tid = str(uuid.uuid4())
# #     r = GRAPH.invoke({"resume_text": g["text"], "domain": req.domain,
# #                       "location": req.location}, _cfg(tid))
# #     itr = _interrupt(r)
# #     return {"thread_id": tid, "matches": itr["matches"] if itr else [],
# #             "guardrail_flags": g["flags"]}


# # @app.post("/api/select")
# # def select(req: SelectReq):
# #     r = GRAPH.invoke(Command(resume=req.job_id), _cfg(req.thread_id))
# #     itr = _interrupt(r)
# #     return {"ats": itr["ats"] if itr else None}


# # @app.post("/api/action")
# # def action(req: ActionReq):
# #     cmd = {"action": req.action, "text": req.text}
# #     if req.action == "upload_resume":
# #         g = guardrails.check_resume(req.resume_text)
# #         if not g["ok"]:
# #             raise HTTPException(400, g["reason"])
# #         cmd["resume_text"] = g["text"]
# #     r = GRAPH.invoke(Command(resume=cmd), _cfg(req.thread_id))
# #     itr = _interrupt(r)
# #     return {"ats": itr["ats"] if itr else None, "how_to_add": r.get("how_to_add"),
# #             "answer": r.get("answer"), "artifacts": r.get("artifacts", {}),
# #             "done": itr is None}


# # @app.get("/api/audit")
# # def audit():
# #     return {"available": rq_tools.available(), "calls": rq_tools.audit_log()}


# # # ── pages ──
# # @app.get("/")
# # def landing():
# #     return FileResponse(os.path.join(HERE, "web", "landing.html"))


# # @app.get("/login")
# # def login_page():
# #     return FileResponse(os.path.join(HERE, "web", "login.html"))


# # @app.get("/app")
# # def app_page():
# #     return FileResponse(os.path.join(HERE, "web", "index.html"))
# # # """FastAPI app: LangGraph agent + auth + jobs/filters + market intel + saved jobs.

# # # Scoring maps onto the graph's interrupts:
# # #   /api/start -> select_job interrupt ; /api/select -> review ; /api/action -> loop
# # # Auth is JWT (bcrypt hashing). Jobs/filters/market/saved/dashboard use SQLite.
# # # Guardrails sanitise resume text before it reaches the LLM/scorer.
# # # """
# # # import os
# # # import uuid
# # # from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# # # from fastapi.responses import FileResponse
# # # from pydantic import BaseModel
# # # from langgraph.types import Command

# # # from agent.graph import build_graph
# # # from agent import db, auth, guardrails, rq_tools, userstore, crypto, llm_user

# # # import logging as _logging
# # # import re as _re
# # # class _RedactSecrets(_logging.Filter):
# # #     _pat = _re.compile(r'(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|"api_key"\s*:\s*"[^"]*")')
# # #     def filter(self, rec):
# # #         try:
# # #             if isinstance(rec.msg, str):
# # #                 rec.msg = self._pat.sub("[REDACTED]", rec.msg)
# # #             if rec.args:
# # #                 rec.args = tuple(self._pat.sub("[REDACTED]", str(a)) if isinstance(a, str) else a
# # #                                  for a in rec.args)
# # #         except Exception:
# # #             pass
# # #         return True
# # # for _n in ("", "uvicorn", "uvicorn.access", "uvicorn.error"):
# # #     _logging.getLogger(_n).addFilter(_RedactSecrets())

# # # app = FastAPI(title="Job Scout ATS Agent")
# # # GRAPH = build_graph()
# # # userstore.init()  # create user tables (sqlite fallback or Postgres)

# # # # Seed a built-in admin account (replaces OAuth). Configure via env.
# # # _ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@local")
# # # _ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin12345")
# # # if not userstore.get_user_by_email(_ADMIN_EMAIL):
# # #     userstore.create_user(_ADMIN_EMAIL, auth.hash_password(_ADMIN_PASSWORD),
# # #                           name="Admin", is_admin=True)
# # # HERE = os.path.dirname(__file__)


# # # def _cfg(tid): return {"configurable": {"thread_id": tid}}
# # # def _interrupt(r): return r["__interrupt__"][0].value if "__interrupt__" in r else None


# # # # ── schemas ──
# # # class StartReq(BaseModel):
# # #     resume_text: str = ""
# # #     domain: str | None = None
# # #     location: str | None = None


# # # class SelectReq(BaseModel):
# # #     thread_id: str
# # #     job_id: str


# # # class ActionReq(BaseModel):
# # #     thread_id: str
# # #     action: str
# # #     text: str = ""
# # #     resume_text: str = ""


# # # class RegisterReq(BaseModel):
# # #     email: str
# # #     password: str
# # #     name: str = ""


# # # class LoginReq(BaseModel):
# # #     email: str
# # #     password: str


# # # class AnalyzeReq(BaseModel):
# # #     resume_text: str
# # #     source: str | None = None
# # #     location: str | None = None
# # #     domain: str | None = None
# # #     remote: bool | None = None
# # #     min_exp: int | None = None
# # #     max_exp: int | None = None
# # #     skills: str = ""          # comma-separated, from the filter box


# # # class ApiKeyReq(BaseModel):
# # #     provider: str          # openai | anthropic | gemini | custom | test
# # #     model: str = ""
# # #     api_key: str
# # #     base_url: str = ""     # required when provider == custom (OpenAI-compatible)


# # # class ChatReq(BaseModel):
# # #     message: str
# # #     job_id: str | None = None
# # #     history: list = []


# # # class ProfileReq(BaseModel):
# # #     location: str = ""
# # #     exp_years: int | None = None
# # #     skills: list[str] = []
# # #     preferred_titles: list[str] = []
# # #     preferred_locations: list[str] = []
# # #     work_pref: str = ""          # remote | hybrid | onsite
# # #     experience_level: str = ""   # entry | mid | senior


# # # # ── auth ──
# # # @app.post("/api/auth/register")
# # # def register(req: RegisterReq):
# # #     if userstore.get_user_by_email(req.email):
# # #         raise HTTPException(409, "Email already registered")
# # #     if len(req.password) < 8:
# # #         raise HTTPException(400, "Password must be at least 8 characters")
# # #     uid = userstore.create_user(req.email, auth.hash_password(req.password), req.name)
# # #     return {"token": auth.make_token(uid), "user": userstore.get_user(uid)}


# # # @app.post("/api/auth/login")
# # # def login(req: LoginReq):
# # #     u = userstore.get_user_by_email(req.email)
# # #     if not u or not auth.verify_password(req.password, u["password_hash"]):
# # #         raise HTTPException(401, "Invalid credentials")
# # #     return {"token": auth.make_token(u["id"]), "user": userstore.get_user(u["id"])}


# # # @app.get("/api/auth/me")
# # # def me(user=Depends(auth.current_user)):
# # #     return user



# # # def require_admin(user=Depends(auth.current_user)):
# # #     if not user.get("is_admin"):
# # #         raise HTTPException(403, "Admin only")
# # #     return user


# # # @app.get("/api/admin/summary")
# # # def admin_summary(user=Depends(require_admin)):
# # #     import sqlite3
# # #     total_jobs = db.market_intel()["total_jobs"]
# # #     return {"admin": user["email"], "total_jobs": total_jobs}




# # # # ── profile ──
# # # @app.get("/api/profile")
# # # def get_profile(user=Depends(auth.current_user)):
# # #     p = userstore.get_profile(user["id"])
# # #     p.pop("encrypted_api_key", None)   # never expose the stored ciphertext
# # #     return p


# # # @app.put("/api/profile")
# # # def put_profile(req: ProfileReq, user=Depends(auth.current_user)):
# # #     return userstore.upsert_profile(user["id"], **req.model_dump())


# # # # ── jobs / filters / facets / market ──
# # # @app.get("/api/facets")
# # # def facets():
# # #     return db.facets()


# # # @app.get("/api/jobs")
# # # def jobs(source: str = None, location: str = None, domain: str = None,
# # #          remote: bool = None, min_exp: int = None, max_exp: int = None,
# # #          skills: str = None, q: str = None, limit: int = 50, offset: int = 0):
# # #     sk = [s.strip() for s in (skills or "").split(",") if s.strip()]
# # #     return {"jobs": db.query_jobs(source=source, location=location, domain=domain,
# # #                                   remote=remote, min_exp=min_exp, max_exp=max_exp,
# # #                                   skills=sk, q=q, limit=limit, offset=offset)}


# # # @app.get("/api/jobs/{job_id}")
# # # def job_detail(job_id: str):
# # #     j = db.get_job(job_id)
# # #     if not j:
# # #         raise HTTPException(404, "Job not found")
# # #     return j          # includes full description (JD-on-select)


# # # @app.get("/api/market")
# # # def market(domain: str = None):
# # #     return db.market_intel(domain=domain)


# # # @app.get("/api/market/position/{job_id}")
# # # def market_position(job_id: str):
# # #     return db.market_for_position(job_id)


# # # # ── saved jobs / dashboard ──
# # # @app.post("/api/jobs/{job_id}/save")
# # # def save(job_id: str, status: str = "saved", user=Depends(auth.current_user)):
# # #     userstore.save_job(user["id"], job_id, status)
# # #     return {"ok": True}


# # # @app.delete("/api/jobs/{job_id}/save")
# # # def unsave(job_id: str, user=Depends(auth.current_user)):
# # #     userstore.unsave_job(user["id"], job_id)
# # #     return {"ok": True}


# # # @app.get("/api/saved")
# # # def saved(user=Depends(auth.current_user)):
# # #     return {"saved": db.saved_jobs(user["id"])}


# # # @app.get("/api/dashboard")
# # # def dashboard(user=Depends(auth.current_user)):
# # #     d = db.dashboard(user["id"])
# # #     d["name"] = user.get("name") or user["email"].split("@")[0]
# # #     return d


# # # @app.get("/api/recommended")
# # # def recommended(user=Depends(auth.current_user)):
# # #     return {"jobs": db.recommend_jobs(user["id"], limit=40)}


# # # @app.post("/api/analyze")
# # # def analyze(req: AnalyzeReq, user=Depends(auth.current_user)):
# # #     """Parse the uploaded resume, store the detected skills on the profile, and
# # #     return recommendations. Recommendations are only produced AFTER this runs."""
# # #     g = guardrails.check_resume(req.resume_text)
# # #     if not g["ok"]:
# # #         raise HTTPException(400, g["reason"])
# # #     from agent.nodes import _parse_resume
# # #     parsed = _parse_resume(g["text"])
# # #     skills = parsed.get("skills", [])
# # #     prof = userstore.get_profile(user["id"])
# # #     merged = sorted(set((prof.get("skills") or []) + skills), key=str.lower)
# # #     userstore.upsert_profile(
# # #         user["id"], location=prof.get("location", ""),
# # #         exp_years=parsed.get("exp_years") or prof.get("exp_years"),
# # #         skills=merged, preferred_titles=prof.get("preferred_titles", []),
# # #         preferred_locations=prof.get("preferred_locations", []),
# # #         work_pref=prof.get("work_pref", ""),
# # #         experience_level=parsed.get("seniority") or prof.get("experience_level", ""))
# # #     return {"skills": merged, "summary": parsed.get("summary", ""),
# # #             "recommended": db.recommend_jobs(
# # #                 user["id"], limit=40, source=req.source, location=req.location,
# # #                 domain=req.domain, remote=req.remote, min_exp=req.min_exp,
# # #                 max_exp=req.max_exp,
# # #                 extra_skills=[s.strip() for s in req.skills.split(",") if s.strip()])}


# # # # ── BYO LLM key: settings (validate -> encrypt -> store), masked read, delete ──
# # # @app.get("/api/settings/apikey")
# # # def get_apikey(user=Depends(auth.current_user)):
# # #     creds = userstore.get_api_credentials(user["id"])
# # #     if not creds or not creds.get("encrypted_api_key"):
# # #         return {"has_key": False, "provider": None, "model": None, "masked": None}
# # #     plain = crypto.decrypt(creds["encrypted_api_key"])
# # #     return {"has_key": True, "provider": creds["provider"], "model": creds["model"],
# # #             "base_url": creds.get("base_url"), "masked": crypto.mask(plain or "")}


# # # @app.put("/api/settings/apikey")
# # # def put_apikey(req: ApiKeyReq, user=Depends(auth.current_user)):
# # #     if req.provider not in llm_user.PROVIDERS:
# # #         raise HTTPException(400, f"provider must be one of {llm_user.PROVIDERS}")
# # #     ok, msg = llm_user.validate_key(req.provider, req.api_key, req.model, req.base_url)
# # #     if not ok:
# # #         # 401-ish invalid vs quota — surface cleanly, never crash
# # #         raise HTTPException(400, f"Key validation failed: {msg}")
# # #     userstore.set_api_key(user["id"], req.provider, req.model,
# # #                           crypto.encrypt(req.api_key), req.base_url)
# # #     return {"ok": True, "provider": req.provider, "model": req.model,
# # #             "masked": crypto.mask(req.api_key), "message": msg}


# # # @app.delete("/api/settings/apikey")
# # # def delete_apikey(user=Depends(auth.current_user)):
# # #     userstore.clear_api_key(user["id"])
# # #     return {"ok": True}


# # # @app.post("/api/chat")
# # # def chat(req: ChatReq, user=Depends(auth.current_user)):
# # #     creds = userstore.get_api_credentials(user["id"])
# # #     if not creds or not creds.get("encrypted_api_key"):
# # #         raise HTTPException(400, "No API key set. Add one in Settings.")
# # #     key = crypto.decrypt(creds["encrypted_api_key"])
# # #     if not key:
# # #         raise HTTPException(400, "Stored key could not be read; please re-enter it.")
# # #     msg, _ = guardrails.sanitize(req.message)
# # #     prof = userstore.get_profile(user["id"])
# # #     ctx = f"Candidate skills: {prof.get('skills')}. "
# # #     if req.job_id:
# # #         j = db.get_job(req.job_id)
# # #         if j:
# # #             ctx += (f"Job of interest: {j['title']} at {j['company']} "
# # #                     f"(skills: {j.get('skills')}).")
# # #     system = ("You are a career assistant. Help the user with available jobs, "
# # #               "their profile, and interview/application preparation. Be concise "
# # #               "and specific. Context: " + ctx)
# # #     try:
# # #         reply = llm_user.chat(creds["provider"], key, creds["model"], system,
# # #                               msg, req.history, base_url=creds.get("base_url", ""))
# # #         return {"reply": reply}
# # #     except llm_user.LLMError as e:
# # #         status = {"invalid_key": 401, "quota_exceeded": 429}.get(e.kind, 502)
# # #         raise HTTPException(status, e.message)


# # # # ── resume parsing helper ──
# # # @app.post("/api/upload")
# # # async def upload(file: UploadFile = File(...)):
# # #     raw = await file.read()
# # #     name = (file.filename or "").lower()
# # #     if name.endswith(".pdf"):
# # #         import io
# # #         from pypdf import PdfReader
# # #         text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
# # #     elif name.endswith(".docx"):
# # #         import io, docx
# # #         text = "\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)
# # #     else:
# # #         text = raw.decode("utf-8", errors="ignore")
# # #     return {"resume_text": text}


# # # # ── scoring graph (with guardrails) ──
# # # @app.post("/api/start")
# # # def start(req: StartReq):
# # #     g = guardrails.check_resume(req.resume_text)
# # #     if not g["ok"]:
# # #         raise HTTPException(400, g["reason"])
# # #     tid = str(uuid.uuid4())
# # #     r = GRAPH.invoke({"resume_text": g["text"], "domain": req.domain,
# # #                       "location": req.location}, _cfg(tid))
# # #     itr = _interrupt(r)
# # #     return {"thread_id": tid, "matches": itr["matches"] if itr else [],
# # #             "guardrail_flags": g["flags"]}


# # # @app.post("/api/select")
# # # def select(req: SelectReq):
# # #     r = GRAPH.invoke(Command(resume=req.job_id), _cfg(req.thread_id))
# # #     itr = _interrupt(r)
# # #     return {"ats": itr["ats"] if itr else None}


# # # @app.post("/api/action")
# # # def action(req: ActionReq):
# # #     cmd = {"action": req.action, "text": req.text}
# # #     if req.action == "upload_resume":
# # #         g = guardrails.check_resume(req.resume_text)
# # #         if not g["ok"]:
# # #             raise HTTPException(400, g["reason"])
# # #         cmd["resume_text"] = g["text"]
# # #     r = GRAPH.invoke(Command(resume=cmd), _cfg(req.thread_id))
# # #     itr = _interrupt(r)
# # #     return {"ats": itr["ats"] if itr else None, "how_to_add": r.get("how_to_add"),
# # #             "answer": r.get("answer"), "artifacts": r.get("artifacts", {}),
# # #             "done": itr is None}


# # # @app.get("/api/audit")
# # # def audit():
# # #     return {"available": rq_tools.available(), "calls": rq_tools.audit_log()}


# # # # ── pages ──
# # # @app.get("/")
# # # def landing():
# # #     return FileResponse(os.path.join(HERE, "web", "landing.html"))


# # # @app.get("/login")
# # # def login_page():
# # #     return FileResponse(os.path.join(HERE, "web", "login.html"))


# # # @app.get("/app")
# # # def app_page():
# # #     return FileResponse(os.path.join(HERE, "web", "index.html"))


# # # # """FastAPI app: LangGraph agent + auth + jobs/filters + market intel + saved jobs.

# # # # Scoring maps onto the graph's interrupts:
# # # #   /api/start -> select_job interrupt ; /api/select -> review ; /api/action -> loop
# # # # Auth is JWT (bcrypt hashing). Jobs/filters/market/saved/dashboard use SQLite.
# # # # Guardrails sanitise resume text before it reaches the LLM/scorer.
# # # # """
# # # # import os
# # # # import uuid
# # # # from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# # # # from fastapi.responses import FileResponse
# # # # from pydantic import BaseModel
# # # # from langgraph.types import Command

# # # # from agent.graph import build_graph
# # # # from agent import db, auth, guardrails, rq_tools, userstore, crypto, llm_user

# # # # import logging as _logging
# # # # import re as _re
# # # # class _RedactSecrets(_logging.Filter):
# # # #     _pat = _re.compile(r'(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|"api_key"\s*:\s*"[^"]*")')
# # # #     def filter(self, rec):
# # # #         try:
# # # #             if isinstance(rec.msg, str):
# # # #                 rec.msg = self._pat.sub("[REDACTED]", rec.msg)
# # # #             if rec.args:
# # # #                 rec.args = tuple(self._pat.sub("[REDACTED]", str(a)) if isinstance(a, str) else a
# # # #                                  for a in rec.args)
# # # #         except Exception:
# # # #             pass
# # # #         return True
# # # # for _n in ("", "uvicorn", "uvicorn.access", "uvicorn.error"):
# # # #     _logging.getLogger(_n).addFilter(_RedactSecrets())

# # # # app = FastAPI(title="Job Scout ATS Agent")
# # # # GRAPH = build_graph()
# # # # userstore.init()  # create user tables (sqlite fallback or Postgres)

# # # # # Seed a built-in admin account (replaces OAuth). Configure via env.
# # # # _ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@local")
# # # # _ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin12345")
# # # # if not userstore.get_user_by_email(_ADMIN_EMAIL):
# # # #     userstore.create_user(_ADMIN_EMAIL, auth.hash_password(_ADMIN_PASSWORD),
# # # #                           name="Admin", is_admin=True)
# # # # HERE = os.path.dirname(__file__)


# # # # def _cfg(tid): return {"configurable": {"thread_id": tid}}
# # # # def _interrupt(r): return r["__interrupt__"][0].value if "__interrupt__" in r else None


# # # # # ── schemas ──
# # # # class StartReq(BaseModel):
# # # #     resume_text: str = ""
# # # #     domain: str | None = None
# # # #     location: str | None = None


# # # # class SelectReq(BaseModel):
# # # #     thread_id: str
# # # #     job_id: str


# # # # class ActionReq(BaseModel):
# # # #     thread_id: str
# # # #     action: str
# # # #     text: str = ""
# # # #     resume_text: str = ""


# # # # class RegisterReq(BaseModel):
# # # #     email: str
# # # #     password: str
# # # #     name: str = ""


# # # # class LoginReq(BaseModel):
# # # #     email: str
# # # #     password: str


# # # # class AnalyzeReq(BaseModel):
# # # #     resume_text: str


# # # # class ApiKeyReq(BaseModel):
# # # #     provider: str          # openai | anthropic | gemini | custom | test
# # # #     model: str = ""
# # # #     api_key: str
# # # #     base_url: str = ""     # required when provider == custom (OpenAI-compatible)


# # # # class ChatReq(BaseModel):
# # # #     message: str
# # # #     job_id: str | None = None
# # # #     history: list = []


# # # # class ProfileReq(BaseModel):
# # # #     location: str = ""
# # # #     exp_years: int | None = None
# # # #     skills: list[str] = []
# # # #     preferred_titles: list[str] = []
# # # #     preferred_locations: list[str] = []
# # # #     work_pref: str = ""          # remote | hybrid | onsite
# # # #     experience_level: str = ""   # entry | mid | senior


# # # # # ── auth ──
# # # # @app.post("/api/auth/register")
# # # # def register(req: RegisterReq):
# # # #     if userstore.get_user_by_email(req.email):
# # # #         raise HTTPException(409, "Email already registered")
# # # #     if len(req.password) < 8:
# # # #         raise HTTPException(400, "Password must be at least 8 characters")
# # # #     uid = userstore.create_user(req.email, auth.hash_password(req.password), req.name)
# # # #     return {"token": auth.make_token(uid), "user": userstore.get_user(uid)}


# # # # @app.post("/api/auth/login")
# # # # def login(req: LoginReq):
# # # #     u = userstore.get_user_by_email(req.email)
# # # #     if not u or not auth.verify_password(req.password, u["password_hash"]):
# # # #         raise HTTPException(401, "Invalid credentials")
# # # #     return {"token": auth.make_token(u["id"]), "user": userstore.get_user(u["id"])}


# # # # @app.get("/api/auth/me")
# # # # def me(user=Depends(auth.current_user)):
# # # #     return user



# # # # def require_admin(user=Depends(auth.current_user)):
# # # #     if not user.get("is_admin"):
# # # #         raise HTTPException(403, "Admin only")
# # # #     return user


# # # # @app.get("/api/admin/summary")
# # # # def admin_summary(user=Depends(require_admin)):
# # # #     import sqlite3
# # # #     total_jobs = db.market_intel()["total_jobs"]
# # # #     return {"admin": user["email"], "total_jobs": total_jobs}




# # # # # ── profile ──
# # # # @app.get("/api/profile")
# # # # def get_profile(user=Depends(auth.current_user)):
# # # #     p = userstore.get_profile(user["id"])
# # # #     p.pop("encrypted_api_key", None)   # never expose the stored ciphertext
# # # #     return p


# # # # @app.put("/api/profile")
# # # # def put_profile(req: ProfileReq, user=Depends(auth.current_user)):
# # # #     return userstore.upsert_profile(user["id"], **req.model_dump())


# # # # # ── jobs / filters / facets / market ──
# # # # @app.get("/api/facets")
# # # # def facets():
# # # #     return db.facets()


# # # # @app.get("/api/jobs")
# # # # def jobs(source: str = None, location: str = None, domain: str = None,
# # # #          remote: bool = None, min_exp: int = None, max_exp: int = None,
# # # #          skills: str = None, q: str = None, limit: int = 50, offset: int = 0):
# # # #     sk = [s.strip() for s in (skills or "").split(",") if s.strip()]
# # # #     return {"jobs": db.query_jobs(source=source, location=location, domain=domain,
# # # #                                   remote=remote, min_exp=min_exp, max_exp=max_exp,
# # # #                                   skills=sk, q=q, limit=limit, offset=offset)}


# # # # @app.get("/api/jobs/{job_id}")
# # # # def job_detail(job_id: str):
# # # #     j = db.get_job(job_id)
# # # #     if not j:
# # # #         raise HTTPException(404, "Job not found")
# # # #     return j          # includes full description (JD-on-select)


# # # # @app.get("/api/market")
# # # # def market(domain: str = None):
# # # #     return db.market_intel(domain=domain)


# # # # @app.get("/api/market/position/{job_id}")
# # # # def market_position(job_id: str):
# # # #     return db.market_for_position(job_id)


# # # # # ── saved jobs / dashboard ──
# # # # @app.post("/api/jobs/{job_id}/save")
# # # # def save(job_id: str, status: str = "saved", user=Depends(auth.current_user)):
# # # #     userstore.save_job(user["id"], job_id, status)
# # # #     return {"ok": True}


# # # # @app.delete("/api/jobs/{job_id}/save")
# # # # def unsave(job_id: str, user=Depends(auth.current_user)):
# # # #     userstore.unsave_job(user["id"], job_id)
# # # #     return {"ok": True}


# # # # @app.get("/api/saved")
# # # # def saved(user=Depends(auth.current_user)):
# # # #     return {"saved": db.saved_jobs(user["id"])}


# # # # @app.get("/api/dashboard")
# # # # def dashboard(user=Depends(auth.current_user)):
# # # #     d = db.dashboard(user["id"])
# # # #     d["name"] = user.get("name") or user["email"].split("@")[0]
# # # #     return d


# # # # @app.get("/api/recommended")
# # # # def recommended(user=Depends(auth.current_user)):
# # # #     return {"jobs": db.recommend_jobs(user["id"], limit=40)}


# # # # @app.post("/api/analyze")
# # # # def analyze(req: AnalyzeReq, user=Depends(auth.current_user)):
# # # #     """Parse the uploaded resume, store the detected skills on the profile, and
# # # #     return recommendations. Recommendations are only produced AFTER this runs."""
# # # #     g = guardrails.check_resume(req.resume_text)
# # # #     if not g["ok"]:
# # # #         raise HTTPException(400, g["reason"])
# # # #     from agent.nodes import _parse_resume
# # # #     parsed = _parse_resume(g["text"])
# # # #     skills = parsed.get("skills", [])
# # # #     prof = userstore.get_profile(user["id"])
# # # #     merged = sorted(set((prof.get("skills") or []) + skills), key=str.lower)
# # # #     userstore.upsert_profile(
# # # #         user["id"], location=prof.get("location", ""),
# # # #         exp_years=parsed.get("exp_years") or prof.get("exp_years"),
# # # #         skills=merged, preferred_titles=prof.get("preferred_titles", []),
# # # #         preferred_locations=prof.get("preferred_locations", []),
# # # #         work_pref=prof.get("work_pref", ""),
# # # #         experience_level=parsed.get("seniority") or prof.get("experience_level", ""))
# # # #     return {"skills": merged, "summary": parsed.get("summary", ""),
# # # #             "recommended": db.recommend_jobs(user["id"], limit=40)}


# # # # # ── BYO LLM key: settings (validate -> encrypt -> store), masked read, delete ──
# # # # @app.get("/api/settings/apikey")
# # # # def get_apikey(user=Depends(auth.current_user)):
# # # #     creds = userstore.get_api_credentials(user["id"])
# # # #     if not creds or not creds.get("encrypted_api_key"):
# # # #         return {"has_key": False, "provider": None, "model": None, "masked": None}
# # # #     plain = crypto.decrypt(creds["encrypted_api_key"])
# # # #     return {"has_key": True, "provider": creds["provider"], "model": creds["model"],
# # # #             "base_url": creds.get("base_url"), "masked": crypto.mask(plain or "")}


# # # # @app.put("/api/settings/apikey")
# # # # def put_apikey(req: ApiKeyReq, user=Depends(auth.current_user)):
# # # #     if req.provider not in llm_user.PROVIDERS:
# # # #         raise HTTPException(400, f"provider must be one of {llm_user.PROVIDERS}")
# # # #     ok, msg = llm_user.validate_key(req.provider, req.api_key, req.model, req.base_url)
# # # #     if not ok:
# # # #         # 401-ish invalid vs quota — surface cleanly, never crash
# # # #         raise HTTPException(400, f"Key validation failed: {msg}")
# # # #     userstore.set_api_key(user["id"], req.provider, req.model,
# # # #                           crypto.encrypt(req.api_key), req.base_url)
# # # #     return {"ok": True, "provider": req.provider, "model": req.model,
# # # #             "masked": crypto.mask(req.api_key), "message": msg}


# # # # @app.delete("/api/settings/apikey")
# # # # def delete_apikey(user=Depends(auth.current_user)):
# # # #     userstore.clear_api_key(user["id"])
# # # #     return {"ok": True}


# # # # @app.post("/api/chat")
# # # # def chat(req: ChatReq, user=Depends(auth.current_user)):
# # # #     creds = userstore.get_api_credentials(user["id"])
# # # #     if not creds or not creds.get("encrypted_api_key"):
# # # #         raise HTTPException(400, "No API key set. Add one in Settings.")
# # # #     key = crypto.decrypt(creds["encrypted_api_key"])
# # # #     if not key:
# # # #         raise HTTPException(400, "Stored key could not be read; please re-enter it.")
# # # #     msg, _ = guardrails.sanitize(req.message)
# # # #     prof = userstore.get_profile(user["id"])
# # # #     ctx = f"Candidate skills: {prof.get('skills')}. "
# # # #     if req.job_id:
# # # #         j = db.get_job(req.job_id)
# # # #         if j:
# # # #             ctx += (f"Job of interest: {j['title']} at {j['company']} "
# # # #                     f"(skills: {j.get('skills')}).")
# # # #     system = ("You are a career assistant. Help the user with available jobs, "
# # # #               "their profile, and interview/application preparation. Be concise "
# # # #               "and specific. Context: " + ctx)
# # # #     try:
# # # #         reply = llm_user.chat(creds["provider"], key, creds["model"], system,
# # # #                               msg, req.history, base_url=creds.get("base_url", ""))
# # # #         return {"reply": reply}
# # # #     except llm_user.LLMError as e:
# # # #         status = {"invalid_key": 401, "quota_exceeded": 429}.get(e.kind, 502)
# # # #         raise HTTPException(status, e.message)


# # # # # ── resume parsing helper ──
# # # # @app.post("/api/upload")
# # # # async def upload(file: UploadFile = File(...)):
# # # #     raw = await file.read()
# # # #     name = (file.filename or "").lower()
# # # #     if name.endswith(".pdf"):
# # # #         import io
# # # #         from pypdf import PdfReader
# # # #         text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
# # # #     elif name.endswith(".docx"):
# # # #         import io, docx
# # # #         text = "\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)
# # # #     else:
# # # #         text = raw.decode("utf-8", errors="ignore")
# # # #     return {"resume_text": text}


# # # # # ── scoring graph (with guardrails) ──
# # # # @app.post("/api/start")
# # # # def start(req: StartReq):
# # # #     g = guardrails.check_resume(req.resume_text)
# # # #     if not g["ok"]:
# # # #         raise HTTPException(400, g["reason"])
# # # #     tid = str(uuid.uuid4())
# # # #     r = GRAPH.invoke({"resume_text": g["text"], "domain": req.domain,
# # # #                       "location": req.location}, _cfg(tid))
# # # #     itr = _interrupt(r)
# # # #     return {"thread_id": tid, "matches": itr["matches"] if itr else [],
# # # #             "guardrail_flags": g["flags"]}


# # # # @app.post("/api/select")
# # # # def select(req: SelectReq):
# # # #     r = GRAPH.invoke(Command(resume=req.job_id), _cfg(req.thread_id))
# # # #     itr = _interrupt(r)
# # # #     return {"ats": itr["ats"] if itr else None}


# # # # @app.post("/api/action")
# # # # def action(req: ActionReq):
# # # #     cmd = {"action": req.action, "text": req.text}
# # # #     if req.action == "upload_resume":
# # # #         g = guardrails.check_resume(req.resume_text)
# # # #         if not g["ok"]:
# # # #             raise HTTPException(400, g["reason"])
# # # #         cmd["resume_text"] = g["text"]
# # # #     r = GRAPH.invoke(Command(resume=cmd), _cfg(req.thread_id))
# # # #     itr = _interrupt(r)
# # # #     return {"ats": itr["ats"] if itr else None, "how_to_add": r.get("how_to_add"),
# # # #             "answer": r.get("answer"), "artifacts": r.get("artifacts", {}),
# # # #             "done": itr is None}


# # # # @app.get("/api/audit")
# # # # def audit():
# # # #     return {"available": rq_tools.available(), "calls": rq_tools.audit_log()}


# # # # # ── pages ──
# # # # @app.get("/")
# # # # def landing():
# # # #     return FileResponse(os.path.join(HERE, "web", "landing.html"))


# # # # @app.get("/login")
# # # # def login_page():
# # # #     return FileResponse(os.path.join(HERE, "web", "login.html"))


# # # # @app.get("/app")
# # # # def app_page():
# # # #     return FileResponse(os.path.join(HERE, "web", "index.html"))
# # # # # """FastAPI app: LangGraph agent + auth + jobs/filters + market intel + saved jobs.

# # # # # Scoring maps onto the graph's interrupts:
# # # # #   /api/start -> select_job interrupt ; /api/select -> review ; /api/action -> loop
# # # # # Auth is JWT (bcrypt hashing). Jobs/filters/market/saved/dashboard use SQLite.
# # # # # Guardrails sanitise resume text before it reaches the LLM/scorer.
# # # # # """
# # # # # import os
# # # # # import uuid
# # # # # from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# # # # # from fastapi.responses import FileResponse
# # # # # from pydantic import BaseModel
# # # # # from langgraph.types import Command

# # # # # from agent.graph import build_graph
# # # # # from agent import db, auth, guardrails, rq_tools, userstore

# # # # # app = FastAPI(title="Job Scout ATS Agent")
# # # # # GRAPH = build_graph()
# # # # # userstore.init()  # create user tables (sqlite fallback or Postgres)

# # # # # # Seed a built-in admin account (replaces OAuth). Configure via env.
# # # # # _ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@local")
# # # # # _ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin12345")
# # # # # if not userstore.get_user_by_email(_ADMIN_EMAIL):
# # # # #     userstore.create_user(_ADMIN_EMAIL, auth.hash_password(_ADMIN_PASSWORD),
# # # # #                           name="Admin", is_admin=True)
# # # # # HERE = os.path.dirname(__file__)


# # # # # def _cfg(tid): return {"configurable": {"thread_id": tid}}
# # # # # def _interrupt(r): return r["__interrupt__"][0].value if "__interrupt__" in r else None


# # # # # # ── schemas ──
# # # # # class StartReq(BaseModel):
# # # # #     resume_text: str = ""
# # # # #     domain: str | None = None
# # # # #     location: str | None = None


# # # # # class SelectReq(BaseModel):
# # # # #     thread_id: str
# # # # #     job_id: str


# # # # # class ActionReq(BaseModel):
# # # # #     thread_id: str
# # # # #     action: str
# # # # #     text: str = ""
# # # # #     resume_text: str = ""


# # # # # class RegisterReq(BaseModel):
# # # # #     email: str
# # # # #     password: str
# # # # #     name: str = ""


# # # # # class LoginReq(BaseModel):
# # # # #     email: str
# # # # #     password: str


# # # # # class ProfileReq(BaseModel):
# # # # #     location: str = ""
# # # # #     exp_years: int | None = None
# # # # #     skills: list[str] = []
# # # # #     preferred_titles: list[str] = []
# # # # #     preferred_locations: list[str] = []
# # # # #     work_pref: str = ""          # remote | hybrid | onsite
# # # # #     experience_level: str = ""   # entry | mid | senior


# # # # # # ── auth ──
# # # # # @app.post("/api/auth/register")
# # # # # def register(req: RegisterReq):
# # # # #     if userstore.get_user_by_email(req.email):
# # # # #         raise HTTPException(409, "Email already registered")
# # # # #     if len(req.password) < 8:
# # # # #         raise HTTPException(400, "Password must be at least 8 characters")
# # # # #     uid = userstore.create_user(req.email, auth.hash_password(req.password), req.name)
# # # # #     return {"token": auth.make_token(uid), "user": userstore.get_user(uid)}


# # # # # @app.post("/api/auth/login")
# # # # # def login(req: LoginReq):
# # # # #     u = userstore.get_user_by_email(req.email)
# # # # #     if not u or not auth.verify_password(req.password, u["password_hash"]):
# # # # #         raise HTTPException(401, "Invalid credentials")
# # # # #     return {"token": auth.make_token(u["id"]), "user": userstore.get_user(u["id"])}


# # # # # @app.get("/api/auth/me")
# # # # # def me(user=Depends(auth.current_user)):
# # # # #     return user


# # # # # def require_admin(user=Depends(auth.current_user)):
# # # # #     if not user.get("is_admin"):
# # # # #         raise HTTPException(403, "Admin only")
# # # # #     return user


# # # # # @app.get("/api/admin/summary")
# # # # # def admin_summary(user=Depends(require_admin)):
# # # # #     import sqlite3
# # # # #     total_jobs = db.market_intel()["total_jobs"]
# # # # #     return {"admin": user["email"], "total_jobs": total_jobs}




# # # # # # ── profile ──
# # # # # @app.get("/api/profile")
# # # # # def get_profile(user=Depends(auth.current_user)):
# # # # #     return userstore.get_profile(user["id"])


# # # # # @app.put("/api/profile")
# # # # # def put_profile(req: ProfileReq, user=Depends(auth.current_user)):
# # # # #     return userstore.upsert_profile(user["id"], **req.model_dump())


# # # # # # ── jobs / filters / facets / market ──
# # # # # @app.get("/api/facets")
# # # # # def facets():
# # # # #     return db.facets()


# # # # # @app.get("/api/jobs")
# # # # # def jobs(source: str = None, location: str = None, domain: str = None,
# # # # #          remote: bool = None, min_exp: int = None, max_exp: int = None,
# # # # #          skills: str = None, q: str = None, limit: int = 50, offset: int = 0):
# # # # #     sk = [s.strip() for s in (skills or "").split(",") if s.strip()]
# # # # #     return {"jobs": db.query_jobs(source=source, location=location, domain=domain,
# # # # #                                   remote=remote, min_exp=min_exp, max_exp=max_exp,
# # # # #                                   skills=sk, q=q, limit=limit, offset=offset)}


# # # # # @app.get("/api/jobs/{job_id}")
# # # # # def job_detail(job_id: str):
# # # # #     j = db.get_job(job_id)
# # # # #     if not j:
# # # # #         raise HTTPException(404, "Job not found")
# # # # #     return j          # includes full description (JD-on-select)


# # # # # @app.get("/api/market")
# # # # # def market(domain: str = None):
# # # # #     return db.market_intel(domain=domain)


# # # # # @app.get("/api/market/position/{job_id}")
# # # # # def market_position(job_id: str):
# # # # #     return db.market_for_position(job_id)


# # # # # # ── saved jobs / dashboard ──
# # # # # @app.post("/api/jobs/{job_id}/save")
# # # # # def save(job_id: str, status: str = "saved", user=Depends(auth.current_user)):
# # # # #     userstore.save_job(user["id"], job_id, status)
# # # # #     return {"ok": True}


# # # # # @app.delete("/api/jobs/{job_id}/save")
# # # # # def unsave(job_id: str, user=Depends(auth.current_user)):
# # # # #     userstore.unsave_job(user["id"], job_id)
# # # # #     return {"ok": True}


# # # # # @app.get("/api/saved")
# # # # # def saved(user=Depends(auth.current_user)):
# # # # #     return {"saved": db.saved_jobs(user["id"])}


# # # # # @app.get("/api/dashboard")
# # # # # def dashboard(user=Depends(auth.current_user)):
# # # # #     d = db.dashboard(user["id"])
# # # # #     d["name"] = user.get("name") or user["email"].split("@")[0]
# # # # #     return d


# # # # # @app.get("/api/recommended")
# # # # # def recommended(user=Depends(auth.current_user)):
# # # # #     return {"jobs": db.recommend_jobs(user["id"], limit=40)}


# # # # # # ── resume parsing helper ──
# # # # # @app.post("/api/upload")
# # # # # async def upload(file: UploadFile = File(...)):
# # # # #     raw = await file.read()
# # # # #     name = (file.filename or "").lower()
# # # # #     if name.endswith(".pdf"):
# # # # #         import io
# # # # #         from pypdf import PdfReader
# # # # #         text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
# # # # #     elif name.endswith(".docx"):
# # # # #         import io, docx
# # # # #         text = "\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)
# # # # #     else:
# # # # #         text = raw.decode("utf-8", errors="ignore")
# # # # #     return {"resume_text": text}


# # # # # # ── scoring graph (with guardrails) ──
# # # # # @app.post("/api/start")
# # # # # def start(req: StartReq):
# # # # #     g = guardrails.check_resume(req.resume_text)
# # # # #     if not g["ok"]:
# # # # #         raise HTTPException(400, g["reason"])
# # # # #     tid = str(uuid.uuid4())
# # # # #     r = GRAPH.invoke({"resume_text": g["text"], "domain": req.domain,
# # # # #                       "location": req.location}, _cfg(tid))
# # # # #     itr = _interrupt(r)
# # # # #     return {"thread_id": tid, "matches": itr["matches"] if itr else [],
# # # # #             "guardrail_flags": g["flags"]}


# # # # # @app.post("/api/select")
# # # # # def select(req: SelectReq):
# # # # #     r = GRAPH.invoke(Command(resume=req.job_id), _cfg(req.thread_id))
# # # # #     itr = _interrupt(r)
# # # # #     return {"ats": itr["ats"] if itr else None}


# # # # # @app.post("/api/action")
# # # # # def action(req: ActionReq):
# # # # #     cmd = {"action": req.action, "text": req.text}
# # # # #     if req.action == "upload_resume":
# # # # #         g = guardrails.check_resume(req.resume_text)
# # # # #         if not g["ok"]:
# # # # #             raise HTTPException(400, g["reason"])
# # # # #         cmd["resume_text"] = g["text"]
# # # # #     r = GRAPH.invoke(Command(resume=cmd), _cfg(req.thread_id))
# # # # #     itr = _interrupt(r)
# # # # #     return {"ats": itr["ats"] if itr else None, "how_to_add": r.get("how_to_add"),
# # # # #             "answer": r.get("answer"), "artifacts": r.get("artifacts", {}),
# # # # #             "done": itr is None}


# # # # # @app.get("/api/audit")
# # # # # def audit():
# # # # #     return {"available": rq_tools.available(), "calls": rq_tools.audit_log()}


# # # # # # ── pages ──
# # # # # @app.get("/")
# # # # # def landing():
# # # # #     return FileResponse(os.path.join(HERE, "web", "landing.html"))


# # # # # @app.get("/login")
# # # # # def login_page():
# # # # #     return FileResponse(os.path.join(HERE, "web", "login.html"))


# # # # # @app.get("/app")
# # # # # def app_page():
# # # # #     return FileResponse(os.path.join(HERE, "web", "index.html"))
