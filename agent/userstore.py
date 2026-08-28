"""User data (users / profiles / saved_jobs) — MongoDB Atlas.

Single database for the whole app (jobs + users) via MONGODB_URI. No SQLite/
Postgres. User ids are the Mongo _id (ObjectId) rendered as a hex string, which
is what goes into the JWT `sub`.

PRIVACY: encrypted API keys are Fernet ciphertext (agent/crypto.py); never store
plaintext secrets.
"""
import os
from datetime import datetime, timezone

MONGODB_DB = os.environ.get("MONGODB_DB", "jobscout")
_database = None


def _db():
    global _database
    if _database is None:
        from pymongo import MongoClient
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGODB_URI is not set")
        _database = MongoClient(uri, appname="job-scout")[MONGODB_DB]
    return _database


def _oid(user_id):
    from bson import ObjectId
    return ObjectId(user_id)


def _now():
    return datetime.now(timezone.utc).isoformat()


def init():
    """Create indexes (idempotent)."""
    db = _db()
    db.users.create_index("email", unique=True)
    db.saved_jobs.create_index([("user_id", 1), ("job_id", 1)], unique=True)
    db.jobs.create_index("source")
    db.jobs.create_index("domain")
    db.jobs.create_index("remote")
    db.jobs.create_index("posted_month")
    db.jobs.create_index("skills_lc")


# ── users ────────────────────────────────────────────────────────────────────
def create_user(email, password_hash, name="", oauth_provider=None, is_admin=False):
    res = _db().users.insert_one({
        "email": email.lower(), "password_hash": password_hash, "name": name,
        "oauth_provider": oauth_provider, "created_at": _now(), "is_admin": bool(is_admin)})
    return str(res.inserted_id)


def get_user_by_email(email):
    d = _db().users.find_one({"email": email.lower()})
    if not d:
        return None
    return {"id": str(d["_id"]), "email": d["email"], "name": d.get("name", ""),
            "password_hash": d.get("password_hash"), "is_admin": bool(d.get("is_admin"))}


def get_user(user_id):
    try:
        d = _db().users.find_one({"_id": _oid(user_id)})
    except Exception:
        return None
    if not d:
        return None
    return {"id": str(d["_id"]), "email": d["email"], "name": d.get("name", ""),
            "is_admin": bool(d.get("is_admin"))}


# ── profile ──────────────────────────────────────────────────────────────────
_PCOLS = ("location", "exp_years", "skills", "preferred_titles",
          "preferred_locations", "work_pref", "experience_level")


def get_profile(user_id):
    d = _db().profiles.find_one({"_id": str(user_id)}) or {}
    out = {"user_id": str(user_id)}
    for k in _PCOLS:
        out[k] = d.get(k, [] if k in ("skills", "preferred_titles", "preferred_locations") else None)
    return out


def upsert_profile(user_id, **f):
    doc = {c: f.get(c) for c in _PCOLS}
    _db().profiles.update_one({"_id": str(user_id)}, {"$set": doc}, upsert=True)
    return get_profile(user_id)


# ── saved jobs ───────────────────────────────────────────────────────────────
def save_job(user_id, job_id, status="saved"):
    _db().saved_jobs.update_one(
        {"user_id": str(user_id), "job_id": job_id},
        {"$set": {"status": status, "saved_at": _now()}}, upsert=True)


def unsave_job(user_id, job_id):
    _db().saved_jobs.delete_one({"user_id": str(user_id), "job_id": job_id})


def saved_job_ids(user_id):
    cur = _db().saved_jobs.find({"user_id": str(user_id)}).sort("saved_at", -1)
    return [(d["job_id"], d.get("status", "saved"), d.get("saved_at", "")) for d in cur]


def status_counts(user_id):
    out = {}
    for g in _db().saved_jobs.aggregate([
            {"$match": {"user_id": str(user_id)}},
            {"$group": {"_id": "$status", "n": {"$sum": 1}}}]):
        out[g["_id"]] = g["n"]
    return out


# ── LLM API-key settings (Fernet ciphertext stored here) ─────────────────────
def set_api_key(user_id, provider, model, encrypted, base_url=""):
    _db().profiles.update_one(
        {"_id": str(user_id)},
        {"$set": {"llm_provider": provider, "api_model": model,
                  "encrypted_api_key": encrypted, "api_base_url": base_url}}, upsert=True)


def get_api_credentials(user_id):
    d = _db().profiles.find_one({"_id": str(user_id)},
                                {"llm_provider": 1, "api_model": 1,
                                 "encrypted_api_key": 1, "api_base_url": 1})
    if not d:
        return None
    return {"provider": d.get("llm_provider"), "model": d.get("api_model"),
            "encrypted_api_key": d.get("encrypted_api_key"),
            "base_url": d.get("api_base_url")}


def clear_api_key(user_id):
    _db().profiles.update_one(
        {"_id": str(user_id)},
        {"$unset": {"llm_provider": "", "api_model": "",
                    "encrypted_api_key": "", "api_base_url": ""}})# """User data (users / profiles / saved_jobs) — the ONLY mutable data.

# Backend is chosen at runtime:
#   * DATABASE_URL set  -> PostgreSQL (durable, shared across instances). Use this
#     on Render/anywhere the container filesystem is ephemeral.
#   * else              -> SQLite file at USER_DB (local dev / single box).

# Jobs stay in the read-only SQLite (agent/db.py) — they're immutable and rebuilt
# each deploy, so ephemeral storage is fine for them. Only the small, read-write
# user tables need a persistent DB, which is what this module isolates.

# Portable SQL: both engines support `ON CONFLICT ... DO UPDATE SET x=EXCLUDED.x`.
# We only branch on the id placeholder (`?` vs `%s`), the identity column type,
# and how a new id is returned.
# """
# import os
# import json
# from datetime import datetime, timezone

# DATABASE_URL = os.environ.get("DATABASE_URL")
# IS_PG = bool(DATABASE_URL)
# USER_DB = os.environ.get("USER_DB",
#                          os.path.join(os.path.dirname(__file__), "..", "data", "users.db"))


# def _connect():
#     if IS_PG:
#         import psycopg
#         return psycopg.connect(DATABASE_URL)
#     import sqlite3
#     con = sqlite3.connect(USER_DB, check_same_thread=False)
#     con.row_factory = sqlite3.Row
#     return con


# def _ph(sql: str) -> str:
#     """SQLite uses ?, Postgres uses %s."""
#     return sql.replace("?", "%s") if IS_PG else sql


# _ID = ("SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT")

# SCHEMA = f"""
# CREATE TABLE IF NOT EXISTS users (
#   id {_ID}, email TEXT UNIQUE NOT NULL, password_hash TEXT, name TEXT,
#   oauth_provider TEXT, created_at TEXT, is_admin INTEGER DEFAULT 0);
# CREATE TABLE IF NOT EXISTS profiles (
#   user_id INTEGER PRIMARY KEY, location TEXT, exp_years INTEGER, skills TEXT,
#   preferred_titles TEXT, preferred_locations TEXT, work_pref TEXT,
#   experience_level TEXT,
#   llm_provider TEXT, api_model TEXT, encrypted_api_key TEXT, api_base_url TEXT);
# CREATE TABLE IF NOT EXISTS saved_jobs (
#   user_id INTEGER, job_id TEXT, status TEXT DEFAULT 'saved', saved_at TEXT,
#   PRIMARY KEY(user_id, job_id));
# """


# # Columns added after the original schema — auto-added to existing DBs on boot
# # so upgrading never loses accounts and never hits "no such column".
# _MIGRATIONS = {
#     "users": [("is_admin", "INTEGER DEFAULT 0")],
#     "profiles": [("llm_provider", "TEXT"), ("api_model", "TEXT"),
#                  ("encrypted_api_key", "TEXT"), ("api_base_url", "TEXT")],
# }


# def _existing_columns(cur, table):
#     if IS_PG:
#         cur.execute("SELECT column_name FROM information_schema.columns "
#                     "WHERE table_name = %s", (table,))
#         return {r[0] for r in cur.fetchall()}
#     cur.execute(f"PRAGMA table_info({table})")
#     return {r[1] for r in cur.fetchall()}


# def _migrate(con):
#     cur = con.cursor()
#     for table, cols in _MIGRATIONS.items():
#         have = _existing_columns(cur, table)
#         for name, decl in cols:
#             if name not in have:
#                 try:
#                     cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
#                 except Exception:
#                     pass   # already added by a concurrent worker, etc.


# def init():
#     con = _connect()
#     try:
#         if IS_PG:
#             with con.cursor() as cur:
#                 cur.execute(SCHEMA)
#         else:
#             con.executescript(SCHEMA)
#         con.commit()
#         _migrate(con)          # add any columns missing from an older DB
#         con.commit()
#     finally:
#         con.close()


# def _now():
#     return datetime.now(timezone.utc).isoformat()


# def _rowdict(cur, row):
#     if row is None:
#         return None
#     if IS_PG:
#         cols = [c.name for c in cur.description]
#         return dict(zip(cols, row))
#     return dict(row)


# # ── users ──
# def create_user(email, password_hash, name="", oauth_provider=None, is_admin=False):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         if IS_PG:
#             cur.execute(_ph("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
#                             "VALUES (?,?,?,?,?,?) RETURNING id"),
#                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
#             uid = cur.fetchone()[0]
#         else:
#             cur.execute("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
#                         "VALUES (?,?,?,?,?,?)",
#                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
#             uid = cur.lastrowid
#         con.commit()
#         return uid
#     finally:
#         con.close()


# def get_user_by_email(email):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("SELECT id,email,password_hash,name,is_admin FROM users WHERE email=?"),
#                     (email.lower(),))
#         return _rowdict(cur, cur.fetchone())
#     finally:
#         con.close()


# def get_user(user_id):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("SELECT id,email,name,is_admin FROM users WHERE id=?"), (user_id,))
#         return _rowdict(cur, cur.fetchone())
#     finally:
#         con.close()


# # ── profile ──
# _PCOLS = ["location", "exp_years", "skills", "preferred_titles",
#           "preferred_locations", "work_pref", "experience_level"]


# def get_profile(user_id):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("SELECT * FROM profiles WHERE user_id=?"), (user_id,))
#         d = _rowdict(cur, cur.fetchone()) or {"user_id": user_id}
#     finally:
#         con.close()
#     for k in ("skills", "preferred_titles", "preferred_locations"):
#         if isinstance(d.get(k), str):
#             d[k] = json.loads(d[k]) if d[k] else []
#     return d


# def upsert_profile(user_id, **f):
#     vals = {c: (json.dumps(f.get(c)) if isinstance(f.get(c), list) else f.get(c))
#             for c in _PCOLS}
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph(
#             f"INSERT INTO profiles(user_id,{','.join(_PCOLS)}) "
#             f"VALUES (?,{','.join(['?']*len(_PCOLS))}) "
#             f"ON CONFLICT(user_id) DO UPDATE SET "
#             f"{','.join(f'{c}=EXCLUDED.{c}' for c in _PCOLS)}"),
#             [user_id] + [vals[c] for c in _PCOLS])
#         con.commit()
#     finally:
#         con.close()
#     return get_profile(user_id)


# # ── saved jobs ──
# def save_job(user_id, job_id, status="saved"):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("INSERT INTO saved_jobs(user_id,job_id,status,saved_at) VALUES (?,?,?,?) "
#                         "ON CONFLICT(user_id,job_id) DO UPDATE SET status=EXCLUDED.status"),
#                     (user_id, job_id, status, _now()))
#         con.commit()
#     finally:
#         con.close()


# def unsave_job(user_id, job_id):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("DELETE FROM saved_jobs WHERE user_id=? AND job_id=?"), (user_id, job_id))
#         con.commit()
#     finally:
#         con.close()


# def saved_job_ids(user_id):
#     """Return [(job_id, status, saved_at)] — jobs are joined in db.py (jobs live
#     in the read-only SQLite, a different database)."""
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("SELECT job_id,status,saved_at FROM saved_jobs WHERE user_id=? "
#                         "ORDER BY saved_at DESC"), (user_id,))
#         rows = cur.fetchall()
#     finally:
#         con.close()
#     return [(r[0], r[1], r[2]) for r in rows]


# def status_counts(user_id):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("SELECT status, COUNT(*) FROM saved_jobs WHERE user_id=? GROUP BY status"),
#                     (user_id,))
#         return dict(cur.fetchall())
#     finally:
#         con.close()


# # ── LLM API-key settings (encrypted at rest; ciphertext stored here) ────────
# def set_api_key(user_id, provider, model, encrypted, base_url=""):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph(
#             "INSERT INTO profiles(user_id,llm_provider,api_model,encrypted_api_key,api_base_url) "
#             "VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
#             "llm_provider=EXCLUDED.llm_provider, api_model=EXCLUDED.api_model, "
#             "encrypted_api_key=EXCLUDED.encrypted_api_key, api_base_url=EXCLUDED.api_base_url"),
#             (user_id, provider, model, encrypted, base_url))
#         con.commit()
#     finally:
#         con.close()


# def get_api_credentials(user_id):
#     """Raw row incl. ciphertext — server-side use only, never returned to client."""
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("SELECT llm_provider,api_model,encrypted_api_key,api_base_url "
#                         "FROM profiles WHERE user_id=?"), (user_id,))
#         row = cur.fetchone()
#     finally:
#         con.close()
#     if not row:
#         return None
#     return {"provider": row[0], "model": row[1], "encrypted_api_key": row[2],
#             "base_url": row[3]}


# def clear_api_key(user_id):
#     con = _connect()
#     try:
#         cur = con.cursor()
#         cur.execute(_ph("UPDATE profiles SET llm_provider=NULL, api_model=NULL, "
#                         "encrypted_api_key=NULL, api_base_url=NULL WHERE user_id=?"), (user_id,))
#         con.commit()
#     finally:
#         con.close()

# # """BYO-key LLM adapter.

# # Users bring their own API key + model. We call the provider over raw HTTP (no
# # heavy SDKs) for two operations:
# #   * validate_key(...)  -> a lightweight check before saving (list models / tiny prompt)
# #   * chat(...)          -> answer a question grounded in the user's context

# # Errors are mapped to clean outcomes so a bad key or quota never crashes the
# # worker: 401/403 -> invalid_key, 429 -> quota_exceeded, others -> provider_error.

# # A built-in `test` provider (key must start with "test-") is included so the flow
# # can be exercised without network / real keys.
# # """
# # import json
# # import urllib.request
# # import urllib.error

# # PROVIDERS = ("openai", "anthropic", "gemini", "groq", "custom", "test")

# # # "custom" = any OpenAI-compatible endpoint (Groq, Together, OpenRouter, Mistral,
# # # DeepSeek, Fireworks, Azure, local Ollama/LM Studio, ...) via a base_url.


# # GROQ_BASE = "https://api.groq.com/openai/v1"


# # class LLMError(Exception):
# #     def __init__(self, kind, message):
# #         self.kind = kind          # invalid_key | quota_exceeded | provider_error | bad_provider
# #         self.message = message
# #         super().__init__(message)


# # def _http(method, url, headers=None, body=None, timeout=60):
# #     data = json.dumps(body).encode() if body is not None else None
# #     # Some providers (Groq via Cloudflare) reject requests with no User-Agent,
# #     # which surfaced as spurious 401s — send a UA + Accept like a normal client.
# #     req_headers = {"Accept": "application/json", "User-Agent": "job-scout-agent/1.0"}
# #     if headers:
# #         req_headers.update(headers)
# #     req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
# #     try:
# #         with urllib.request.urlopen(req, timeout=timeout) as r:
# #             return r.status, json.loads(r.read().decode())
# #     except urllib.error.HTTPError as e:
# #         status = e.code
# #         try:
# #             payload = json.loads(e.read().decode())
# #         except Exception:
# #             payload = {"error": str(e)}
# #         if status in (401, 403):
# #             raise LLMError("invalid_key", "The API key was rejected (401/403).")
# #         if status == 429:
# #             raise LLMError("quota_exceeded", "Rate limit / quota exceeded (429).")
# #         raise LLMError("provider_error", f"Provider returned {status}: {str(payload)[:200]}")
# #     except urllib.error.URLError as e:
# #         raise LLMError("provider_error", f"Could not reach provider: {e}")


# # def _oai_chat(base, api_key, model, messages, max_out):
# #     """POST /chat/completions, tolerating both token-limit param names.
# #     gpt-oss / o-series need `max_completion_tokens`; older models want
# #     `max_tokens`. Try the new name, fall back on a 400."""
# #     hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
# #     last = None
# #     for pkey in ("max_completion_tokens", "max_tokens"):
# #         try:
# #             _, data = _http("POST", base + "/chat/completions", headers=hdr,
# #                             body={"model": model, "messages": messages, pkey: max_out})
# #             return data
# #         except LLMError as e:
# #             if e.kind == "provider_error" and pkey == "max_completion_tokens":
# #                 last = e            # param not accepted -> try the older name
# #                 continue
# #             raise
# #     raise last


# # # ── validation ─────────────────────────────────────────────────────────────
# # def validate_key(provider, api_key, model="", base_url=""):
# #     api_key = (api_key or "").strip()
# #     """Return (ok: bool, message: str). Never raises."""
# #     try:
# #         if provider == "openai" and api_key.startswith("gsk_"):
# #             return False, ("This looks like a Groq key (gsk_…). Set Provider = Groq, "
# #                            "not OpenAI.")
# #         if provider == "groq" and api_key.startswith("sk-") and not api_key.startswith("gsk_"):
# #             return False, ("This looks like an OpenAI key (sk-…). Set Provider = OpenAI, "
# #                            "not Groq.")
# #         if provider == "test":
# #             if not api_key.startswith("test-"):
# #                 raise LLMError("invalid_key", "test keys must start with 'test-'")
# #             return True, "Test key accepted."
# #         if provider in ("openai", "groq", "custom"):
# #             base = ("https://api.openai.com/v1" if provider == "openai"
# #                     else GROQ_BASE if provider == "groq"
# #                     else (base_url or "").rstrip("/"))
# #             if provider == "custom" and not base:
# #                 return False, "Custom provider needs a base URL (e.g. https://api.groq.com/openai/v1)."
# #             hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
# #             if model:
# #                 # mirror a real client call; tolerates max_tokens vs max_completion_tokens
# #                 _oai_chat(base, api_key, model,
# #                           [{"role": "user", "content": "ping"}], 32)
# #                 return True, f"Key valid ({provider} responded for {model})."
# #             # no model given -> just check the key against the models list
# #             _http("GET", base + "/models", headers={"Authorization": f"Bearer {api_key}"})
# #             return True, f"Key valid ({provider} models listed)."
# #         if provider == "gemini":
# #             # Gemini keys go in the x-goog-api-key HEADER (not ?key=), which works
# #             # for both legacy AIza... keys and the new AQ... authorization keys.
# #             _http("GET", "https://generativelanguage.googleapis.com/v1beta/models",
# #                   headers={"x-goog-api-key": api_key})
# #             return True, "Key valid (Gemini models listed)."
# #         if provider == "anthropic":
# #             # cheapest check: a 1-token message
# #             _http("POST", "https://api.anthropic.com/v1/messages",
# #                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
# #                            "content-type": "application/json"},
# #                   body={"model": model or "claude-3-5-haiku-20241022",
# #                         "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
# #             return True, "Key valid (Anthropic responded)."
# #         return False, f"Unknown provider '{provider}'."
# #     except LLMError as e:
# #         return False, e.message


# # # ── chat ───────────────────────────────────────────────────────────────────
# # def chat(provider, api_key, model, system, message, history=None, base_url=""):
# #     api_key = (api_key or "").strip()
# #     """Return assistant text. Raises LLMError on failure."""
# #     history = history or []
# #     if provider == "test":
# #         return f"[test:{model or 'demo'}] You asked: {message[:200]}"
# #     if provider in ("openai", "groq", "custom"):
# #         base = ("https://api.openai.com/v1" if provider == "openai"
# #                 else GROQ_BASE if provider == "groq"
# #                 else (base_url or "").rstrip("/"))
# #         if provider == "custom" and not base:
# #             raise LLMError("bad_provider", "Custom provider needs a base URL.")
# #         msgs = [{"role": "system", "content": system}] + history + \
# #                [{"role": "user", "content": message}]
# #         data = _oai_chat(base, api_key,
# #                          model or ("llama-3.1-8b-instant" if provider == "groq" else "gpt-4o-mini"),
# #                          msgs, 1500)
# #         m = data["choices"][0]["message"]
# #         # reasoning models put the answer in content after thinking; if the budget
# #         # was consumed by reasoning, surface that so the reply is never blank.
# #         return m.get("content") or m.get("reasoning") or "(model returned no content — try a higher token budget)"
# #     if provider == "anthropic":
# #         _, data = _http("POST", "https://api.anthropic.com/v1/messages",
# #                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
# #                                  "content-type": "application/json"},
# #                         body={"model": model or "claude-3-5-sonnet-20241022",
# #                               "max_tokens": 600, "system": system,
# #                               "messages": history + [{"role": "user", "content": message}]})
# #         return "".join(b.get("text", "") for b in data.get("content", []))
# #     if provider == "gemini":
# #         url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
# #                f"{model or 'gemini-1.5-flash'}:generateContent")
# #         _, data = _http("POST", url,
# #                         headers={"x-goog-api-key": api_key,
# #                                  "Content-Type": "application/json"},
# #                         body={"contents": [{"parts": [{"text": system + "\n\n" + message}]}]})
# #         try:
# #             parts = data["candidates"][0]["content"]["parts"]
# #             return "".join(p.get("text", "") for p in parts) or "(no content returned)"
# #         except (KeyError, IndexError):
# #             return "(no content returned)"
# #     raise LLMError("bad_provider", f"Unknown provider '{provider}'")

# # # """User data (users / profiles / saved_jobs) — the ONLY mutable data.

# # # Backend is chosen at runtime:
# # #   * DATABASE_URL set  -> PostgreSQL (durable, shared across instances). Use this
# # #     on Render/anywhere the container filesystem is ephemeral.
# # #   * else              -> SQLite file at USER_DB (local dev / single box).

# # # Jobs stay in the read-only SQLite (agent/db.py) — they're immutable and rebuilt
# # # each deploy, so ephemeral storage is fine for them. Only the small, read-write
# # # user tables need a persistent DB, which is what this module isolates.

# # # Portable SQL: both engines support `ON CONFLICT ... DO UPDATE SET x=EXCLUDED.x`.
# # # We only branch on the id placeholder (`?` vs `%s`), the identity column type,
# # # and how a new id is returned.
# # # """
# # # import os
# # # import json
# # # from datetime import datetime, timezone

# # # DATABASE_URL = os.environ.get("DATABASE_URL")
# # # IS_PG = bool(DATABASE_URL)
# # # USER_DB = os.environ.get("USER_DB",
# # #                          os.path.join(os.path.dirname(__file__), "..", "data", "users.db"))


# # # def _connect():
# # #     if IS_PG:
# # #         import psycopg
# # #         return psycopg.connect(DATABASE_URL)
# # #     import sqlite3
# # #     con = sqlite3.connect(USER_DB, check_same_thread=False)
# # #     con.row_factory = sqlite3.Row
# # #     return con


# # # def _ph(sql: str) -> str:
# # #     """SQLite uses ?, Postgres uses %s."""
# # #     return sql.replace("?", "%s") if IS_PG else sql


# # # _ID = ("SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT")

# # # SCHEMA = f"""
# # # CREATE TABLE IF NOT EXISTS users (
# # #   id {_ID}, email TEXT UNIQUE NOT NULL, password_hash TEXT, name TEXT,
# # #   oauth_provider TEXT, created_at TEXT, is_admin INTEGER DEFAULT 0);
# # # CREATE TABLE IF NOT EXISTS profiles (
# # #   user_id INTEGER PRIMARY KEY, location TEXT, exp_years INTEGER, skills TEXT,
# # #   preferred_titles TEXT, preferred_locations TEXT, work_pref TEXT,
# # #   experience_level TEXT,
# # #   llm_provider TEXT, api_model TEXT, encrypted_api_key TEXT, api_base_url TEXT);
# # # CREATE TABLE IF NOT EXISTS saved_jobs (
# # #   user_id INTEGER, job_id TEXT, status TEXT DEFAULT 'saved', saved_at TEXT,
# # #   PRIMARY KEY(user_id, job_id));
# # # """


# # # # Columns added after the original schema — auto-added to existing DBs on boot
# # # # so upgrading never loses accounts and never hits "no such column".
# # # _MIGRATIONS = {
# # #     "users": [("is_admin", "INTEGER DEFAULT 0")],
# # #     "profiles": [("llm_provider", "TEXT"), ("api_model", "TEXT"),
# # #                  ("encrypted_api_key", "TEXT"), ("api_base_url", "TEXT")],
# # # }


# # # def _existing_columns(cur, table):
# # #     if IS_PG:
# # #         cur.execute("SELECT column_name FROM information_schema.columns "
# # #                     "WHERE table_name = %s", (table,))
# # #         return {r[0] for r in cur.fetchall()}
# # #     cur.execute(f"PRAGMA table_info({table})")
# # #     return {r[1] for r in cur.fetchall()}


# # # def _migrate(con):
# # #     cur = con.cursor()
# # #     for table, cols in _MIGRATIONS.items():
# # #         have = _existing_columns(cur, table)
# # #         for name, decl in cols:
# # #             if name not in have:
# # #                 try:
# # #                     cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
# # #                 except Exception:
# # #                     pass   # already added by a concurrent worker, etc.


# # # def init():
# # #     con = _connect()
# # #     try:
# # #         if IS_PG:
# # #             with con.cursor() as cur:
# # #                 cur.execute(SCHEMA)
# # #         else:
# # #             con.executescript(SCHEMA)
# # #         con.commit()
# # #         _migrate(con)          # add any columns missing from an older DB
# # #         con.commit()
# # #     finally:
# # #         con.close()


# # # def _now():
# # #     return datetime.now(timezone.utc).isoformat()


# # # def _rowdict(cur, row):
# # #     if row is None:
# # #         return None
# # #     if IS_PG:
# # #         cols = [c.name for c in cur.description]
# # #         return dict(zip(cols, row))
# # #     return dict(row)


# # # # ── users ──
# # # def create_user(email, password_hash, name="", oauth_provider=None, is_admin=False):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         if IS_PG:
# # #             cur.execute(_ph("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
# # #                             "VALUES (?,?,?,?,?,?) RETURNING id"),
# # #                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
# # #             uid = cur.fetchone()[0]
# # #         else:
# # #             cur.execute("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
# # #                         "VALUES (?,?,?,?,?,?)",
# # #                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
# # #             uid = cur.lastrowid
# # #         con.commit()
# # #         return uid
# # #     finally:
# # #         con.close()


# # # def get_user_by_email(email):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("SELECT id,email,password_hash,name,is_admin FROM users WHERE email=?"),
# # #                     (email.lower(),))
# # #         return _rowdict(cur, cur.fetchone())
# # #     finally:
# # #         con.close()


# # # def get_user(user_id):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("SELECT id,email,name,is_admin FROM users WHERE id=?"), (user_id,))
# # #         return _rowdict(cur, cur.fetchone())
# # #     finally:
# # #         con.close()


# # # # ── profile ──
# # # _PCOLS = ["location", "exp_years", "skills", "preferred_titles",
# # #           "preferred_locations", "work_pref", "experience_level"]


# # # def get_profile(user_id):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("SELECT * FROM profiles WHERE user_id=?"), (user_id,))
# # #         d = _rowdict(cur, cur.fetchone()) or {"user_id": user_id}
# # #     finally:
# # #         con.close()
# # #     for k in ("skills", "preferred_titles", "preferred_locations"):
# # #         if isinstance(d.get(k), str):
# # #             d[k] = json.loads(d[k]) if d[k] else []
# # #     return d


# # # def upsert_profile(user_id, **f):
# # #     vals = {c: (json.dumps(f.get(c)) if isinstance(f.get(c), list) else f.get(c))
# # #             for c in _PCOLS}
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph(
# # #             f"INSERT INTO profiles(user_id,{','.join(_PCOLS)}) "
# # #             f"VALUES (?,{','.join(['?']*len(_PCOLS))}) "
# # #             f"ON CONFLICT(user_id) DO UPDATE SET "
# # #             f"{','.join(f'{c}=EXCLUDED.{c}' for c in _PCOLS)}"),
# # #             [user_id] + [vals[c] for c in _PCOLS])
# # #         con.commit()
# # #     finally:
# # #         con.close()
# # #     return get_profile(user_id)


# # # # ── saved jobs ──
# # # def save_job(user_id, job_id, status="saved"):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("INSERT INTO saved_jobs(user_id,job_id,status,saved_at) VALUES (?,?,?,?) "
# # #                         "ON CONFLICT(user_id,job_id) DO UPDATE SET status=EXCLUDED.status"),
# # #                     (user_id, job_id, status, _now()))
# # #         con.commit()
# # #     finally:
# # #         con.close()


# # # def unsave_job(user_id, job_id):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("DELETE FROM saved_jobs WHERE user_id=? AND job_id=?"), (user_id, job_id))
# # #         con.commit()
# # #     finally:
# # #         con.close()


# # # def saved_job_ids(user_id):
# # #     """Return [(job_id, status, saved_at)] — jobs are joined in db.py (jobs live
# # #     in the read-only SQLite, a different database)."""
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("SELECT job_id,status,saved_at FROM saved_jobs WHERE user_id=? "
# # #                         "ORDER BY saved_at DESC"), (user_id,))
# # #         rows = cur.fetchall()
# # #     finally:
# # #         con.close()
# # #     return [(r[0], r[1], r[2]) for r in rows]


# # # def status_counts(user_id):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("SELECT status, COUNT(*) FROM saved_jobs WHERE user_id=? GROUP BY status"),
# # #                     (user_id,))
# # #         return dict(cur.fetchall())
# # #     finally:
# # #         con.close()


# # # # ── LLM API-key settings (encrypted at rest; ciphertext stored here) ────────
# # # def set_api_key(user_id, provider, model, encrypted, base_url=""):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph(
# # #             "INSERT INTO profiles(user_id,llm_provider,api_model,encrypted_api_key,api_base_url) "
# # #             "VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
# # #             "llm_provider=EXCLUDED.llm_provider, api_model=EXCLUDED.api_model, "
# # #             "encrypted_api_key=EXCLUDED.encrypted_api_key, api_base_url=EXCLUDED.api_base_url"),
# # #             (user_id, provider, model, encrypted, base_url))
# # #         con.commit()
# # #     finally:
# # #         con.close()


# # # def get_api_credentials(user_id):
# # #     """Raw row incl. ciphertext — server-side use only, never returned to client."""
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("SELECT llm_provider,api_model,encrypted_api_key,api_base_url "
# # #                         "FROM profiles WHERE user_id=?"), (user_id,))
# # #         row = cur.fetchone()
# # #     finally:
# # #         con.close()
# # #     if not row:
# # #         return None
# # #     return {"provider": row[0], "model": row[1], "encrypted_api_key": row[2],
# # #             "base_url": row[3]}


# # # def clear_api_key(user_id):
# # #     con = _connect()
# # #     try:
# # #         cur = con.cursor()
# # #         cur.execute(_ph("UPDATE profiles SET llm_provider=NULL, api_model=NULL, "
# # #                         "encrypted_api_key=NULL, api_base_url=NULL WHERE user_id=?"), (user_id,))
# # #         con.commit()
# # #     finally:
# # #         con.close()

# # # """User data (users / profiles / saved_jobs) — the ONLY mutable data.

# # # # Backend is chosen at runtime:
# # # #   * DATABASE_URL set  -> PostgreSQL (durable, shared across instances). Use this
# # # #     on Render/anywhere the container filesystem is ephemeral.
# # # #   * else              -> SQLite file at USER_DB (local dev / single box).

# # # # Jobs stay in the read-only SQLite (agent/db.py) — they're immutable and rebuilt
# # # # each deploy, so ephemeral storage is fine for them. Only the small, read-write
# # # # user tables need a persistent DB, which is what this module isolates.

# # # # Portable SQL: both engines support `ON CONFLICT ... DO UPDATE SET x=EXCLUDED.x`.
# # # # We only branch on the id placeholder (`?` vs `%s`), the identity column type,
# # # # and how a new id is returned.
# # # # """
# # # # import os
# # # # import json
# # # # from datetime import datetime, timezone

# # # # DATABASE_URL = os.environ.get("DATABASE_URL")
# # # # IS_PG = bool(DATABASE_URL)
# # # # USER_DB = os.environ.get("USER_DB",
# # # #                          os.path.join(os.path.dirname(__file__), "..", "data", "users.db"))


# # # # def _connect():
# # # #     if IS_PG:
# # # #         import psycopg
# # # #         return psycopg.connect(DATABASE_URL)
# # # #     import sqlite3
# # # #     con = sqlite3.connect(USER_DB, check_same_thread=False)
# # # #     con.row_factory = sqlite3.Row
# # # #     return con


# # # # def _ph(sql: str) -> str:
# # # #     """SQLite uses ?, Postgres uses %s."""
# # # #     return sql.replace("?", "%s") if IS_PG else sql


# # # # _ID = ("SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT")

# # # # SCHEMA = f"""
# # # # CREATE TABLE IF NOT EXISTS users (
# # # #   id {_ID}, email TEXT UNIQUE NOT NULL, password_hash TEXT, name TEXT,
# # # #   oauth_provider TEXT, created_at TEXT, is_admin INTEGER DEFAULT 0);
# # # # CREATE TABLE IF NOT EXISTS profiles (
# # # #   user_id INTEGER PRIMARY KEY, location TEXT, exp_years INTEGER, skills TEXT,
# # # #   preferred_titles TEXT, preferred_locations TEXT, work_pref TEXT,
# # # #   experience_level TEXT,
# # # #   llm_provider TEXT, api_model TEXT, encrypted_api_key TEXT, api_base_url TEXT);
# # # # CREATE TABLE IF NOT EXISTS saved_jobs (
# # # #   user_id INTEGER, job_id TEXT, status TEXT DEFAULT 'saved', saved_at TEXT,
# # # #   PRIMARY KEY(user_id, job_id));
# # # # """


# # # # def init():
# # # #     con = _connect()
# # # #     try:
# # # #         if IS_PG:
# # # #             with con.cursor() as cur:
# # # #                 cur.execute(SCHEMA)
# # # #         else:
# # # #             con.executescript(SCHEMA)
# # # #         con.commit()
# # # #     finally:
# # # #         con.close()


# # # # def _now():
# # # #     return datetime.now(timezone.utc).isoformat()


# # # # def _rowdict(cur, row):
# # # #     if row is None:
# # # #         return None
# # # #     if IS_PG:
# # # #         cols = [c.name for c in cur.description]
# # # #         return dict(zip(cols, row))
# # # #     return dict(row)


# # # # # ── users ──
# # # # def create_user(email, password_hash, name="", oauth_provider=None, is_admin=False):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         if IS_PG:
# # # #             cur.execute(_ph("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
# # # #                             "VALUES (?,?,?,?,?,?) RETURNING id"),
# # # #                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
# # # #             uid = cur.fetchone()[0]
# # # #         else:
# # # #             cur.execute("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
# # # #                         "VALUES (?,?,?,?,?,?)",
# # # #                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
# # # #             uid = cur.lastrowid
# # # #         con.commit()
# # # #         return uid
# # # #     finally:
# # # #         con.close()


# # # # def get_user_by_email(email):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("SELECT id,email,password_hash,name,is_admin FROM users WHERE email=?"),
# # # #                     (email.lower(),))
# # # #         return _rowdict(cur, cur.fetchone())
# # # #     finally:
# # # #         con.close()


# # # # def get_user(user_id):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("SELECT id,email,name,is_admin FROM users WHERE id=?"), (user_id,))
# # # #         return _rowdict(cur, cur.fetchone())
# # # #     finally:
# # # #         con.close()


# # # # # ── profile ──
# # # # _PCOLS = ["location", "exp_years", "skills", "preferred_titles",
# # # #           "preferred_locations", "work_pref", "experience_level"]


# # # # def get_profile(user_id):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("SELECT * FROM profiles WHERE user_id=?"), (user_id,))
# # # #         d = _rowdict(cur, cur.fetchone()) or {"user_id": user_id}
# # # #     finally:
# # # #         con.close()
# # # #     for k in ("skills", "preferred_titles", "preferred_locations"):
# # # #         if isinstance(d.get(k), str):
# # # #             d[k] = json.loads(d[k]) if d[k] else []
# # # #     return d


# # # # def upsert_profile(user_id, **f):
# # # #     vals = {c: (json.dumps(f.get(c)) if isinstance(f.get(c), list) else f.get(c))
# # # #             for c in _PCOLS}
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph(
# # # #             f"INSERT INTO profiles(user_id,{','.join(_PCOLS)}) "
# # # #             f"VALUES (?,{','.join(['?']*len(_PCOLS))}) "
# # # #             f"ON CONFLICT(user_id) DO UPDATE SET "
# # # #             f"{','.join(f'{c}=EXCLUDED.{c}' for c in _PCOLS)}"),
# # # #             [user_id] + [vals[c] for c in _PCOLS])
# # # #         con.commit()
# # # #     finally:
# # # #         con.close()
# # # #     return get_profile(user_id)


# # # # # ── saved jobs ──
# # # # def save_job(user_id, job_id, status="saved"):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("INSERT INTO saved_jobs(user_id,job_id,status,saved_at) VALUES (?,?,?,?) "
# # # #                         "ON CONFLICT(user_id,job_id) DO UPDATE SET status=EXCLUDED.status"),
# # # #                     (user_id, job_id, status, _now()))
# # # #         con.commit()
# # # #     finally:
# # # #         con.close()


# # # # def unsave_job(user_id, job_id):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("DELETE FROM saved_jobs WHERE user_id=? AND job_id=?"), (user_id, job_id))
# # # #         con.commit()
# # # #     finally:
# # # #         con.close()


# # # # def saved_job_ids(user_id):
# # # #     """Return [(job_id, status, saved_at)] — jobs are joined in db.py (jobs live
# # # #     in the read-only SQLite, a different database)."""
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("SELECT job_id,status,saved_at FROM saved_jobs WHERE user_id=? "
# # # #                         "ORDER BY saved_at DESC"), (user_id,))
# # # #         rows = cur.fetchall()
# # # #     finally:
# # # #         con.close()
# # # #     return [(r[0], r[1], r[2]) for r in rows]


# # # # def status_counts(user_id):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("SELECT status, COUNT(*) FROM saved_jobs WHERE user_id=? GROUP BY status"),
# # # #                     (user_id,))
# # # #         return dict(cur.fetchall())
# # # #     finally:
# # # #         con.close()


# # # # # ── LLM API-key settings (encrypted at rest; ciphertext stored here) ────────
# # # # def set_api_key(user_id, provider, model, encrypted, base_url=""):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph(
# # # #             "INSERT INTO profiles(user_id,llm_provider,api_model,encrypted_api_key,api_base_url) "
# # # #             "VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
# # # #             "llm_provider=EXCLUDED.llm_provider, api_model=EXCLUDED.api_model, "
# # # #             "encrypted_api_key=EXCLUDED.encrypted_api_key, api_base_url=EXCLUDED.api_base_url"),
# # # #             (user_id, provider, model, encrypted, base_url))
# # # #         con.commit()
# # # #     finally:
# # # #         con.close()


# # # # def get_api_credentials(user_id):
# # # #     """Raw row incl. ciphertext — server-side use only, never returned to client."""
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("SELECT llm_provider,api_model,encrypted_api_key,api_base_url "
# # # #                         "FROM profiles WHERE user_id=?"), (user_id,))
# # # #         row = cur.fetchone()
# # # #     finally:
# # # #         con.close()
# # # #     if not row:
# # # #         return None
# # # #     return {"provider": row[0], "model": row[1], "encrypted_api_key": row[2],
# # # #             "base_url": row[3]}


# # # # def clear_api_key(user_id):
# # # #     con = _connect()
# # # #     try:
# # # #         cur = con.cursor()
# # # #         cur.execute(_ph("UPDATE profiles SET llm_provider=NULL, api_model=NULL, "
# # # #                         "encrypted_api_key=NULL, api_base_url=NULL WHERE user_id=?"), (user_id,))
# # # #         con.commit()
# # # #     finally:
# # # #         con.close()
# # # # # """User data (users / profiles / saved_jobs) — the ONLY mutable data.

# # # # # Backend is chosen at runtime:
# # # # #   * DATABASE_URL set  -> PostgreSQL (durable, shared across instances). Use this
# # # # #     on Render/anywhere the container filesystem is ephemeral.
# # # # #   * else              -> SQLite file at USER_DB (local dev / single box).

# # # # # Jobs stay in the read-only SQLite (agent/db.py) — they're immutable and rebuilt
# # # # # each deploy, so ephemeral storage is fine for them. Only the small, read-write
# # # # # user tables need a persistent DB, which is what this module isolates.

# # # # # Portable SQL: both engines support `ON CONFLICT ... DO UPDATE SET x=EXCLUDED.x`.
# # # # # We only branch on the id placeholder (`?` vs `%s`), the identity column type,
# # # # # and how a new id is returned.
# # # # # """
# # # # # import os
# # # # # import json
# # # # # from datetime import datetime, timezone

# # # # # DATABASE_URL = os.environ.get("DATABASE_URL")
# # # # # IS_PG = bool(DATABASE_URL)
# # # # # USER_DB = os.environ.get("USER_DB",
# # # # #                          os.path.join(os.path.dirname(__file__), "..", "data", "users.db"))


# # # # # def _connect():
# # # # #     if IS_PG:
# # # # #         import psycopg
# # # # #         return psycopg.connect(DATABASE_URL)
# # # # #     import sqlite3
# # # # #     con = sqlite3.connect(USER_DB, check_same_thread=False)
# # # # #     con.row_factory = sqlite3.Row
# # # # #     return con


# # # # # def _ph(sql: str) -> str:
# # # # #     """SQLite uses ?, Postgres uses %s."""
# # # # #     return sql.replace("?", "%s") if IS_PG else sql


# # # # # _ID = ("SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT")

# # # # # SCHEMA = f"""
# # # # # CREATE TABLE IF NOT EXISTS users (
# # # # #   id {_ID}, email TEXT UNIQUE NOT NULL, password_hash TEXT, name TEXT,
# # # # #   oauth_provider TEXT, created_at TEXT, is_admin INTEGER DEFAULT 0);
# # # # # CREATE TABLE IF NOT EXISTS profiles (
# # # # #   user_id INTEGER PRIMARY KEY, location TEXT, exp_years INTEGER, skills TEXT,
# # # # #   preferred_titles TEXT, preferred_locations TEXT, work_pref TEXT,
# # # # #   experience_level TEXT);
# # # # # CREATE TABLE IF NOT EXISTS saved_jobs (
# # # # #   user_id INTEGER, job_id TEXT, status TEXT DEFAULT 'saved', saved_at TEXT,
# # # # #   PRIMARY KEY(user_id, job_id));
# # # # # """


# # # # # def init():
# # # # #     con = _connect()
# # # # #     try:
# # # # #         if IS_PG:
# # # # #             with con.cursor() as cur:
# # # # #                 cur.execute(SCHEMA)
# # # # #         else:
# # # # #             con.executescript(SCHEMA)
# # # # #         con.commit()
# # # # #     finally:
# # # # #         con.close()


# # # # # def _now():
# # # # #     return datetime.now(timezone.utc).isoformat()


# # # # # def _rowdict(cur, row):
# # # # #     if row is None:
# # # # #         return None
# # # # #     if IS_PG:
# # # # #         cols = [c.name for c in cur.description]
# # # # #         return dict(zip(cols, row))
# # # # #     return dict(row)


# # # # # # ── users ──
# # # # # def create_user(email, password_hash, name="", oauth_provider=None, is_admin=False):
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         if IS_PG:
# # # # #             cur.execute(_ph("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
# # # # #                             "VALUES (?,?,?,?,?,?) RETURNING id"),
# # # # #                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
# # # # #             uid = cur.fetchone()[0]
# # # # #         else:
# # # # #             cur.execute("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
# # # # #                         "VALUES (?,?,?,?,?,?)",
# # # # #                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
# # # # #             uid = cur.lastrowid
# # # # #         con.commit()
# # # # #         return uid
# # # # #     finally:
# # # # #         con.close()


# # # # # def get_user_by_email(email):
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         cur.execute(_ph("SELECT id,email,password_hash,name,is_admin FROM users WHERE email=?"),
# # # # #                     (email.lower(),))
# # # # #         return _rowdict(cur, cur.fetchone())
# # # # #     finally:
# # # # #         con.close()


# # # # # def get_user(user_id):
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         cur.execute(_ph("SELECT id,email,name,is_admin FROM users WHERE id=?"), (user_id,))
# # # # #         return _rowdict(cur, cur.fetchone())
# # # # #     finally:
# # # # #         con.close()


# # # # # # ── profile ──
# # # # # _PCOLS = ["location", "exp_years", "skills", "preferred_titles",
# # # # #           "preferred_locations", "work_pref", "experience_level"]


# # # # # def get_profile(user_id):
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         cur.execute(_ph("SELECT * FROM profiles WHERE user_id=?"), (user_id,))
# # # # #         d = _rowdict(cur, cur.fetchone()) or {"user_id": user_id}
# # # # #     finally:
# # # # #         con.close()
# # # # #     for k in ("skills", "preferred_titles", "preferred_locations"):
# # # # #         if isinstance(d.get(k), str):
# # # # #             d[k] = json.loads(d[k]) if d[k] else []
# # # # #     return d


# # # # # def upsert_profile(user_id, **f):
# # # # #     vals = {c: (json.dumps(f.get(c)) if isinstance(f.get(c), list) else f.get(c))
# # # # #             for c in _PCOLS}
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         cur.execute(_ph(
# # # # #             f"INSERT INTO profiles(user_id,{','.join(_PCOLS)}) "
# # # # #             f"VALUES (?,{','.join(['?']*len(_PCOLS))}) "
# # # # #             f"ON CONFLICT(user_id) DO UPDATE SET "
# # # # #             f"{','.join(f'{c}=EXCLUDED.{c}' for c in _PCOLS)}"),
# # # # #             [user_id] + [vals[c] for c in _PCOLS])
# # # # #         con.commit()
# # # # #     finally:
# # # # #         con.close()
# # # # #     return get_profile(user_id)


# # # # # # ── saved jobs ──
# # # # # def save_job(user_id, job_id, status="saved"):
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         cur.execute(_ph("INSERT INTO saved_jobs(user_id,job_id,status,saved_at) VALUES (?,?,?,?) "
# # # # #                         "ON CONFLICT(user_id,job_id) DO UPDATE SET status=EXCLUDED.status"),
# # # # #                     (user_id, job_id, status, _now()))
# # # # #         con.commit()
# # # # #     finally:
# # # # #         con.close()


# # # # # def unsave_job(user_id, job_id):
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         cur.execute(_ph("DELETE FROM saved_jobs WHERE user_id=? AND job_id=?"), (user_id, job_id))
# # # # #         con.commit()
# # # # #     finally:
# # # # #         con.close()


# # # # # def saved_job_ids(user_id):
# # # # #     """Return [(job_id, status, saved_at)] — jobs are joined in db.py (jobs live
# # # # #     in the read-only SQLite, a different database)."""
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         cur.execute(_ph("SELECT job_id,status,saved_at FROM saved_jobs WHERE user_id=? "
# # # # #                         "ORDER BY saved_at DESC"), (user_id,))
# # # # #         rows = cur.fetchall()
# # # # #     finally:
# # # # #         con.close()
# # # # #     return [(r[0], r[1], r[2]) for r in rows]


# # # # # def status_counts(user_id):
# # # # #     con = _connect()
# # # # #     try:
# # # # #         cur = con.cursor()
# # # # #         cur.execute(_ph("SELECT status, COUNT(*) FROM saved_jobs WHERE user_id=? GROUP BY status"),
# # # # #                     (user_id,))
# # # # #         return dict(cur.fetchall())
# # # # #     finally:
# # # # #         con.close()
