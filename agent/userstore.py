"""User data (users / profiles / saved_jobs) — the ONLY mutable data.

Backend is chosen at runtime:
  * DATABASE_URL set  -> PostgreSQL (durable, shared across instances). Use this
    on Render/anywhere the container filesystem is ephemeral.
  * else              -> SQLite file at USER_DB (local dev / single box).

Jobs stay in the read-only SQLite (agent/db.py) — they're immutable and rebuilt
each deploy, so ephemeral storage is fine for them. Only the small, read-write
user tables need a persistent DB, which is what this module isolates.

Portable SQL: both engines support `ON CONFLICT ... DO UPDATE SET x=EXCLUDED.x`.
We only branch on the id placeholder (`?` vs `%s`), the identity column type,
and how a new id is returned.
"""
import os
import json
from datetime import datetime, timezone

DATABASE_URL = os.environ.get("DATABASE_URL")
IS_PG = bool(DATABASE_URL)
USER_DB = os.environ.get("USER_DB",
                         os.path.join(os.path.dirname(__file__), "..", "data", "users.db"))


def _connect():
    if IS_PG:
        import psycopg
        return psycopg.connect(DATABASE_URL)
    import sqlite3
    con = sqlite3.connect(USER_DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _ph(sql: str) -> str:
    """SQLite uses ?, Postgres uses %s."""
    return sql.replace("?", "%s") if IS_PG else sql


_ID = ("SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
  id {_ID}, email TEXT UNIQUE NOT NULL, password_hash TEXT, name TEXT,
  oauth_provider TEXT, created_at TEXT, is_admin INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS profiles (
  user_id INTEGER PRIMARY KEY, location TEXT, exp_years INTEGER, skills TEXT,
  preferred_titles TEXT, preferred_locations TEXT, work_pref TEXT,
  experience_level TEXT,
  llm_provider TEXT, api_model TEXT, encrypted_api_key TEXT, api_base_url TEXT);
CREATE TABLE IF NOT EXISTS saved_jobs (
  user_id INTEGER, job_id TEXT, status TEXT DEFAULT 'saved', saved_at TEXT,
  PRIMARY KEY(user_id, job_id));
"""


# Columns added after the original schema — auto-added to existing DBs on boot
# so upgrading never loses accounts and never hits "no such column".
_MIGRATIONS = {
    "users": [("is_admin", "INTEGER DEFAULT 0")],
    "profiles": [("llm_provider", "TEXT"), ("api_model", "TEXT"),
                 ("encrypted_api_key", "TEXT"), ("api_base_url", "TEXT")],
}


def _existing_columns(cur, table):
    if IS_PG:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s", (table,))
        return {r[0] for r in cur.fetchall()}
    cur.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cur.fetchall()}


def _migrate(con):
    cur = con.cursor()
    for table, cols in _MIGRATIONS.items():
        have = _existing_columns(cur, table)
        for name, decl in cols:
            if name not in have:
                try:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                except Exception:
                    pass   # already added by a concurrent worker, etc.


def init():
    con = _connect()
    try:
        if IS_PG:
            with con.cursor() as cur:
                cur.execute(SCHEMA)
        else:
            con.executescript(SCHEMA)
        con.commit()
        _migrate(con)          # add any columns missing from an older DB
        con.commit()
    finally:
        con.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _rowdict(cur, row):
    if row is None:
        return None
    if IS_PG:
        cols = [c.name for c in cur.description]
        return dict(zip(cols, row))
    return dict(row)


# ── users ──
def create_user(email, password_hash, name="", oauth_provider=None, is_admin=False):
    con = _connect()
    try:
        cur = con.cursor()
        if IS_PG:
            cur.execute(_ph("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
                            "VALUES (?,?,?,?,?,?) RETURNING id"),
                        (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
            uid = cur.fetchone()[0]
        else:
            cur.execute("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
                        "VALUES (?,?,?,?,?,?)",
                        (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
            uid = cur.lastrowid
        con.commit()
        return uid
    finally:
        con.close()


def get_user_by_email(email):
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("SELECT id,email,password_hash,name,is_admin FROM users WHERE email=?"),
                    (email.lower(),))
        return _rowdict(cur, cur.fetchone())
    finally:
        con.close()


def get_user(user_id):
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("SELECT id,email,name,is_admin FROM users WHERE id=?"), (user_id,))
        return _rowdict(cur, cur.fetchone())
    finally:
        con.close()


# ── profile ──
_PCOLS = ["location", "exp_years", "skills", "preferred_titles",
          "preferred_locations", "work_pref", "experience_level"]


def get_profile(user_id):
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("SELECT * FROM profiles WHERE user_id=?"), (user_id,))
        d = _rowdict(cur, cur.fetchone()) or {"user_id": user_id}
    finally:
        con.close()
    for k in ("skills", "preferred_titles", "preferred_locations"):
        if isinstance(d.get(k), str):
            d[k] = json.loads(d[k]) if d[k] else []
    return d


def upsert_profile(user_id, **f):
    vals = {c: (json.dumps(f.get(c)) if isinstance(f.get(c), list) else f.get(c))
            for c in _PCOLS}
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph(
            f"INSERT INTO profiles(user_id,{','.join(_PCOLS)}) "
            f"VALUES (?,{','.join(['?']*len(_PCOLS))}) "
            f"ON CONFLICT(user_id) DO UPDATE SET "
            f"{','.join(f'{c}=EXCLUDED.{c}' for c in _PCOLS)}"),
            [user_id] + [vals[c] for c in _PCOLS])
        con.commit()
    finally:
        con.close()
    return get_profile(user_id)


# ── saved jobs ──
def save_job(user_id, job_id, status="saved"):
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("INSERT INTO saved_jobs(user_id,job_id,status,saved_at) VALUES (?,?,?,?) "
                        "ON CONFLICT(user_id,job_id) DO UPDATE SET status=EXCLUDED.status"),
                    (user_id, job_id, status, _now()))
        con.commit()
    finally:
        con.close()


def unsave_job(user_id, job_id):
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("DELETE FROM saved_jobs WHERE user_id=? AND job_id=?"), (user_id, job_id))
        con.commit()
    finally:
        con.close()


def saved_job_ids(user_id):
    """Return [(job_id, status, saved_at)] — jobs are joined in db.py (jobs live
    in the read-only SQLite, a different database)."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("SELECT job_id,status,saved_at FROM saved_jobs WHERE user_id=? "
                        "ORDER BY saved_at DESC"), (user_id,))
        rows = cur.fetchall()
    finally:
        con.close()
    return [(r[0], r[1], r[2]) for r in rows]


def status_counts(user_id):
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("SELECT status, COUNT(*) FROM saved_jobs WHERE user_id=? GROUP BY status"),
                    (user_id,))
        return dict(cur.fetchall())
    finally:
        con.close()


# ── LLM API-key settings (encrypted at rest; ciphertext stored here) ────────
def set_api_key(user_id, provider, model, encrypted, base_url=""):
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph(
            "INSERT INTO profiles(user_id,llm_provider,api_model,encrypted_api_key,api_base_url) "
            "VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
            "llm_provider=EXCLUDED.llm_provider, api_model=EXCLUDED.api_model, "
            "encrypted_api_key=EXCLUDED.encrypted_api_key, api_base_url=EXCLUDED.api_base_url"),
            (user_id, provider, model, encrypted, base_url))
        con.commit()
    finally:
        con.close()


def get_api_credentials(user_id):
    """Raw row incl. ciphertext — server-side use only, never returned to client."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("SELECT llm_provider,api_model,encrypted_api_key,api_base_url "
                        "FROM profiles WHERE user_id=?"), (user_id,))
        row = cur.fetchone()
    finally:
        con.close()
    if not row:
        return None
    return {"provider": row[0], "model": row[1], "encrypted_api_key": row[2],
            "base_url": row[3]}


def clear_api_key(user_id):
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(_ph("UPDATE profiles SET llm_provider=NULL, api_model=NULL, "
                        "encrypted_api_key=NULL, api_base_url=NULL WHERE user_id=?"), (user_id,))
        con.commit()
    finally:
        con.close()

"""User data (users / profiles / saved_jobs) — the ONLY mutable data.

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


# def init():
#     con = _connect()
#     try:
#         if IS_PG:
#             with con.cursor() as cur:
#                 cur.execute(SCHEMA)
#         else:
#             con.executescript(SCHEMA)
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
# # """User data (users / profiles / saved_jobs) — the ONLY mutable data.

# # Backend is chosen at runtime:
# #   * DATABASE_URL set  -> PostgreSQL (durable, shared across instances). Use this
# #     on Render/anywhere the container filesystem is ephemeral.
# #   * else              -> SQLite file at USER_DB (local dev / single box).

# # Jobs stay in the read-only SQLite (agent/db.py) — they're immutable and rebuilt
# # each deploy, so ephemeral storage is fine for them. Only the small, read-write
# # user tables need a persistent DB, which is what this module isolates.

# # Portable SQL: both engines support `ON CONFLICT ... DO UPDATE SET x=EXCLUDED.x`.
# # We only branch on the id placeholder (`?` vs `%s`), the identity column type,
# # and how a new id is returned.
# # """
# # import os
# # import json
# # from datetime import datetime, timezone

# # DATABASE_URL = os.environ.get("DATABASE_URL")
# # IS_PG = bool(DATABASE_URL)
# # USER_DB = os.environ.get("USER_DB",
# #                          os.path.join(os.path.dirname(__file__), "..", "data", "users.db"))


# # def _connect():
# #     if IS_PG:
# #         import psycopg
# #         return psycopg.connect(DATABASE_URL)
# #     import sqlite3
# #     con = sqlite3.connect(USER_DB, check_same_thread=False)
# #     con.row_factory = sqlite3.Row
# #     return con


# # def _ph(sql: str) -> str:
# #     """SQLite uses ?, Postgres uses %s."""
# #     return sql.replace("?", "%s") if IS_PG else sql


# # _ID = ("SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT")

# # SCHEMA = f"""
# # CREATE TABLE IF NOT EXISTS users (
# #   id {_ID}, email TEXT UNIQUE NOT NULL, password_hash TEXT, name TEXT,
# #   oauth_provider TEXT, created_at TEXT, is_admin INTEGER DEFAULT 0);
# # CREATE TABLE IF NOT EXISTS profiles (
# #   user_id INTEGER PRIMARY KEY, location TEXT, exp_years INTEGER, skills TEXT,
# #   preferred_titles TEXT, preferred_locations TEXT, work_pref TEXT,
# #   experience_level TEXT);
# # CREATE TABLE IF NOT EXISTS saved_jobs (
# #   user_id INTEGER, job_id TEXT, status TEXT DEFAULT 'saved', saved_at TEXT,
# #   PRIMARY KEY(user_id, job_id));
# # """


# # def init():
# #     con = _connect()
# #     try:
# #         if IS_PG:
# #             with con.cursor() as cur:
# #                 cur.execute(SCHEMA)
# #         else:
# #             con.executescript(SCHEMA)
# #         con.commit()
# #     finally:
# #         con.close()


# # def _now():
# #     return datetime.now(timezone.utc).isoformat()


# # def _rowdict(cur, row):
# #     if row is None:
# #         return None
# #     if IS_PG:
# #         cols = [c.name for c in cur.description]
# #         return dict(zip(cols, row))
# #     return dict(row)


# # # ── users ──
# # def create_user(email, password_hash, name="", oauth_provider=None, is_admin=False):
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         if IS_PG:
# #             cur.execute(_ph("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
# #                             "VALUES (?,?,?,?,?,?) RETURNING id"),
# #                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
# #             uid = cur.fetchone()[0]
# #         else:
# #             cur.execute("INSERT INTO users(email,password_hash,name,oauth_provider,created_at,is_admin) "
# #                         "VALUES (?,?,?,?,?,?)",
# #                         (email.lower(), password_hash, name, oauth_provider, _now(), int(is_admin)))
# #             uid = cur.lastrowid
# #         con.commit()
# #         return uid
# #     finally:
# #         con.close()


# # def get_user_by_email(email):
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         cur.execute(_ph("SELECT id,email,password_hash,name,is_admin FROM users WHERE email=?"),
# #                     (email.lower(),))
# #         return _rowdict(cur, cur.fetchone())
# #     finally:
# #         con.close()


# # def get_user(user_id):
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         cur.execute(_ph("SELECT id,email,name,is_admin FROM users WHERE id=?"), (user_id,))
# #         return _rowdict(cur, cur.fetchone())
# #     finally:
# #         con.close()


# # # ── profile ──
# # _PCOLS = ["location", "exp_years", "skills", "preferred_titles",
# #           "preferred_locations", "work_pref", "experience_level"]


# # def get_profile(user_id):
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         cur.execute(_ph("SELECT * FROM profiles WHERE user_id=?"), (user_id,))
# #         d = _rowdict(cur, cur.fetchone()) or {"user_id": user_id}
# #     finally:
# #         con.close()
# #     for k in ("skills", "preferred_titles", "preferred_locations"):
# #         if isinstance(d.get(k), str):
# #             d[k] = json.loads(d[k]) if d[k] else []
# #     return d


# # def upsert_profile(user_id, **f):
# #     vals = {c: (json.dumps(f.get(c)) if isinstance(f.get(c), list) else f.get(c))
# #             for c in _PCOLS}
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         cur.execute(_ph(
# #             f"INSERT INTO profiles(user_id,{','.join(_PCOLS)}) "
# #             f"VALUES (?,{','.join(['?']*len(_PCOLS))}) "
# #             f"ON CONFLICT(user_id) DO UPDATE SET "
# #             f"{','.join(f'{c}=EXCLUDED.{c}' for c in _PCOLS)}"),
# #             [user_id] + [vals[c] for c in _PCOLS])
# #         con.commit()
# #     finally:
# #         con.close()
# #     return get_profile(user_id)


# # # ── saved jobs ──
# # def save_job(user_id, job_id, status="saved"):
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         cur.execute(_ph("INSERT INTO saved_jobs(user_id,job_id,status,saved_at) VALUES (?,?,?,?) "
# #                         "ON CONFLICT(user_id,job_id) DO UPDATE SET status=EXCLUDED.status"),
# #                     (user_id, job_id, status, _now()))
# #         con.commit()
# #     finally:
# #         con.close()


# # def unsave_job(user_id, job_id):
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         cur.execute(_ph("DELETE FROM saved_jobs WHERE user_id=? AND job_id=?"), (user_id, job_id))
# #         con.commit()
# #     finally:
# #         con.close()


# # def saved_job_ids(user_id):
# #     """Return [(job_id, status, saved_at)] — jobs are joined in db.py (jobs live
# #     in the read-only SQLite, a different database)."""
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         cur.execute(_ph("SELECT job_id,status,saved_at FROM saved_jobs WHERE user_id=? "
# #                         "ORDER BY saved_at DESC"), (user_id,))
# #         rows = cur.fetchall()
# #     finally:
# #         con.close()
# #     return [(r[0], r[1], r[2]) for r in rows]


# # def status_counts(user_id):
# #     con = _connect()
# #     try:
# #         cur = con.cursor()
# #         cur.execute(_ph("SELECT status, COUNT(*) FROM saved_jobs WHERE user_id=? GROUP BY status"),
# #                     (user_id,))
# #         return dict(cur.fetchall())
# #     finally:
# #         con.close()
