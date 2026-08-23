"""Build the app SQLite DB from the raw job dump.

    python build_db.py --source /path/to/query_result.json --out app.db
    python build_db.py --source ... --limit 4000 --out app_sample.db   # small demo db

Creates: jobs (with description + source + remote + posted_month) and the auth
tables (users, profiles, saved_jobs). stdlib only (ijson optional for streaming).
"""
import argparse
import json
import re
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, title TEXT, company TEXT, location TEXT, domain TEXT,
  source TEXT, emp TEXT, schedule_type TEXT, min_exp INTEGER, max_exp INTEGER,
  remote INTEGER, posted_month TEXT, skills TEXT, description TEXT, apply_link TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_source ON jobs(source);
CREATE INDEX IF NOT EXISTS ix_jobs_domain ON jobs(domain);
CREATE INDEX IF NOT EXISTS ix_jobs_remote ON jobs(remote);
CREATE INDEX IF NOT EXISTS ix_jobs_month  ON jobs(posted_month);

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
  password_hash TEXT, name TEXT, oauth_provider TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS profiles (
  user_id INTEGER PRIMARY KEY, location TEXT, exp_years INTEGER, skills TEXT,
  preferred_titles TEXT, preferred_locations TEXT, work_pref TEXT,
  experience_level TEXT,
  FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS saved_jobs (
  user_id INTEGER, job_id TEXT, status TEXT DEFAULT 'saved', saved_at TEXT,
  PRIMARY KEY(user_id, job_id)
);
"""


def clean(x):
    s = str(x or "").strip()
    return "" if s.lower() in ("none", "null", "") else s


def num(x):
    try:
        return int(float(clean(x)))
    except Exception:
        return None


def parse_month(*vals):
    for v in vals:
        m = re.match(r"(\d{4})/(\d{1,2})", str(v or ""))
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}"
    return ""


def is_remote(loc, lr, desc):
    blob = f"{loc} {lr} {desc[:400]}".lower()
    return int("remote" in blob or "anywhere" in blob or "work from home" in blob)


def apply_link(apply_options):
    try:
        opts = json.loads(clean(apply_options).replace('\\"', '"'))
        if isinstance(opts, list) and opts:
            return opts[0].get("link", "")
    except Exception:
        pass
    return ""


def rows(source, limit):
    try:
        import ijson
        with open(source, "rb") as f:
            for i, r in enumerate(ijson.items(f, "item")):
                if limit and i >= limit:
                    break
                yield r
    except ImportError:
        data = json.load(open(source))
        for i, r in enumerate(data):
            if limit and i >= limit:
                break
            yield r


def build(source, out, limit, desc_cap):
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)
    n = 0
    for r in rows(source, limit):
        desc = clean(r.get("description"))
        con.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.get("job_id"), clean(r.get("title")), clean(r.get("company_name")),
             clean(r.get("location")), clean(r.get("domain")),
             clean(r.get("via")).replace("via ", "").strip() or "Unknown",
             clean(r.get("employmentType")), clean(r.get("schedule_type")),
             num(r.get("minExperienceRequired")), num(r.get("maxExperienceRequired")),
             is_remote(clean(r.get("location")), clean(r.get("locationRequirement")), desc),
             parse_month(r.get("posted_at"), r.get("createdAt"), r.get("publishedAt")),
             clean(r.get("skills")), desc[:desc_cap], apply_link(r.get("apply_options"))))
        n += 1
        if n % 5000 == 0:
            con.commit()
    con.commit()
    con.close()
    print(f"built {out} with {n} jobs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="app.db")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--desc-cap", type=int, default=6000)
    a = ap.parse_args()
    build(a.source, a.out, a.limit, a.desc_cap)
