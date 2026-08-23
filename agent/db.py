"""SQLite data access — stdlib sqlite3 only (keeps the app container light).

Covers: filtered job search, dropdown facets, market-intelligence aggregates,
single job with full description, and the auth/profile/saved-jobs/dashboard data.
"""
import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from .trace import traceable
from . import userstore

DB_PATH = os.environ.get(
    "APP_DB", os.path.join(os.path.dirname(__file__), "..", "data", "app_sample.db"))


def _con():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _skills_list(s):
    return [t.strip() for t in re.split(r"[,;|]", s or "") if t.strip()]


# ── jobs ───────────────────────────────────────────────────────────────────
def query_jobs(source=None, location=None, domain=None, remote=None,
               min_exp=None, max_exp=None, skills=None, q=None, limit=50, offset=0):
    where, args = [], []
    if source and source != "All Sources":
        where.append("source = ?"); args.append(source)
    if location:
        where.append("location LIKE ?"); args.append(f"%{location}%")
    if domain:
        where.append("domain = ?"); args.append(domain)
    if remote is not None:
        where.append("remote = ?"); args.append(1 if remote else 0)
    if min_exp is not None:
        where.append("(max_exp IS NULL OR max_exp >= ?)"); args.append(min_exp)
    if max_exp is not None:
        where.append("(min_exp IS NULL OR min_exp <= ?)"); args.append(max_exp)
    if q:
        where.append("(title LIKE ? OR company LIKE ?)"); args += [f"%{q}%", f"%{q}%"]
    for sk in (skills or []):
        where.append("skills LIKE ?"); args.append(f"%{sk}%")
    sql = "SELECT id,title,company,location,domain,source,emp,min_exp,max_exp,remote,skills FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " LIMIT ? OFFSET ?"; args += [limit, offset]
    con = _con()
    try:
        rows = [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()
    for r in rows:
        r["skills"] = _skills_list(r["skills"])
    return rows


def get_job(job_id):
    con = _con()
    try:
        r = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        con.close()
    if not r:
        return None
    d = dict(r)
    d["skills"] = _skills_list(d["skills"])
    return d


def facets():
    con = _con()
    try:
        src = [r[0] for r in con.execute(
            "SELECT source FROM jobs GROUP BY source ORDER BY COUNT(*) DESC LIMIT 12")]
        dom = [r[0] for r in con.execute(
            "SELECT domain FROM jobs GROUP BY domain ORDER BY COUNT(*) DESC")]
    finally:
        con.close()
    return {"sources": ["All Sources"] + src, "domains": dom}


def market_intel(domain=None, month=None):
    """Aggregates for the market-intelligence dashboard (grounded in the data)."""
    con = _con()
    try:
        base = "FROM jobs" + (" WHERE domain = ?" if domain else "")
        a = [domain] if domain else []
        total = con.execute("SELECT COUNT(*) " + base, a).fetchone()[0]
        companies = [dict(name=r[0], jobs=r[1]) for r in con.execute(
            "SELECT company, COUNT(*) c " + base + " GROUP BY company "
            "ORDER BY c DESC LIMIT 10", a) if r[0]]
        locations = [dict(name=r[0], jobs=r[1]) for r in con.execute(
            "SELECT location, COUNT(*) c " + base + " GROUP BY location "
            "ORDER BY c DESC LIMIT 10", a) if r[0]]
        rem = con.execute("SELECT SUM(remote), COUNT(*) " + base, a).fetchone()
        # skill frequency (overall) and month-over-month trend
        skill_rows = con.execute("SELECT skills, posted_month " + base, a).fetchall()
    finally:
        con.close()

    from collections import Counter, defaultdict
    freq = Counter(); by_month = defaultdict(Counter)
    for sk, m in skill_rows:
        for s in _skills_list(sk):
            s = s.lower(); freq[s] += 1
            if m:
                by_month[m][s] += 1
    top_skills = [dict(skill=s, count=c) for s, c in freq.most_common(15)]

    trend = []
    months = sorted(by_month)
    if len(months) >= 2:
        cur, prev = by_month[months[-1]], by_month[months[-2]]
        for s, c in cur.most_common(30):
            p = prev.get(s, 0)
            if p >= 3:
                pct = round((c - p) / p * 100)
                trend.append(dict(skill=s, change_pct=pct, month=months[-1]))
        trend = sorted(trend, key=lambda x: -x["change_pct"])[:10]

    return {
        "total_jobs": total,
        "top_skills": top_skills,
        "top_companies": companies,
        "top_locations": locations,
        "remote_vs_onsite": {"remote": rem[0] or 0, "onsite": (rem[1] or 0) - (rem[0] or 0)},
        "emerging_skills": trend,   # month-over-month % change
        "note": "Salary data is not present in this dataset (12/56,769 rows).",
    }


# ── saved / dashboard (mutable user data lives in agent/userstore) ──────────
def market_for_position(job_id):
    """Most-requested skills across postings for the SAME role as this job.
    Role key = title with seniority words stripped, matched via LIKE."""
    job = get_job(job_id)
    if not job:
        return {"error": "job_not_found"}
    import re as _re
    from collections import Counter
    title = (job.get("title") or "").lower()
    stop = {"senior", "junior", "lead", "staff", "principal", "intern",
            "associate", "sr", "jr", "i", "ii", "iii", "trainee", "entry", "level",
            "intermediate", "mid", "expert", "experienced", "fresher", "the", "and",
            "of", "for", "a", "an", "remote", "hybrid", "onsite", "fulltime"}
    words = [w for w in _re.split(r"[^a-z]+", title) if w and w not in stop]
    key = " ".join(words[:3]).strip() or title[:24]
    con = _con()
    try:
        rows = con.execute(
            "SELECT skills FROM jobs WHERE LOWER(title) LIKE ? LIMIT 3000",
            (f"%{key}%",)).fetchall()
        # too specific? fall back to the core 2-word role (e.g. "data scientist")
        if len(rows) < 10 and len(words) >= 2:
            key = " ".join(words[-2:])
            rows = con.execute(
                "SELECT skills FROM jobs WHERE LOWER(title) LIKE ? LIMIT 3000",
                (f"%{key}%",)).fetchall()
    finally:
        con.close()
    freq = Counter()
    for (sk,) in rows:
        for s in _skills_list(sk):
            freq[s.lower()] += 1
    return {
        "position": job.get("title"),
        "role_key": key,
        "postings": len(rows),
        "top_skills": [{"skill": s, "count": c} for s, c in freq.most_common(15)],
    }


def get_jobs_by_ids(ids):
    if not ids:
        return []
    q = ",".join("?" * len(ids))
    con = _con()
    try:
        rows = con.execute(
            f"SELECT id,title,company,location,domain,emp,min_exp,max_exp,skills "
            f"FROM jobs WHERE id IN ({q})", list(ids)).fetchall()
    finally:
        con.close()
    out = [dict(r) for r in rows]
    for r in out:
        r["skills"] = _skills_list(r["skills"])
    return out


def ranking_pool(domain=None, location=None, limit=5000):
    """Candidate pool for the scout/match ranking (id+skills+meta)."""
    return query_jobs(domain=domain, location=location, limit=limit)


def saved_jobs(user_id):
    """Join the user's saved job IDs (userstore/Postgres) with job rows (jobs DB)."""
    rows = userstore.saved_job_ids(user_id)
    by_id = {j["id"]: j for j in get_jobs_by_ids([r[0] for r in rows])}
    out = []
    for jid, status, saved_at in rows:
        j = by_id.get(jid)
        if j:
            out.append({"id": jid, "title": j["title"], "company": j["company"],
                        "location": j["location"], "status": status, "saved_at": saved_at})
    return out


@traceable(run_type="retriever", name="recommend_jobs")
def recommend_jobs(user_id, limit=40, source=None, location=None, domain=None,
                   remote=None, min_exp=None, max_exp=None, extra_skills=None):
    """Rank jobs against the profile (skill overlap + preferred-title match),
    now also honoring the UI filters so Analyze + Search is ONE call/list."""
    p = userstore.get_profile(user_id)
    pskills = {s.lower() for s in (p.get("skills") or [])}
    pskills |= {s.lower() for s in (extra_skills or [])}   # filter-box skills count too
    titles = [t.lower() for t in (p.get("preferred_titles") or [])]
    locs = p.get("preferred_locations") or []
    remote_only = (p.get("work_pref") or "").lower() == "remote"
    if not pskills and not titles:
        return []

    eff_location = location or (locs[0] if locs else None)
    eff_remote = remote if remote is not None else (True if remote_only else None)
    cand = query_jobs(source=source, location=eff_location, domain=domain,
                      remote=eff_remote, min_exp=min_exp, max_exp=max_exp,
                      skills=list(extra_skills) if extra_skills else None, limit=3000)
    out = []
    for j in cand:
        js = {s.lower() for s in j["skills"]}
        inter = pskills & js
        title_hit = any(t in (j["title"] or "").lower() for t in titles)
        if not inter and not title_hit:
            continue
        score = 0.0
        if js:
            score += 0.7 * (len(inter) / len(js))
        if pskills:
            score += 0.2 * (len(inter) / len(pskills))
        if title_hit:
            score += 0.25
        out.append({**j, "score": round(min(score, 1.0) * 100, 1),
                    "matched": sorted(inter)})
    out.sort(key=lambda x: -x["score"])
    return out[:limit]


def recommend_count(user_id):
    return len(recommend_jobs(user_id, limit=40))


def dashboard(user_id):
    counts = userstore.status_counts(user_id)
    con = _con()
    try:
        total_jobs = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    finally:
        con.close()
    return {
        "new_jobs": total_jobs,
        "recommended": recommend_count(user_id),
        "saved": counts.get("saved", 0),
        "applied": counts.get("applied", 0),
        "interviews": counts.get("interview", 0),
    }

# """SQLite data access — stdlib sqlite3 only (keeps the app container light).

# Covers: filtered job search, dropdown facets, market-intelligence aggregates,
# single job with full description, and the auth/profile/saved-jobs/dashboard data.
# """
# import os
# import re
# import json
# import sqlite3
# from datetime import datetime, timezone
# from .trace import traceable
# from . import userstore

# DB_PATH = os.environ.get(
#     "APP_DB", os.path.join(os.path.dirname(__file__), "..", "data", "app_sample.db"))


# def _con():
#     con = sqlite3.connect(DB_PATH, check_same_thread=False)
#     con.row_factory = sqlite3.Row
#     return con


# def _skills_list(s):
#     return [t.strip() for t in re.split(r"[,;|]", s or "") if t.strip()]


# # ── jobs ───────────────────────────────────────────────────────────────────
# def query_jobs(source=None, location=None, domain=None, remote=None,
#                min_exp=None, max_exp=None, skills=None, q=None, limit=50, offset=0):
#     where, args = [], []
#     if source and source != "All Sources":
#         where.append("source = ?"); args.append(source)
#     if location:
#         where.append("location LIKE ?"); args.append(f"%{location}%")
#     if domain:
#         where.append("domain = ?"); args.append(domain)
#     if remote is not None:
#         where.append("remote = ?"); args.append(1 if remote else 0)
#     if min_exp is not None:
#         where.append("(max_exp IS NULL OR max_exp >= ?)"); args.append(min_exp)
#     if max_exp is not None:
#         where.append("(min_exp IS NULL OR min_exp <= ?)"); args.append(max_exp)
#     if q:
#         where.append("(title LIKE ? OR company LIKE ?)"); args += [f"%{q}%", f"%{q}%"]
#     for sk in (skills or []):
#         where.append("skills LIKE ?"); args.append(f"%{sk}%")
#     sql = "SELECT id,title,company,location,domain,source,emp,min_exp,max_exp,remote,skills FROM jobs"
#     if where:
#         sql += " WHERE " + " AND ".join(where)
#     sql += " LIMIT ? OFFSET ?"; args += [limit, offset]
#     con = _con()
#     try:
#         rows = [dict(r) for r in con.execute(sql, args).fetchall()]
#     finally:
#         con.close()
#     for r in rows:
#         r["skills"] = _skills_list(r["skills"])
#     return rows


# def get_job(job_id):
#     con = _con()
#     try:
#         r = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
#     finally:
#         con.close()
#     if not r:
#         return None
#     d = dict(r)
#     d["skills"] = _skills_list(d["skills"])
#     return d


# def facets():
#     con = _con()
#     try:
#         src = [r[0] for r in con.execute(
#             "SELECT source FROM jobs GROUP BY source ORDER BY COUNT(*) DESC LIMIT 12")]
#         dom = [r[0] for r in con.execute(
#             "SELECT domain FROM jobs GROUP BY domain ORDER BY COUNT(*) DESC")]
#     finally:
#         con.close()
#     return {"sources": ["All Sources"] + src, "domains": dom}


# def market_intel(domain=None, month=None):
#     """Aggregates for the market-intelligence dashboard (grounded in the data)."""
#     con = _con()
#     try:
#         base = "FROM jobs" + (" WHERE domain = ?" if domain else "")
#         a = [domain] if domain else []
#         total = con.execute("SELECT COUNT(*) " + base, a).fetchone()[0]
#         companies = [dict(name=r[0], jobs=r[1]) for r in con.execute(
#             "SELECT company, COUNT(*) c " + base + " GROUP BY company "
#             "ORDER BY c DESC LIMIT 10", a) if r[0]]
#         locations = [dict(name=r[0], jobs=r[1]) for r in con.execute(
#             "SELECT location, COUNT(*) c " + base + " GROUP BY location "
#             "ORDER BY c DESC LIMIT 10", a) if r[0]]
#         rem = con.execute("SELECT SUM(remote), COUNT(*) " + base, a).fetchone()
#         # skill frequency (overall) and month-over-month trend
#         skill_rows = con.execute("SELECT skills, posted_month " + base, a).fetchall()
#     finally:
#         con.close()

#     from collections import Counter, defaultdict
#     freq = Counter(); by_month = defaultdict(Counter)
#     for sk, m in skill_rows:
#         for s in _skills_list(sk):
#             s = s.lower(); freq[s] += 1
#             if m:
#                 by_month[m][s] += 1
#     top_skills = [dict(skill=s, count=c) for s, c in freq.most_common(15)]

#     trend = []
#     months = sorted(by_month)
#     if len(months) >= 2:
#         cur, prev = by_month[months[-1]], by_month[months[-2]]
#         for s, c in cur.most_common(30):
#             p = prev.get(s, 0)
#             if p >= 3:
#                 pct = round((c - p) / p * 100)
#                 trend.append(dict(skill=s, change_pct=pct, month=months[-1]))
#         trend = sorted(trend, key=lambda x: -x["change_pct"])[:10]

#     return {
#         "total_jobs": total,
#         "top_skills": top_skills,
#         "top_companies": companies,
#         "top_locations": locations,
#         "remote_vs_onsite": {"remote": rem[0] or 0, "onsite": (rem[1] or 0) - (rem[0] or 0)},
#         "emerging_skills": trend,   # month-over-month % change
#         "note": "Salary data is not present in this dataset (12/56,769 rows).",
#     }


# # ── saved / dashboard (mutable user data lives in agent/userstore) ──────────
# def market_for_position(job_id):
#     """Most-requested skills across postings for the SAME role as this job.
#     Role key = title with seniority words stripped, matched via LIKE."""
#     job = get_job(job_id)
#     if not job:
#         return {"error": "job_not_found"}
#     import re as _re
#     from collections import Counter
#     title = (job.get("title") or "").lower()
#     stop = {"senior", "junior", "lead", "staff", "principal", "intern",
#             "associate", "sr", "jr", "i", "ii", "iii", "trainee", "entry", "level",
#             "intermediate", "mid", "expert", "experienced", "fresher", "the", "and",
#             "of", "for", "a", "an", "remote", "hybrid", "onsite", "fulltime"}
#     words = [w for w in _re.split(r"[^a-z]+", title) if w and w not in stop]
#     key = " ".join(words[:3]).strip() or title[:24]
#     con = _con()
#     try:
#         rows = con.execute(
#             "SELECT skills FROM jobs WHERE LOWER(title) LIKE ? LIMIT 3000",
#             (f"%{key}%",)).fetchall()
#         # too specific? fall back to the core 2-word role (e.g. "data scientist")
#         if len(rows) < 10 and len(words) >= 2:
#             key = " ".join(words[-2:])
#             rows = con.execute(
#                 "SELECT skills FROM jobs WHERE LOWER(title) LIKE ? LIMIT 3000",
#                 (f"%{key}%",)).fetchall()
#     finally:
#         con.close()
#     freq = Counter()
#     for (sk,) in rows:
#         for s in _skills_list(sk):
#             freq[s.lower()] += 1
#     return {
#         "position": job.get("title"),
#         "role_key": key,
#         "postings": len(rows),
#         "top_skills": [{"skill": s, "count": c} for s, c in freq.most_common(15)],
#     }


# def get_jobs_by_ids(ids):
#     if not ids:
#         return []
#     q = ",".join("?" * len(ids))
#     con = _con()
#     try:
#         rows = con.execute(
#             f"SELECT id,title,company,location,domain,emp,min_exp,max_exp,skills "
#             f"FROM jobs WHERE id IN ({q})", list(ids)).fetchall()
#     finally:
#         con.close()
#     out = [dict(r) for r in rows]
#     for r in out:
#         r["skills"] = _skills_list(r["skills"])
#     return out


# def ranking_pool(domain=None, location=None, limit=5000):
#     """Candidate pool for the scout/match ranking (id+skills+meta)."""
#     return query_jobs(domain=domain, location=location, limit=limit)


# def saved_jobs(user_id):
#     """Join the user's saved job IDs (userstore/Postgres) with job rows (jobs DB)."""
#     rows = userstore.saved_job_ids(user_id)
#     by_id = {j["id"]: j for j in get_jobs_by_ids([r[0] for r in rows])}
#     out = []
#     for jid, status, saved_at in rows:
#         j = by_id.get(jid)
#         if j:
#             out.append({"id": jid, "title": j["title"], "company": j["company"],
#                         "location": j["location"], "status": status, "saved_at": saved_at})
#     return out


# @traceable(run_type="retriever", name="recommend_jobs")
# def recommend_jobs(user_id, limit=40):
#     """Rank jobs against the saved profile: skill overlap + preferred-title match,
#     filtered by preferred locations / remote preference when set."""
#     p = userstore.get_profile(user_id)
#     pskills = {s.lower() for s in (p.get("skills") or [])}
#     titles = [t.lower() for t in (p.get("preferred_titles") or [])]
#     locs = p.get("preferred_locations") or []
#     remote_only = (p.get("work_pref") or "").lower() == "remote"
#     if not pskills and not titles:
#         return []

#     # coarse candidate set (respect one preferred location if given; else all)
#     cand = query_jobs(location=(locs[0] if locs else None),
#                       remote=True if remote_only else None, limit=3000)
#     out = []
#     for j in cand:
#         js = {s.lower() for s in j["skills"]}
#         inter = pskills & js
#         title_hit = any(t in (j["title"] or "").lower() for t in titles)
#         if not inter and not title_hit:
#             continue
#         score = 0.0
#         if js:
#             score += 0.7 * (len(inter) / len(js))
#         if pskills:
#             score += 0.2 * (len(inter) / len(pskills))
#         if title_hit:
#             score += 0.25
#         out.append({**j, "score": round(min(score, 1.0) * 100, 1),
#                     "matched": sorted(inter)})
#     out.sort(key=lambda x: -x["score"])
#     return out[:limit]


# def recommend_count(user_id):
#     return len(recommend_jobs(user_id, limit=40))


# def dashboard(user_id):
#     counts = userstore.status_counts(user_id)
#     con = _con()
#     try:
#         total_jobs = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
#     finally:
#         con.close()
#     return {
#         "new_jobs": total_jobs,
#         "recommended": recommend_count(user_id),
#         "saved": counts.get("saved", 0),
#         "applied": counts.get("applied", 0),
#         "interviews": counts.get("interview", 0),
#     }
