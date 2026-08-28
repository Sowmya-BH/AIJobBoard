"""Jobs data access — MongoDB Atlas (replaces SQLite).

Set MONGODB_URI (and optional MONGODB_DB, default "jobscout"). Jobs live in the
`jobs` collection, indexed on source/domain/remote/posted_month + a lowercased
`skills_lc` array for fast skill filtering. Nothing is shipped in the image.

The offline indexer data/build_mongo.py loads the raw JSON into Atlas once.
"""
import os
import re
from collections import Counter

from .trace import traceable
from . import userstore, vectors

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


def _jobs():
    return _db().jobs


def _shape(d, full=False):
    if not d:
        return d
    out = {"id": d.get("_id"), "title": d.get("title", ""), "company": d.get("company", ""),
           "location": d.get("location", ""), "domain": d.get("domain", ""),
           "source": d.get("source", ""), "emp": d.get("emp", ""),
           "min_exp": d.get("min_exp"), "max_exp": d.get("max_exp"),
           "remote": d.get("remote", 0), "skills": d.get("skills", []) or []}
    if full:
        out["description"] = d.get("description", "")
        out["apply_link"] = d.get("apply_link", "")
        out["schedule_type"] = d.get("schedule_type", "")
        out["posted_month"] = d.get("posted_month", "")
    return out


_LIST_PROJ = {"title": 1, "company": 1, "location": 1, "domain": 1, "source": 1,
              "emp": 1, "min_exp": 1, "max_exp": 1, "remote": 1, "skills": 1}


def _build_filter(source=None, location=None, domain=None, remote=None,
                  min_exp=None, max_exp=None, skills=None, q=None):
    flt, ands = {}, []
    if source and source != "All Sources":
        flt["source"] = source
    if domain:
        flt["domain"] = domain
    if remote is not None:
        flt["remote"] = 1 if remote else 0
    if location:
        flt["location"] = {"$regex": re.escape(location), "$options": "i"}
    if q:
        rx = {"$regex": re.escape(q), "$options": "i"}
        ands.append({"$or": [{"title": rx}, {"company": rx}]})
    if min_exp is not None:
        ands.append({"$or": [{"max_exp": None}, {"max_exp": {"$gte": min_exp}}]})
    if max_exp is not None:
        ands.append({"$or": [{"min_exp": None}, {"min_exp": {"$lte": max_exp}}]})
    if skills:
        flt["skills_lc"] = {"$all": [s.lower() for s in skills]}
    if ands:
        flt["$and"] = ands
    return flt


# ── jobs ────────────────────────────────────────────────────────────────────
def query_jobs(source=None, location=None, domain=None, remote=None,
               min_exp=None, max_exp=None, skills=None, q=None, limit=50, offset=0):
    flt = _build_filter(source, location, domain, remote, min_exp, max_exp, skills, q)
    cur = _jobs().find(flt, _LIST_PROJ).skip(offset).limit(limit)
    return [_shape(d) for d in cur]


def get_job(job_id):
    return _shape(_jobs().find_one({"_id": job_id}), full=True)


def get_jobs_by_ids(ids):
    if not ids:
        return []
    return [_shape(d) for d in _jobs().find({"_id": {"$in": list(ids)}}, _LIST_PROJ)]


def ranking_pool(domain=None, location=None, limit=5000):
    return query_jobs(domain=domain, location=location, limit=limit)


def facets():
    j = _jobs()
    src = [g["_id"] for g in j.aggregate([
        {"$group": {"_id": "$source", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}}, {"$limit": 12}]) if g["_id"]]
    dom = [g["_id"] for g in j.aggregate([
        {"$group": {"_id": "$domain", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}}]) if g["_id"]]
    return {"sources": ["All Sources"] + src, "domains": dom}


def market_intel(domain=None, month=None):
    j = _jobs()
    match = {"domain": domain} if domain else {}
    total = j.count_documents(match)
    companies = [{"name": g["_id"], "jobs": g["n"]} for g in j.aggregate([
        {"$match": match}, {"$group": {"_id": "$company", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}}, {"$limit": 10}]) if g["_id"]]
    locations = [{"name": g["_id"], "jobs": g["n"]} for g in j.aggregate([
        {"$match": match}, {"$group": {"_id": "$location", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}}, {"$limit": 10}]) if g["_id"]]
    rem = next(iter(j.aggregate([
        {"$match": match},
        {"$group": {"_id": None, "remote": {"$sum": "$remote"}, "total": {"$sum": 1}}}])),
        {"remote": 0, "total": total})
    top_skills = [{"skill": g["_id"], "count": g["n"]} for g in j.aggregate([
        {"$match": match}, {"$unwind": "$skills_lc"},
        {"$group": {"_id": "$skills_lc", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}}, {"$limit": 15}]) if g["_id"]]

    months = sorted(m for m in j.distinct("posted_month", match) if m)
    trend = []
    if len(months) >= 2:
        def month_counts(m):
            return {g["_id"]: g["n"] for g in j.aggregate([
                {"$match": {**match, "posted_month": m}}, {"$unwind": "$skills_lc"},
                {"$group": {"_id": "$skills_lc", "n": {"$sum": 1}}}])}
        cur, prev = month_counts(months[-1]), month_counts(months[-2])
        for s, c in sorted(cur.items(), key=lambda x: -x[1])[:30]:
            p = prev.get(s, 0)
            if p >= 3:
                trend.append({"skill": s, "change_pct": round((c - p) / p * 100),
                              "month": months[-1]})
        trend = sorted(trend, key=lambda x: -x["change_pct"])[:10]

    return {"total_jobs": total, "top_skills": top_skills, "top_companies": companies,
            "top_locations": locations,
            "remote_vs_onsite": {"remote": rem.get("remote", 0),
                                 "onsite": (rem.get("total", total) - rem.get("remote", 0))},
            "emerging_skills": trend,
            "note": "Salary data is not present in this dataset."}


def market_for_position(job_id):
    job = get_job(job_id)
    if not job:
        return {"error": "job_not_found"}
    title = (job.get("title") or "").lower()
    stop = {"senior", "junior", "lead", "staff", "principal", "intern", "associate",
            "sr", "jr", "i", "ii", "iii", "trainee", "entry", "level", "intermediate",
            "mid", "expert", "experienced", "fresher", "the", "and", "of", "for", "a", "an"}
    words = [w for w in re.split(r"[^a-z]+", title) if w and w not in stop]
    key = " ".join(words[:3]).strip() or title[:24]
    j = _jobs()

    def rows_for(k):
        rx = {"title": {"$regex": re.escape(k), "$options": "i"}}
        return list(j.find(rx, {"skills_lc": 1}).limit(3000))
    rows = rows_for(key)
    if len(rows) < 10 and len(words) >= 2:
        key = " ".join(words[-2:])
        rows = rows_for(key)
    freq = Counter()
    for r in rows:
        for s in (r.get("skills_lc") or []):
            freq[s] += 1
    return {"position": job.get("title"), "role_key": key, "postings": len(rows),
            "top_skills": [{"skill": s, "count": c} for s, c in freq.most_common(15)]}


# ── recommendations (semantic vector search -> skill-overlap fallback) ────────
@traceable(run_type="retriever", name="recommend_jobs")
def recommend_jobs(user_id, limit=40, source=None, location=None, domain=None,
                   remote=None, min_exp=None, max_exp=None, extra_skills=None,
                   resume_text=None):
    if resume_text and vectors.available():
        try:
            vres = vectors.recommend(
                resume_text, {"source": source, "domain": domain, "remote": remote}, limit)
            if vres:
                return vres
        except Exception:
            pass

    p = userstore.get_profile(user_id)
    pskills = {s.lower() for s in (p.get("skills") or [])}
    pskills |= {s.lower() for s in (extra_skills or [])}
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
    for jb in cand:
        js = {s.lower() for s in jb["skills"]}
        inter = pskills & js
        title_hit = any(t in (jb["title"] or "").lower() for t in titles)
        if not inter and not title_hit:
            continue
        score = 0.0
        if js:
            score += 0.7 * (len(inter) / len(js))
        if pskills:
            score += 0.2 * (len(inter) / len(pskills))
        if title_hit:
            score += 0.25
        out.append({**jb, "score": round(min(score, 1.0) * 100, 1), "matched": sorted(inter)})
    out.sort(key=lambda x: -x["score"])
    return out[:limit]


def recommend_count(user_id):
    return len(recommend_jobs(user_id, limit=40))


# ── saved jobs + dashboard (user data lives in userstore, also Mongo) ─────────
def saved_jobs(user_id):
    rows = userstore.saved_job_ids(user_id)
    by_id = {jb["id"]: jb for jb in get_jobs_by_ids([r[0] for r in rows])}
    out = []
    for jid, status, saved_at in rows:
        jb = by_id.get(jid)
        if jb:
            out.append({"id": jid, "title": jb["title"], "company": jb["company"],
                        "location": jb["location"], "status": status, "saved_at": saved_at})
    return out


def dashboard(user_id):
    counts = userstore.status_counts(user_id)
    return {"new_jobs": _jobs().count_documents({}),
            "recommended": recommend_count(user_id),
            "saved": counts.get("saved", 0), "applied": counts.get("applied", 0),
            "interviews": counts.get("interview", 0)}

