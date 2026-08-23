"""Job ranking over SQLite (no JSONL). Skill-overlap scoring for the scout/match
nodes. Resume *scoring* is ResumeHQ (rq_tools) — this is only retrieval ranking.

Reads jobs from the app DB via agent.db, so the large source JSON/JSONL is not
needed at runtime — the database is the single store.
"""
from . import db
from .trace import traceable

SYN = {"ml": "machine learning", "dl": "deep learning", "js": "javascript",
       "ts": "typescript", "reactjs": "react", "nodejs": "node", "py": "python",
       "sklearn": "scikit-learn", "tf": "tensorflow", "k8s": "kubernetes",
       "nlp": "natural language processing", "postgres": "postgresql"}


def norm(s):
    s = str(s).lower().strip(" .")
    return SYN.get(s, s)


def normset(items):
    return {norm(x) for x in items if str(x).strip()}


def filter_pool(domain=None, location=None, limit=5000):
    """Coarse retrieval (scout branch) — from SQLite, no profile needed."""
    return db.ranking_pool(domain=domain, location=location, limit=limit)


def get_job(job_id):
    return db.get_job(job_id)


@traceable(run_type="tool", name="rank_jobs")
def score_pool(profile: dict, pool, top=12):
    """Fine ranking (join node): score the scouted pool vs the parsed profile."""
    pskills = normset(profile.get("skills", []))
    exp = profile.get("exp_years")
    out = []
    for j in pool:
        js = normset(j.get("skills", []))
        if not js:
            continue
        inter = pskills & js
        if not inter:
            continue
        job_cov = len(inter) / len(js)
        prof_cov = len(inter) / max(len(pskills), 1)
        score = 0.7 * job_cov + 0.3 * prof_cov
        if exp is not None and j.get("min_exp") is not None and exp < j["min_exp"] - 1:
            score *= 0.85
        out.append({"id": j["id"], "title": j["title"], "company": j["company"],
                    "location": j["location"], "domain": j["domain"],
                    "score": round(score * 100, 1),
                    "matched": sorted(inter), "missing": sorted(js - pskills)})
    out.sort(key=lambda x: -x["score"])
    return out[:top]


def match_jobs(profile: dict, top=12, domain=None, location=None):
    return score_pool(profile, filter_pool(domain, location), top)
