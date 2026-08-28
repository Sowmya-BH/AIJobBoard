"""One-time (offline) ingestion: push the raw job JSON into MongoDB Atlas.

Replaces data/build_db.py (SQLite). Run locally, NOT on Render.

    export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority"
    export MONGODB_DB=jobscout            # optional
    python data/build_mongo.py --source /path/to/query_result.json --desc-cap 6000

Creates the `jobs` collection with indexes. Stores skills as a list plus a
lowercased `skills_lc` list for fast filtering/aggregation.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean(x):
    s = str(x or "").strip()
    return "" if s.lower() in ("none", "null", "") else s


def num(x):
    try:
        return int(float(clean(x)))
    except Exception:
        return None


# --- ADD THE SMART CAP FUNCTION HERE ---
def smart_cap(text, limit=3000):
    """Truncates text at the last full sentence to stay under the limit."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    # Cut at the limit
    truncated = text[:limit]
    # Find the last period in the last 20% of the truncated text
    last_period = truncated.rfind(".")
    if last_period > limit * 0.8: 
        return truncated[:last_period + 1]
    # Fallback if no sentence boundary is found
    return truncated.rsplit(' ', 1)[0] + "..."




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


def skills_list(raw):
    return [s.strip() for s in re.split(r"[,;|]", raw or "") if s.strip()]


def rows(source, limit):
    try:
        import ijson
        with open(source, "rb") as f:
            for i, r in enumerate(ijson.items(f, "item")):
                if limit and i >= limit:
                    break
                yield r
    except ImportError:
        for i, r in enumerate(json.load(open(source))):
            if limit and i >= limit:
                break
            yield r


def to_doc(r, desc_cap):
    desc = clean(r.get("description"))
    sk = skills_list(clean(r.get("skills")))
    return {
        "_id": r.get("job_id"),
        "title": clean(r.get("title")), "company": clean(r.get("company_name")),
        "location": clean(r.get("location")), "domain": clean(r.get("domain")),
        "source": clean(r.get("via")).replace("via ", "").strip() or "Unknown",
        "emp": clean(r.get("employmentType")), "schedule_type": clean(r.get("schedule_type")),
        "min_exp": num(r.get("minExperienceRequired")),
        "max_exp": num(r.get("maxExperienceRequired")),
        "remote": is_remote(clean(r.get("location")), clean(r.get("locationRequirement")), desc),
        "posted_month": parse_month(r.get("posted_at"), r.get("createdAt"), r.get("publishedAt")),
        "skills": sk, "skills_lc": [s.lower() for s in sk],
        "description": smart_cap(desc, desc_cap), 
        
        "apply_link": apply_link(r.get("apply_options")),
        # "description": desc[:desc_cap], "apply_link": apply_link(r.get("apply_options")),
    }


def main(source, desc_cap, limit, batch, drop):
    from pymongo import MongoClient, ReplaceOne
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        sys.exit("Set MONGODB_URI first.")
    db = MongoClient(uri, appname="job-scout-ingest")[os.environ.get("MONGODB_DB", "jobscout")]
    jobs = db.jobs
    if drop:
        jobs.drop()
        print("dropped existing jobs collection")

    ops, n = [], 0
    for r in rows(source, limit):
        if not r.get("job_id"):
            continue
        ops.append(ReplaceOne({"_id": r["job_id"]}, to_doc(r, desc_cap), upsert=True))
        if len(ops) >= batch:
            jobs.bulk_write(ops, ordered=False)
            n += len(ops); ops = []
            print(f"  upserted {n}")
    if ops:
        jobs.bulk_write(ops, ordered=False)
        n += len(ops)

    print("creating indexes...")
    jobs.create_index("source"); jobs.create_index("domain")
    jobs.create_index("remote"); jobs.create_index("posted_month")
    jobs.create_index("skills_lc")
    print(f"done — {n} jobs in MongoDB ({db.name}.jobs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--desc-cap", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--drop", action="store_true", help="drop the collection first")
    a = ap.parse_args()
    main(a.source, a.desc_cap, a.limit, a.batch, a.drop)