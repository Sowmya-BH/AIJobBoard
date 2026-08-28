"""OFFLINE: embed all jobs with LOCAL SBERT and upsert into Qdrant Cloud.

Run on your Mac (NOT on Render). Uses the SAME model as the query side
(agent/vectors.EMBED_MODEL, default all-mpnet-base-v2, 768-dim) so the resume
vectors from Render's HF Inference call live in the same space.

    pip install sentence-transformers pymongo
    export MONGODB_URI="mongodb+srv://..."
    export QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
    export QDRANT_API_KEY=...
    python data/build_embeddings.py --limit 2 --recreate --batch 2   # dry run
    python data/build_embeddings.py --recreate                        # full 57k
"""
import argparse
import os
import sys
import json
import uuid
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import vectors   # reuse QDRANT_* config + EMBED_MODEL/EMBED_DIM

_MODEL = None


def sbert():
    """Load the local sentence-transformers model once."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"loading local SBERT: {vectors.EMBED_MODEL} ...")
        _MODEL = SentenceTransformer(vectors.EMBED_MODEL)
    return _MODEL


def _put(url, headers, body, timeout=60):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json", **headers},
                                 method="PUT")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def create_collection(recreate=False):
    base = f"{vectors.QDRANT_URL}/collections/{vectors.QDRANT_COLLECTION}"
    hdr = {"api-key": vectors.QDRANT_API_KEY}
    if recreate:
        try:
            urllib.request.urlopen(
                urllib.request.Request(base, headers=hdr, method="DELETE"), timeout=15)
            print("dropped existing collection")
        except Exception:
            pass
    _put(base, hdr, {"vectors": {"size": vectors.EMBED_DIM, "distance": "Cosine"}})
    print(f"collection '{vectors.QDRANT_COLLECTION}' ready ({vectors.EMBED_DIM}-dim, Cosine)")


def job_text(r):
    sk = r.get("skills", [])
    skills = ", ".join(sk) if isinstance(sk, list) else str(sk)
    desc = (r.get("description") or "")[:1000]
    return f"Role: {r.get('title', 'Untitled')}. Skills: {skills}. Description: {desc}"


def upsert(points):
    url = f"{vectors.QDRANT_URL}/collections/{vectors.QDRANT_COLLECTION}/points"
    _put(url, {"api-key": vectors.QDRANT_API_KEY}, {"points": points})


def run(batch_size, limit, recreate):
    if not (vectors.QDRANT_URL and vectors.QDRANT_API_KEY):
        sys.exit("Set QDRANT_URL and QDRANT_API_KEY first.")
    from pymongo import MongoClient
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        sys.exit("Set MONGODB_URI first.")
    col = MongoClient(uri)[os.environ.get("MONGODB_DB", "jobscout")].jobs

    cur = col.find({})
    if limit > 0:
        cur = cur.limit(limit)
    all_jobs = list(cur)
    total = len(all_jobs)
    print(f"{total} jobs loaded from MongoDB")

    model = sbert()                      # load once, up front
    create_collection(recreate=recreate)

    done = 0
    for i in range(0, total, batch_size):
        chunk = all_jobs[i:i + batch_size]
        vecs = model.encode([job_text(j) for j in chunk],
                            batch_size=min(batch_size, 64),
                            show_progress_bar=False, 
                            convert_to_tensor=False).tolist()
        points = []
        for job, vec in zip(chunk, vecs):
            jid = str(job["_id"])
            points.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, jid)),  # valid Qdrant id
                "vector": vec,
                "payload": {"job_id": jid, "title": job.get("title"),
                            "company": job.get("company"), "location": job.get("location"),
                            "domain": job.get("domain"), "source": job.get("source"),
                            "remote": int(job.get("remote", 0))},
            })
        try:
            upsert(points)
            done += len(points)
            print(f"  indexed {done}/{total}")
        except Exception as e:
            print(f"  upsert failed at {i}: {e}")
    print(f"done — {done} jobs indexed into Qdrant")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recreate", action="store_true")
    a = ap.parse_args()
    run(a.batch, a.limit, a.recreate)