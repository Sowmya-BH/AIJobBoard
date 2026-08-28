"""Semantic job matching via Qdrant Cloud + SBERT embeddings.

Same model on both sides so the vectors are comparable:
  * job indexing  (offline, your Mac)  -> local sentence-transformers  [build_embeddings.py]
  * resume query  (Render, RAM-tiny)   -> HF Inference API (this file, urllib only)

Model: sentence-transformers/all-mpnet-base-v2 (768-dim). No torch in the app.
Enable with: QDRANT_URL, QDRANT_API_KEY, HF_TOKEN. Any missing -> callers fall
back to skill-overlap ranking.
"""
import os
import json
import urllib.request
import urllib.error

QDRANT_URL = (os.environ.get("QDRANT_URL", "") or "").rstrip("/")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "jobs")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("RQ_SCORER_HF_TOKEN") or ""
# Query-time embedding endpoint (feature-extraction). Override if HF changes URLs.
HF_EMBED_URL = os.environ.get(
    "HF_EMBED_URL", f"https://api-inference.huggingface.co/models/{EMBED_MODEL}")


def available() -> bool:
    return bool(QDRANT_URL and QDRANT_API_KEY and HF_TOKEN)


def _post(url, headers, body, timeout=30):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json", **headers},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")[:300]
        print(f"ERROR: embedding/search call failed ({e.code}): {body}")
        raise


# ── SBERT embeddings via HF Inference (query time) ──────────────────────────
def _pool(v):
    """Collapse an HF feature-extraction response to ONE flat vector of floats.
    Handles: a pooled vector [768]; a token matrix [tokens][768] (mean-pool, which
    also covers the single-row [[...]] case); and an extra batch wrapper
    [[tokens][768]] returned for some models/inputs."""
    if not isinstance(v, list) or not v:
        return v
    # already a flat vector: [f, f, ...]
    if isinstance(v[0], (int, float)):
        return v
    # 2-D matrix of floats: [[f,...],[f,...]] -> mean-pool rows (1 row -> itself)
    if isinstance(v[0], list) and v[0] and isinstance(v[0][0], (int, float)):
        n, dim = len(v), len(v[0])
        return [sum(row[i] for row in v) / n for i in range(dim)]
    # 3-D (batch wrapper around a matrix): [[[f,...]]] -> unwrap then pool
    if isinstance(v[0], list) and v[0] and isinstance(v[0][0], list):
        return _pool(v[0])
    return v


def embed(text: str) -> list:
    """Embed one text (resume) via the HF Inference API."""
    data = _post(HF_EMBED_URL, {"Authorization": f"Bearer {HF_TOKEN}"},
                 {"inputs": text[:3000], "options": {"wait_for_model": True}})
    return _pool(data)


def embed_batch(texts: list) -> list:
    """Embed several texts in one HF call (optional; indexing uses local SBERT)."""
    data = _post(HF_EMBED_URL, {"Authorization": f"Bearer {HF_TOKEN}"},
                 {"inputs": [t[:3000] for t in texts], "options": {"wait_for_model": True}})
    return [_pool(x) for x in data]


# ── Qdrant ──────────────────────────────────────────────────────────────────
def _qdrant_filter(filters: dict):
    must = []
    if filters.get("source") and filters["source"] != "All Sources":
        must.append({"key": "source", "match": {"value": filters["source"]}})
    if filters.get("domain"):
        must.append({"key": "domain", "match": {"value": filters["domain"]}})
    if filters.get("remote"):
        must.append({"key": "remote", "match": {"value": 1}})
    return {"must": must} if must else None


def search(vector: list, filters: dict, limit: int = 40) -> list:
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search"
    body = {"vector": vector, "limit": limit, "with_payload": True}
    flt = _qdrant_filter(filters or {})
    if flt:
        body["filter"] = flt
    data = _post(url, {"api-key": QDRANT_API_KEY}, body)
    return data.get("result", [])


def recommend(resume_text: str, filters: dict, limit: int = 40) -> list:
    """Embed the resume (HF), search Qdrant, return jobs by semantic similarity.
    Any failure returns [] so callers fall back to skill-overlap ranking."""
    if not available():
        return []
    try:
        vec = embed(resume_text)
        hits = search(vec, filters or {}, limit)
        out = []
        for h in hits:
            p = h.get("payload", {})
            out.append({
                "id": p.get("job_id") or str(h.get("id")),
                "title": p.get("title", ""), "company": p.get("company", ""),
                "location": p.get("location", ""), "source": p.get("source", ""),
                "domain": p.get("domain", ""), "remote": p.get("remote", 0),
                "score": round(float(h.get("score", 0)) * 100, 1),
            })
        return out
    except Exception as e:
        print(f"vector recommend error: {e}")
        return []