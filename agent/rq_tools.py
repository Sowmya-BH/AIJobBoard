"""ResumeHQ strict-tool boundary.

This is the ONLY place the app is allowed to touch ResumeHQ. It enforces the
three MCP disciplines the design calls for, while running in-process (no
separate MCP server) via a direct import of ResumeHQ's `mcp_scorer` module:

1. Enumerate strict tools — TOOLS is a fixed registry. Any call to a name not
   in it is rejected. Required args are validated against a rigid schema before
   the underlying function runs.
2. Force ground truths — the raw dict ResumeHQ returns is passed back UNCHANGED.
   This module never invents a score, keyword, or suggestion. If ResumeHQ (or
   its deps) is unavailable, or a tool has no data, callers get an explicit
   error/empty result — never a fabricated one.
3. Audit trail — every call and its return are logged (discrete, timestamped,
   hash-stamped) to an in-memory list and an append-only JSONL file, so there
   is verifiable proof of exactly what the scorer saw and returned.

NOTE ON IMPORT NAME: ResumeHQ installs FLAT top-level modules, not a `resumehq`
package. The correct import is `import mcp_scorer` (console script:
`resumehq-mcp`). `import resumehq` / `resumehq.ResumeBuilder()` do NOT exist.
"""
from __future__ import annotations
import os
import json
import time
import hashlib
import itertools
from .trace import traceable
from datetime import datetime, timezone

AUDIT_PATH = os.environ.get("RQ_AUDIT_LOG", "/tmp/resumehq_audit.jsonl")

def _norm_url(u: str) -> str:
    """Accept a full URL or a bare host:port (Render `fromService: hostport`
    injects the latter) and return a usable http URL."""
    u = (u or "").strip().rstrip("/")
    if u and not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u


# Self-hosted scorer sidecar. If set, ResumeHQ scoring is done over HTTP by a
# separate process (its own RAM budget, full torch/SBERT) instead of importing
# the heavy engine here. This keeps the app container small AND bypasses the
# cloud free-tier cap, because the sidecar runs LOCAL scoring (no quota).
SCORER_URL = _norm_url(os.environ.get("RQ_SCORER_URL", ""))
SCORER_TIMEOUT = float(os.environ.get("RQ_SCORER_TIMEOUT", "60"))
REMOTE_ENDPOINTS = {
    "score_ats": "/score/ats",
    "score_hr": "/score/hr",
    "explain_score": "/explain",
    "generate_cover_letter": "/cover-letter",
}

_audit: list[dict] = []
_counter = itertools.count(1)

# ── lazy import of ResumeHQ's strict-tool module ───────────────────────────
_mcp = None
_import_error = None


def _load_scorer():
    """Import ResumeHQ's mcp_scorer once. Returns module or None.

    `import mcp_scorer` is light (~76 MB with fastmcp; torch is NOT loaded — the
    local scorers are imported lazily only when a score falls through to local).
    Set RQ_CLOUD_ONLY=1 to hard-disable that local fallback so a cloud miss
    returns an explicit error instead of importing torch/SBERT and breaching a
    small (e.g. 512 MB) container — which would OOM-kill the process.
    """
    global _mcp, _import_error
    if _mcp is not None or _import_error is not None:
        return _mcp
    try:
        import mcp_scorer  # ResumeHQ — flat module name, NOT `resumehq`
        _mcp = mcp_scorer
    except Exception as e:  # ImportError, or heavy-dep failure
        _import_error = f"{type(e).__name__}: {e}"
        return _mcp

    if os.environ.get("RQ_CLOUD_ONLY", "").lower() in ("1", "true", "yes"):
        def _blocked(*_a, **_k):
            raise RuntimeError(
                "RQ_CLOUD_ONLY is set: local (torch/SBERT) scoring is disabled to "
                "stay within the container RAM budget. Configure SCORER_CLOUD_API_KEY "
                "for cloud scoring, or run local scoring in a separate process.")
        _mcp._ensure_scorers = _blocked          # any local fallback raises cleanly
        _mcp._local_available = lambda: False     # never import ats_scorer to probe
    return _mcp


def available() -> bool:
    return _load_scorer() is not None


# ── enumerated tool registry (rigid schemas) ───────────────────────────────
# Each entry: required arg names + the documented return keys we rely on.
TOOLS = {
    "score_ats": {
        "args": ["resume_text", "jd_text"],
        "returns": ["total_score", "matched_keywords", "missing_keywords",
                    "domain", "format_risk", "rating", "likelihood"],
    },
    "score_hr": {
        "args": ["resume_text", "jd_text"],
        "returns": ["overall_score", "recommendation", "strengths",
                    "concerns", "suggested_questions"],
    },
    "explain_score": {
        "args": ["resume_text", "jd_text"],
        "returns": ["current_score", "explanation"],
    },
    "generate_cover_letter": {
        "args": ["resume_text", "jd_text"],
        "optional": ["company_name", "job_title"],
        "returns": ["paragraphs", "full_text", "word_count"],
    },
    "extract_text": {
        "args": ["file_path"],
        "returns": ["text"],
    },
}


def _sha(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:12]


def _record(entry: dict):
    _audit.append(entry)
    try:
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


@traceable(run_type="tool", name="resumehq_tool")
def call(tool: str, **kwargs) -> dict:
    """Invoke a ResumeHQ tool through the strict boundary.

    Returns the scorer's raw dict, or an explicit error dict. Always audited.
    """
    call_id = next(_counter)
    ts = datetime.now(timezone.utc).isoformat()

    # (1) enumerate strict tools
    spec = TOOLS.get(tool)
    if spec is None:
        out = {"error": "unknown_tool", "message": f"'{tool}' is not an enumerated ResumeHQ tool.",
               "allowed": list(TOOLS)}
        _record({"id": call_id, "ts": ts, "tool": tool, "ok": False, "reason": "unknown_tool"})
        return out

    # rigid arg validation
    missing = [a for a in spec["args"] if a not in kwargs or kwargs[a] in (None, "")]
    if missing:
        out = {"error": "missing_args", "message": f"{tool} requires {missing}"}
        _record({"id": call_id, "ts": ts, "tool": tool, "ok": False, "reason": f"missing:{missing}"})
        return out

    # audit the inputs (hash + length, not raw text)
    in_stamp = {k: {"sha": _sha(str(v)), "len": len(str(v))} for k, v in kwargs.items()}

    # (remote) self-hosted sidecar: HTTP call, app stays torch-free, no quota
    if SCORER_URL and tool in REMOTE_ENDPOINTS:
        t0 = time.time()
        result = _call_remote(REMOTE_ENDPOINTS[tool],
                              {k: kwargs[k] for k in spec["args"]
                               + spec.get("optional", []) if k in kwargs})
        _record({"id": call_id, "ts": ts, "tool": tool, "via": "remote",
                 "ok": "error" not in result, "ms": round((time.time() - t0) * 1000, 1),
                 "inputs": in_stamp, "returned_keys": sorted(result.keys()),
                 "score": result.get("total_score", result.get("overall_score",
                          result.get("current_score")))})
        return result

    # (2) force ground truth: no scorer -> explicit empty/error, never invented
    mcp = _load_scorer()
    if mcp is None:
        out = {"error": "resumehq_unavailable",
               "message": ("ResumeHQ is not importable in this runtime "
                           f"({_import_error}). Install it, or set RQ_SCORER_URL "
                           "to a scorer sidecar."),
               "tool": tool}
        _record({"id": call_id, "ts": ts, "tool": tool, "ok": False, "reason": "unavailable"})
        return out

    fn = getattr(mcp, tool, None)
    if fn is None:
        out = {"error": "tool_not_found", "message": f"mcp_scorer has no '{tool}'"}
        _record({"id": call_id, "ts": ts, "tool": tool, "ok": False, "reason": "no_attr"})
        return out

    t0 = time.time()
    try:
        result = fn(**{k: kwargs[k] for k in spec["args"]},
                    **{k: kwargs[k] for k in spec.get("optional", []) if k in kwargs})
    except Exception as e:
        out = {"error": "tool_raised", "message": f"{type(e).__name__}: {e}", "tool": tool}
        _record({"id": call_id, "ts": ts, "tool": tool, "ok": False,
                 "inputs": in_stamp, "reason": f"raised:{type(e).__name__}"})
        return out

    if not isinstance(result, dict):
        result = {"error": "bad_return", "message": "tool did not return a dict"}

    # (3) audit trail: discrete record of what the scorer saw + returned
    _record({
        "id": call_id, "ts": ts, "tool": tool, "ok": "error" not in result,
        "ms": round((time.time() - t0) * 1000, 1),
        "inputs": in_stamp,
        "returned_keys": sorted(result.keys()),
        "score": result.get("total_score", result.get("overall_score", result.get("current_score"))),
    })
    return result


def audit_log() -> list[dict]:
    """Return the in-memory audit trail (verifiable proof of tool I/O)."""
    return list(_audit)


def _call_remote(endpoint: str, payload: dict) -> dict:
    """POST to the scorer sidecar. Stdlib only — no torch, no extra deps."""
    import urllib.request
    import urllib.error
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SCORER_URL + endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=SCORER_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body if isinstance(body, dict) else {"error": "bad_return"}
    except urllib.error.HTTPError as e:
        return {"error": "scorer_http_error", "message": f"{e.code} {e.reason}"}
    except Exception as e:
        return {"error": "scorer_unreachable",
                "message": f"{type(e).__name__}: {e} (RQ_SCORER_URL={SCORER_URL})"}
