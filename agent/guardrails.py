"""Lightweight guardrails for text that reaches the LLM / scorer.

Not a substitute for a full policy engine — it caps size, strips common
prompt-injection patterns, and flags empty/garbage input so the agent fails
loudly instead of feeding junk to ResumeHQ or the LLM.
"""
import re

MAX_CHARS = 20000
_INJECTION = re.compile(
    r"(ignore (all |the )?previous instructions|disregard (all|the) above|"
    r"system prompt|you are now|act as (an?|the) |reveal your (prompt|instructions)|"
    r"</?(system|assistant|tool)>)", re.I)


def sanitize(text: str) -> tuple[str, list]:
    """Return (clean_text, flags). Never raises."""
    flags = []
    text = text or ""
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        flags.append("truncated")
    hits = _INJECTION.findall(text)
    if hits:
        text = _INJECTION.sub("[removed]", text)
        flags.append("injection_stripped")
    return text, flags


def check_resume(text: str) -> dict:
    """Validate a resume before scoring. Returns {ok, text, flags, reason?}."""
    clean, flags = sanitize(text)
    if len(clean.strip()) < 30:
        return {"ok": False, "reason": "resume_too_short", "text": clean, "flags": flags}
    return {"ok": True, "text": clean, "flags": flags}
