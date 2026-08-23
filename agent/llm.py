"""Gemini wrapper. Degrades to deterministic stubs when no key is present,
so the whole graph runs (with weaker text) even before you wire a key."""
import json
import re
from .config import GEMINI_API_KEY, GEMINI_MODEL

from .trace import traceable

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _client = genai.GenerativeModel(GEMINI_MODEL)
        return _client
    except Exception:
        return None


def available() -> bool:
    return _get_client() is not None


@traceable(run_type="llm", name="gemini_generate")
def generate(prompt: str, system: str = "", as_json: bool = False):
    """Single-shot generation. Returns str, or dict/list if as_json."""
    client = _get_client()
    if client is None:
        return _stub(prompt, as_json)
    full = (system + "\n\n" + prompt) if system else prompt
    try:
        resp = client.generate_content(full)
        text = resp.text
    except Exception as e:
        return _stub(prompt, as_json, err=str(e))
    if as_json:
        return _parse_json(text)
    return text.strip()


def _parse_json(text: str):
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"[\[{].*[\]}]", cleaned, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {} if "{" in text else []


# --- deterministic fallbacks (no key) --------------------------------------
def _stub(prompt: str, as_json: bool, err: str = ""):
    if as_json:
        return {} if "skills" not in prompt.lower() else {"skills": [], "exp_years": None}
    tag = f" (LLM unavailable{': ' + err if err else ''} — stub output)"
    return "[Gemini not configured] Set GEMINI_API_KEY to enable rich text." + tag
