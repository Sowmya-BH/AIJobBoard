import json
import urllib.request
import urllib.error
import os

from google import genai 

PROVIDERS = ("openai", "anthropic", "gemini", "groq", "custom", "test")
GROQ_BASE = "https://api.groq.com/openai/v1"

class LLMError(Exception):
    def __init__(self, kind, message):
        self.kind = kind          # invalid_key | quota_exceeded | provider_error | bad_provider
        self.message = message
        super().__init__(message)

def _http(method, url, headers=None, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req_headers = {
        "Accept": "application/json", 
        "User-Agent": "job-scout-agent/1.0",
        "Content-Type": "application/json"
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        status = e.code
        error_body = e.read().decode()
        print(f"DEBUG: API Error Body ({status}): {error_body}")
        try:
            payload = json.loads(error_body)
        except Exception:
            payload = {"error": str(e)}
        
        if status in (401, 403):
            raise LLMError("invalid_key", f"The API key was rejected (401/403). Details: {error_body[:200]}")
        if status == 429:
            raise LLMError("quota_exceeded", "Rate limit / quota exceeded (429).")
        raise LLMError("provider_error", f"Provider returned {status}: {str(payload)[:200]}")
    except urllib.error.URLError as e:
        raise LLMError("provider_error", f"Could not reach provider: {e}")

def _oai_chat(base, api_key, model, messages, max_out):
    hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last = None
    for pkey in ("max_completion_tokens", "max_tokens"):
        try:
            _, data = _http("POST", base + "/chat/completions", headers=hdr,
                            body={"model": model, "messages": messages, pkey: max_out})
            return data
        except LLMError as e:
            if e.kind == "provider_error" and pkey == "max_completion_tokens":
                last = e
                continue
            raise
    raise last

# ── validation ─────────────────────────────────────────────────────────────
def validate_key(provider, api_key, model="", base_url=""):
    api_key = (api_key or "").strip()
    try:
        if provider == "test":
            if not api_key.startswith("test-"):
                raise LLMError("invalid_key", "test keys must start with 'test-'")
            return True, "Test key accepted."

        if provider in ("openai", "groq", "custom"):
            base = ("https://api.openai.com/v1" if provider == "openai"
                    else GROQ_BASE if provider == "groq"
                    else (base_url or "").rstrip("/"))
            hdr = {"Authorization": f"Bearer {api_key}"}
            _http("GET", base + "/models", headers=hdr)
            return True, f"Key valid ({provider} models listed)."

        if provider == "gemini":
            # NATIVE ROUTE: Pass key as a query parameter
            # This is the most compatible way for both AIza and AQ keys.
            # https://generativelanguage.googleapis.com
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            
            _http("GET", url)
            return True, "Native Gemini key validated successfully."
        # if provider == "gemini":
        #     # Logic: Simple GET call to list models. Faster/cheaper than GenerateContent.
        #     url = "https://generativelanguage.googleapis.com/v1beta/models"
            
        #     # BRANCH LOGIC for AQ keys
        #     if api_key.startswith("AIza"):
        #         headers = {"x-goog-api-key": api_key}
        #     else:
        #         # New AQ keys often require the Bearer format
        #         headers = {"Authorization": f"Bearer {api_key}"}

        #     _http("GET", url, headers=headers)
        #     return True, "Gemini key validated successfully."
        # if provider == "gemini":
        #     url = (
        #         f"https://generativelanguage.googleapis.com/v1beta/models/"
        #         f"{model or 'gemini-3.6-flash'}:generateContent"
        #     )

        #     headers = {
        #         "Content-Type": "application/json",
        #         "x-goog-api-key": api_key,
        #     }

        #     body = {
        #         "contents": [
        #             {
        #                 "parts": [
        #                     {"text": "Reply with OK"}
        #                 ]
        #             }
        #         ]
        #     }

        #     try:
        #         _, data = _http(
        #             "POST",
        #             url,
        #             headers=headers,
        #             body=body
        #         )

        #         return True, "Gemini credential is valid."

        #     except LLMError as e:
        #         return False, e.message
        # if provider == "gemini":
        #     # Correct Validation Logic: Just check if we can see the models list
        #     url = "https://generativelanguage.googleapis.com/v1beta/models"
            
        #     # Try API Key Header (Standard AIza... keys)
        #     headers = {"x-goog-api-key": api_key}
        #     try:
        #         _http("GET", url, headers=headers)
        #         return True, "Key valid (Gemini models listed)."
        #     except LLMError:
        #         # Fallback for OAuth/Bearer (AQ.A... tokens)
        #         print("DEBUG: validate_key failed with x-goog-api-key, retrying with Bearer...")
        #         headers = {"Authorization": f"Bearer {api_key}"}
        #         _http("GET", url, headers=headers)
        #         return True, "Token valid (Authenticated via Bearer)."

        if provider == "anthropic":
            _http("POST", "https://api.anthropic.com/v1/messages",
                  headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                  body={"model": model or "claude-3-5-haiku-20241022",
                        "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
            return True, "Key valid (Anthropic responded)."

        return False, f"Unknown provider '{provider}'."
    except LLMError as e:
        return False, e.message

# ── chat ───────────────────────────────────────────────────────────────────
def chat(provider, api_key, model, system, message, history=None, base_url=""):
    api_key = (api_key or "").strip()
    history = history or []
    
    if provider == "test":
        return f"[test:{model or 'demo'}] You asked: {message[:200]}"

    if provider in ("openai", "groq", "custom"):
        base = ("https://api.openai.com/v1" if provider == "openai"
                else GROQ_BASE if provider == "groq"
                else (base_url or "").rstrip("/"))
        msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
        data = _oai_chat(base, api_key, model or "gpt-4o-mini", msgs, 1500)
        return data["choices"][0]["message"].get("content", "")

    if provider == "anthropic":
        _, data = _http("POST", "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                        body={"model": model or "claude-3-5-sonnet-20241022",
                              "max_tokens": 600, "system": system,
                              "messages": history + [{"role": "user", "content": message}]})
        return "".join(b.get("text", "") for b in data.get("content", []))

    if provider == "gemini":
        # Correct Chat Logic: Send the prompt to generateContent
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or 'gemini-1.5-flash'}:generateContent"
        
        if api_key.startswith("AIza"):
            headers = {"x-goog-api-key": api_key}
        else:
            headers = {"Authorization": f"Bearer {api_key}"}
            
        body = {
            "contents": [{"parts": [{"text": f"{system}\n\n{message}"}]}]
        }
        _, data = _http("POST", url, headers=headers, body=body)
        return data["candidates"][0]["content"]["parts"][0]["text"]

    raise LLMError("bad_provider", f"Unknown provider '{provider}'")
