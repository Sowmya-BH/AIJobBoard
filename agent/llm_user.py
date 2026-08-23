"""BYO-key LLM adapter.

Users bring their own API key + model. We call the provider over raw HTTP (no
heavy SDKs) for two operations:
  * validate_key(...)  -> a lightweight check before saving (list models / tiny prompt)
  * chat(...)          -> answer a question grounded in the user's context

Errors are mapped to clean outcomes so a bad key or quota never crashes the
worker: 401/403 -> invalid_key, 429 -> quota_exceeded, others -> provider_error.

A built-in `test` provider (key must start with "test-") is included so the flow
can be exercised without network / real keys.
"""
import json
import urllib.request
import urllib.error

PROVIDERS = ("openai", "anthropic", "gemini", "groq", "custom", "test")

# "custom" = any OpenAI-compatible endpoint (Groq, Together, OpenRouter, Mistral,
# DeepSeek, Fireworks, Azure, local Ollama/LM Studio, ...) via a base_url.


GROQ_BASE = "https://api.groq.com/openai/v1"


class LLMError(Exception):
    def __init__(self, kind, message):
        self.kind = kind          # invalid_key | quota_exceeded | provider_error | bad_provider
        self.message = message
        super().__init__(message)


def _http(method, url, headers=None, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    # Some providers (Groq via Cloudflare) reject requests with no User-Agent,
    # which surfaced as spurious 401s — send a UA + Accept like a normal client.
    req_headers = {"Accept": "application/json", "User-Agent": "job-scout-agent/1.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = {"error": str(e)}
        if status in (401, 403):
            raise LLMError("invalid_key", "The API key was rejected (401/403).")
        if status == 429:
            raise LLMError("quota_exceeded", "Rate limit / quota exceeded (429).")
        raise LLMError("provider_error", f"Provider returned {status}: {str(payload)[:200]}")
    except urllib.error.URLError as e:
        raise LLMError("provider_error", f"Could not reach provider: {e}")


def _oai_chat(base, api_key, model, messages, max_out):
    """POST /chat/completions, tolerating both token-limit param names.
    gpt-oss / o-series need `max_completion_tokens`; older models want
    `max_tokens`. Try the new name, fall back on a 400."""
    hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last = None
    for pkey in ("max_completion_tokens", "max_tokens"):
        try:
            _, data = _http("POST", base + "/chat/completions", headers=hdr,
                            body={"model": model, "messages": messages, pkey: max_out})
            return data
        except LLMError as e:
            if e.kind == "provider_error" and pkey == "max_completion_tokens":
                last = e            # param not accepted -> try the older name
                continue
            raise
    raise last


# ── validation ─────────────────────────────────────────────────────────────
def validate_key(provider, api_key, model="", base_url=""):
    api_key = (api_key or "").strip()
    """Return (ok: bool, message: str). Never raises."""
    try:
        if provider == "openai" and api_key.startswith("gsk_"):
            return False, ("This looks like a Groq key (gsk_…). Set Provider = Groq, "
                           "not OpenAI.")
        if provider == "groq" and api_key.startswith("sk-") and not api_key.startswith("gsk_"):
            return False, ("This looks like an OpenAI key (sk-…). Set Provider = OpenAI, "
                           "not Groq.")
        if provider == "test":
            if not api_key.startswith("test-"):
                raise LLMError("invalid_key", "test keys must start with 'test-'")
            return True, "Test key accepted."
        if provider in ("openai", "groq", "custom"):
            base = ("https://api.openai.com/v1" if provider == "openai"
                    else GROQ_BASE if provider == "groq"
                    else (base_url or "").rstrip("/"))
            if provider == "custom" and not base:
                return False, "Custom provider needs a base URL (e.g. https://api.groq.com/openai/v1)."
            hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            if model:
                # mirror a real client call; tolerates max_tokens vs max_completion_tokens
                _oai_chat(base, api_key, model,
                          [{"role": "user", "content": "ping"}], 32)
                return True, f"Key valid ({provider} responded for {model})."
            # no model given -> just check the key against the models list
            _http("GET", base + "/models", headers={"Authorization": f"Bearer {api_key}"})
            return True, f"Key valid ({provider} models listed)."
        if provider == "gemini":
            _http("GET", f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
            return True, "Key valid (Gemini models listed)."
        if provider == "anthropic":
            # cheapest check: a 1-token message
            _http("POST", "https://api.anthropic.com/v1/messages",
                  headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                           "content-type": "application/json"},
                  body={"model": model or "claude-3-5-haiku-20241022",
                        "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
            return True, "Key valid (Anthropic responded)."
        return False, f"Unknown provider '{provider}'."
    except LLMError as e:
        return False, e.message


# ── chat ───────────────────────────────────────────────────────────────────
def chat(provider, api_key, model, system, message, history=None, base_url=""):
    api_key = (api_key or "").strip()
    """Return assistant text. Raises LLMError on failure."""
    history = history or []
    if provider == "test":
        return f"[test:{model or 'demo'}] You asked: {message[:200]}"
    if provider in ("openai", "groq", "custom"):
        base = ("https://api.openai.com/v1" if provider == "openai"
                else GROQ_BASE if provider == "groq"
                else (base_url or "").rstrip("/"))
        if provider == "custom" and not base:
            raise LLMError("bad_provider", "Custom provider needs a base URL.")
        msgs = [{"role": "system", "content": system}] + history + \
               [{"role": "user", "content": message}]
        data = _oai_chat(base, api_key,
                         model or ("llama-3.1-8b-instant" if provider == "groq" else "gpt-4o-mini"),
                         msgs, 1500)
        m = data["choices"][0]["message"]
        # reasoning models put the answer in content after thinking; if the budget
        # was consumed by reasoning, surface that so the reply is never blank.
        return m.get("content") or m.get("reasoning") or "(model returned no content — try a higher token budget)"
    if provider == "anthropic":
        _, data = _http("POST", "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                        body={"model": model or "claude-3-5-sonnet-20241022",
                              "max_tokens": 600, "system": system,
                              "messages": history + [{"role": "user", "content": message}]})
        return "".join(b.get("text", "") for b in data.get("content", []))
    if provider == "gemini":
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model or 'gemini-1.5-flash'}:generateContent?key={api_key}")
        _, data = _http("POST", url, headers={"Content-Type": "application/json"},
                        body={"contents": [{"parts": [{"text": system + "\n\n" + message}]}]})
        return data["candidates"][0]["content"]["parts"][0]["text"]
    raise LLMError("bad_provider", f"Unknown provider '{provider}'")

# """BYO-key LLM adapter.

# Users bring their own API key + model. We call the provider over raw HTTP (no
# heavy SDKs) for two operations:
#   * validate_key(...)  -> a lightweight check before saving (list models / tiny prompt)
#   * chat(...)          -> answer a question grounded in the user's context

# Errors are mapped to clean outcomes so a bad key or quota never crashes the
# worker: 401/403 -> invalid_key, 429 -> quota_exceeded, others -> provider_error.

# A built-in `test` provider (key must start with "test-") is included so the flow
# can be exercised without network / real keys.
# """
# import json
# import urllib.request
# import urllib.error

# PROVIDERS = ("openai", "anthropic", "gemini", "groq", "custom", "test")

# # "custom" = any OpenAI-compatible endpoint (Groq, Together, OpenRouter, Mistral,
# # DeepSeek, Fireworks, Azure, local Ollama/LM Studio, ...) via a base_url.


# GROQ_BASE = "https://api.groq.com/openai/v1"


# class LLMError(Exception):
#     def __init__(self, kind, message):
#         self.kind = kind          # invalid_key | quota_exceeded | provider_error | bad_provider
#         self.message = message
#         super().__init__(message)


# def _http(method, url, headers=None, body=None, timeout=60):
#     data = json.dumps(body).encode() if body is not None else None

#     req_headers = {
#         "Accept": "application/json",
#         "User-Agent": "job-scout-agent/1.0",
#     }

#     if headers:
#         req_headers.update(headers)

#     req = urllib.request.Request(
#         url,
#         data=data,
#         headers=req_headers,
#         method=method,
#     )

#     try:
#         with urllib.request.urlopen(req, timeout=timeout) as r:
#             return r.status, json.loads(r.read().decode())

#     except urllib.error.HTTPError as e:
#         status = e.code

#         try:
#             payload = json.loads(e.read().decode())
#         except Exception:
#             payload = {"error": str(e)}

#         if status in (401, 403):
#             raise LLMError(
#                 "invalid_key",
#                 f"Provider returned {status}: {str(payload)[:500]}"
#             )

#         if status == 429:
#             raise LLMError(
#                 "quota_exceeded",
#                 "Rate limit / quota exceeded (429)."
#             )

#         raise LLMError(
#             "provider_error",
#             f"Provider returned {status}: {str(payload)[:500]}"
#         )

#     except urllib.error.URLError as e:
#         raise LLMError(
#             "provider_error",
#             f"Could not reach provider: {e}"
#         )

# # def _http(method, url, headers=None, body=None, timeout=60):
# #     data = json.dumps(body).encode() if body is not None else None
# #     req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
# #     try:
# #         with urllib.request.urlopen(req, timeout=timeout) as r:
# #             return r.status, json.loads(r.read().decode())
# #     except urllib.error.HTTPError as e:
# #         status = e.code
# #         try:
# #             payload = json.loads(e.read().decode())
# #         except Exception:
# #             payload = {"error": str(e)}
# #         if status in (401, 403):
# #             raise LLMError("invalid_key",
# #             f"Provider returned {status}: {str(payload)[:500]}"
# #             )
# #         # if status in (401, 403):
# #             # raise LLMError("invalid_key", "The API key was rejected (401/403).")
# #         if status == 429:
# #             raise LLMError("quota_exceeded", "Rate limit / quota exceeded (429).")
# #         raise LLMError("provider_error", f"Provider returned {status}: {str(payload)[:200]}")
# #     except urllib.error.URLError as e:
# #         raise LLMError("provider_error", f"Could not reach provider: {e}")


# def _oai_chat(base, api_key, model, messages, max_out):
#     """Call an OpenAI-compatible /chat/completions endpoint."""
#     hdr = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json",
#     }

#     _, data = _http(
#         "POST",
#         base + "/chat/completions",
#         headers=hdr,
#         body={
#             "model": model,
#             "messages": messages,
#             "max_tokens": max_out,
#         },
#     )

#     return data

# # def _oai_chat(base, api_key, model, messages, max_out):
# #     """POST /chat/completions, tolerating both token-limit param names.
# #     gpt-oss / o-series need `max_completion_tokens`; older models want
# #     `max_tokens`. Try the new name, fall back on a 400."""
# #     hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
# #     last = None
# #     for pkey in ("max_completion_tokens", "max_tokens"):
# #         try:
# #             _, data = _http("POST", base + "/chat/completions", headers=hdr,
# #                             body={"model": model, "messages": messages, pkey: max_out})
# #             return data
# #         except LLMError as e:
# #             if e.kind == "provider_error" and pkey == "max_completion_tokens":
# #                 last = e            # param not accepted -> try the older name
# #                 continue
# #             raise
# #     raise last


# # ── validation ─────────────────────────────────────────────────────────────
# def validate_key(provider, api_key, model="", base_url=""):
#     api_key = (api_key or "").strip()
#     """Return (ok: bool, message: str). Never raises."""
#     try:
#         if provider == "test":
#             if not api_key.startswith("test-"):
#                 raise LLMError("invalid_key", "test keys must start with 'test-'")
#             return True, "Test key accepted."
#         if provider in ("openai", "groq", "custom"):
#             base = ("https://api.openai.com/v1" if provider == "openai"
#                     else GROQ_BASE if provider == "groq"
#                     else (base_url or "").rstrip("/"))
#             if provider == "custom" and not base:
#                 return False, "Custom provider needs a base URL (e.g. https://api.groq.com/openai/v1)."
#             hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#             if model:
#                 # mirror a real client call; tolerates max_tokens vs max_completion_tokens
#                 _oai_chat(base, api_key, model,
#                           [{"role": "user", "content": "ping"}], 32)
#                 return True, f"Key valid ({provider} responded for {model})."
#             # no model given -> just check the key against the models list
#             _http("GET", base + "/models", headers={"Authorization": f"Bearer {api_key}"})
#             return True, f"Key valid ({provider} models listed)."
#         if provider == "gemini":
#             _http("GET", f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
#             return True, "Key valid (Gemini models listed)."
#         if provider == "anthropic":
#             # cheapest check: a 1-token message
#             _http("POST", "https://api.anthropic.com/v1/messages",
#                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
#                            "content-type": "application/json"},
#                   body={"model": model or "claude-3-5-haiku-20241022",
#                         "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
#             return True, "Key valid (Anthropic responded)."
#         return False, f"Unknown provider '{provider}'."
#     except LLMError as e:
#         return False, e.message


# # ── chat ───────────────────────────────────────────────────────────────────
# def chat(provider, api_key, model, system, message, history=None, base_url=""):
#     api_key = (api_key or "").strip()
#     """Return assistant text. Raises LLMError on failure."""
#     history = history or []
#     if provider == "test":
#         return f"[test:{model or 'demo'}] You asked: {message[:200]}"
#     if provider in ("openai", "groq", "custom"):
#         base = ("https://api.openai.com/v1" if provider == "openai"
#                 else GROQ_BASE if provider == "groq"
#                 else (base_url or "").rstrip("/"))
#         if provider == "custom" and not base:
#             raise LLMError("bad_provider", "Custom provider needs a base URL.")
#         msgs = [{"role": "system", "content": system}] + history + \
#                [{"role": "user", "content": message}]
#         data = _oai_chat(base, api_key,
#                          model or ("llama-3.1-8b-instant" if provider == "groq" else "gpt-4o-mini"),
#                          msgs, 1500)
#         m = data["choices"][0]["message"]
#         # reasoning models put the answer in content after thinking; if the budget
#         # was consumed by reasoning, surface that so the reply is never blank.
#         return m.get("content") or m.get("reasoning") or "(model returned no content — try a higher token budget)"
#     if provider == "anthropic":
#         _, data = _http("POST", "https://api.anthropic.com/v1/messages",
#                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
#                                  "content-type": "application/json"},
#                         body={"model": model or "claude-3-5-sonnet-20241022",
#                               "max_tokens": 600, "system": system,
#                               "messages": history + [{"role": "user", "content": message}]})
#         return "".join(b.get("text", "") for b in data.get("content", []))
#     if provider == "gemini":
#         url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
#                f"{model or 'gemini-1.5-flash'}:generateContent?key={api_key}")
#         _, data = _http("POST", url, headers={"Content-Type": "application/json"},
#                         body={"contents": [{"parts": [{"text": system + "\n\n" + message}]}]})
#         return data["candidates"][0]["content"]["parts"][0]["text"]
#     raise LLMError("bad_provider", f"Unknown provider '{provider}'")

# # """BYO-key LLM adapter.

# # Users bring their own API key + model. We call the provider over raw HTTP (no
# # heavy SDKs) for two operations:
# #   * validate_key(...)  -> a lightweight check before saving (list models / tiny prompt)
# #   * chat(...)          -> answer a question grounded in the user's context

# # Errors are mapped to clean outcomes so a bad key or quota never crashes the
# # worker: 401/403 -> invalid_key, 429 -> quota_exceeded, others -> provider_error.

# # A built-in `test` provider (key must start with "test-") is included so the flow
# # can be exercised without network / real keys.
# # """
# # import json
# # import urllib.request
# # import urllib.error

# # PROVIDERS = ("openai", "anthropic", "gemini", "groq", "custom", "test")

# # # "custom" = any OpenAI-compatible endpoint (Groq, Together, OpenRouter, Mistral,
# # # DeepSeek, Fireworks, Azure, local Ollama/LM Studio, ...) via a base_url.


# # GROQ_BASE = "https://api.groq.com/openai/v1"


# # class LLMError(Exception):
# #     def __init__(self, kind, message):
# #         self.kind = kind          # invalid_key | quota_exceeded | provider_error | bad_provider
# #         self.message = message
# #         super().__init__(message)


# # def _http(method, url, headers=None, body=None, timeout=60):
# #     data = json.dumps(body).encode() if body is not None else None
# #     req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
# #     try:
# #         with urllib.request.urlopen(req, timeout=timeout) as r:
# #             return r.status, json.loads(r.read().decode())
# #     except urllib.error.HTTPError as e:
# #         status = e.code
# #         try:
# #             payload = json.loads(e.read().decode())
# #         except Exception:
# #             payload = {"error": str(e)}
# #         if status in (401, 403):
# #             raise LLMError("invalid_key", "The API key was rejected (401/403).")
# #         if status == 429:
# #             raise LLMError("quota_exceeded", "Rate limit / quota exceeded (429).")
# #         raise LLMError("provider_error", f"Provider returned {status}: {str(payload)[:200]}")
# #     except urllib.error.URLError as e:
# #         raise LLMError("provider_error", f"Could not reach provider: {e}")


# # # ── validation ─────────────────────────────────────────────────────────────
# # def validate_key(provider, api_key, model="", base_url=""):
# #     api_key = (api_key or "").strip()
# #     """Return (ok: bool, message: str). Never raises."""
# #     try:
# #         if provider == "test":
# #             if not api_key.startswith("test-"):
# #                 raise LLMError("invalid_key", "test keys must start with 'test-'")
# #             return True, "Test key accepted."
# #         if provider in ("openai", "groq", "custom"):
# #             base = ("https://api.openai.com/v1" if provider == "openai"
# #                     else GROQ_BASE if provider == "groq"
# #                     else (base_url or "").rstrip("/"))
# #             if provider == "custom" and not base:
# #                 return False, "Custom provider needs a base URL (e.g. https://api.groq.com/openai/v1)."
# #             hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
# #             if model:
# #                 # mirror exactly what a working client does: a real chat completion
# #                 # with the user's model (avoids /models and default-model quirks).
# #                 _http("POST", base + "/chat/completions", headers=hdr,
# #                       body={"model": model, "max_tokens": 32,
# #                             "messages": [{"role": "user", "content": "ping"}]})
# #                 return True, f"Key valid ({provider} responded for {model})."
# #             # no model given -> just check the key against the models list
# #             _http("GET", base + "/models", headers={"Authorization": f"Bearer {api_key}"})
# #             return True, f"Key valid ({provider} models listed)."
# #         if provider == "gemini":
# #             _http("GET", f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
# #             return True, "Key valid (Gemini models listed)."
# #         if provider == "anthropic":
# #             # cheapest check: a 1-token message
# #             _http("POST", "https://api.anthropic.com/v1/messages",
# #                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
# #                            "content-type": "application/json"},
# #                   body={"model": model or "claude-3-5-haiku-20241022",
# #                         "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
# #             return True, "Key valid (Anthropic responded)."
# #         return False, f"Unknown provider '{provider}'."
# #     except LLMError as e:
# #         return False, e.message


# # # ── chat ───────────────────────────────────────────────────────────────────
# # def chat(provider, api_key, model, system, message, history=None, base_url=""):
# #     api_key = (api_key or "").strip()
# #     """Return assistant text. Raises LLMError on failure."""
# #     history = history or []
# #     if provider == "test":
# #         return f"[test:{model or 'demo'}] You asked: {message[:200]}"
# #     if provider in ("openai", "groq", "custom"):
# #         base = ("https://api.openai.com/v1" if provider == "openai"
# #                 else GROQ_BASE if provider == "groq"
# #                 else (base_url or "").rstrip("/"))
# #         if provider == "custom" and not base:
# #             raise LLMError("bad_provider", "Custom provider needs a base URL.")
# #         msgs = [{"role": "system", "content": system}] + history + \
# #                [{"role": "user", "content": message}]
# #         _, data = _http("POST", base + "/chat/completions",
# #                         headers={"Authorization": f"Bearer {api_key}",
# #                                  "Content-Type": "application/json"},
# #                         body={"model": model or ("llama-3.1-8b-instant" if provider == "groq" else "gpt-4o-mini"),
# #                               "messages": msgs, "max_tokens": 600})
# #         m = data["choices"][0]["message"]
# #         return m.get("content") or m.get("reasoning") or ""
# #     if provider == "anthropic":
# #         _, data = _http("POST", "https://api.anthropic.com/v1/messages",
# #                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
# #                                  "content-type": "application/json"},
# #                         body={"model": model or "claude-3-5-sonnet-20241022",
# #                               "max_tokens": 600, "system": system,
# #                               "messages": history + [{"role": "user", "content": message}]})
# #         return "".join(b.get("text", "") for b in data.get("content", []))
# #     if provider == "gemini":
# #         url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
# #                f"{model or 'gemini-1.5-flash'}:generateContent?key={api_key}")
# #         _, data = _http("POST", url, headers={"Content-Type": "application/json"},
# #                         body={"contents": [{"parts": [{"text": system + "\n\n" + message}]}]})
# #         return data["candidates"][0]["content"]["parts"][0]["text"]
# #     raise LLMError("bad_provider", f"Unknown provider '{provider}'")



# # # """BYO-key LLM adapter.

# # # Users bring their own API key + model. We call the provider over raw HTTP (no
# # # heavy SDKs) for two operations:
# # #   * validate_key(...)  -> a lightweight check before saving (list models / tiny prompt)
# # #   * chat(...)          -> answer a question grounded in the user's context

# # # Errors are mapped to clean outcomes so a bad key or quota never crashes the
# # # worker: 401/403 -> invalid_key, 429 -> quota_exceeded, others -> provider_error.

# # # A built-in `test` provider (key must start with "test-") is included so the flow
# # # can be exercised without network / real keys.
# # # """
# # # import json
# # # import urllib.request
# # # import urllib.error

# # # PROVIDERS = ("openai", "anthropic", "gemini", "groq", "custom", "test")

# # # # "custom" = any OpenAI-compatible endpoint (Groq, Together, OpenRouter, Mistral,
# # # # DeepSeek, Fireworks, Azure, local Ollama/LM Studio, ...) via a base_url.


# # # GROQ_BASE = "https://api.groq.com/openai/v1"


# # # class LLMError(Exception):
# # #     def __init__(self, kind, message):
# # #         self.kind = kind          # invalid_key | quota_exceeded | provider_error | bad_provider
# # #         self.message = message
# # #         super().__init__(message)


# # # def _http(method, url, headers=None, body=None, timeout=25):
# # #     data = json.dumps(body).encode() if body is not None else None
# # #     req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
# # #     try:
# # #         with urllib.request.urlopen(req, timeout=timeout) as r:
# # #             return r.status, json.loads(r.read().decode())
# # #     except urllib.error.HTTPError as e:
# # #         status = e.code
# # #         try:
# # #             payload = json.loads(e.read().decode())
# # #         except Exception:
# # #             payload = {"error": str(e)}
# # #         if status in (401, 403):
# # #             raise LLMError("invalid_key", "The API key was rejected (401/403).")
# # #         if status == 429:
# # #             raise LLMError("quota_exceeded", "Rate limit / quota exceeded (429).")
# # #         raise LLMError("provider_error", f"Provider returned {status}: {str(payload)[:200]}")
# # #     except urllib.error.URLError as e:
# # #         raise LLMError("provider_error", f"Could not reach provider: {e}")


# # # # ── validation ─────────────────────────────────────────────────────────────
# # # def validate_key(provider, api_key, model="", base_url=""):
# # #     """Return (ok: bool, message: str). Never raises."""
# # #     try:
# # #         if provider == "test":
# # #             if not api_key.startswith("test-"):
# # #                 raise LLMError("invalid_key", "test keys must start with 'test-'")
# # #             return True, "Test key accepted."
# # #         if provider == "openai":
# # #             _http("GET", "https://api.openai.com/v1/models",
# # #                   headers={"Authorization": f"Bearer {api_key}"})
# # #             return True, "Key valid (OpenAI models listed)."
# # #         if provider in ("groq", "custom"):
# # #             base = GROQ_BASE if provider == "groq" else (base_url or "").rstrip("/")
# # #             if not base:
# # #                 return False, "Custom provider needs a base URL (e.g. https://api.groq.com/openai/v1)."
# # #             _http("POST", base + "/chat/completions",
# # #                   headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
# # #                   body={"model": model or ("llama-3.1-8b-instant" if provider == "groq" else "gpt-3.5-turbo"),
# # #                         "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
# # #             return True, f"Key valid ({provider} responded)."
# # #         if provider == "gemini":
# # #             _http("GET", f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
# # #             return True, "Key valid (Gemini models listed)."
# # #         if provider == "anthropic":
# # #             # cheapest check: a 1-token message
# # #             _http("POST", "https://api.anthropic.com/v1/messages",
# # #                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
# # #                            "content-type": "application/json"},
# # #                   body={"model": model or "claude-3-5-haiku-20241022",
# # #                         "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
# # #             return True, "Key valid (Anthropic responded)."
# # #         return False, f"Unknown provider '{provider}'."
# # #     except LLMError as e:
# # #         return False, e.message


# # # # ── chat ───────────────────────────────────────────────────────────────────
# # # def chat(provider, api_key, model, system, message, history=None, base_url=""):
# # #     """Return assistant text. Raises LLMError on failure."""
# # #     history = history or []
# # #     if provider == "test":
# # #         return f"[test:{model or 'demo'}] You asked: {message[:200]}"
# # #     if provider in ("openai", "groq", "custom"):
# # #         base = ("https://api.openai.com/v1" if provider == "openai"
# # #                 else GROQ_BASE if provider == "groq"
# # #                 else (base_url or "").rstrip("/"))
# # #         if provider == "custom" and not base:
# # #             raise LLMError("bad_provider", "Custom provider needs a base URL.")
# # #         msgs = [{"role": "system", "content": system}] + history + \
# # #                [{"role": "user", "content": message}]
# # #         _, data = _http("POST", base + "/chat/completions",
# # #                         headers={"Authorization": f"Bearer {api_key}",
# # #                                  "Content-Type": "application/json"},
# # #                         body={"model": model or ("llama-3.1-8b-instant" if provider == "groq" else "gpt-4o-mini"),
# # #                               "messages": msgs, "max_tokens": 600})
# # #         return data["choices"][0]["message"]["content"]
# # #     if provider == "anthropic":
# # #         _, data = _http("POST", "https://api.anthropic.com/v1/messages",
# # #                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
# # #                                  "content-type": "application/json"},
# # #                         body={"model": model or "claude-3-5-sonnet-20241022",
# # #                               "max_tokens": 600, "system": system,
# # #                               "messages": history + [{"role": "user", "content": message}]})
# # #         return "".join(b.get("text", "") for b in data.get("content", []))
# # #     if provider == "gemini":
# # #         url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
# # #                f"{model or 'gemini-1.5-flash'}:generateContent?key={api_key}")
# # #         _, data = _http("POST", url, headers={"Content-Type": "application/json"},
# # #                         body={"contents": [{"parts": [{"text": system + "\n\n" + message}]}]})
# # #         return data["candidates"][0]["content"]["parts"][0]["text"]
# # #     raise LLMError("bad_provider", f"Unknown provider '{provider}'")

# # # # """BYO-key LLM adapter.

# # # # Users bring their own API key + model. We call the provider over raw HTTP (no
# # # # heavy SDKs) for two operations:
# # # #   * validate_key(...)  -> a lightweight check before saving (list models / tiny prompt)
# # # #   * chat(...)          -> answer a question grounded in the user's context

# # # # Errors are mapped to clean outcomes so a bad key or quota never crashes the
# # # # worker: 401/403 -> invalid_key, 429 -> quota_exceeded, others -> provider_error.

# # # # A built-in `test` provider (key must start with "test-") is included so the flow
# # # # can be exercised without network / real keys.
# # # # """
# # # # import json
# # # # import urllib.request
# # # # import urllib.error

# # # # PROVIDERS = ("openai", "anthropic", "gemini", "groq", "custom", "test")

# # # # # "custom" = any OpenAI-compatible endpoint (Groq, Together, OpenRouter, Mistral,
# # # # # DeepSeek, Fireworks, Azure, local Ollama/LM Studio, ...) via a base_url.


# # # # GROQ_BASE = "https://api.groq.com/openai/v1"


# # # # class LLMError(Exception):
# # # #     def __init__(self, kind, message):
# # # #         self.kind = kind          # invalid_key | quota_exceeded | provider_error | bad_provider
# # # #         self.message = message
# # # #         super().__init__(message)


# # # # def _http(method, url, headers=None, body=None, timeout=25):
# # # #     data = json.dumps(body).encode() if body is not None else None
# # # #     req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
# # # #     try:
# # # #         with urllib.request.urlopen(req, timeout=timeout) as r:
# # # #             return r.status, json.loads(r.read().decode())
# # # #     except urllib.error.HTTPError as e:
# # # #         status = e.code
# # # #         try:
# # # #             payload = json.loads(e.read().decode())
# # # #         except Exception:
# # # #             payload = {"error": str(e)}
# # # #         if status in (401, 403):
# # # #             raise LLMError("invalid_key", "The API key was rejected (401/403).")
# # # #         if status == 429:
# # # #             raise LLMError("quota_exceeded", "Rate limit / quota exceeded (429).")
# # # #         raise LLMError("provider_error", f"Provider returned {status}: {str(payload)[:200]}")
# # # #     except urllib.error.URLError as e:
# # # #         raise LLMError("provider_error", f"Could not reach provider: {e}")


# # # # # ── validation ─────────────────────────────────────────────────────────────
# # # # def validate_key(provider, api_key, model="", base_url=""):
# # # #     """Return (ok: bool, message: str). Never raises."""
# # # #     try:
# # # #         if provider == "test":
# # # #             if not api_key.startswith("test-"):
# # # #                 raise LLMError("invalid_key", "test keys must start with 'test-'")
# # # #             return True, "Test key accepted."
# # # #         if provider == "openai":
# # # #             _http("GET", "https://api.openai.com/v1/models",
# # # #                   headers={"Authorization": f"Bearer {api_key}"})
# # # #             return True, "Key valid (OpenAI models listed)."
# # # #         if provider in ("groq", "custom"):
# # # #             base = GROQ_BASE if provider == "groq" else (base_url or "").rstrip("/")
# # # #             if not base:
# # # #                 return False, "Custom provider needs a base URL (e.g. https://api.groq.com/openai/v1)."
# # # #             _http("POST", base + "/chat/completions",
# # # #                   headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
# # # #                   body={"model": model or ("llama-3.1-8b-instant" if provider == "groq" else "gpt-3.5-turbo"),
# # # #                         "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
# # # #             return True, f"Key valid ({provider} responded)."
# # # #         if provider == "gemini":
# # # #             _http("GET", f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
# # # #             return True, "Key valid (Gemini models listed)."
# # # #         if provider == "anthropic":
# # # #             # cheapest check: a 1-token message
# # # #             _http("POST", "https://api.anthropic.com/v1/messages",
# # # #                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
# # # #                            "content-type": "application/json"},
# # # #                   body={"model": model or "claude-3-5-haiku-20241022",
# # # #                         "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]})
# # # #             return True, "Key valid (Anthropic responded)."
# # # #         return False, f"Unknown provider '{provider}'."
# # # #     except LLMError as e:
# # # #         return False, e.message


# # # # # ── chat ───────────────────────────────────────────────────────────────────
# # # # def chat(provider, api_key, model, system, message, history=None, base_url=""):
# # # #     """Return assistant text. Raises LLMError on failure."""
# # # #     history = history or []
# # # #     if provider == "test":
# # # #         return f"[test:{model or 'demo'}] You asked: {message[:200]}"
# # # #     if provider in ("openai", "groq", "custom"):
# # # #         base = ("https://api.openai.com/v1" if provider == "openai"
# # # #                 else GROQ_BASE if provider == "groq"
# # # #                 else (base_url or "").rstrip("/"))
# # # #         if provider == "custom" and not base:
# # # #             raise LLMError("bad_provider", "Custom provider needs a base URL.")
# # # #         msgs = [{"role": "system", "content": system}] + history + \
# # # #                [{"role": "user", "content": message}]
# # # #         _, data = _http("POST", base + "/chat/completions",
# # # #                         headers={"Authorization": f"Bearer {api_key}",
# # # #                                  "Content-Type": "application/json"},
# # # #                         body={"model": model or ("llama-3.1-8b-instant" if provider == "groq" else "gpt-4o-mini"),
# # # #                               "messages": msgs, "max_tokens": 600})
# # # #         return data["choices"][0]["message"]["content"]
# # # #     if provider == "anthropic":
# # # #         _, data = _http("POST", "https://api.anthropic.com/v1/messages",
# # # #                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
# # # #                                  "content-type": "application/json"},
# # # #                         body={"model": model or "claude-3-5-sonnet-20241022",
# # # #                               "max_tokens": 600, "system": system,
# # # #                               "messages": history + [{"role": "user", "content": message}]})
# # # #         return "".join(b.get("text", "") for b in data.get("content", []))
# # # #     if provider == "gemini":
# # # #         url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
# # # #                f"{model or 'gemini-1.5-flash'}:generateContent?key={api_key}")
# # # #         _, data = _http("POST", url, headers={"Content-Type": "application/json"},
# # # #                         body={"contents": [{"parts": [{"text": system + "\n\n" + message}]}]})
# # # #         return data["candidates"][0]["content"]["parts"][0]["text"]
# # # #     raise LLMError("bad_provider", f"Unknown provider '{provider}'")