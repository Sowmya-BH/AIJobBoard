"""Thin LangSmith wrapper.

`@traceable(...)` becomes a real LangSmith span when the `langsmith` package is
installed AND LANGCHAIN_TRACING_V2 is truthy; otherwise it's a zero-overhead
identity decorator, so the app runs identically with or without tracing.

Enable tracing:
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=ls__...
    export LANGCHAIN_PROJECT=job-scout-agent   # optional

See TRACING.md for what every node/function emits.
"""
import os
import functools

_ENABLED = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes")
_real = None
if _ENABLED:
    try:
        from langsmith import traceable as _real
    except Exception:
        _real = None


def traceable(*d_args, **d_kwargs):
    """Usable as @traceable or @traceable(run_type=..., name=...)."""
    # bare @traceable
    if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
        fn = d_args[0]
        return _real(fn) if _real else fn

    def wrap(fn):
        if _real:
            return _real(*d_args, **d_kwargs)(fn)
        @functools.wraps(fn)
        def inner(*a, **k):
            return fn(*a, **k)
        return inner
    return wrap


def enabled() -> bool:
    return bool(_real)
