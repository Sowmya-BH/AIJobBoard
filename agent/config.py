"""Config. The job data now lives in SQLite (see agent/db.py, APP_DB); the large
source JSON / JSONL index are no longer used at runtime."""
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
