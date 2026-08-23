"""Symmetric encryption for secrets stored at rest (user LLM API keys).

Uses Fernet (AES-128-CBC + HMAC). The encryption key comes from
APP_ENCRYPTION_KEY (a Fernet key, or any string we hash into one). In dev it
falls back to a key derived from JWT_SECRET — set APP_ENCRYPTION_KEY explicitly
in production so keys survive restarts and are not tied to the JWT secret.
"""
import os
import base64
import hashlib
from cryptography.fernet import Fernet, InvalidToken


def _fernet_key() -> bytes:
    raw = os.environ.get("APP_ENCRYPTION_KEY")
    if raw:
        try:                       # already a valid Fernet key?
            Fernet(raw.encode() if isinstance(raw, str) else raw)
            return raw.encode() if isinstance(raw, str) else raw
        except Exception:          # derive one from an arbitrary secret
            return base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest())
    secret = os.environ.get("JWT_SECRET", "dev-only-change-me")
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())


def encrypt(plaintext: str) -> str:
    return Fernet(_fernet_key()).encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str | None:
    try:
        return Fernet(_fernet_key()).decrypt(ciphertext.encode()).decode()
    except (InvalidToken, Exception):
        return None


def mask(secret: str) -> str:
    """Show only the last 4 chars, e.g. '••••••••3f2a'."""
    if not secret:
        return ""
    tail = secret[-4:]
    return "•" * 8 + tail