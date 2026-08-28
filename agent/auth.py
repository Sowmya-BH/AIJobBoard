"""Auth: secure password hashing (bcrypt) + JWT. Session-less, Render-friendly.

OAuth (Google/GitHub) routes are scaffolded in server.py and need client
credentials to function — see README.
"""
import os
import time
import bcrypt
import jwt          # PyJWT
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from . import userstore

SECRET = os.environ.get("JWT_SECRET", "dev-only-change-me")
ALGO = "HS256"
TTL = int(os.environ.get("JWT_TTL_SECONDS", str(60 * 60 * 24 * 7)))  # 7 days

oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def hash_password(pw: str) -> str:
    # bcrypt caps at 72 bytes; truncate deterministically.
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode("utf-8"))
    except ValueError:
        return False


def make_token(user_id) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(user_id), "iat": now, "exp": now + TTL}, SECRET, algorithm=ALGO)


def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGO])
    except jwt.PyJWTError:
        return None


def current_user(token: str = Depends(oauth2)):
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    user = userstore.get_user(payload["sub"])
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user
