"""
JWT-based authentication (Improvement #10).

WHY THIS MATTERS: before this change, anyone who could guess or enumerate
a session/index ID (a UUID - hard to guess, but not access-controlled)
could read someone else's chat history and documents. There was no
concept of "whose data is this" anywhere in the system. This adds real
user accounts and ties every session/index to its owner.

DESIGN CHOICES:
- Password hashing uses the STANDARD LIBRARY (hashlib.pbkdf2_hmac with a
  random salt, industry-standard parameters) rather than adding a bcrypt
  dependency - this project already tries to keep its dependency
  footprint deliberately small, and pbkdf2_hmac is a legitimate, widely
  used, NIST-recommended choice for this.
- JWT (via PyJWT) for stateless auth tokens - no server-side session
  store needed, works cleanly with horizontal scaling (see the Redis
  cache and multi-instance considerations elsewhere in this project).
- Backward compatibility: existing sessions/indexes created before auth
  existed have user_id = NULL. Rather than break them, `get_current_user`
  is REQUIRED for creating new resources, but ownership checks on
  existing resources treat a NULL owner as "unowned/legacy, accessible
  to anyone" - a deliberate, documented tradeoff rather than silently
  locking users out of their pre-existing data. A future migration could
  backfill ownership if desired.
"""

import hashlib
import hmac
import logging
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Header, HTTPException

from rag_system.config import settings

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 600_000  # OWASP-recommended minimum as of 2023 for PBKDF2-HMAC-SHA256


# --- Password hashing ---


def hash_password(password: str) -> tuple[str, str]:
    """Returns (password_hash, salt), both hex-encoded strings suitable
    for storing in the database."""
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    )
    return hashed.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Constant-time comparison to avoid leaking timing information about
    how much of the hash matched (a real, if minor, security concern)."""
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(candidate.hex(), password_hash)


# --- JWT tokens ---


def create_access_token(user_id: str, email: str) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expiry_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """Raises jwt.InvalidTokenError (or a subclass, e.g. ExpiredSignatureError)
    if the token is invalid, expired, or tampered with."""
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


# --- FastAPI dependencies ---


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    """
    Required-auth dependency: raises 401 if there's no valid bearer
    token. Use this on routes that create or modify resources (e.g.
    creating a session) where an owner must be known.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token") from None

    return {"user_id": payload["sub"], "email": payload["email"]}


def get_current_user_optional(authorization: str | None = Header(default=None)) -> dict | None:
    """
    Optional-auth dependency: returns None instead of raising if there's
    no token, rather than blocking the request. Use this on routes that
    should remain accessible for legacy/unowned data (see module
    docstring) while still identifying the caller when a token IS
    provided, for ownership checks.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return get_current_user(authorization)
    except HTTPException:
        return None


def check_ownership(resource_user_id: str | None, current_user: dict | None) -> None:
    """
    Raises 403 if the resource has a known owner that doesn't match the
    current user. A resource with NO owner (user_id is None - created
    before auth existed) is treated as accessible to anyone, per the
    backward-compatibility policy documented at the top of this file.
    """
    if resource_user_id is None:
        return  # legacy/unowned resource - accessible to anyone
    if current_user is None or current_user["user_id"] != resource_user_id:
        raise HTTPException(status_code=403, detail="You do not have access to this resource")
