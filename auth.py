from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import secrets
from threading import Lock
import re

from passlib.context import CryptContext

from config import settings


pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
)
USERNAME_PATTERN = re.compile(r"^[a-z0-9_]{3,32}$")
_sign_in_attempts: dict[str, list[float]] = {}
_sign_in_attempts_lock = Lock()


@dataclass
class SessionRecord:
    token: str
    token_hash: str
    expires_at: str


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def hash_session_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def create_session_record(days_valid: int = 30) -> SessionRecord:
    raw_token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=days_valid)

    return SessionRecord(
        token=raw_token,
        token_hash=hash_session_token(raw_token),
        expires_at=expires_at.isoformat(),
    )


def validate_username(username: str) -> str:
    normalized = username.strip().lower()

    if not USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Username must be 3 to 32 characters and contain only lowercase letters, numbers, or underscores."
        )

    return normalized


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must include at least one uppercase letter.")

    if not re.search(r"[a-z]", password):
        raise ValueError("Password must include at least one lowercase letter.")

    if not re.search(r"\d", password):
        raise ValueError("Password must include at least one number.")


def _prune_attempts(now_timestamp: float, attempt_timestamps: list[float]) -> list[float]:
    window_start = now_timestamp - settings.auth_rate_limit_window_seconds
    return [timestamp for timestamp in attempt_timestamps if timestamp >= window_start]


def enforce_sign_in_rate_limit(identifier: str) -> None:
    now_timestamp = datetime.now(timezone.utc).timestamp()

    with _sign_in_attempts_lock:
        pruned_attempts = _prune_attempts(now_timestamp, _sign_in_attempts.get(identifier, []))
        _sign_in_attempts[identifier] = pruned_attempts

        if len(pruned_attempts) >= settings.auth_rate_limit_attempts:
            raise ValueError("Too many sign-in attempts. Please wait a few minutes and try again.")


def record_failed_sign_in(identifier: str) -> None:
    now_timestamp = datetime.now(timezone.utc).timestamp()

    with _sign_in_attempts_lock:
        attempts = _prune_attempts(now_timestamp, _sign_in_attempts.get(identifier, []))
        attempts.append(now_timestamp)
        _sign_in_attempts[identifier] = attempts


def reset_sign_in_rate_limit(identifier: str) -> None:
    with _sign_in_attempts_lock:
        _sign_in_attempts.pop(identifier, None)
