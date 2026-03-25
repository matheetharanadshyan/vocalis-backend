from __future__ import annotations

from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace

import pytest


def build_config_stub(*, attempts: int = 2, window_seconds: int = 60) -> ModuleType:
    config_module = ModuleType("config")
    config_module.settings = SimpleNamespace(
        auth_rate_limit_attempts=attempts,
        auth_rate_limit_window_seconds=window_seconds,
    )
    return config_module


def build_passlib_stubs() -> dict[str, ModuleType]:
    passlib_module = ModuleType("passlib")
    context_module = ModuleType("passlib.context")

    class FakeCryptContext:
        def __init__(self, *args, **kwargs) -> None:
            del args
            del kwargs

        def hash(self, password: str) -> str:
            return f"hashed::{password}"

        def verify(self, password: str, password_hash: str) -> bool:
            return password_hash == f"hashed::{password}"

    context_module.CryptContext = FakeCryptContext
    passlib_module.context = context_module
    return {
        "passlib": passlib_module,
        "passlib.context": context_module,
    }


def load_auth_module(load_backend_module, *, attempts: int = 2, window_seconds: int = 60):
    return load_backend_module(
        "auth",
        {
            "config": build_config_stub(attempts=attempts, window_seconds=window_seconds),
            **build_passlib_stubs(),
        },
    )


def test_validate_username_normalizes_and_rejects_invalid_values(load_backend_module) -> None:
    auth = load_auth_module(load_backend_module)

    assert auth.validate_username("  User_Name  ") == "user_name"

    with pytest.raises(ValueError, match="Username must be 3 to 32 characters"):
        auth.validate_username("bad-name")


def test_validate_password_enforces_strength_rules(load_backend_module) -> None:
    auth = load_auth_module(load_backend_module)

    auth.validate_password("StrongPass1")

    with pytest.raises(ValueError, match="uppercase"):
        auth.validate_password("lowercase1")

    with pytest.raises(ValueError, match="lowercase"):
        auth.validate_password("UPPERCASE1")

    with pytest.raises(ValueError, match="number"):
        auth.validate_password("MissingNumber")


def test_hash_password_and_verify_password_use_context(load_backend_module) -> None:
    auth = load_auth_module(load_backend_module)

    password_hash = auth.hash_password("SecretPass1")

    assert password_hash == "hashed::SecretPass1"
    assert auth.verify_password("SecretPass1", password_hash) is True
    assert auth.verify_password("WrongPass1", password_hash) is False


def test_create_session_record_hashes_token_and_sets_expiry(load_backend_module) -> None:
    auth = load_auth_module(load_backend_module)

    session_record = auth.create_session_record(days_valid=1)

    assert session_record.token
    assert session_record.token_hash == auth.hash_session_token(session_record.token)
    assert datetime.fromisoformat(session_record.expires_at) > datetime.now(timezone.utc)


def test_sign_in_rate_limit_blocks_until_reset(load_backend_module) -> None:
    auth = load_auth_module(load_backend_module, attempts=2)

    auth.record_failed_sign_in("learner")
    auth.record_failed_sign_in("learner")

    with pytest.raises(ValueError, match="Too many sign-in attempts"):
        auth.enforce_sign_in_rate_limit("learner")

    auth.reset_sign_in_rate_limit("learner")
    auth.enforce_sign_in_rate_limit("learner")
