from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest


def build_database_stub() -> tuple[ModuleType, list[tuple[str, tuple]]]:
    calls: list[tuple[str, tuple]] = []
    database_module = ModuleType("database")

    def execute(query: str, params: tuple = ()):
        calls.append((query, params))
        return SimpleNamespace(rowcount=1)

    def fetch_one(query: str, params: tuple = ()):
        calls.append((query, params))
        return None

    database_module.execute = execute
    database_module.fetch_one = fetch_one
    return database_module, calls


def test_replace_session_updates_existing_row(load_backend_module) -> None:
    database_stub, calls = build_database_stub()
    auth_repository = load_backend_module("auth_repository", {"database": database_stub})

    rowcount = auth_repository.replace_session("old-hash", 7, "new-hash", "2026-03-26T00:00:00+00:00")

    assert rowcount == 1
    assert len(calls) == 1
    query, params = calls[0]
    assert "UPDATE sessions" in query
    assert "SET token_hash = ?" in query
    assert "last_used_at = CURRENT_TIMESTAMP" in query
    assert params == ("new-hash", "2026-03-26T00:00:00+00:00", "old-hash", 7)


def test_replace_session_raises_when_no_session_was_rotated(load_backend_module) -> None:
    database_stub, _calls = build_database_stub()

    def execute(query: str, params: tuple = ()):
        del query
        del params
        return SimpleNamespace(rowcount=0)

    database_stub.execute = execute
    auth_repository = load_backend_module("auth_repository", {"database": database_stub})

    with pytest.raises(ValueError, match="Session could not be rotated"):
        auth_repository.replace_session("old-hash", 7, "new-hash", "2026-03-26T00:00:00+00:00")


def test_delete_expired_sessions_normalizes_timestamp_before_compare(load_backend_module) -> None:
    database_stub, calls = build_database_stub()
    auth_repository = load_backend_module("auth_repository", {"database": database_stub})

    auth_repository.delete_expired_sessions()

    assert len(calls) == 1
    query, params = calls[0]
    assert "datetime(expires_at) <= CURRENT_TIMESTAMP" in query
    assert params == ()


def test_update_session_last_used_touches_timestamp(load_backend_module) -> None:
    database_stub, calls = build_database_stub()
    auth_repository = load_backend_module("auth_repository", {"database": database_stub})

    auth_repository.update_session_last_used("token-hash")

    assert len(calls) == 1
    query, params = calls[0]
    assert "UPDATE sessions" in query
    assert "SET last_used_at = CURRENT_TIMESTAMP" in query
    assert params == ("token-hash",)
