from __future__ import annotations

from database import execute, fetch_one


def create_user(username: str, password_hash: str) -> int:
    cursor = execute(
        """
        INSERT INTO users (username, password_hash)
        VALUES (?, ?)
        """,
        (username, password_hash),
    )
    return int(cursor.lastrowid)


def get_user_by_username(username: str):
    return fetch_one(
        """
        SELECT id, username, password_hash, created_at, updated_at, last_seen_at
        FROM users
        WHERE username = ?
        """,
        (username,),
    )


def get_user_by_session_token_hash(token_hash: str):
    return fetch_one(
        """
        SELECT
            users.id,
            users.username,
            users.created_at,
            users.updated_at,
            users.last_seen_at,
            sessions.id,
            sessions.expires_at
        FROM sessions
        INNER JOIN users ON users.id = sessions.user_id
        WHERE sessions.token_hash = ?
        """,
        (token_hash,),
    )


def create_session(user_id: int, token_hash: str, expires_at: str) -> int:
    cursor = execute(
        """
        INSERT INTO sessions (user_id, token_hash, expires_at)
        VALUES (?, ?, ?)
        """,
        (user_id, token_hash, expires_at),
    )
    return int(cursor.lastrowid)


def replace_session(old_token_hash: str, user_id: int, new_token_hash: str, expires_at: str) -> int:
    cursor = execute(
        """
        UPDATE sessions
        SET token_hash = ?,
            expires_at = ?,
            last_used_at = CURRENT_TIMESTAMP
        WHERE token_hash = ? AND user_id = ?
        """,
        (new_token_hash, expires_at, old_token_hash, user_id),
    )
    if cursor.rowcount < 1:
        raise ValueError("Session could not be rotated.")
    return int(cursor.rowcount)


def delete_session(token_hash: str) -> None:
    execute(
        """
        DELETE FROM sessions
        WHERE token_hash = ?
        """,
        (token_hash,),
    )


def delete_expired_sessions() -> None:
    execute(
        """
        DELETE FROM sessions
        WHERE datetime(expires_at) <= CURRENT_TIMESTAMP
        """
    )


def update_session_last_used(token_hash: str) -> None:
    execute(
        """
        UPDATE sessions
        SET last_used_at = CURRENT_TIMESTAMP
        WHERE token_hash = ?
        """,
        (token_hash,),
    )


def update_user_last_seen(user_id: int) -> None:
    execute(
        """
        UPDATE users
        SET last_seen_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (user_id,),
    )
