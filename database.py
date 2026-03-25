from __future__ import annotations

from threading import Lock

import libsql

from config import settings


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_seen_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token_hash TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at TEXT NOT NULL,
        last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        target_text TEXT NOT NULL,
        target_difficulty TEXT NOT NULL,
        normalized_target_text TEXT NOT NULL,
        overall_score REAL NOT NULL,
        performance_band TEXT NOT NULL,
        feedback_summary TEXT NOT NULL,
        feedback_action_items_json TEXT NOT NULL,
        feedback_encouragement TEXT NOT NULL,
        word_scores_json TEXT NOT NULL,
        phoneme_results_json TEXT NOT NULL,
        phoneme_model_transcript TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS phoneme_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        phoneme TEXT NOT NULL,
        total_occurrences INTEGER NOT NULL DEFAULT 0,
        weak_occurrences INTEGER NOT NULL DEFAULT 0,
        average_score REAL NOT NULL DEFAULT 0,
        average_severity_score REAL NOT NULL DEFAULT 0,
        recent_weighted_score REAL NOT NULL DEFAULT 0,
        common_error_types_json TEXT NOT NULL DEFAULT '[]',
        last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, phoneme),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS phoneme_attempt_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        phoneme TEXT NOT NULL,
        average_phoneme_score REAL NOT NULL,
        average_severity_score REAL NOT NULL,
        occurrence_count INTEGER NOT NULL DEFAULT 0,
        weak_occurrence_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (attempt_id) REFERENCES attempts(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash);",
    "CREATE INDEX IF NOT EXISTS idx_attempts_user_id_created_at ON attempts(user_id, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_phoneme_memory_user_id ON phoneme_memory(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_phoneme_attempt_summaries_user_id_phoneme_created_at ON phoneme_attempt_summaries(user_id, phoneme, created_at DESC);",
]


_database_connection = None
_database_lock = Lock()


def _build_connection():
    if not settings.turso_database_url:
        raise RuntimeError("TURSO_DATABASE_URL is not configured.")

    if not settings.turso_auth_token:
        raise RuntimeError("TURSO_AUTH_TOKEN is not configured.")

    return libsql.connect(
        database=settings.turso_database_url,
        auth_token=settings.turso_auth_token,
    )


def get_db_connection():
    global _database_connection

    if _database_connection is not None:
        return _database_connection

    with _database_lock:
        if _database_connection is None:
            _database_connection = _build_connection()

    return _database_connection


def reset_db_connection() -> None:
    global _database_connection

    with _database_lock:
        _database_connection = None


def _is_retryable_database_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "timed out" in message
        or "connection error" in message
        or "dns error" in message
        or "failed to lookup" in message
        or "nodename nor servname" in message
        or "temporarily unavailable" in message
        or "broken pipe" in message
    )


def _run_with_connection_retry(operation):
    try:
        return operation(get_db_connection())
    except Exception as error:
        if not _is_retryable_database_error(error):
            raise

        reset_db_connection()
        return operation(get_db_connection())


def init_database() -> None:
    def init_operation(connection):
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)

        connection.commit()

    _run_with_connection_retry(init_operation)


def ping_database() -> None:
    def operation(connection):
        connection.execute("SELECT 1")

    _run_with_connection_retry(operation)


def fetch_one(query: str, params: tuple = ()):
    def operation(connection):
        cursor = connection.execute(query, params)
        return cursor.fetchone()

    return _run_with_connection_retry(operation)


def fetch_all(query: str, params: tuple = ()):
    def operation(connection):
        cursor = connection.execute(query, params)
        return cursor.fetchall()

    return _run_with_connection_retry(operation)


def execute(query: str, params: tuple = ()):
    def operation(connection):
        cursor = connection.execute(query, params)
        connection.commit()
        return cursor

    return _run_with_connection_retry(operation)


def execute_many(query: str, params_seq: list[tuple] | tuple[tuple, ...]):
    def operation(connection):
        cursor = connection.executemany(query, params_seq)
        connection.commit()
        return cursor

    return _run_with_connection_retry(operation)
