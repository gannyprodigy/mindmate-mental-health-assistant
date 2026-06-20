"""SQLite persistence layer for MindMate.

The schema is intentionally normalised so it can be documented as an
Entity-Relationship model in the project report:

    users (1) ──< (N) mood_logs
    users (1) ──< (N) chat_messages
    users (1) ──< (N) screening_results
    users (1) ──< (N) recommendation_logs

All access goes through small, well-named helper functions; the rest of the
application never writes raw SQL.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import SETTINGS

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    age             INTEGER,
    course          TEXT,
    year_of_study   INTEGER,
    segment         TEXT,                 -- assigned ML segment label
    created_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS mood_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    mood_score      INTEGER NOT NULL,      -- 1 (very low) .. 5 (very good)
    energy_score    INTEGER,               -- 1 .. 5
    sleep_hours     REAL,
    note            TEXT,
    sentiment       REAL,                  -- compound sentiment of the note
    logged_at       TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    role            TEXT    NOT NULL,       -- 'user' | 'assistant'
    content         TEXT    NOT NULL,
    sentiment       REAL,
    risk_level      TEXT,                   -- 'none' | 'elevated' | 'crisis'
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS screening_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    instrument      TEXT    NOT NULL,       -- 'PHQ-9' | 'GAD-7'
    total_score     INTEGER NOT NULL,
    severity        TEXT    NOT NULL,
    taken_at        TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendation_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    strategy        TEXT    NOT NULL,
    category        TEXT,
    context         TEXT,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Yield a SQLite connection with foreign keys enabled."""
    path = Path(db_path) if db_path else SETTINGS.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Create all tables if they do not already exist."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
def create_user(name: str, age: Optional[int] = None, course: Optional[str] = None,
                year_of_study: Optional[int] = None, db_path: Optional[Path] = None) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO users (name, age, course, year_of_study, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, age, course, year_of_study, _now()),
        )
        return int(cur.lastrowid)


def get_user(user_id: int, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def list_users(db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def set_user_segment(user_id: int, segment: str, db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("UPDATE users SET segment = ? WHERE id = ?", (segment, user_id))


# --------------------------------------------------------------------------- #
# Mood logs
# --------------------------------------------------------------------------- #
def add_mood_log(user_id: int, mood_score: int, energy_score: Optional[int] = None,
                 sleep_hours: Optional[float] = None, note: Optional[str] = None,
                 sentiment: Optional[float] = None, db_path: Optional[Path] = None) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO mood_logs (user_id, mood_score, energy_score, sleep_hours, "
            "note, sentiment, logged_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, mood_score, energy_score, sleep_hours, note, sentiment, _now()),
        )
        return int(cur.lastrowid)


def get_mood_logs(user_id: int, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM mood_logs WHERE user_id = ? ORDER BY logged_at ASC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Chat messages
# --------------------------------------------------------------------------- #
def add_chat_message(user_id: int, role: str, content: str,
                     sentiment: Optional[float] = None, risk_level: str = "none",
                     db_path: Optional[Path] = None) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages (user_id, role, content, sentiment, risk_level, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, role, content, sentiment, risk_level, _now()),
        )
        return int(cur.lastrowid)


def get_chat_history(user_id: int, limit: Optional[int] = None,
                     db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM chat_messages WHERE user_id = ? ORDER BY created_at ASC"
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, (user_id,)).fetchall()
        history = [dict(r) for r in rows]
        return history[-limit:] if limit else history


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
def add_screening_result(user_id: int, instrument: str, total_score: int,
                         severity: str, db_path: Optional[Path] = None) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO screening_results (user_id, instrument, total_score, severity, "
            "taken_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, instrument, total_score, severity, _now()),
        )
        return int(cur.lastrowid)


def get_screening_results(user_id: int, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM screening_results WHERE user_id = ? ORDER BY taken_at ASC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Recommendation logs
# --------------------------------------------------------------------------- #
def add_recommendation_log(user_id: int, strategy: str, category: Optional[str] = None,
                           context: Optional[str] = None, db_path: Optional[Path] = None) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO recommendation_logs (user_id, strategy, category, context, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, strategy, category, context, _now()),
        )
        return int(cur.lastrowid)


def get_recommendation_logs(user_id: int, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM recommendation_logs WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def execute_many(sql: str, rows: Iterable[tuple], db_path: Optional[Path] = None) -> None:
    """Utility for bulk inserts used by the data-seeding script."""
    with get_connection(db_path) as conn:
        conn.executemany(sql, rows)
