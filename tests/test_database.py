"""Tests for the SQLite persistence layer."""
from __future__ import annotations

from src import database as db


def test_create_and_fetch_user(temp_db):
    uid = db.create_user("Test Student", 21, "MSc DS", 1, db_path=temp_db)
    user = db.get_user(uid, db_path=temp_db)
    assert user is not None
    assert user["name"] == "Test Student"
    assert user["course"] == "MSc DS"


def test_mood_log_roundtrip(temp_db):
    uid = db.create_user("M", db_path=temp_db)
    db.add_mood_log(uid, 4, 3, 7.5, "felt okay", 0.2, db_path=temp_db)
    logs = db.get_mood_logs(uid, db_path=temp_db)
    assert len(logs) == 1
    assert logs[0]["mood_score"] == 4
    assert logs[0]["sleep_hours"] == 7.5


def test_chat_history_order(temp_db):
    uid = db.create_user("C", db_path=temp_db)
    db.add_chat_message(uid, "user", "hello", 0.0, "none", db_path=temp_db)
    db.add_chat_message(uid, "assistant", "hi there", None, "none", db_path=temp_db)
    history = db.get_chat_history(uid, db_path=temp_db)
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_segment_update(temp_db):
    uid = db.create_user("S", db_path=temp_db)
    db.set_user_segment(uid, "At-Risk", db_path=temp_db)
    assert db.get_user(uid, db_path=temp_db)["segment"] == "At-Risk"


def test_screening_persisted(temp_db):
    uid = db.create_user("P", db_path=temp_db)
    db.add_screening_result(uid, "PHQ-9", 12, "Moderate", db_path=temp_db)
    results = db.get_screening_results(uid, db_path=temp_db)
    assert results[0]["instrument"] == "PHQ-9"
    assert results[0]["total_score"] == 12
