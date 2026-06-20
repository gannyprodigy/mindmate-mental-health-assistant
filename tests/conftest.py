"""Shared pytest fixtures."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src import database as db


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    """Provide an isolated, initialised SQLite database for a test."""
    path = tmp_path / "test_mindmate.db"
    db.init_db(path)
    return path
