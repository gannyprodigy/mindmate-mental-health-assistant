"""Central configuration for the MindMate application.

All tunable settings are read from environment variables (optionally loaded
from a local ``.env`` file) so that no secret is ever hard-coded in source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass

# Project root = parent of the ``src`` package.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models" / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings resolved from the environment."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip())
    db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MINDMATE_DB_PATH", str(DATA_DIR / "mindmate.db"))
        )
    )

    @property
    def llm_enabled(self) -> bool:
        """True when a non-empty API key is configured."""
        return bool(self.openai_api_key)


SETTINGS = Settings()

# Files produced by scripts/train_models.py
SENTIMENT_DATASET = DATA_DIR / "synthetic_students.csv"
SEGMENTER_PATH = MODEL_DIR / "student_segmenter.joblib"
STRESS_MODEL_PATH = MODEL_DIR / "stress_classifier.joblib"
EVALUATION_DATASET = DATA_DIR / "evaluation_study.csv"
