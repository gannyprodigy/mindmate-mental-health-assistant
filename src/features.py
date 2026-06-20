"""Feature engineering: turn a user's stored activity into an ML feature row.

Both the segmenter and the stress classifier consume the same six features,
defined once here so the training script and the live application stay in
sync.
"""
from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional

from . import database as db

FEATURE_COLUMNS = [
    "avg_mood",
    "avg_sleep_hours",
    "phq9_score",
    "gad7_score",
    "avg_sentiment",
    "engagement",
]

# Neutral defaults used when a brand-new user has no history yet.
DEFAULTS: Dict[str, float] = {
    "avg_mood": 3.0,
    "avg_sleep_hours": 7.0,
    "phq9_score": 5.0,
    "gad7_score": 5.0,
    "avg_sentiment": 0.0,
    "engagement": 1.0,
}


def _safe_mean(values: List[float], default: float) -> float:
    cleaned = [v for v in values if v is not None]
    return float(mean(cleaned)) if cleaned else default


def build_feature_row(user_id: int) -> Dict[str, float]:
    """Aggregate a user's mood logs, screenings and chats into features."""
    moods = db.get_mood_logs(user_id)
    chats = db.get_chat_history(user_id)
    screens = db.get_screening_results(user_id)

    avg_mood = _safe_mean([m["mood_score"] for m in moods], DEFAULTS["avg_mood"])
    avg_sleep = _safe_mean([m["sleep_hours"] for m in moods], DEFAULTS["avg_sleep_hours"])

    mood_sent = [m["sentiment"] for m in moods if m.get("sentiment") is not None]
    chat_sent = [c["sentiment"] for c in chats if c.get("sentiment") is not None]
    avg_sentiment = _safe_mean(mood_sent + chat_sent, DEFAULTS["avg_sentiment"])

    # Most recent screening score per instrument (fall back to defaults).
    phq9 = next((s["total_score"] for s in reversed(screens)
                 if s["instrument"] == "PHQ-9"), DEFAULTS["phq9_score"])
    gad7 = next((s["total_score"] for s in reversed(screens)
                 if s["instrument"] == "GAD-7"), DEFAULTS["gad7_score"])

    # Engagement: number of user-authored chat + mood interactions.
    engagement = float(len([c for c in chats if c["role"] == "user"]) + len(moods))
    engagement = engagement or DEFAULTS["engagement"]

    return {
        "avg_mood": round(avg_mood, 3),
        "avg_sleep_hours": round(avg_sleep, 3),
        "phq9_score": float(phq9),
        "gad7_score": float(gad7),
        "avg_sentiment": round(avg_sentiment, 3),
        "engagement": float(engagement),
    }


def feature_vector(row: Dict[str, float]) -> List[float]:
    """Return the feature values in canonical column order."""
    return [row.get(col, DEFAULTS[col]) for col in FEATURE_COLUMNS]
