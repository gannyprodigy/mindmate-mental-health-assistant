"""Application service layer.

Sits between the Streamlit UI and the lower-level modules (database, ML
models, chat engine). Centralising this logic keeps the UI thin and makes
the core behaviour unit-testable without Streamlit.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from . import database as db
from . import features
from .config import SEGMENTER_PATH, STRESS_MODEL_PATH
from .ml.personalization import SEGMENT_STRATEGIES, StudentSegmenter
from .ml.recommender import Recommendation, recommend_from_signals
from .ml.stress import StressClassifier


@lru_cache(maxsize=1)
def get_segmenter() -> Optional[StudentSegmenter]:
    """Load the persisted segmenter, or None if it has not been trained."""
    if SEGMENTER_PATH.exists():
        return StudentSegmenter.load(SEGMENTER_PATH)
    return None


@lru_cache(maxsize=1)
def get_stress_classifier() -> Optional[StressClassifier]:
    if STRESS_MODEL_PATH.exists():
        return StressClassifier.load(STRESS_MODEL_PATH)
    return None


def models_available() -> bool:
    return get_segmenter() is not None and get_stress_classifier() is not None


def profile_for_user(user_id: int) -> Dict:
    """Compute the live ML profile (segment + stress) for a user."""
    row = features.build_feature_row(user_id)
    segmenter = get_segmenter()
    classifier = get_stress_classifier()

    segment = segmenter.predict(row) if segmenter else "Coping"
    if classifier:
        stress_level = classifier.predict(row)
        stress_proba = classifier.predict_proba(row)
    else:
        stress_level, stress_proba = "Moderate", {}

    return {
        "features": row,
        "segment": segment,
        "segment_strategy": SEGMENT_STRATEGIES.get(segment, ""),
        "stress_level": stress_level,
        "stress_proba": stress_proba,
    }


def wellness_score(user_id: int) -> int:
    """A single 0-100 wellbeing index for the dashboard headline metric."""
    row = features.build_feature_row(user_id)
    mood = (row["avg_mood"] / 5.0)
    sleep = min(row["avg_sleep_hours"], 9.0) / 9.0
    sentiment = (row["avg_sentiment"] + 1.0) / 2.0
    phq = 1.0 - (row["phq9_score"] / 27.0)
    gad = 1.0 - (row["gad7_score"] / 21.0)
    composite = (mood + sleep + sentiment + phq + gad) / 5.0
    return int(round(composite * 100))


def personalised_recommendations(user_id: int, emotion: str = "neutral",
                                 top_k: int = 3) -> List[Recommendation]:
    profile = profile_for_user(user_id)
    recs = recommend_from_signals(
        emotion=emotion,
        segment=profile["segment"],
        stress_level=profile["stress_level"],
        top_k=top_k,
    )
    for rec in recs:
        db.add_recommendation_log(
            user_id, rec.title, rec.category,
            context=f"emotion={emotion};segment={profile['segment']}",
        )
    return recs
