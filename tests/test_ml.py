"""Tests for the ML components: segmentation, stress, recommender."""
from __future__ import annotations

import numpy as np

from src.ml.personalization import FEATURE_COLUMNS, train_segmenter
from src.ml.recommender import RecommendationContext, recommend
from src.ml.stress import train_classifier


def _toy_matrix(n=120, seed=0):
    rng = np.random.default_rng(seed)
    # Two well-separated wellbeing groups.
    good = rng.normal([4.3, 8, 3, 3, 0.4, 6], 0.3, size=(n // 2, 6))
    poor = rng.normal([2.0, 5, 20, 16, -0.4, 1], 0.3, size=(n // 2, 6))
    return np.vstack([good, poor])


def test_segmenter_predicts_known_label():
    X = _toy_matrix()
    seg = train_segmenter(X, n_clusters=4)
    high = dict(zip(FEATURE_COLUMNS, [4.5, 8.5, 1, 1, 0.6, 8]))
    low = dict(zip(FEATURE_COLUMNS, [1.5, 4.5, 24, 19, -0.7, 0]))
    # A clearly-thriving profile should never be the worst segment.
    assert seg.predict(high) in {"Thriving", "Coping"}
    assert seg.predict(low) in {"At-Risk", "High-Need"}


def test_stress_classifier_learns_signal():
    rng = np.random.default_rng(1)
    X, y = [], []
    for _ in range(300):
        level = rng.choice(["Low", "Moderate", "High"])
        base = {"Low": (4.5, 8, 2, 2, 0.5),
                "Moderate": (3, 6.5, 10, 9, 0.0),
                "High": (1.8, 5, 22, 17, -0.5)}[level]
        X.append([base[0], base[1], base[2], base[3], base[4], rng.integers(0, 10)])
        y.append(level)
    clf = train_classifier(np.array(X), np.array(y))
    pred = clf.predict(dict(zip(
        ["avg_mood", "avg_sleep_hours", "phq9_score", "gad7_score",
         "avg_sentiment", "engagement"], [1.8, 5, 22, 17, -0.5, 3])))
    assert pred == "High"


def test_recommender_prioritises_emotion_match():
    ctx = RecommendationContext(emotion="anxiety", segment="At-Risk",
                                stress_level="High")
    recs = recommend(ctx, top_k=3)
    assert recs[0].score >= recs[-1].score
    # Top picks for anxious + high stress should include calming/grounding work.
    titles = {r.title for r in recs}
    assert titles & {"4-7-8 Breathing", "5-4-3-2-1 Grounding"}
