"""Hybrid coping-strategy recommender.

The recommender personalises which wellbeing strategies a student sees by
combining three signals:

    1. *Emotion*, the dominant emotion detected in the student's recent text
       (from :mod:`src.ml.sentiment`).
    2. *Segment*, the student's behavioural cluster
       (from :mod:`src.ml.personalization`).
    3. *Stress level*, the predicted stress class
       (from :mod:`src.ml.stress`).

Strategies are scored against these signals and the top items are returned.
This "content-based + rule-weighted" design is transparent (every score can
be explained), which is important in a wellbeing context where black-box
recommendations would be inappropriate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..content import COPING_STRATEGIES

# How strongly each stress level biases towards calming/grounding content.
_STRESS_WEIGHT = {"Low": 0.0, "Moderate": 0.5, "High": 1.0}

# Categories that should be boosted when a student is highly stressed.
_DEESCALATION_CATEGORIES = {"Calming", "Grounding"}

# Segment-specific category preferences.
_SEGMENT_PREFERENCES: Dict[str, set] = {
    "Thriving": {"Academic", "Mood"},
    "Coping": {"Cognitive", "Academic", "Sleep"},
    "At-Risk": {"Grounding", "Cognitive", "Mood"},
    "High-Need": {"Calming", "Grounding", "Mood"},
}


@dataclass
class Recommendation:
    title: str
    category: str
    steps: List[str]
    score: float
    rationale: str = ""


@dataclass
class RecommendationContext:
    emotion: str = "neutral"
    segment: str = "Coping"
    stress_level: str = "Moderate"
    extra: Dict = field(default_factory=dict)


def _score_strategy(strategy: Dict, ctx: RecommendationContext) -> tuple[float, str]:
    """Return (score, human-readable rationale) for one strategy."""
    score = 0.0
    reasons: List[str] = []

    # 1. Emotion match, the strongest single signal.
    if ctx.emotion in strategy["emotions"]:
        score += 2.0
        reasons.append(f"matches your current feeling ({ctx.emotion})")

    # 2. Segment preference.
    preferred = _SEGMENT_PREFERENCES.get(ctx.segment, set())
    if strategy["category"] in preferred:
        score += 1.0
        reasons.append(f"suits the '{ctx.segment}' support plan")

    # 3. Stress-driven de-escalation boost.
    stress_w = _STRESS_WEIGHT.get(ctx.stress_level, 0.5)
    if strategy["category"] in _DEESCALATION_CATEGORIES:
        score += 1.5 * stress_w
        if stress_w > 0:
            reasons.append("helps settle high stress quickly")

    rationale = "; ".join(reasons) if reasons else "a generally helpful technique"
    return score, rationale


def recommend(ctx: RecommendationContext, top_k: int = 3) -> List[Recommendation]:
    """Return the top-k personalised coping strategies for the context."""
    scored: List[Recommendation] = []
    for strategy in COPING_STRATEGIES:
        score, rationale = _score_strategy(strategy, ctx)
        scored.append(
            Recommendation(
                title=strategy["title"],
                category=strategy["category"],
                steps=strategy["steps"],
                score=round(score, 2),
                rationale=rationale,
            )
        )
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def recommend_from_signals(emotion: str = "neutral", segment: str = "Coping",
                           stress_level: str = "Moderate",
                           top_k: int = 3) -> List[Recommendation]:
    """Convenience wrapper used by the UI and chat engine."""
    ctx = RecommendationContext(emotion=emotion, segment=segment,
                                stress_level=stress_level)
    return recommend(ctx, top_k=top_k)
