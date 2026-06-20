"""Sentiment and emotion analysis for user-generated text.

The primary scorer is VADER (Valence Aware Dictionary and sEntiment
Reasoner), a lexicon and rule-based model that is well suited to short,
informal text such as chat messages and mood-journal notes. On top of the
raw VADER compound score we add a light-weight emotion-keyword layer that
maps text onto a small set of student-relevant emotional states used by the
recommender.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()

# Coarse emotion lexicon, keywords that signal a dominant emotional state.
EMOTION_LEXICON: Dict[str, list[str]] = {
    "anxiety": [
        "anxious", "anxiety", "panic", "nervous", "worried", "worry", "scared",
        "afraid", "overwhelm", "overwhelmed", "tense", "restless", "dread",
    ],
    "sadness": [
        "sad", "down", "depressed", "hopeless", "lonely", "empty", "cry",
        "crying", "miserable", "worthless", "numb", "unmotivated",
    ],
    "stress": [
        "stress", "stressed", "pressure", "deadline", "exam", "exams",
        "assignment", "burnout", "burnt out", "workload", "overworked",
    ],
    "anger": [
        "angry", "frustrated", "frustration", "irritated", "annoyed", "mad",
        "furious", "resent",
    ],
    "positive": [
        "happy", "good", "great", "better", "calm", "relaxed", "hopeful",
        "grateful", "confident", "motivated", "proud", "excited",
    ],
}


@dataclass
class SentimentResult:
    """Structured output of :func:`analyze`."""

    compound: float       # -1.0 (very negative) .. +1.0 (very positive)
    positive: float
    neutral: float
    negative: float
    label: str            # 'positive' | 'neutral' | 'negative'
    emotion: str          # dominant emotion from EMOTION_LEXICON or 'neutral'

    def as_dict(self) -> dict:
        return {
            "compound": self.compound,
            "positive": self.positive,
            "neutral": self.neutral,
            "negative": self.negative,
            "label": self.label,
            "emotion": self.emotion,
        }


def _label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def detect_emotion(text: str) -> str:
    """Return the dominant emotion keyword group present in ``text``."""
    lowered = f" {text.lower()} "
    counts: Dict[str, int] = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        counts[emotion] = sum(lowered.count(f"{kw}") for kw in keywords)
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "neutral"


def analyze(text: str) -> SentimentResult:
    """Run sentiment + emotion analysis on a single piece of text."""
    if not text or not text.strip():
        return SentimentResult(0.0, 0.0, 1.0, 0.0, "neutral", "neutral")

    scores = _ANALYZER.polarity_scores(text)
    compound = float(scores["compound"])
    return SentimentResult(
        compound=compound,
        positive=float(scores["pos"]),
        neutral=float(scores["neu"]),
        negative=float(scores["neg"]),
        label=_label_from_compound(compound),
        emotion=detect_emotion(text),
    )


def compound_score(text: str) -> float:
    """Convenience helper returning only the VADER compound score."""
    return analyze(text).compound
