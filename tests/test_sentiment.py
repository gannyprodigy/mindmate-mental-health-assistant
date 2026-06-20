"""Tests for sentiment + emotion analysis."""
from __future__ import annotations

from src.ml import sentiment


def test_positive_text_scores_positive():
    result = sentiment.analyze("I feel great and really hopeful today")
    assert result.compound > 0
    assert result.label == "positive"


def test_negative_text_scores_negative():
    result = sentiment.analyze("I am miserable and everything feels hopeless")
    assert result.compound < 0
    assert result.label == "negative"


def test_emotion_detection_anxiety():
    assert sentiment.analyze("I am so anxious and worried about exams").emotion == "anxiety"


def test_emotion_detection_stress():
    assert sentiment.analyze("the workload and deadlines are too much").emotion == "stress"


def test_empty_text_is_neutral():
    result = sentiment.analyze("")
    assert result.label == "neutral"
    assert result.emotion == "neutral"
