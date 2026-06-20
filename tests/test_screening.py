"""Tests for the PHQ-9 / GAD-7 scoring logic."""
from __future__ import annotations

import pytest

from src import screening


def test_phq9_minimal():
    out = screening.score_phq9([0] * 9)
    assert out.total_score == 0
    assert out.severity == "Minimal"
    assert out.flags_self_harm is False


def test_phq9_severe_and_self_harm_flag():
    out = screening.score_phq9([3] * 9)
    assert out.total_score == 27
    assert out.severity == "Severe"
    assert out.flags_self_harm is True  # item 9 non-zero


def test_phq9_self_harm_flag_only_on_item9():
    responses = [3, 3, 3, 3, 3, 0, 0, 0, 0]  # item 9 = 0
    assert screening.score_phq9(responses).flags_self_harm is False


def test_gad7_thresholds():
    assert screening.score_gad7([0] * 7).severity == "Minimal"
    assert screening.score_gad7([3] * 7).severity == "Severe"


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        screening.score_phq9([0] * 5)
    with pytest.raises(ValueError):
        screening.score_gad7([0] * 9)
