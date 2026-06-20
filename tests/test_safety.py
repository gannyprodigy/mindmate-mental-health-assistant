"""Tests for the crisis-detection safety layer."""
from __future__ import annotations

import pytest

from src.assistant import safety


@pytest.mark.parametrize("text", [
    "I want to kill myself",
    "Sometimes I think about ending my life",
    "I feel like I want to die",
    "I've been thinking about hurting myself",
    "there is no reason to live anymore",
])
def test_crisis_phrases_flagged(text):
    assert safety.assess_risk(text).level == "crisis"


@pytest.mark.parametrize("text", [
    "I feel so hopeless",
    "I just can't cope with all this",
    "I feel worthless lately",
])
def test_elevated_phrases_flagged(text):
    assert safety.assess_risk(text).level == "elevated"


@pytest.mark.parametrize("text", [
    "I'm killing it in my exams!",
    "This deadline is killing me but I'll manage",
    "I had a great day today",
    "",
])
def test_non_crisis_not_overflagged(text):
    assert safety.assess_risk(text).level != "crisis"


def test_crisis_response_contains_helpline():
    response = safety.crisis_response()
    assert any(token in response for token in ("14416", "Vandrevala", "AASRA"))
