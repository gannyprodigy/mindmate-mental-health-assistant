"""Validated self-report screening instruments.

MindMate uses two widely cited public-domain screening questionnaires:

    * PHQ-9, Patient Health Questionnaire (depressive symptoms).
    * GAD-7, Generalised Anxiety Disorder scale (anxiety symptoms).

These are screening aids only and are clearly framed as such in the UI; they
do not provide a diagnosis. Scoring thresholds follow the standard published
cut-offs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

RESPONSE_OPTIONS: List[Tuple[str, int]] = [
    ("Not at all", 0),
    ("Several days", 1),
    ("More than half the days", 2),
    ("Nearly every day", 3),
]

PHQ9_ITEMS: List[str] = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself, or that you are a failure",
    "Trouble concentrating on things, such as reading or studying",
    "Moving or speaking slowly, or being restless and fidgety",
    "Thoughts that you would be better off dead, or of hurting yourself",
]

GAD7_ITEMS: List[str] = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]


@dataclass
class ScreeningOutcome:
    instrument: str
    total_score: int
    severity: str
    advice: str
    flags_self_harm: bool = False


def _phq9_severity(score: int) -> str:
    if score <= 4:
        return "Minimal"
    if score <= 9:
        return "Mild"
    if score <= 14:
        return "Moderate"
    if score <= 19:
        return "Moderately severe"
    return "Severe"


def _gad7_severity(score: int) -> str:
    if score <= 4:
        return "Minimal"
    if score <= 9:
        return "Mild"
    if score <= 14:
        return "Moderate"
    return "Severe"


_ADVICE = {
    "Minimal": "Your responses suggest few symptoms right now. Keep up the "
               "habits that support you.",
    "Mild": "Your responses suggest some mild symptoms. Self-help strategies "
            "and regular check-ins may help.",
    "Moderate": "Your responses suggest a moderate level of symptoms. "
                "Consider speaking with a campus counsellor.",
    "Moderately severe": "Your responses suggest a noticeable level of "
                         "symptoms. Reaching out to a counsellor or doctor is "
                         "strongly encouraged.",
    "Severe": "Your responses suggest a high level of symptoms. Please "
              "consider professional support soon, you don't have to manage "
              "this alone.",
}


def score_phq9(responses: List[int]) -> ScreeningOutcome:
    """Score a 9-item PHQ-9 response vector (each 0-3)."""
    if len(responses) != len(PHQ9_ITEMS):
        raise ValueError("PHQ-9 requires exactly 9 responses")
    total = int(sum(responses))
    severity = _phq9_severity(total)
    # Item 9 is the self-harm ideation item; any non-zero score flags it.
    flags = responses[8] > 0
    return ScreeningOutcome("PHQ-9", total, severity, _ADVICE[severity], flags)


def score_gad7(responses: List[int]) -> ScreeningOutcome:
    """Score a 7-item GAD-7 response vector (each 0-3)."""
    if len(responses) != len(GAD7_ITEMS):
        raise ValueError("GAD-7 requires exactly 7 responses")
    total = int(sum(responses))
    severity = _gad7_severity(total)
    return ScreeningOutcome("GAD-7", total, severity, _ADVICE[severity], False)
