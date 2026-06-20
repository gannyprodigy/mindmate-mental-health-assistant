"""Safety and crisis-detection layer.

Before any response is generated, every user message is screened for signs
of acute risk (self-harm, suicidal ideation, abuse). When risk is detected
the assistant must *not* attempt therapy; instead it responds with a calm,
validating message and prominently surfaces professional crisis resources.

This module is deliberately rule-based and conservative: in a wellbeing
product, a false positive (showing helplines unnecessarily) is far cheaper
than a false negative (missing a genuine crisis).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..content import SUPPORT_RESOURCES

# Phrases that indicate an acute crisis. Matched on word boundaries to avoid
# false hits (e.g. "killing it in exams" is excluded by the patterns below).
CRISIS_PATTERNS: List[str] = [
    r"\bkill myself\b",
    r"\bkilling myself\b",
    r"\bend my life\b",
    r"\bending my life\b",
    r"\btake my (own )?life\b",
    r"\bsuicid(e|al)\b",
    r"\bwant to die\b",
    r"\bwish i (was|were) dead\b",
    r"\bdon'?t want to (be alive|live)\b",
    r"\bno reason to live\b",
    r"\bharm(ing)? myself\b",
    r"\bhurt(ing)? myself\b",
    r"\bself[- ]harm\b",
    r"\bcut(ting)? myself\b",
    r"\boverdose\b",
    r"\bcan'?t go on\b",
    r"\bbetter off without me\b",
]

# Phrases indicating elevated (but not acute) distress.
ELEVATED_PATTERNS: List[str] = [
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\bhate myself\b",
    r"\bcan'?t cope\b",
    r"\bcan'?t take (it|this) anymore\b",
    r"\bgiving up\b",
    r"\bempty inside\b",
    r"\bnumb\b",
    r"\bbreaking down\b",
]

_CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)
_ELEVATED_RE = re.compile("|".join(ELEVATED_PATTERNS), re.IGNORECASE)


@dataclass
class RiskAssessment:
    level: str            # 'none' | 'elevated' | 'crisis'
    matched: List[str]

    @property
    def is_crisis(self) -> bool:
        return self.level == "crisis"


def assess_risk(text: str) -> RiskAssessment:
    """Classify the risk level of a user message."""
    if not text:
        return RiskAssessment("none", [])

    crisis_hits = _CRISIS_RE.findall(text)
    if crisis_hits:
        flat = [h if isinstance(h, str) else next(filter(None, h), "") for h in crisis_hits]
        return RiskAssessment("crisis", [m for m in flat if m])

    elevated_hits = _ELEVATED_RE.findall(text)
    if elevated_hits:
        flat = [h if isinstance(h, str) else next(filter(None, h), "") for h in elevated_hits]
        return RiskAssessment("elevated", [m for m in flat if m])

    return RiskAssessment("none", [])


def crisis_response() -> str:
    """A safe, validating crisis message with helplines appended."""
    lines = [
        "I'm really glad you told me how you're feeling, and I'm concerned "
        "about you. You deserve support from someone who can be there with "
        "you right now.",
        "",
        "If you might act on thoughts of harming yourself, please reach out "
        "immediately to a trusted person or one of these services:",
        "",
    ]
    for r in SUPPORT_RESOURCES:
        if r["type"] in {"Crisis & Counselling", "Suicide Prevention"}:
            lines.append(f"• **{r['name']}**, {r['contact']} ({r['hours']})")
    lines += [
        "",
        "If you are in immediate danger, please contact your local emergency "
        "number right away. You don't have to go through this alone.",
    ]
    return "\n".join(lines)
