"""The conversational core of MindMate.

The engine exposes a single :func:`generate_reply` function. Internally it:

    1. Screens the user message for crisis risk (always, regardless of mode).
    2. If a crisis is detected, returns the safe crisis response immediately.
    3. Otherwise builds a personalised reply using either:
         * the OpenAI Chat Completions API (when an API key is configured), or
         * a fully offline, rule-based fallback engine.

Because the fallback engine is always available, the application demonstrates
end-to-end without any external dependency or API cost, important for a
reproducible academic submission and a smooth live demo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import SETTINGS
from ..ml import sentiment
from ..ml.recommender import recommend_from_signals
from . import prompts, safety


@dataclass
class AssistantReply:
    text: str
    risk_level: str
    sentiment: float
    emotion: str
    source: str          # 'openai' | 'offline' | 'safety'


# --------------------------------------------------------------------------- #
# OpenAI path
# --------------------------------------------------------------------------- #
def _openai_reply(message: str, history: List[Dict], note: str) -> Optional[str]:
    """Call the OpenAI API; return None on any failure so we can fall back."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=SETTINGS.openai_api_key)
        messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "system", "content": note}]
        # Include a trimmed window of prior turns for continuity.
        for turn in history[-8:]:
            role = "assistant" if turn["role"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=SETTINGS.openai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Any error (missing package, network, quota) -> use offline engine.
        return None


# --------------------------------------------------------------------------- #
# Offline rule-based path
# --------------------------------------------------------------------------- #
_EMOTION_OPENERS = {
    "anxiety": "It sounds like anxiety is weighing on you right now, and that's really hard.",
    "sadness": "I'm sorry you're feeling so low, thank you for sharing that with me.",
    "stress": "It sounds like you're under a lot of pressure at the moment.",
    "anger": "It's completely understandable to feel frustrated about this.",
    "positive": "I'm really glad to hear there's some lightness for you today.",
    "neutral": "Thank you for telling me what's on your mind.",
}


def _offline_reply(message: str, emotion: str, segment: str, stress_level: str) -> str:
    """Construct a supportive, personalised reply with no external calls."""
    opener = _EMOTION_OPENERS.get(emotion, _EMOTION_OPENERS["neutral"])
    recs = recommend_from_signals(emotion=emotion, segment=segment,
                                  stress_level=stress_level, top_k=1)
    parts = [opener]

    parts.append(
        "Feelings like these are a normal response to everything students "
        "are juggling, and they do pass. You don't have to fix everything at "
        "once."
    )

    if recs:
        rec = recs[0]
        steps = " ".join(f"{i+1}) {s}" for i, s in enumerate(rec.steps[:3]))
        parts.append(
            f"One small thing that might help right now is **{rec.title}**: "
            f"{steps}"
        )

    parts.append(
        "Would you like to talk a bit more about what's behind this, or shall "
        "we try a short exercise together?"
    )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_reply(message: str, history: Optional[List[Dict]] = None,
                   segment: str = "Coping", stress_level: str = "Moderate") -> AssistantReply:
    """Generate the assistant's reply to a single user message."""
    history = history or []

    risk = safety.assess_risk(message)
    sent = sentiment.analyze(message)

    if risk.is_crisis:
        return AssistantReply(
            text=safety.crisis_response(),
            risk_level="crisis",
            sentiment=sent.compound,
            emotion=sent.emotion,
            source="safety",
        )

    note = prompts.build_personalization_note(sent.emotion, segment, stress_level)

    text = None
    source = "offline"
    if SETTINGS.llm_enabled:
        text = _openai_reply(message, history, note)
        if text is not None:
            source = "openai"

    if text is None:
        text = _offline_reply(message, sent.emotion, segment, stress_level)
        source = "offline"

    # For elevated (non-crisis) distress, append a gentle resource nudge.
    if risk.level == "elevated":
        text += (
            "\n\n_If these feelings stick around or get heavier, please "
            "consider reaching out to a counsellor or one of the support "
            "lines on the Resources page. You deserve support._"
        )

    return AssistantReply(
        text=text,
        risk_level=risk.level,
        sentiment=sent.compound,
        emotion=sent.emotion,
        source=source,
    )
