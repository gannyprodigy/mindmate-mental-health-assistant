"""Tests for the assistant chat engine (offline path + safety routing)."""
from __future__ import annotations

from src.assistant import chat_engine


def test_offline_reply_is_supportive():
    reply = chat_engine.generate_reply(
        "I'm really stressed about my exams",
        segment="At-Risk", stress_level="High",
    )
    assert reply.source == "offline"          # no API key in test env
    assert reply.risk_level == "none"
    assert len(reply.text) > 40


def test_crisis_message_routes_to_safety():
    reply = chat_engine.generate_reply("I want to kill myself")
    assert reply.source == "safety"
    assert reply.risk_level == "crisis"
    assert "helpline" in reply.text.lower() or "14416" in reply.text


def test_elevated_reply_appends_resource_nudge():
    reply = chat_engine.generate_reply("I feel hopeless about everything")
    assert reply.risk_level == "elevated"
    assert "resources" in reply.text.lower()
