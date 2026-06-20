"""Curated, evidence-informed wellbeing content for MindMate.

All coping strategies are drawn from widely used, non-clinical self-help
techniques (CBT-style reframing, grounding, breathing, behavioural
activation, sleep hygiene, time management). Nothing here is a substitute
for professional care, the assistant always surfaces that distinction.
"""
from __future__ import annotations

from typing import Dict, List

# Each strategy: title, category, the emotions it best addresses, and steps.
COPING_STRATEGIES: List[Dict] = [
    {
        "title": "4-7-8 Breathing",
        "category": "Calming",
        "emotions": ["anxiety", "stress", "anger"],
        "steps": [
            "Breathe in quietly through the nose for 4 seconds.",
            "Hold your breath for 7 seconds.",
            "Exhale slowly through the mouth for 8 seconds.",
            "Repeat for 4 cycles and notice your body settle.",
        ],
    },
    {
        "title": "5-4-3-2-1 Grounding",
        "category": "Grounding",
        "emotions": ["anxiety", "stress"],
        "steps": [
            "Name 5 things you can see.",
            "Name 4 things you can feel.",
            "Name 3 things you can hear.",
            "Name 2 things you can smell.",
            "Name 1 thing you can taste.",
        ],
    },
    {
        "title": "Thought Reframing (CBT)",
        "category": "Cognitive",
        "emotions": ["sadness", "anxiety", "anger"],
        "steps": [
            "Write the upsetting thought exactly as it appears.",
            "Ask: what is the evidence for and against it?",
            "Identify the thinking trap (e.g. catastrophising).",
            "Write a more balanced, kinder alternative thought.",
        ],
    },
    {
        "title": "Pomodoro Study Sprints",
        "category": "Academic",
        "emotions": ["stress"],
        "steps": [
            "Pick one small task and set a 25-minute timer.",
            "Work with no distractions until it rings.",
            "Take a 5-minute break, then repeat.",
            "After 4 sprints, take a longer 20-minute break.",
        ],
    },
    {
        "title": "Behavioural Activation",
        "category": "Mood",
        "emotions": ["sadness"],
        "steps": [
            "List 3 small activities that used to bring you energy.",
            "Schedule one of them for today, however briefly.",
            "Do it before you feel motivated, action comes first.",
            "Note how your mood shifts afterwards.",
        ],
    },
    {
        "title": "Sleep Wind-Down Routine",
        "category": "Sleep",
        "emotions": ["stress", "anxiety"],
        "steps": [
            "Stop screens 45 minutes before bed.",
            "Dim lights and do something calming (reading, stretching).",
            "Write tomorrow's top 3 tasks to park your worries.",
            "Keep a consistent wake-up time, even after late nights.",
        ],
    },
    {
        "title": "Self-Compassion Break",
        "category": "Mood",
        "emotions": ["sadness", "anger"],
        "steps": [
            "Acknowledge: 'This is a moment of difficulty.'",
            "Remind yourself: 'Difficulty is part of being human.'",
            "Place a hand on your chest and offer yourself kindness.",
            "Ask what you would say to a friend in your situation.",
        ],
    },
    {
        "title": "Worry Time Scheduling",
        "category": "Cognitive",
        "emotions": ["anxiety", "stress"],
        "steps": [
            "Set aside 15 minutes later today as 'worry time'.",
            "When worries arise now, note them and postpone them.",
            "During worry time, review the list, many will have faded.",
            "Turn the remaining ones into concrete next actions.",
        ],
    },
    {
        "title": "Gratitude & Wins Journal",
        "category": "Mood",
        "emotions": ["positive", "sadness"],
        "steps": [
            "Write 3 things that went okay today, however small.",
            "Note one personal strength you used.",
            "Name one thing you are looking forward to.",
            "Re-read past entries on hard days.",
        ],
    },
]

# Curated support resources. International + India-specific helplines are
# included because the project brief targets students in India.
SUPPORT_RESOURCES: List[Dict] = [
    {
        "name": "iCall Psychosocial Helpline (India)",
        "contact": "9152987821",
        "hours": "Mon-Sat, 8 AM - 10 PM",
        "type": "Counselling",
    },
    {
        "name": "Vandrevala Foundation Helpline (India)",
        "contact": "1860-2662-345 / 1800-2333-330",
        "hours": "24x7",
        "type": "Crisis & Counselling",
    },
    {
        "name": "Tele-MANAS (Govt. of India Mental Health)",
        "contact": "14416 / 1-800-891-4416",
        "hours": "24x7",
        "type": "Crisis & Counselling",
    },
    {
        "name": "AASRA",
        "contact": "+91-9820466726",
        "hours": "24x7",
        "type": "Suicide Prevention",
    },
    {
        "name": "International Association for Suicide Prevention",
        "contact": "https://www.iasp.info/resources/Crisis_Centres/",
        "hours": "Directory",
        "type": "Global directory",
    },
]

# Psychoeducation snippets surfaced on the Resources page.
PSYCHOEDUCATION: List[Dict] = [
    {
        "title": "Understanding Exam Stress",
        "body": (
            "A degree of stress before exams is normal and can sharpen focus. "
            "It becomes a problem when it is constant, disrupts sleep, or stops "
            "you functioning. Breaking work into small sprints and protecting "
            "sleep are two of the most effective countermeasures."
        ),
    },
    {
        "title": "Sleep and Mental Health",
        "body": (
            "Sleep and mood are tightly linked. Even one week of restricted "
            "sleep measurably lowers mood and concentration. A consistent "
            "wake-up time matters more than a fixed bedtime."
        ),
    },
    {
        "title": "When to Seek Professional Help",
        "body": (
            "Consider reaching out to a counsellor if low mood, anxiety, or "
            "hopelessness last more than two weeks, affect your studies or "
            "relationships, or you have thoughts of harming yourself. Seeking "
            "help is a sign of strength, not weakness."
        ),
    },
]


def strategies_for_emotion(emotion: str) -> List[Dict]:
    """Return all coping strategies that target a given emotion."""
    matches = [s for s in COPING_STRATEGIES if emotion in s["emotions"]]
    return matches or COPING_STRATEGIES
