"""Self-check page: PHQ-9 and GAD-7 screening questionnaires."""
from __future__ import annotations

import streamlit as st

from .. import database as db
from ..content import SUPPORT_RESOURCES
from ..screening import (GAD7_ITEMS, PHQ9_ITEMS, RESPONSE_OPTIONS,
                         score_gad7, score_phq9)

_OPTION_LABELS = [label for label, _ in RESPONSE_OPTIONS]
_OPTION_VALUES = {label: value for label, value in RESPONSE_OPTIONS}


def _render_instrument(key: str, title: str, items: list[str], scorer, user_id: int):
    st.subheader(title)
    st.caption(
        "Over the last 2 weeks, how often have you been bothered by the "
        "following? This is a self-check, not a diagnosis."
    )
    responses = []
    with st.form(f"{key}_form"):
        for i, item in enumerate(items):
            choice = st.radio(f"{i+1}. {item}", _OPTION_LABELS,
                              horizontal=True, index=0, key=f"{key}_{i}")
            responses.append(_OPTION_VALUES[choice])
        submitted = st.form_submit_button(f"Score my {title.split(', ')[0].strip()}")

    if submitted:
        outcome = scorer(responses)
        db.add_screening_result(user_id, outcome.instrument,
                                outcome.total_score, outcome.severity)
        st.metric(f"{outcome.instrument} score", f"{outcome.total_score}",
                  outcome.severity)
        st.info(outcome.advice)

        if getattr(outcome, "flags_self_harm", False):
            st.error(
                "You indicated thoughts of being better off dead or of hurting "
                "yourself. Please talk to someone you trust or contact a "
                "helpline right now, you deserve support."
            )
            for r in SUPPORT_RESOURCES:
                if r["type"] in {"Crisis & Counselling", "Suicide Prevention"}:
                    st.markdown(f"- **{r['name']}**, {r['contact']} ({r['hours']})")


def render() -> None:
    user_id = st.session_state.user_id
    st.title("📝 Self-Check")
    st.markdown(
        "These two short, widely used questionnaires can help you reflect on "
        "your mood (PHQ-9) and anxiety (GAD-7). **They are screening aids "
        "only and do not provide a diagnosis.** Your scores feed into your "
        "personalised insights."
    )

    tab1, tab2 = st.tabs(["PHQ-9 (Mood)", "GAD-7 (Anxiety)"])
    with tab1:
        _render_instrument("phq9", "PHQ-9, Mood Self-Check", PHQ9_ITEMS,
                           score_phq9, user_id)
    with tab2:
        _render_instrument("gad7", "GAD-7, Anxiety Self-Check", GAD7_ITEMS,
                           score_gad7, user_id)

    history = db.get_screening_results(user_id)
    if history:
        st.divider()
        st.subheader("Your previous self-checks")
        for r in reversed(history[-6:]):
            st.markdown(
                f"- **{r['instrument']}** · score {r['total_score']} "
                f"({r['severity']}) · {r['taken_at'][:10]}"
            )
