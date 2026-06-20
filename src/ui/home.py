"""Home / dashboard landing page."""
from __future__ import annotations

import streamlit as st

from .. import service
from ..config import SETTINGS


def render() -> None:
    user_id = st.session_state.user_id
    name = st.session_state.get("user_name", "there")

    st.title(f"Hi {name} 👋")
    st.markdown("Here's a snapshot of how you've been doing.")

    profile = service.profile_for_user(user_id)
    score = service.wellness_score(user_id)

    c1, c2, c3 = st.columns(3)
    c1.metric("Wellness index", f"{score}/100")
    c2.metric("Support segment", profile["segment"])
    c3.metric("Predicted stress", profile["stress_level"])

    if not service.models_available():
        st.warning(
            "ML models are not trained yet. Run "
            "`python -m scripts.generate_data && python -m scripts.train_models` "
            "to enable personalised insights."
        )

    st.divider()
    st.subheader("Your support plan")
    st.info(profile["segment_strategy"] or "Keep logging your mood to personalise this.")

    st.subheader("Suggested for you right now")
    recs = service.personalised_recommendations(user_id, emotion="neutral", top_k=3)
    cols = st.columns(len(recs))
    for col, rec in zip(cols, recs):
        with col:
            st.markdown(f"**{rec.title}**")
            st.caption(rec.category)
            with st.expander("How to do it"):
                for step in rec.steps:
                    st.markdown(f"- {step}")

    st.divider()
    mode = "live OpenAI responses" if SETTINGS.llm_enabled else "the built-in offline engine"
    st.markdown("#### About this session")
    st.caption(
        f"MindMate is currently using {mode}. Use the navigation on the left "
        "to talk to the assistant, log your mood, take a self-check, or "
        "review your insights. All your data stays in a local database on "
        "this machine."
    )
