"""Insights page: surfaces the ML personalisation in a transparent way."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .. import database as db, service


def render() -> None:
    user_id = st.session_state.user_id
    st.title("🧠 Your Insights")
    st.caption(
        "MindMate uses machine learning to personalise your support. Here is "
        "exactly what it sees and why it suggests what it does, no black boxes."
    )

    if not service.models_available():
        st.warning(
            "Personalisation models are not trained yet. Run "
            "`python -m scripts.generate_data && python -m scripts.train_models`."
        )
        return

    profile = service.profile_for_user(user_id)
    feats = profile["features"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Support segment", profile["segment"])
    c2.metric("Predicted stress", profile["stress_level"])
    c3.metric("Wellness index", f"{service.wellness_score(user_id)}/100")

    st.info(f"**Why this segment?** {profile['segment_strategy']}")

    # Stress probability breakdown.
    if profile["stress_proba"]:
        st.subheader("How confident is the stress model?")
        proba_df = pd.DataFrame({
            "Stress level": list(profile["stress_proba"].keys()),
            "Probability": list(profile["stress_proba"].values()),
        })
        fig = px.bar(proba_df, x="Stress level", y="Probability",
                     range_y=[0, 1], color="Stress level",
                     color_discrete_map={"Low": "#2ca02c", "Moderate": "#ff7f0e",
                                         "High": "#d62728"})
        st.plotly_chart(fig, use_container_width=True)

    # The feature vector the models use.
    st.subheader("The signals behind your profile")
    nice = {
        "avg_mood": "Average mood (1-5)",
        "avg_sleep_hours": "Average sleep (hours)",
        "phq9_score": "Latest PHQ-9 score",
        "gad7_score": "Latest GAD-7 score",
        "avg_sentiment": "Average note sentiment (-1 to +1)",
        "engagement": "Engagement (interactions)",
    }
    feat_df = pd.DataFrame(
        [{"Signal": nice.get(k, k), "Value": round(v, 2)} for k, v in feats.items()]
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # Personalised recommendations with rationale.
    st.subheader("Recommended coping strategies")
    recs = service.personalised_recommendations(user_id, emotion="neutral", top_k=4)
    for rec in recs:
        with st.expander(f"{rec.title}  ·  {rec.category}  (match score {rec.score})"):
            st.caption(f"Why this: {rec.rationale}")
            for step in rec.steps:
                st.markdown(f"- {step}")

    # Activity summary.
    st.divider()
    st.subheader("Your activity")
    moods = db.get_mood_logs(user_id)
    chats = db.get_chat_history(user_id)
    screens = db.get_screening_results(user_id)
    a, b, c = st.columns(3)
    a.metric("Mood check-ins", len(moods))
    b.metric("Chat messages", len([m for m in chats if m["role"] == "user"]))
    c.metric("Self-checks", len(screens))
