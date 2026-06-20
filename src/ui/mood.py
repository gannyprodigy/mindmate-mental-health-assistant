"""Mood-tracking page with trend visualisation."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .. import database as db
from ..ml import sentiment

MOOD_LABELS = {1: "Very low", 2: "Low", 3: "Okay", 4: "Good", 5: "Very good"}


def render() -> None:
    user_id = st.session_state.user_id
    st.title("📈 Mood Tracker")
    st.caption("A quick daily check-in. Over time this builds a picture of "
               "your wellbeing that powers your personalised insights.")

    with st.form("mood_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            mood = st.slider("How is your mood right now?", 1, 5, 3,
                             help="1 = very low, 5 = very good")
            st.caption(f"Selected: **{MOOD_LABELS[mood]}**")
            energy = st.slider("Energy level", 1, 5, 3)
        with col2:
            sleep = st.number_input("Hours of sleep last night", 0.0, 16.0, 7.0, 0.5)
            note = st.text_area("Anything you'd like to note? (optional)",
                                placeholder="e.g. Big assignment due, feeling the pressure...")
        submitted = st.form_submit_button("Save check-in")
        if submitted:
            sent = sentiment.analyze(note).compound if note.strip() else None
            db.add_mood_log(user_id, mood, energy, sleep, note.strip() or None, sent)
            st.success("Check-in saved. Thank you for taking a moment for yourself.")

    logs = db.get_mood_logs(user_id)
    if not logs:
        st.info("No check-ins yet. Log your first one above to see trends here.")
        return

    df = pd.DataFrame(logs)
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    df = df.sort_values("logged_at")

    st.divider()
    st.subheader("Your trends")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average mood", f"{df['mood_score'].mean():.1f}/5")
    c2.metric("Average sleep", f"{df['sleep_hours'].mean():.1f} h")
    c3.metric("Check-ins logged", len(df))

    fig = px.line(df, x="logged_at", y="mood_score", markers=True,
                  title="Mood over time", range_y=[0.5, 5.5])
    fig.update_layout(yaxis_title="Mood (1-5)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    if df["sentiment"].notna().any():
        fig2 = px.bar(df[df["sentiment"].notna()], x="logged_at", y="sentiment",
                      title="Sentiment of your notes",
                      color="sentiment", color_continuous_scale="RdYlGn",
                      range_color=[-1, 1])
        fig2.update_layout(yaxis_title="Sentiment (-1 to +1)", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("See your check-in history"):
        display = df[["logged_at", "mood_score", "energy_score", "sleep_hours", "note"]]
        display = display.rename(columns={
            "logged_at": "When", "mood_score": "Mood", "energy_score": "Energy",
            "sleep_hours": "Sleep (h)", "note": "Note",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)
