"""MindMate, AI-Powered Mental Health Assistant for Students.

Streamlit entry point. Run with::

    streamlit run app.py

The app is organised as a multi-page experience (Home, Chat, Mood Tracker,
Self-Check, Insights, Resources) built on top of the service layer in
``src/service.py``.
"""
from __future__ import annotations

import streamlit as st

from src import __app_name__, __version__, database as db
from src.config import SETTINGS
from src.ui import chat, home, insights, mood, resources, screening_page

st.set_page_config(
    page_title="MindMate, Student Wellbeing Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure the database schema exists on first launch.
db.init_db()


def _ensure_user() -> None:
    """Onboard a user into session state if none is active."""
    if "user_id" in st.session_state and st.session_state.user_id:
        return

    st.title("🌱 Welcome to MindMate")
    st.markdown(
        "MindMate is a supportive wellbeing companion for students. It offers "
        "a friendly space to talk, track your mood, and discover coping "
        "techniques tailored to you.\n\n"
        "**MindMate is not a medical service and does not provide a diagnosis.** "
        "If you are in crisis, please use the helplines on the Resources page."
    )
    with st.form("onboarding"):
        st.subheader("Let's set up your profile")
        name = st.text_input("Your name or a nickname", max_chars=40)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=15, max_value=80, value=20)
            course = st.text_input("Course / programme", value="")
        with col2:
            year = st.number_input("Year of study", min_value=1, max_value=7, value=1)
            consent = st.checkbox(
                "I understand MindMate is a self-help tool, not a substitute "
                "for professional care.", value=False
            )
        submitted = st.form_submit_button("Start")
        if submitted:
            if not name.strip():
                st.error("Please enter a name or nickname.")
            elif not consent:
                st.error("Please acknowledge the disclaimer to continue.")
            else:
                uid = db.create_user(name.strip(), int(age), course.strip() or None, int(year))
                st.session_state.user_id = uid
                st.session_state.user_name = name.strip()
                st.rerun()
    st.stop()


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### 🌱 {__app_name__}")
        st.caption(f"v{__version__} · Student Wellbeing Assistant")
        if st.session_state.get("user_name"):
            st.success(f"Signed in as **{st.session_state.user_name}**")
        mode = "🟢 Live AI (OpenAI)" if SETTINGS.llm_enabled else "🟡 Offline engine"
        st.caption(f"Assistant mode: {mode}")
        st.divider()
        if st.button("Switch user / sign out", use_container_width=True):
            for key in ("user_id", "user_name", "chat_messages"):
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        st.caption(
            "In crisis? You are not alone. See the **Resources** page for "
            "24x7 helplines."
        )


def main() -> None:
    _ensure_user()
    _sidebar()

    pages = [
        st.Page(home.render, title="Home", icon="🏠", url_path="home", default=True),
        st.Page(chat.render, title="Talk to MindMate", icon="💬", url_path="chat"),
        st.Page(mood.render, title="Mood Tracker", icon="📈", url_path="mood"),
        st.Page(screening_page.render, title="Self-Check", icon="📝", url_path="self-check"),
        st.Page(insights.render, title="Insights", icon="🧠", url_path="insights"),
        st.Page(resources.render, title="Resources", icon="📚", url_path="resources"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
