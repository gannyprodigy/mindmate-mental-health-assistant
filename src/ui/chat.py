"""Conversational assistant page."""
from __future__ import annotations

import streamlit as st

from .. import database as db, service
from ..assistant import chat_engine


def _load_history(user_id: int) -> list[dict]:
    """Pull stored chat history into the session on first load."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in db.get_chat_history(user_id)
        ]
    return st.session_state.chat_messages


def render() -> None:
    user_id = st.session_state.user_id
    st.title("💬 Talk to MindMate")
    st.caption(
        "A supportive, judgement-free space. MindMate is a self-help tool, "
        "not a therapist or emergency service."
    )

    messages = _load_history(user_id)

    if not messages:
        st.info(
            "Hi, I'm MindMate. How are you feeling today? You can tell me "
            "what's on your mind, big or small."
        )

    for msg in messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    prompt = st.chat_input("Type how you're feeling...")
    if not prompt:
        return

    # Display + persist the user's message.
    with st.chat_message("user"):
        st.markdown(prompt)
    messages.append({"role": "user", "content": prompt})

    profile = service.profile_for_user(user_id)

    with st.chat_message("assistant"):
        with st.spinner("MindMate is thinking..."):
            reply = chat_engine.generate_reply(
                prompt,
                history=messages[:-1],
                segment=profile["segment"],
                stress_level=profile["stress_level"],
            )
        st.markdown(reply.text)
        if reply.risk_level == "crisis":
            st.error("It looks like you might be going through something very "
                     "serious. Please reach out to a helpline on the Resources "
                     "page or someone you trust right now.")

    messages.append({"role": "assistant", "content": reply.text})

    # Persist both turns with their sentiment / risk metadata.
    db.add_chat_message(user_id, "user", prompt,
                        sentiment=reply.sentiment, risk_level=reply.risk_level)
    db.add_chat_message(user_id, "assistant", reply.text,
                        sentiment=None, risk_level=reply.risk_level)
