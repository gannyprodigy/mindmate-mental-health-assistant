"""Resources page: helplines, psychoeducation and the full coping library."""
from __future__ import annotations

import streamlit as st

from ..content import COPING_STRATEGIES, PSYCHOEDUCATION, SUPPORT_RESOURCES


def render() -> None:
    st.title("📚 Resources & Support")

    st.error(
        "**If you are in immediate danger or thinking about harming yourself, "
        "please reach out now.** You are not alone, and support is available "
        "24x7 through the helplines below."
    )

    st.subheader("Helplines")
    for r in SUPPORT_RESOURCES:
        st.markdown(
            f"**{r['name']}**  \n"
            f"📞 {r['contact']}  ·  🕒 {r['hours']}  ·  _{r['type']}_"
        )
    st.caption(
        "Helplines are provided for convenience. MindMate is not affiliated "
        "with these services and cannot guarantee availability."
    )

    st.divider()
    st.subheader("Understanding your wellbeing")
    for item in PSYCHOEDUCATION:
        with st.expander(item["title"]):
            st.write(item["body"])

    st.divider()
    st.subheader("Coping strategy library")
    st.caption("Browse every technique MindMate can recommend.")
    categories = sorted({s["category"] for s in COPING_STRATEGIES})
    chosen = st.multiselect("Filter by category", categories, default=categories)
    for s in COPING_STRATEGIES:
        if s["category"] not in chosen:
            continue
        with st.expander(f"{s['title']} · {s['category']}"):
            for step in s["steps"]:
                st.markdown(f"- {step}")

    st.divider()
    st.caption(
        "MindMate is a self-help tool and does not provide medical advice, "
        "diagnosis, or treatment. Always seek the advice of a qualified health "
        "professional with any questions about a medical condition."
    )
