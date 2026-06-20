"""Smoke tests for the Streamlit app via the AppTest harness.

These exercise the app *past* onboarding so that ``st.navigation`` actually
runs, catching issues such as duplicate page URL pathnames that a simple
HTTP health check would miss.
"""
from __future__ import annotations

import pytest

from streamlit.testing.v1 import AppTest

from src import database as db

PAGE_MODULES = {
    "home": "home",
    "chat": "chat",
    "mood": "mood",
    "self-check": "screening_page",
    "insights": "insights",
    "resources": "resources",
}


def test_app_runs_past_onboarding():
    """The full app must run past onboarding (exercises st.navigation)."""
    db.init_db()
    uid = db.create_user("Test Runner", 22, "MSc DS", 1)
    at = AppTest.from_file("app.py", default_timeout=60)
    at.session_state["user_id"] = uid
    at.session_state["user_name"] = "Test Runner"
    at.run()
    assert not at.exception, f"App raised: {at.exception}"


@pytest.mark.parametrize("page", list(PAGE_MODULES))
def test_each_page_renders(page):
    """Each page's render function must run without raising."""
    db.init_db()
    uid = db.create_user("Test Runner", 22, "MSc DS", 1)
    module = PAGE_MODULES[page]
    script = (
        "import streamlit as st\n"
        f"st.session_state['user_id'] = {uid}\n"
        "st.session_state['user_name'] = 'Test Runner'\n"
        f"from src.ui import {module} as page\n"
        "page.render()\n"
    )
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, f"Page '{page}' raised: {at.exception}"
