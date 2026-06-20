"""Capture screenshots of every MindMate page for the report and slides.

This uses Playwright to drive a headless browser through the running app,
completing onboarding and visiting each page. Screenshots are written to
``docs/screenshots/``.

Prerequisites (one-time)::

    pip install playwright
    python -m playwright install chromium

Then, with the app running in another terminal (``streamlit run app.py``)::

    python -m scripts.capture_screenshots

If you prefer, you can simply take the screenshots manually: run the app,
sign in, and use your operating system's screenshot tool on each page.
"""
from __future__ import annotations

import time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8501"
PAGES = ["Home", "Talk to MindMate", "Mood Tracker", "Self-Check",
         "Insights", "Resources"]


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Playwright is not installed. Run:\n"
            "  pip install playwright && python -m playwright install chromium"
        )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE, wait_until="networkidle")
        time.sleep(2)

        # Complete onboarding if present.
        try:
            page.get_by_label("Your name or a nickname").fill("Aarav", timeout=4000)
            page.get_by_text("I understand MindMate is a self-help tool").click()
            page.get_by_role("button", name="Start").click()
            time.sleep(3)
        except Exception:
            pass  # already onboarded

        for name in PAGES:
            try:
                page.get_by_role("link", name=name).click()
                page.wait_for_load_state("networkidle")
                time.sleep(2)
                fn = OUT / f"{name.lower().replace(' ', '_')}.png"
                page.screenshot(path=str(fn), full_page=True)
                print(f"  captured {fn.name}")
            except Exception as exc:
                print(f"  skipped {name}: {exc}")

        browser.close()
    print(f"Screenshots saved to {OUT}")


if __name__ == "__main__":
    main()
