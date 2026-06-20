"""Render the MindMate presentation as a 16:9 landscape PDF.

A self-contained PDF version of the deck (mirrors build_slides.py) so a PDF
presentation is available alongside the editable PPTX.

Usage::

    python -m scripts.build_slides_pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "presentation" / "MindMate_Final_Presentation_PDF.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 960, 540  # 16:9 points
NAVY = HexColor("#1f3b63")
GREEN = HexColor("#2c6e49")
MINT = HexColor("#9ec4a8")
GREY = HexColor("#555555")
LIGHT = HexColor("#eef2f7")

c = canvas.Canvas(str(OUT), pagesize=(W, H))


def _wrap(text, font, size, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if c.stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def title_slide():
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 52)
    c.drawCentredString(W / 2, H - 180, "MindMate")
    c.setFillColor(MINT)
    c.setFont("Helvetica-Oblique", 24)
    c.drawCentredString(W / 2, H - 225, "An AI-Powered Mental Health Assistant for Students")
    c.setFillColor(white)
    lines = [
        ("Helvetica-Bold", 16, "M.Sc. (Data Science), Final Project Report"),
        ("Helvetica", 13, "Submitted by: Ganesh L   |   Enrolment No: O24MSD110165"),
        ("Helvetica", 13, "Guide: Mr. Prashant Stephens   |   In association with Plag Pro, Noida"),
        ("Helvetica", 13, "Chandigarh University · 2026"),
    ]
    y = 230
    for font, size, txt in lines:
        c.setFont(font, size)
        c.drawCentredString(W / 2, y, txt)
        y -= 30
    c.showPage()


def _frame(title, subtitle=None):
    c.setFillColor(white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(GREEN)
    c.rect(0, 0, 13, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(40, H - 55, title)
    if subtitle:
        c.setFillColor(GREEN)
        c.setFont("Helvetica-Oblique", 15)
        c.drawString(42, H - 80, subtitle)
    c.setFillColor(GREY)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 18, "MindMate, An AI-Powered Mental Health Assistant for Students")


def content_slide(title, bullets, subtitle=None):
    _frame(title, subtitle)
    y = H - (140 if subtitle else 115)
    c.setFillColor(NAVY)
    for b in bullets:
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(NAVY)
        c.drawString(50, y, "▸")
        for i, ln in enumerate(_wrap(b, "Helvetica", 15, 820)):
            c.setFont("Helvetica", 15)
            c.setFillColor(HexColor("#333333"))
            c.drawString(72, y, ln)
            y -= 24
        y -= 12
    c.showPage()


def image_slide(title, image_name, caption=None, bullets=None):
    _frame(title)
    img = FIG / image_name
    if bullets and img.exists():
        iw, ih = ImageReader(str(img)).getSize()
        aspect = ih / iw
        draw_w = 480
        draw_h = draw_w * aspect
        if draw_h > 360:
            draw_h = 360
            draw_w = draw_h / aspect
        c.drawImage(str(img), 40, (H - 110 - draw_h) / 2 + 30, width=draw_w,
                    height=draw_h, preserveAspectRatio=True, mask="auto")
        y = H - 150
        for b in bullets:
            c.setFillColor(NAVY)
            c.setFont("Helvetica-Bold", 15)
            c.drawString(560, y, "▸")
            for ln in _wrap(b, "Helvetica-Bold", 15, 330):
                c.drawString(580, y, ln)
                y -= 22
            y -= 12
    elif img.exists():
        iw, ih = ImageReader(str(img)).getSize()
        aspect = ih / iw
        draw_h = 360
        draw_w = draw_h / aspect
        if draw_w > 820:
            draw_w = 820
            draw_h = draw_w * aspect
        c.drawImage(str(img), (W - draw_w) / 2, 110, width=draw_w, height=draw_h,
                    preserveAspectRatio=True, mask="auto")
        if caption:
            c.setFillColor(GREY)
            c.setFont("Helvetica-Oblique", 11)
            for k, ln in enumerate(_wrap(caption, "Helvetica-Oblique", 11, 820)):
                c.drawCentredString(W / 2, 80 - k * 14, ln)
    c.showPage()


def divider(title, n):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(MINT)
    c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(W / 2, H / 2 + 30, f"{n:02d}")
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(W / 2, H / 2 - 20, title)
    c.showPage()


def closing():
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 48)
    c.drawCentredString(W / 2, H / 2 + 30, "Thank You")
    c.setFillColor(MINT)
    c.setFont("Helvetica-Oblique", 22)
    c.drawCentredString(W / 2, H / 2 - 20, "Questions & Discussion")
    c.setFillColor(LIGHT)
    c.setFont("Helvetica", 14)
    c.drawCentredString(W / 2, H / 2 - 60,
                        "Project repository: github.com/gannyprodigy/mindmate-mental-health-assistant")
    c.showPage()


def build():
    title_slide()
    content_slide("The Problem", [
        "Student mental-health needs are rising, but counselling capacity is limited (WHO, 2022).",
        "Stigma, cost and waiting times stop many students seeking help early.",
        "Generic apps rarely personalise support or handle crises transparently.",
        "Need: accessible, stigma-free, personalised first-line support, distinct from clinical care.",
    ], "Why MindMate?")
    content_slide("Project Objectives", [
        "Design and develop an AI mental-health assistant for students.",
        "Personalise its support using machine-learning algorithms.",
        "Evaluate its effectiveness on mental-health and academic outcomes.",
    ], "Three core objectives")
    content_slide("MindMate, Solution Overview", [
        "Streamlit web app: Home, Chat, Mood Tracker, Self-Check, Insights, Resources.",
        "Empathetic assistant (OpenAI API with a robust offline fallback engine).",
        "ML personalisation: sentiment, student segmentation, stress prediction, recommendations.",
        "Safety-first: every message screened for crisis signals before any reply.",
        "PHQ-9 and GAD-7 self-checks with validated severity bands.",
    ], "What we built")
    image_slide("System Architecture", "diagram_architecture.png", bullets=[
        "Presentation layer (Streamlit)",
        "Service layer orchestrates logic",
        "Assistant + ML domain layer",
        "Local SQLite persistence",
        "OpenAI API is optional",
    ])
    content_slide("Technology Stack", [
        "Python 3.12, Streamlit (multipage UI).",
        "scikit-learn (K-Means, Logistic Regression, PCA); VADER sentiment.",
        "OpenAI Chat Completions API + offline rule-based engine.",
        "pandas, numpy, scipy; plotly and matplotlib for visualisation.",
        "SQLite persistence; joblib models; 34 pytest tests.",
    ], "Open-source and reproducible")
    divider("Machine-Learning Personalisation", 1)
    image_slide("Student Segmentation (K-Means)", "fig_segment_pca.png", bullets=[
        "Segments: Thriving, Coping, At-Risk, High-Need",
        "Six wellbeing features",
        "Silhouette score 0.209",
        "Labels ranked by wellbeing index",
        "Each segment maps to a plan",
    ])
    image_slide("Stress Classification", "fig_confusion_matrix.png", bullets=[
        "Logistic Regression (Low/Mod/High)",
        "Test accuracy: 90.0%",
        "Macro-F1: 0.889",
        "Feeds dashboard + recommender",
        "Transparent probabilities",
    ])
    content_slide("Safety-First Design", [
        "Every message screened for crisis signals BEFORE a reply is generated.",
        "Three risk levels: none, elevated, crisis.",
        "Crisis -> calm message + 24x7 helplines (Tele-MANAS, Vandrevala, AASRA, iCall).",
        "Conservative by design: a missed crisis is far costlier than a false alarm.",
        "Never diagnoses; defers clinical concerns to professionals.",
    ], "Ethics and user safety")
    divider("Evaluation of Effectiveness", 2)
    image_slide("Pilot Evaluation Results", "fig_evaluation_prepost.png",
                caption="Simulated 4-week pilot (120 treatment vs 120 control). "
                        "Synthetic data, demonstrates the evaluation methodology, not clinical efficacy.")
    image_slide("Effect Sizes", "fig_effect_sizes.png", bullets=[
        "PHQ-9: d = 1.62 (large)",
        "GAD-7: d = 1.47 (large)",
        "Wellbeing: d = 1.85 (large)",
        "Focus hours: d = 1.44 (large)",
        "All p < 0.001 (synthetic)",
    ])
    content_slide("Testing & Validation", [
        "34 automated pytest tests (unit, integration, system), all passing.",
        "Safety-critical tests (crisis detection; no over-flagging of 'killing it in exams').",
        "Hold-out validation: accuracy 0.900, macro-F1 0.889; silhouette 0.209.",
        "Reproducible end-to-end via a fixed random seed.",
    ], "Quality assurance")
    content_slide("Conclusion & Future Scope", [
        "All three objectives met: working assistant, ML personalisation, reproducible evaluation.",
        "Contributions: transparent personalisation, safety-first + offline design, reproducible pipeline.",
        "Future: real IRB-approved user study; multilingual support; mobile app.",
        "Future: transformer emotion models; counselling-referral integration; secure cloud deployment.",
    ], "Outcomes and next steps")
    closing()
    c.save()
    print(f"Slides PDF written to {OUT}")
    print(f"Size: {OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    build()
