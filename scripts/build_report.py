"""Assemble the MindMate Final Project Report into a formatted PDF.

Reads the Markdown chapter files in ``docs/report/`` and renders a single
academic PDF that follows the prescribed formatting:

    * Times-style serif body, 12 pt, 1.5 line spacing, justified.
    * 1-inch margins, page numbers bottom-right.
    * Roman-numeral front matter, Arabic body.
    * Cover, certificate, declaration, acknowledgement, abstract.
    * Auto-generated, page-numbered Table of Contents, List of Figures and
      List of Tables.
    * Chapters 1-8, References (Chapter 9) and Appendices (Chapter 10).

Pure reportlab, no LaTeX or system PDF tooling required.

Usage::

    python -m scripts.build_report
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Preformatted,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "docs" / "report"
FIG_DIR = ROOT / "docs" / "figures"
OUT_PDF = REPORT_DIR / "MindMate_Final_Project_Report.pdf"

PAGE_W, PAGE_H = A4
MARGIN = inch
CONTENT_W = PAGE_W - 2 * MARGIN

NAVY = HexColor("#1f3b63")
ACCENT = HexColor("#2c6e49")
LIGHT = HexColor("#eef2f7")
GREY = HexColor("#666666")

# Module-level marker for where the Arabic body numbering begins.
_STATE = {"body_start": None}


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def build_styles():
    ss = getSampleStyleSheet()
    serif = "Times-Roman"
    serif_b = "Times-Bold"

    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontName=serif,
                          fontSize=12, leading=18, alignment=TA_JUSTIFY,
                          spaceAfter=8))
    ss.add(ParagraphStyle("CSChapter", parent=ss["Normal"], fontName=serif_b,
                          fontSize=18, leading=24, textColor=NAVY,
                          spaceBefore=6, spaceAfter=14, keepWithNext=True))
    ss.add(ParagraphStyle("CSH2", parent=ss["Normal"], fontName=serif_b,
                          fontSize=14, leading=19, textColor=NAVY,
                          spaceBefore=12, spaceAfter=6, keepWithNext=True))
    ss.add(ParagraphStyle("CSH3", parent=ss["Normal"], fontName=serif_b,
                          fontSize=12.5, leading=17, textColor=ACCENT,
                          spaceBefore=8, spaceAfter=4, keepWithNext=True))
    ss.add(ParagraphStyle("FigCaption", parent=ss["Normal"], fontName="Times-Italic",
                          fontSize=10, leading=13, alignment=TA_CENTER,
                          textColor=GREY, spaceBefore=4, spaceAfter=12))
    ss.add(ParagraphStyle("TblCaption", parent=ss["Normal"], fontName="Times-Italic",
                          fontSize=10, leading=13, alignment=TA_LEFT,
                          textColor=GREY, spaceBefore=8, spaceAfter=4,
                          keepWithNext=True))
    ss.add(ParagraphStyle("CodeBlock", parent=ss["Normal"], fontName="Courier",
                          fontSize=8.3, leading=10.4, textColor=HexColor("#222222"),
                          backColor=LIGHT, borderPadding=5, spaceBefore=6,
                          spaceAfter=10, leftIndent=2, firstLineIndent=0))
    ss.add(ParagraphStyle("BulletItem", parent=ss["Body"], leftIndent=18,
                          bulletIndent=6, spaceAfter=3))
    ss.add(ParagraphStyle("TableCell", parent=ss["Normal"], fontName=serif,
                          fontSize=9.5, leading=12, alignment=TA_LEFT))
    ss.add(ParagraphStyle("TableHead", parent=ss["Normal"], fontName=serif_b,
                          fontSize=9.5, leading=12, textColor=white,
                          alignment=TA_LEFT))
    # Cover / front-matter styles
    ss.add(ParagraphStyle("CoverTitle", parent=ss["Normal"], fontName=serif_b,
                          fontSize=30, leading=36, alignment=TA_CENTER,
                          textColor=NAVY, spaceAfter=8))
    ss.add(ParagraphStyle("CoverSub", parent=ss["Normal"], fontName="Times-Italic",
                          fontSize=15, leading=20, alignment=TA_CENTER,
                          textColor=ACCENT, spaceAfter=20))
    ss.add(ParagraphStyle("CoverText", parent=ss["Normal"], fontName=serif,
                          fontSize=13, leading=20, alignment=TA_CENTER,
                          spaceAfter=6))
    ss.add(ParagraphStyle("CoverBold", parent=ss["Normal"], fontName=serif_b,
                          fontSize=13, leading=20, alignment=TA_CENTER,
                          spaceAfter=6))
    ss.add(ParagraphStyle("FMHeading", parent=ss["Normal"], fontName=serif_b,
                          fontSize=16, leading=22, alignment=TA_CENTER,
                          textColor=NAVY, spaceAfter=16))
    ss.add(ParagraphStyle("TOCTitle", parent=ss["Normal"], fontName=serif_b,
                          fontSize=16, leading=22, textColor=NAVY, spaceAfter=12))
    return ss


STYLES = build_styles()


# --------------------------------------------------------------------------- #
# Inline markup
# --------------------------------------------------------------------------- #
def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="10">\1</font>', text)
    return text


# --------------------------------------------------------------------------- #
# Figure + table helpers
# --------------------------------------------------------------------------- #
def make_image(filename: str, max_h: float = 7.2 * inch):
    path = FIG_DIR / filename
    if not path.exists():
        return Paragraph(f"[missing figure: {filename}]", STYLES["FigCaption"])
    iw, ih = ImageReader(str(path)).getSize()
    aspect = ih / float(iw)
    draw_w = CONTENT_W * 0.92
    draw_h = draw_w * aspect
    if draw_h > max_h:
        draw_h = max_h
        draw_w = draw_h / aspect
    img = Image(str(path), width=draw_w, height=draw_h)
    img.hAlign = "CENTER"
    return img


def make_table(rows: list[list[str]]):
    header, *body = rows
    ncols = len(header)
    data = [[Paragraph(inline(c), STYLES["TableHead"]) for c in header]]
    for r in body:
        # pad/truncate to header width
        r = (r + [""] * ncols)[:ncols]
        data.append([Paragraph(inline(c), STYLES["TableCell"]) for c in r])
    col_w = CONTENT_W / ncols
    tbl = Table(data, colWidths=[col_w] * ncols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#b8c4d4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


# --------------------------------------------------------------------------- #
# Markdown -> flowables
# --------------------------------------------------------------------------- #
FIG_RE = re.compile(r"\[\[FIG:\s*([^|\]]+?)\s*\|\s*(.+?)\s*\]\]")


def parse_markdown(text: str, tbl_counter: dict, current_chapter: list) -> list:
    flow = []
    lines = text.split("\n")
    i = 0
    pending_list = []
    list_type = None
    last_heading = ""

    def flush_list():
        nonlocal pending_list, list_type
        if not pending_list:
            return
        from reportlab.platypus import ListFlowable, ListItem
        items = [ListItem(Paragraph(inline(t), STYLES["BulletItem"]),
                          value=None) for t in pending_list]
        bt = "bullet" if list_type == "ul" else "1"
        flow.append(ListFlowable(items, bulletType=bt,
                                 start="1" if list_type == "ol" else None,
                                 leftIndent=18))
        flow.append(Spacer(1, 4))
        pending_list = []
        list_type = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Code block
        if stripped.startswith("```"):
            flush_list()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code_text = "\n".join(code_lines)
            for chunk in _split_code(code_text):
                flow.append(Preformatted(chunk, STYLES["CodeBlock"]))
            continue

        # Figure directive
        m = FIG_RE.search(stripped)
        if m:
            flush_list()
            fname, caption = m.group(1).strip(), m.group(2).strip()
            flow.append(Spacer(1, 6))
            flow.append(make_image(fname))
            cap = Paragraph(inline(caption), STYLES["FigCaption"])
            flow.append(cap)
            i += 1
            continue

        # Table block
        if "|" in stripped and i + 1 < len(lines) and re.match(
                r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]) and "-" in lines[i + 1]:
            flush_list()
            tbl_rows = []
            while i < len(lines) and "|" in lines[i]:
                raw = lines[i].strip().strip("|")
                if re.match(r"^[\s:\-|]+$", raw):  # separator
                    i += 1
                    continue
                cells = [c.strip() for c in raw.split("|")]
                tbl_rows.append(cells)
                i += 1
            if tbl_rows:
                ch = current_chapter[0]
                tbl_counter[ch] = tbl_counter.get(ch, 0) + 1
                tnum = f"{ch}.{tbl_counter[ch]}"
                title = re.sub(r"^\d+(\.\d+)*\s+", "", last_heading or "Data table")
                flow.append(Paragraph(f"Table {tnum}: {title}", STYLES["TblCaption"]))
                flow.append(make_table(tbl_rows))
                flow.append(Spacer(1, 8))
            continue

        # Headings
        if stripped.startswith("# "):
            flush_list()
            htext = stripped[2:].strip()
            mnum = re.match(r"Chapter\s+(\d+)", htext)
            if mnum:
                current_chapter[0] = mnum.group(1)
            flow.append(Paragraph(inline(htext), STYLES["CSChapter"]))
            last_heading = htext
            i += 1
            continue
        if stripped.startswith("## "):
            flush_list()
            htext = stripped[3:].strip()
            flow.append(Paragraph(inline(htext), STYLES["CSH2"]))
            last_heading = htext
            i += 1
            continue
        if stripped.startswith("### "):
            flush_list()
            htext = stripped[4:].strip()
            flow.append(Paragraph(inline(htext), STYLES["CSH3"]))
            last_heading = htext
            i += 1
            continue

        # Lists
        mul = re.match(r"^[-*]\s+(.*)", stripped)
        mol = re.match(r"^\d+\.\s+(.*)", stripped)
        if mul:
            if list_type == "ol":
                flush_list()
            list_type = "ul"
            pending_list.append(mul.group(1))
            i += 1
            continue
        if mol:
            if list_type == "ul":
                flush_list()
            list_type = "ol"
            pending_list.append(mol.group(1))
            i += 1
            continue

        # Blank line
        if not stripped:
            flush_list()
            i += 1
            continue

        # Paragraph (gather consecutive non-blank, non-special lines)
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "```", "- ", "* "))
                    or re.match(r"^\d+\.\s", nxt) or FIG_RE.search(nxt)
                    or ("|" in nxt and i + 1 < len(lines)
                        and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]))):
                break
            para_lines.append(nxt)
            i += 1
        flush_list()
        flow.append(Paragraph(inline(" ".join(para_lines)), STYLES["Body"]))

    flush_list()
    return flow


def _split_code(code: str, max_lines: int = 46):
    lines = code.split("\n")
    for j in range(0, len(lines), max_lines):
        yield "\n".join(lines[j:j + max_lines])


# --------------------------------------------------------------------------- #
# Front matter
# --------------------------------------------------------------------------- #
def cover_page():
    P = lambda t, s: Paragraph(t, STYLES[s])
    el = [Spacer(1, 0.5 * inch),
          P("MindMate", "CoverTitle"),
          P("An AI-Powered Mental Health Assistant for Students", "CoverSub"),
          Spacer(1, 0.3 * inch),
          P("A Project Report submitted in partial fulfilment of the "
            "requirements for the award of the degree of", "CoverText"),
          P("Master of Science (Data Science)", "CoverBold"),
          Spacer(1, 0.35 * inch),
          P("Submitted by", "CoverText"),
          P("Ganesh L", "CoverBold"),
          P("Enrolment No: O24MSD110165", "CoverText"),
          Spacer(1, 0.3 * inch),
          P("Under the guidance of", "CoverText"),
          P("Mr. Prashant Stephens (Mentor)", "CoverBold"),
          Spacer(1, 0.3 * inch),
          P("In association with", "CoverText"),
          P("Plag Pro, Noida", "CoverBold"),
          Spacer(1, 0.5 * inch),
          P("Chandigarh University", "CoverBold"),
          P("2026", "CoverText")]
    el.append(NextPageTemplate("front"))
    el.append(PageBreak())
    return el


def simple_section(title: str, paragraphs: list[str], bold_title=True):
    el = [Paragraph(title, STYLES["FMHeading"]), Spacer(1, 6)]
    for p in paragraphs:
        el.append(Paragraph(inline(p), STYLES["Body"]))
    el.append(PageBreak())
    return el


def certificate_page():
    paras = [
        "This is to certify that the project report titled "
        "**MindMate, An AI-Powered Mental Health Assistant for "
        "Students** is a bona fide record of the project work carried "
        "out by **Ganesh L** (Enrolment No: O24MSD110165) in "
        "partial fulfilment of the requirements for the award of the degree "
        "of Master of Science (Data Science).",
        "The work was carried out under my supervision during the academic "
        "year 2026 in association with Plag Pro, Noida, and has not been "
        "submitted previously for the award of any other degree or diploma.",
        "",
        "_Copy of the certificate received from Qollabb to be pasted here._",
        "",
        "Mr. Prashant Stephens",
        "(Mentor / Guide)",
    ]
    return simple_section("Certificate", paras)


def declaration_page():
    paras = [
        "I, **Ganesh L**, hereby solemnly declare that the project "
        "report titled **MindMate, An AI-Powered Mental Health "
        "Assistant for Students**, submitted in partial fulfilment of "
        "the requirements for the award of the degree of Master of Science "
        "(Data Science), is my original work.",
        "I further declare that:",
        "• This project has been carried out by me during the academic "
        "year 2026 under the supervision of Mr. Prashant Stephens (Mentor).",
        "• The work has not been submitted previously to any other "
        "university, institution, or examination body for the award of any "
        "degree, diploma, or certification.",
        "• All sources of information used in this report have been duly "
        "acknowledged and referenced in accordance with academic ethics and "
        "plagiarism norms.",
        "• The data presented in this report is authentic to the best of "
        "my knowledge; the datasets used for model training and evaluation "
        "are clearly identified as synthetically generated for demonstration "
        "and no real personal data has been collected or used.",
        "",
        "Place: ____________________        Date: ____________________",
        "",
        "Student Signature: ____________________",
        "Student Name: Ganesh L        Enrolment No: O24MSD110165",
    ]
    return simple_section("Declaration", paras)


def acknowledgement_page():
    paras = [
        "The successful completion of this project would not have been "
        "possible without the guidance and support of several individuals, "
        "and I take this opportunity to express my sincere gratitude to them.",
        "I am deeply grateful to my mentor, Mr. Prashant Stephens, for his "
        "valuable guidance, constructive feedback, and continuous "
        "encouragement throughout the course of this project. His insights "
        "shaped both the technical direction and the ethical grounding of "
        "this work.",
        "I extend my sincere thanks to Plag Pro, Noida, for providing the "
        "opportunity and platform to undertake this project, and to the "
        "faculty of Chandigarh University for their academic support.",
        "Finally, I thank my family and friends for their patience and "
        "unwavering support during the development of MindMate.",
    ]
    return simple_section("Acknowledgement", paras)


def make_toc(title: str, kind: str, levels):
    toc = TableOfContents()
    toc._notifyKind = kind
    toc.levelStyles = levels
    return toc


def build_toc_objects():
    base = ParagraphStyle("toc0", fontName="Times-Roman", fontSize=12, leading=18)
    l1 = ParagraphStyle("toc1", parent=base, fontName="Times-Bold", spaceBefore=4)
    l2 = ParagraphStyle("toc2", parent=base, leftIndent=18, fontSize=11, leading=16)
    l3 = ParagraphStyle("toc3", parent=base, leftIndent=36, fontSize=10.5, leading=15,
                        textColor=GREY)
    contents = make_toc("Table of Contents", "TOCEntry", [l1, l2, l3])
    figs = make_toc("List of Figures", "LOFEntry", [base])
    tbls = make_toc("List of Tables", "LOTEntry", [base])
    return contents, figs, tbls


def abbreviations_page():
    rows = [
        ["Abbreviation", "Expansion"],
        ["AI", "Artificial Intelligence"],
        ["ML", "Machine Learning"],
        ["NLP", "Natural Language Processing"],
        ["VADER", "Valence Aware Dictionary and sEntiment Reasoner"],
        ["PHQ-9", "Patient Health Questionnaire (9-item)"],
        ["GAD-7", "Generalised Anxiety Disorder scale (7-item)"],
        ["CBT", "Cognitive Behavioural Therapy"],
        ["API", "Application Programming Interface"],
        ["UI / UX", "User Interface / User Experience"],
        ["DFD", "Data-Flow Diagram"],
        ["ER", "Entity-Relationship"],
        ["UML", "Unified Modelling Language"],
        ["PCA", "Principal Component Analysis"],
        ["CSV", "Comma-Separated Values"],
        ["SQL", "Structured Query Language"],
    ]
    el = [Paragraph("List of Abbreviations", STYLES["FMHeading"]), Spacer(1, 6),
          make_table(rows), PageBreak()]
    return el


# --------------------------------------------------------------------------- #
# References + appendices
# --------------------------------------------------------------------------- #
def references_flowables():
    text = (REPORT_DIR / "REFERENCES.md").read_text()
    el = [Paragraph("Chapter 9: References", STYLES["CSChapter"]),
          Paragraph("The following sources are cited in APA style. In-text "
                    "citations throughout the report refer to this list.",
                    STYLES["Body"]), Spacer(1, 6)]
    ref_style = ParagraphStyle("Ref", parent=STYLES["Body"], leftIndent=20,
                               firstLineIndent=-20, spaceAfter=7)
    for line in text.split("\n"):
        m = re.match(r"^\d+\.\s+(.*)", line.strip())
        if m:
            el.append(Paragraph(inline(m.group(1)), ref_style))
    return el


def appendices_flowables():
    el = [Paragraph("Chapter 10: Appendices", STYLES["CSChapter"])]
    el.append(Paragraph("Appendix A: Database Schema", STYLES["CSH2"]))
    el.append(Paragraph("The complete SQLite schema used by the application "
                        "is reproduced below.", STYLES["Body"]))
    from src.database import SCHEMA
    for chunk in _split_code(SCHEMA.strip()):
        el.append(Preformatted(chunk, STYLES["CodeBlock"]))

    # Key source listings.
    appendix_files = [
        ("Appendix B: Sentiment Analysis Module (src/ml/sentiment.py)", "src/ml/sentiment.py"),
        ("Appendix C: Student Segmentation Module (src/ml/personalization.py)", "src/ml/personalization.py"),
        ("Appendix D: Safety / Crisis-Detection Layer (src/assistant/safety.py)", "src/assistant/safety.py"),
        ("Appendix E: Conversational Engine (src/assistant/chat_engine.py)", "src/assistant/chat_engine.py"),
    ]
    for title, rel in appendix_files:
        path = ROOT / rel
        if not path.exists():
            continue
        el.append(Paragraph(title, STYLES["CSH2"]))
        code = path.read_text()
        for chunk in _split_code(code):
            el.append(Preformatted(chunk, STYLES["CodeBlock"]))

    el.append(Paragraph("Appendix F: Running the Project", STYLES["CSH2"]))
    el.append(Paragraph(
        "The application and its full source code are available in the "
        "project repository. After creating a virtual environment and "
        "installing the dependencies from <font face='Courier'>requirements.txt</font>, "
        "the data and models are produced with "
        "<font face='Courier'>python -m scripts.generate_data</font> and "
        "<font face='Courier'>python -m scripts.train_models</font>, and the "
        "application is launched with "
        "<font face='Courier'>streamlit run app.py</font>. The complete "
        "instructions are provided in the project README.", STYLES["Body"]))
    return el


# --------------------------------------------------------------------------- #
# Document template with numbering + TOC notifications
# --------------------------------------------------------------------------- #
def _roman(n: int) -> str:
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
            (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
            (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


from reportlab.platypus import Flowable


class BodyStart(Flowable):
    """A zero-size flowable that records the page where the body begins."""

    def wrap(self, *a):
        return (0, 0)

    def draw(self):
        _STATE["body_start"] = self.canv.getPageNumber()


from reportlab.pdfgen import canvas as _canvas


class NumberedCanvas(_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        body_start = _STATE.get("body_start")
        total = len(self._saved)
        for idx, state in enumerate(self._saved, start=1):
            self.__dict__.update(state)
            self._draw_number(idx, body_start)
            super().showPage()
        super().save()

    def _draw_number(self, page_idx: int, body_start):
        if page_idx == 1:
            return  # cover page: no number
        if body_start and page_idx >= body_start:
            label = str(page_idx - body_start + 1)
        else:
            label = _roman(page_idx)
        self.setFont("Times-Roman", 10)
        self.setFillColor(GREY)
        self.drawRightString(PAGE_W - MARGIN, MARGIN * 0.6, label)


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4,
                         leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN, **kw)
        frame = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="main")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame]),
            PageTemplate(id="front", frames=[frame]),
            PageTemplate(id="body", frames=[frame]),
        ])

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        text = flowable.getPlainText()
        if style == "CSChapter":
            self.notify("TOCEntry", (0, text, self.page))
        elif style == "CSH2":
            self.notify("TOCEntry", (1, text, self.page))
        elif style == "CSH3":
            self.notify("TOCEntry", (2, text, self.page))
        elif style == "FigCaption":
            self.notify("LOFEntry", (0, text, self.page))
        elif style == "TblCaption":
            self.notify("LOTEntry", (0, text, self.page))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build():
    _STATE["body_start"] = None
    contents_toc, figs_toc, tbls_toc = build_toc_objects()

    story = []
    # --- Cover (template 'cover') ---
    story += cover_page()           # ends with NextPageTemplate('front') + PageBreak

    # --- Front matter (roman) ---
    story += certificate_page()
    story += declaration_page()
    story += acknowledgement_page()

    # Abstract (split from chapter_1.md)
    ch1 = (REPORT_DIR / "chapter_1.md").read_text()
    if "# Chapter 1" in ch1:
        abstract_md, chapter1_md = ch1.split("# Chapter 1", 1)
        chapter1_md = "# Chapter 1" + chapter1_md
    else:
        abstract_md, chapter1_md = ch1, ""
    story.append(Paragraph("Abstract", STYLES["FMHeading"]))
    tbl_counter, cur_ch = {}, ["0"]
    abs_body = abstract_md.replace("# Abstract", "").strip()
    for para in parse_markdown(abs_body, tbl_counter, cur_ch):
        story.append(para)
    story.append(PageBreak())

    # TOC / LOF / LOT / Abbreviations
    story.append(Paragraph("Table of Contents", STYLES["TOCTitle"]))
    story.append(contents_toc)
    story.append(PageBreak())
    story.append(Paragraph("List of Figures", STYLES["TOCTitle"]))
    story.append(figs_toc)
    story.append(PageBreak())
    story.append(Paragraph("List of Tables", STYLES["TOCTitle"]))
    story.append(tbls_toc)
    story.append(PageBreak())
    story += abbreviations_page()

    # --- Body (arabic) ---
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    story.append(BodyStart())

    chapter_md = [chapter1_md]
    for n in range(2, 9):
        chapter_md.append((REPORT_DIR / f"chapter_{n}.md").read_text())

    for idx, md in enumerate(chapter_md):
        cur_ch = [str(idx + 1)]
        flow = parse_markdown(md, tbl_counter, cur_ch)
        story += flow
        story.append(PageBreak())

    story += references_flowables()
    story.append(PageBreak())
    story += appendices_flowables()

    doc = ReportDoc(str(OUT_PDF))
    doc.multiBuild(story, canvasmaker=NumberedCanvas)
    print(f"Report written to {OUT_PDF}")
    print(f"Size: {OUT_PDF.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    build()
