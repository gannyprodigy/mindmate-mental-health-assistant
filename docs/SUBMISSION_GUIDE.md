# MindMate, Submission Guide (Qollabb)

Everything you need to complete the project on the Qollabb portal before the
**30 June 2026** deadline.

## 1. Deliverables produced (where to find them)

| Deliverable | File |
|-------------|------|
| Final Project Report (PDF) | `docs/report/MindMate_Final_Project_Report.pdf` |
| Final Presentation (PPTX) | `docs/presentation/MindMate_Final_Presentation.pptx` |
| Final Presentation (PDF)  | `docs/presentation/MindMate_Final_Presentation_PDF.pdf` |
| Working application | `app.py` (run with `streamlit run app.py`) |
| Source code link | your GitHub repo (see `docs/GITHUB_SETUP.md`) |

The report PDF is ~1.1 MB and the presentation files are <0.5 MB each, all
comfortably under the 20 MB Qollabb limit.

## 2. Before submitting, personalise

1. Replace `[Student Name]` and `[Enrolment Number]` (see `docs/GITHUB_SETUP.md`,
   last section) and rebuild the report and slides.
2. Paste your Qollabb certificate image onto the **Certificate** page of the
   report (the page currently holds a placeholder note, per the template).
3. (Recommended) Capture app screenshots and add them to the report's
   Implementation/Results chapters, run the app and use your screenshot tool,
   or `python -m scripts.capture_screenshots` (needs Playwright). Save them in
   `docs/screenshots/`.

## 3. Map to the six milestones (mark each complete on Qollabb)

| Milestone | Covered by |
|-----------|-----------|
| 1. Literature Review | Report Chapter 2 (Literature Review & System Study) |
| 2. Design AI Assistant | Report Chapter 4 (System Design); UI in `src/ui/` |
| 3. Develop Personalization Algorithms | `src/ml/` (segmentation, stress, recommender); Chapter 5 |
| 4. User Testing | `tests/` (34 tests); Chapter 6 (Testing) |
| 5. Impact Evaluation | `scripts/analyze_evaluation.py`; Chapter 7 (Results & Discussion) |
| 6. Final Adjustments | Final report, slides, README, repo |

## 4. Submit on the "Final Report Submission" tab

1. **Final Project Report** → upload `MindMate_Final_Project_Report.pdf`.
2. **Final Presentation** → upload the `.pptx` (or the `_PDF.pdf` if you prefer PDF).
3. **Video (optional)** → record a 2-3 minute screen capture of the app if you wish.
4. **Link to your work** → paste the GitHub repo URL (already created & public):
   **<https://github.com/gannyprodigy/mindmate-mental-health-assistant>**
   (and a live demo URL too, if you deploy one, see `docs/GITHUB_SETUP.md`).
5. Click **Submit** (use **Clear** only to reset).

## 5. After submission
- The **Evaluation Status** tab unlocks once you submit.
- Your mentor evaluates the work and conducts a **viva**, use the slides to present.
- A **certificate** is issued after evaluation and viva.

## 6. Confirm scope with your mentor (recommended)
The mentor's group-chat instructions describe a formal written report
(15,000+ words, 10-part structure, PDF only). This submission satisfies that
**and** delivers the working build. If unsure whether the practical build, the
written report, or both are expected, confirm with Mr. Stephens, but this
package covers all three.

## How this report meets the stated requirements

- **Length:** ~24,000 words / 117 pages (MSc range is 18,000-30,000 words).
- **Format:** PDF, Times serif 12 pt, 1.5 spacing, 1-inch margins, justified.
- **Structure:** cover, certificate, declaration, acknowledgement, abstract,
  table of contents, list of figures, list of tables, list of abbreviations,
  Chapters 1-10, 28 references (APA), appendices with source code.
- **Page numbers:** Roman front matter, Arabic body, bottom-right.
- **Working demo:** full Streamlit app + GitHub link (avoids the "theory only"
  pitfall the guidelines warn against).
- **Originality:** all prose is original; sources are paraphrased and cited.
