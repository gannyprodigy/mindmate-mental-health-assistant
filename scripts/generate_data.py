"""Generate the synthetic student-wellbeing datasets used by MindMate.

Two datasets are produced:

1. ``data/synthetic_students.csv``, one row per simulated student with the
   six wellbeing features plus a derived stress label. This trains the
   K-Means segmenter and the Logistic-Regression stress classifier.

2. ``data/evaluation_study.csv``, a simulated 4-week pilot in which a
   treatment group uses MindMate and a control group does not, with pre/post
   PHQ-9, GAD-7 and wellbeing scores. This drives the Data Analysis &
   Interpretation chapter of the report.

The data is *synthetic but plausibly structured* (it encodes realistic
correlations, e.g. poor sleep ↔ higher anxiety). It is clearly labelled as
synthetic wherever it is used, so no real or fabricated human data is
involved. A fixed random seed makes every run fully reproducible.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.config import EVALUATION_DATASET, SENTIMENT_DATASET

RNG = np.random.default_rng(42)
N_STUDENTS = 600
N_EVAL = 240  # 120 treatment + 120 control


def _clip(value, lo, hi):
    return float(np.clip(value, lo, hi))


def _stress_label(avg_mood, sleep, phq9, gad7, sentiment) -> str:
    """Derive a stress label from features via an interpretable rule + noise.

    A latent 'distress index' is computed and thresholded. Gaussian noise is
    added so the relationship is learnable but not perfectly separable, 
    realistic for a supervised model.
    """
    distress = (
        (5 - avg_mood) * 0.9
        + max(0, 7 - sleep) * 0.6
        + (phq9 / 27) * 6
        + (gad7 / 21) * 5
        + (-sentiment) * 2
        + RNG.normal(0, 0.8)
    )
    if distress < 4.0:
        return "Low"
    if distress < 7.5:
        return "Moderate"
    return "High"


def generate_students(n: int = N_STUDENTS) -> list[dict]:
    rows = []
    for _ in range(n):
        # Latent wellbeing drives correlated features.
        wellbeing = RNG.beta(2.2, 2.2)  # 0 (poor) .. 1 (good)

        avg_mood = _clip(1 + wellbeing * 4 + RNG.normal(0, 0.4), 1, 5)
        sleep = _clip(4.5 + wellbeing * 4 + RNG.normal(0, 0.8), 3, 10)
        phq9 = _clip((1 - wellbeing) * 24 + RNG.normal(0, 2.5), 0, 27)
        gad7 = _clip((1 - wellbeing) * 18 + RNG.normal(0, 2.2), 0, 21)
        sentiment = _clip((wellbeing - 0.5) * 1.6 + RNG.normal(0, 0.2), -1, 1)
        engagement = _clip(RNG.poisson(3 + wellbeing * 4), 0, 20)

        label = _stress_label(avg_mood, sleep, phq9, gad7, sentiment)
        rows.append(
            {
                "avg_mood": round(avg_mood, 3),
                "avg_sleep_hours": round(sleep, 3),
                "phq9_score": round(phq9, 1),
                "gad7_score": round(gad7, 1),
                "avg_sentiment": round(sentiment, 3),
                "engagement": int(engagement),
                "stress_label": label,
            }
        )
    return rows


def generate_evaluation(n: int = N_EVAL) -> list[dict]:
    """Simulate a pre/post pilot study with treatment and control arms."""
    rows = []
    half = n // 2
    for i in range(n):
        group = "treatment" if i < half else "control"
        baseline_wb = RNG.beta(2.0, 2.6)  # both arms start similarly low-ish

        pre_phq9 = _clip((1 - baseline_wb) * 24 + RNG.normal(0, 2.0), 0, 27)
        pre_gad7 = _clip((1 - baseline_wb) * 18 + RNG.normal(0, 1.8), 0, 21)
        pre_wb = _clip(20 + baseline_wb * 60 + RNG.normal(0, 6), 0, 100)

        if group == "treatment":
            # MindMate use produces a moderate improvement with variance.
            phq9_drop = RNG.normal(4.2, 2.0)
            gad7_drop = RNG.normal(3.6, 1.8)
            wb_gain = RNG.normal(11.0, 5.0)
        else:
            # Control improves slightly (natural regression / time).
            phq9_drop = RNG.normal(1.1, 1.8)
            gad7_drop = RNG.normal(0.9, 1.6)
            wb_gain = RNG.normal(2.5, 4.5)

        post_phq9 = _clip(pre_phq9 - phq9_drop, 0, 27)
        post_gad7 = _clip(pre_gad7 - gad7_drop, 0, 21)
        post_wb = _clip(pre_wb + wb_gain, 0, 100)

        # A simple academic proxy: self-reported focus hours/week.
        pre_focus = _clip(8 + baseline_wb * 14 + RNG.normal(0, 3), 0, 40)
        focus_gain = RNG.normal(4.5, 2.5) if group == "treatment" else RNG.normal(1.0, 2.5)
        post_focus = _clip(pre_focus + focus_gain, 0, 45)

        rows.append(
            {
                "participant_id": i + 1,
                "group": group,
                "pre_phq9": round(pre_phq9, 1),
                "post_phq9": round(post_phq9, 1),
                "pre_gad7": round(pre_gad7, 1),
                "post_gad7": round(post_gad7, 1),
                "pre_wellbeing": round(pre_wb, 1),
                "post_wellbeing": round(post_wb, 1),
                "pre_focus_hours": round(pre_focus, 1),
                "post_focus_hours": round(post_focus, 1),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    students = generate_students()
    _write_csv(SENTIMENT_DATASET, students)
    print(f"Wrote {len(students)} student rows -> {SENTIMENT_DATASET}")

    evaluation = generate_evaluation()
    _write_csv(EVALUATION_DATASET, evaluation)
    print(f"Wrote {len(evaluation)} evaluation rows -> {EVALUATION_DATASET}")


if __name__ == "__main__":
    main()
