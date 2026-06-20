"""Supervised stress-level classifier.

A multinomial Logistic Regression predicts a student's stress level
(``Low`` / ``Moderate`` / ``High``) from the same wellbeing features used by
the segmenter. The model is trained on the synthetic-but-realistic dataset
produced by ``scripts/generate_data.py`` and persisted with joblib.

The classifier feeds two parts of the product:
    * the dashboard, which shows the predicted stress level and probability;
    * the recommender, which escalates strategy intensity with stress.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import STRESS_MODEL_PATH

FEATURE_COLUMNS: List[str] = [
    "avg_mood",
    "avg_sleep_hours",
    "phq9_score",
    "gad7_score",
    "avg_sentiment",
    "engagement",
]

STRESS_LEVELS = ["Low", "Moderate", "High"]


@dataclass
class StressClassifier:
    pipeline: Pipeline
    classes: List[str]

    def predict(self, features: Dict[str, float]) -> str:
        vector = np.array([[features.get(c, 0.0) for c in FEATURE_COLUMNS]])
        return str(self.pipeline.predict(vector)[0])

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        vector = np.array([[features.get(c, 0.0) for c in FEATURE_COLUMNS]])
        probs = self.pipeline.predict_proba(vector)[0]
        ordered = list(self.pipeline.classes_)
        return {str(cls): float(p) for cls, p in zip(ordered, probs)}

    def save(self, path: Optional[Path] = None) -> None:
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes},
                    path or STRESS_MODEL_PATH)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "StressClassifier":
        payload = joblib.load(path or STRESS_MODEL_PATH)
        return cls(payload["pipeline"], payload["classes"])


def train_classifier(matrix: np.ndarray, labels: np.ndarray,
                     random_state: int = 42) -> StressClassifier:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    pipeline.fit(matrix, labels)
    return StressClassifier(pipeline=pipeline, classes=STRESS_LEVELS)
