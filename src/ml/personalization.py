"""Student segmentation for personalised support.

An unsupervised K-Means model groups students into behavioural *segments*
based on their wellbeing indicators (average mood, sleep, screening scores,
sentiment trend and engagement). Each segment is given a human-readable
label and a tailored support strategy that the recommender draws upon.

The model is trained offline by ``scripts/train_models.py`` and persisted
with joblib; at runtime we simply load the pipeline and call
:meth:`StudentSegmenter.predict`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import SEGMENTER_PATH

# Feature order used everywhere, keep stable for serialisation.
FEATURE_COLUMNS: List[str] = [
    "avg_mood",          # 1..5
    "avg_sleep_hours",   # hours
    "phq9_score",        # 0..27
    "gad7_score",        # 0..21
    "avg_sentiment",     # -1..1
    "engagement",        # interactions per week
]

# Human-readable interpretation attached to each cluster after training.
# These labels are assigned in :func:`label_clusters` by ranking the cluster
# centroids on a composite wellbeing index, so they remain meaningful even
# though K-Means itself produces arbitrary cluster ids.
SEGMENT_LABELS = [
    "Thriving",            # high wellbeing
    "Coping",              # moderate wellbeing
    "At-Risk",             # low wellbeing
    "High-Need",           # very low wellbeing / high distress
]

SEGMENT_STRATEGIES: Dict[str, str] = {
    "Thriving": (
        "Maintain healthy routines. Offer growth-oriented content "
        "(focus techniques, goal setting) and light check-ins."
    ),
    "Coping": (
        "Reinforce protective habits and provide proactive stress-management "
        "tools before exam peaks. Encourage regular mood logging."
    ),
    "At-Risk": (
        "Prioritise grounding and emotional-regulation exercises, suggest "
        "structured routines, and surface counselling resources gently."
    ),
    "High-Need": (
        "Lead with validation and safety. Strongly but compassionately "
        "surface professional-support pathways and crisis resources."
    ),
}


@dataclass
class StudentSegmenter:
    """Wrapper around a fitted K-Means pipeline + cluster->label mapping."""

    pipeline: Pipeline
    cluster_to_label: Dict[int, str]

    def predict(self, features: Dict[str, float]) -> str:
        """Return the segment label for a single feature dictionary."""
        vector = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]])
        cluster = int(self.pipeline.predict(vector)[0])
        return self.cluster_to_label.get(cluster, "Coping")

    def predict_many(self, matrix: np.ndarray) -> List[str]:
        clusters = self.pipeline.predict(matrix)
        return [self.cluster_to_label.get(int(c), "Coping") for c in clusters]

    def strategy_for(self, label: str) -> str:
        return SEGMENT_STRATEGIES.get(label, SEGMENT_STRATEGIES["Coping"])

    def save(self, path: Optional[Path] = None) -> None:
        joblib.dump(
            {"pipeline": self.pipeline, "cluster_to_label": self.cluster_to_label},
            path or SEGMENTER_PATH,
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "StudentSegmenter":
        payload = joblib.load(path or SEGMENTER_PATH)
        return cls(payload["pipeline"], payload["cluster_to_label"])


def _wellbeing_index(centroid: np.ndarray) -> float:
    """Composite score: higher = better wellbeing.

    centroid is in the *original* feature space (after inverse transform).
    Higher mood / sleep / sentiment is good; higher PHQ-9 / GAD-7 is bad.
    """
    avg_mood, avg_sleep, phq9, gad7, sentiment, _engagement = centroid
    return (
        (avg_mood / 5.0)
        + (min(avg_sleep, 9.0) / 9.0)
        + ((sentiment + 1.0) / 2.0)
        - (phq9 / 27.0)
        - (gad7 / 21.0)
    )


def label_clusters(pipeline: Pipeline, n_clusters: int) -> Dict[int, str]:
    """Map arbitrary K-Means cluster ids to ordered wellbeing labels."""
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    kmeans: KMeans = pipeline.named_steps["kmeans"]
    # Bring centroids back to original feature units for interpretation.
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    indices = list(range(n_clusters))
    # Rank clusters by wellbeing (descending) and assign ordered labels.
    ranked = sorted(indices, key=lambda i: _wellbeing_index(centroids[i]), reverse=True)
    labels = SEGMENT_LABELS[:n_clusters]
    return {cluster_id: labels[rank] for rank, cluster_id in enumerate(ranked)}


def train_segmenter(matrix: np.ndarray, n_clusters: int = 4,
                    random_state: int = 42) -> StudentSegmenter:
    """Fit the standardisation + K-Means pipeline on a feature matrix."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=10,
                              random_state=random_state)),
        ]
    )
    pipeline.fit(matrix)
    mapping = label_clusters(pipeline, n_clusters)
    return StudentSegmenter(pipeline=pipeline, cluster_to_label=mapping)
