"""Train and persist the MindMate machine-learning models.

Run order::

    python -m scripts.generate_data   # creates the datasets
    python -m scripts.train_models    # trains + evaluates + saves models

The script prints evaluation metrics (silhouette score for clustering;
accuracy, macro-F1 and a classification report for the stress classifier) so
the numbers quoted in the project report can be reproduced exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, f1_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split

from src.config import SEGMENTER_PATH, SENTIMENT_DATASET, STRESS_MODEL_PATH
from src.ml.personalization import (FEATURE_COLUMNS as SEG_FEATURES,
                                    train_segmenter)
from src.ml.stress import FEATURE_COLUMNS as CLF_FEATURES, train_classifier


def load_dataset() -> pd.DataFrame:
    if not SENTIMENT_DATASET.exists():
        raise FileNotFoundError(
            f"{SENTIMENT_DATASET} not found. Run `python -m scripts.generate_data` first."
        )
    return pd.read_csv(SENTIMENT_DATASET)


def train_segmentation(df: pd.DataFrame) -> None:
    X = df[SEG_FEATURES].to_numpy(dtype=float)
    segmenter = train_segmenter(X, n_clusters=4)

    labels = segmenter.pipeline.named_steps["kmeans"].labels_
    scaled = segmenter.pipeline.named_steps["scaler"].transform(X)
    sil = silhouette_score(scaled, labels)

    segmenter.save(SEGMENTER_PATH)
    print("\n=== Student Segmenter (K-Means, k=4) ===")
    print(f"Silhouette score : {sil:.3f}")
    print("Cluster -> label : ", segmenter.cluster_to_label)
    counts = pd.Series([segmenter.cluster_to_label[c] for c in labels]).value_counts()
    print("Segment sizes    :")
    for name, n in counts.items():
        print(f"   {name:<10} {n}")


def train_stress(df: pd.DataFrame) -> None:
    X = df[CLF_FEATURES].to_numpy(dtype=float)
    y = df["stress_label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = train_classifier(X_train, y_train)
    preds = clf.pipeline.predict(X_test)

    acc = float((preds == y_test).mean())
    macro_f1 = f1_score(y_test, preds, average="macro")

    clf.save(STRESS_MODEL_PATH)
    print("\n=== Stress Classifier (Logistic Regression) ===")
    print(f"Test accuracy : {acc:.3f}")
    print(f"Macro F1      : {macro_f1:.3f}")
    print(classification_report(y_test, preds, digits=3))


def main() -> None:
    df = load_dataset()
    print(f"Loaded {len(df)} rows from {SENTIMENT_DATASET}")
    train_segmentation(df)
    train_stress(df)
    print("\nModels saved to:")
    print(f"   {SEGMENTER_PATH}")
    print(f"   {STRESS_MODEL_PATH}")


if __name__ == "__main__":
    main()
