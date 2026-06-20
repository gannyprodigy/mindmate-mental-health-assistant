"""Generate the statistical figures used in the project report.

Every figure is produced from the *actual* trained models and datasets, so
the report's plots stay consistent with its quoted numbers. Outputs are PNGs
in ``docs/figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.config import EVALUATION_DATASET, SENTIMENT_DATASET
from src.ml.personalization import FEATURE_COLUMNS, train_segmenter
from src.ml.stress import train_classifier

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,  # use plain hyphen-minus on tick labels
})

SEG_COLORS = {"Thriving": "#2ca02c", "Coping": "#1f77b4",
              "At-Risk": "#ff7f0e", "High-Need": "#d62728"}
STRESS_COLORS = {"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728"}


def _save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIG_DIR.parent.parent)}")


def fig_segment_distribution(df: pd.DataFrame):
    seg = train_segmenter(df[FEATURE_COLUMNS].to_numpy(float), 4)
    labels = [seg.cluster_to_label[c]
              for c in seg.pipeline.named_steps["kmeans"].labels_]
    counts = pd.Series(labels).value_counts().reindex(
        ["Thriving", "Coping", "At-Risk", "High-Need"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values,
           color=[SEG_COLORS[s] for s in counts.index])
    ax.set_title("Student distribution across support segments")
    ax.set_ylabel("Number of students")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, int(v), ha="center")
    _save(fig, "fig_segment_distribution.png")
    return seg, labels


def fig_segment_pca(df: pd.DataFrame, seg, labels):
    X = df[FEATURE_COLUMNS].to_numpy(float)
    scaled = seg.pipeline.named_steps["scaler"].transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["Thriving", "Coping", "At-Risk", "High-Need"]:
        mask = np.array(labels) == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.7,
                   label=name, color=SEG_COLORS[name])
    ax.set_title("Student segments (PCA projection)")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(title="Segment")
    _save(fig, "fig_segment_pca.png")


def fig_correlation(df: pd.DataFrame):
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    short = ["mood", "sleep", "PHQ-9", "GAD-7", "sentiment", "engagement"]
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation matrix")
    _save(fig, "fig_correlation.png")


def fig_confusion(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].to_numpy(float)
    y = df["stress_label"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
    clf = train_classifier(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(
        yte, clf.pipeline.predict(Xte),
        labels=["Low", "Moderate", "High"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Stress classifier, confusion matrix (test set)")
    ax.grid(False)
    _save(fig, "fig_confusion_matrix.png")


def fig_evaluation_prepost(ev: pd.DataFrame):
    metrics = [("phq9", "PHQ-9 (lower = better)"),
               ("gad7", "GAD-7 (lower = better)"),
               ("wellbeing", "Wellbeing (higher = better)"),
               ("focus_hours", "Weekly focus hours (higher = better)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (key, title) in zip(axes.flat, metrics):
        data = []
        for grp in ["treatment", "control"]:
            sub = ev[ev.group == grp]
            data.append((sub[f"pre_{key}"].mean(), sub[f"post_{key}"].mean()))
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, [d[0] for d in data], w, label="Pre", color="#9ecae1")
        ax.bar(x + w / 2, [d[1] for d in data], w, label="Post", color="#3182bd")
        ax.set_xticks(x)
        ax.set_xticklabels(["Treatment", "Control"])
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Pre/post comparison: MindMate (treatment) vs control",
                 fontsize=13)
    _save(fig, "fig_evaluation_prepost.png")


def fig_effect_sizes():
    summary_path = EVALUATION_DATASET.parent / "evaluation_summary.csv"
    if not summary_path.exists():
        print("  (skip effect sizes, run analyze_evaluation first)")
        return
    s = pd.read_csv(summary_path)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    d = s["cohens_d"].abs()
    colors = ["#2ca02c" if v >= 0.8 else "#ff7f0e" if v >= 0.5 else "#d62728"
              for v in d]
    ax.barh(s["metric"], d, color=colors)
    ax.axvline(0.8, ls="--", color="grey", lw=1)
    ax.text(0.81, -0.4, "large effect (0.8)", fontsize=8, color="grey")
    ax.set_xlabel("|Cohen's d| (treatment vs control change)")
    ax.set_title("Effect sizes by outcome measure")
    _save(fig, "fig_effect_sizes.png")


def fig_sentiment_demo():
    """Illustrative sentiment trace for a sample of journal notes."""
    samples = [
        ("Day 1", "Exam stress is crushing me, can't sleep"),
        ("Day 3", "Tried the breathing exercise, a little calmer"),
        ("Day 5", "Still anxious but managing my study sprints"),
        ("Day 8", "Felt okay today, got some good revision done"),
        ("Day 12", "Actually feeling hopeful about the exam now"),
    ]
    from src.ml.sentiment import compound_score
    scores = [compound_score(t) for _, t in samples]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if v < -0.05 else "#2ca02c" if v > 0.05 else "#ff7f0e"
              for v in scores]
    ax.plot(range(len(scores)), scores, color="#555", zorder=1)
    ax.scatter(range(len(scores)), scores, color=colors, s=80, zorder=2)
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([d for d, _ in samples])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("VADER compound sentiment")
    ax.set_title("Sentiment of a student's journal notes over time")
    _save(fig, "fig_sentiment_trend.png")


def main():
    df = pd.read_csv(SENTIMENT_DATASET)
    ev = pd.read_csv(EVALUATION_DATASET)
    print("Generating figures...")
    seg, labels = fig_segment_distribution(df)
    fig_segment_pca(df, seg, labels)
    fig_correlation(df)
    fig_confusion(df)
    fig_evaluation_prepost(ev)
    fig_effect_sizes()
    fig_sentiment_demo()
    print(f"Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
