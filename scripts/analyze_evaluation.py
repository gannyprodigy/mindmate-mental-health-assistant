"""Statistical analysis of the simulated MindMate pilot study.

Produces the figures and numbers reported in the *Data Analysis &
Interpretation* chapter:

    * Group means (pre/post) for PHQ-9, GAD-7, wellbeing and focus hours.
    * Within-group change scores.
    * Welch's t-test comparing treatment vs control change scores.
    * Cohen's d effect sizes.
    * A paired t-test on the treatment group's pre/post wellbeing.

Results are printed and also written to ``data/evaluation_summary.csv`` so
the report tables can be reproduced exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_DIR, EVALUATION_DATASET


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for the difference in means of two independent samples."""
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    )
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd else 0.0


def analyse_metric(df: pd.DataFrame, pre: str, post: str, lower_is_better: bool):
    """Compare treatment vs control change for one metric."""
    df = df.copy()
    df["change"] = df[post] - df[pre]

    t = df[df["group"] == "treatment"]["change"].to_numpy()
    c = df[df["group"] == "control"]["change"].to_numpy()

    tstat, pval = stats.ttest_ind(t, c, equal_var=False)
    d = cohens_d(t, c)

    return {
        "metric": pre.replace("pre_", ""),
        "treatment_pre": round(df[df.group == "treatment"][pre].mean(), 2),
        "treatment_post": round(df[df.group == "treatment"][post].mean(), 2),
        "control_pre": round(df[df.group == "control"][pre].mean(), 2),
        "control_post": round(df[df.group == "control"][post].mean(), 2),
        "treatment_change": round(t.mean(), 2),
        "control_change": round(c.mean(), 2),
        "t_stat": round(float(tstat), 3),
        "p_value": round(float(pval), 5),
        "cohens_d": round(d, 3),
        "direction": "improvement" if (lower_is_better and t.mean() < 0)
                     or (not lower_is_better and t.mean() > 0) else ", ",
    }


def main() -> None:
    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(
            f"{EVALUATION_DATASET} not found. Run `python -m scripts.generate_data` first."
        )
    df = pd.read_csv(EVALUATION_DATASET)
    print(f"Loaded {len(df)} participants "
          f"({(df.group == 'treatment').sum()} treatment / "
          f"{(df.group == 'control').sum()} control)\n")

    specs = [
        ("pre_phq9", "post_phq9", True),
        ("pre_gad7", "post_gad7", True),
        ("pre_wellbeing", "post_wellbeing", False),
        ("pre_focus_hours", "post_focus_hours", False),
    ]
    results = [analyse_metric(df, pre, post, lib) for pre, post, lib in specs]
    summary = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))

    # Paired t-test: treatment wellbeing pre vs post.
    treat = df[df.group == "treatment"]
    tstat, pval = stats.ttest_rel(treat["post_wellbeing"], treat["pre_wellbeing"])
    print(
        f"\nPaired t-test (treatment wellbeing pre vs post): "
        f"t = {tstat:.3f}, p = {pval:.6f}"
    )

    out = DATA_DIR / "evaluation_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
