"""
evaluate.py
===========
Standalone evaluation script. Loads cv_results.pkl and prints a
formatted result table + generates comparison charts.

Usage:
    python evaluate.py
"""

import os
import pickle
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")

RESULTS_PATH = "results/cv_results.pkl"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results() -> dict:
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)


def print_table(all_results: dict):
    print("\n" + "=" * 80)
    print("LEAVE-ONE-EVENT-OUT EVALUATION")
    print("=" * 80)

    rows = []
    for n_key in [5, 10, 20, "full"]:
        for mt in ["lr", "rf", "xgboost"]:
            if n_key not in all_results or mt not in all_results[n_key]:
                continue
            ov = all_results[n_key][mt]["overall"]
            rows.append({
                "N": n_key,
                "Model": {"lr": "Logistic Reg", "rf": "Random Forest", "xgboost": "XGBoost"}[mt],
                "Acc": ov["accuracy"],
                "Precision": ov["precision"],
                "Recall": ov["recall"],
                "F1": ov["f1"],
                "F1-macro": ov["f1_macro"],
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.3f"))


def plot_window_comparison(all_results: dict):
    """Line chart: F1 vs early-window size for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    window_keys = [5, 10, 20, "full"]
    window_labels = ["N=5", "N=10", "N=20", "Full"]
    model_styles = {
        "lr": ("Logistic Reg", "#94a3b8", "o--"),
        "rf": ("Random Forest", "#f59e0b", "s-"),
        "xgboost": ("XGBoost", "#6366f1", "D-"),
    }

    for metric_key, metric_label, ax in [
        ("f1", "F1 (binary)", axes[0]),
        ("f1_macro", "F1 (macro)", axes[1]),
    ]:
        for mt, (label, color, style) in model_styles.items():
            values = []
            for nk in window_keys:
                try:
                    v = all_results[nk][mt]["overall"][metric_key]
                    values.append(v)
                except KeyError:
                    values.append(np.nan)
            ax.plot(window_labels, values, style, label=label, color=color, lw=2, ms=7)

        ax.set_title(f"{metric_label} vs Early Window Size", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Early Detection Performance vs Window Size — PHEME Dataset", fontsize=13)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "window_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", path)
    return fig


def plot_per_event(all_results: dict, n_key: int = 10, model_type: str = "xgboost"):
    """Bar chart: per-event F1 for the primary model."""
    folds = all_results[n_key][model_type]["folds"]
    events = [f["event"] for f in folds]
    f1s = [f["f1"] for f in folds]
    f1_macros = [f["f1_macro"] for f in folds]

    x = np.arange(len(events))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, f1s, width, label="F1 (binary)", color="#6366f1", alpha=0.85)
    b2 = ax.bar(x + width / 2, f1_macros, width, label="F1 (macro)", color="#f59e0b", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=20, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title(
        f"Per-Event Performance — XGBoost N={n_key} (LOEO CV)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"per_event_N{n_key}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", path)
    return fig


if __name__ == "__main__":
    if not os.path.exists(RESULTS_PATH):
        print(f"Results file not found: {RESULTS_PATH}")
        print("Run train.py first.")
        raise SystemExit(1)

    all_results = load_results()
    print_table(all_results)
    plot_window_comparison(all_results)
    plot_per_event(all_results)
    print(f"\nPlots saved to {PLOTS_DIR}/")
