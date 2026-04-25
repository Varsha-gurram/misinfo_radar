"""
explainer.py
============
SHAP-based explainability wrapper — compatible with SHAP >= 0.45.
Provides global feature importance and local per-thread waterfall charts.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# ── Human-readable feature labels ───────────────────────────────────────────
FEATURE_LABELS = {
    "txt_src_vader_compound":  "Source tweet sentiment",
    "txt_src_vader_pos":       "Source positive tone",
    "txt_src_vader_neg":       "Source negative tone",
    "txt_src_vader_neu":       "Source neutral tone",
    "txt_mean_reply_compound": "Mean reply sentiment",
    "txt_sentiment_std":       "Sentiment variability",
    "txt_neg_reply_ratio":     "Negative reply ratio",
    "txt_pos_reply_ratio":     "Positive reply ratio",
    "txt_uncertainty_count":   "Uncertainty word count",
    "txt_negation_count":      "Negation word count",
    "txt_question_count":      "Question marks",
    "txt_exclaim_count":       "Exclamation marks",
    "txt_url_count":           "URL count",
    "txt_mention_count":       "@mention count",
    "txt_hashtag_count":       "Hashtag count",
    "txt_src_word_count":      "Source tweet length",
    "txt_avg_reply_word_count":"Avg reply length",
    "txt_total_word_count":    "Total word count",
    "tmp_n_replies":           "Reply count",
    "tmp_time_to_first_reply": "Time to first reply (s)",
    "tmp_mean_gap":            "Mean inter-reply gap (s)",
    "tmp_std_gap":             "Gap variability (s)",
    "tmp_max_gap":             "Max inter-reply gap (s)",
    "tmp_reply_velocity":      "Reply velocity (rpm)",
    "tmp_burst_score":         "Burst score",
    "tmp_span_seconds":        "Window span (s)",
    "str_tree_depth":          "Conversation depth",
    "str_avg_branching":       "Avg branching factor",
    "str_n_leaf_nodes":        "Leaf nodes",
    "str_n_nodes":             "Total nodes",
    "str_n_edges":             "Total edges",
    "str_structural_virality": "Structural virality",
    "str_max_outdegree":       "Max branching",
    "str_avg_outdegree":       "Avg outdegree",
    "usr_src_followers":       "Source followers",
    "usr_src_verified":        "Source verified",
    "usr_max_followers":       "Max follower count",
    "usr_mean_followers":      "Mean follower count",
    "usr_log_max_followers":   "Log max followers",
    "usr_verified_ratio":      "Verified user ratio",
    "usr_n_verified":          "# Verified users",
    "usr_mean_account_age_days":"Mean account age (days)",
    "usr_min_account_age_days": "Min account age (days)",
}

FEATURE_TOOLTIPS = {
    "txt_uncertainty_count":   "Words like 'allegedly', 'reportedly' — high counts signal epistemic doubt.",
    "txt_negation_count":      "Words like 'false', 'not', 'deny' — negation-heavy threads dispute claims.",
    "txt_sentiment_std":       "High variability in reply sentiment suggests disagreement or controversy.",
    "tmp_reply_velocity":      "Replies per minute — viral claims attract fast responses.",
    "tmp_burst_score":         "High score = replies arriving in rapid bursts (low time variation).",
    "tmp_time_to_first_reply": "Seconds until the first reply — shorter times signal higher engagement.",
    "str_structural_virality": "Avg shortest path between all users — high=chain-like; low=star shape.",
    "str_tree_depth":          "How deep the reply chain goes — deeper trees suggest sustained debate.",
    "str_avg_branching":       "How many replies branch per node — wide trees = mass reaction to source.",
    "usr_verified_ratio":      "Proportion of verified users — credible accounts may dampen spread.",
    "usr_log_max_followers":   "Logarithm of most-followed user — high-influence users change dynamics.",
    "usr_mean_account_age_days":"Older accounts tend to be more credible; very new ones may be bots.",
}


def get_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def get_tooltip(feature_name: str) -> str:
    return FEATURE_TOOLTIPS.get(feature_name, "")


# ── SHAP helpers ─────────────────────────────────────────────────────────────

def _extract_shap_array(shap_result) -> np.ndarray:
    """
    Normalise whatever shap returns into a 2-D (n_samples, n_features) array.
    Handles: ndarray, list-of-arrays (old binary), shap.Explanation objects.
    """
    # shap.Explanation object (SHAP >= 0.45 new API)
    if hasattr(shap_result, "values"):
        vals = shap_result.values
        if isinstance(vals, np.ndarray):
            # 3-D (n, features, classes) → take class-1
            if vals.ndim == 3:
                vals = vals[:, :, 1]
            return vals
        return np.array(vals)

    # old-style list [class0_array, class1_array]
    if isinstance(shap_result, list):
        arr = np.array(shap_result)
        if arr.ndim == 3:
            return arr[1]          # class-1
        return arr

    return np.array(shap_result)


def _get_base_value(explainer_obj, shap_result=None) -> float:
    """Robustly extract the base (expected) value."""
    # From Explanation object
    if shap_result is not None and hasattr(shap_result, "base_values"):
        bv = shap_result.base_values
        if isinstance(bv, np.ndarray):
            bv = bv.ravel()
            return float(bv[1]) if len(bv) > 1 else float(bv[0])
        return float(bv)

    ev = getattr(explainer_obj, "expected_value", 0.0)
    if isinstance(ev, (list, np.ndarray)):
        ev = np.array(ev).ravel()
        return float(ev[1]) if len(ev) > 1 else float(ev[0])
    return float(ev)


# ── Explainer class ───────────────────────────────────────────────────────────

class MisinfoExplainer:
    """SHAP wrapper — works with XGBoost, RF, and sklearn Pipelines."""

    def __init__(self, model, feature_cols: List[str]):
        self.model = model
        self.feature_cols = feature_cols
        self._explainer = None

    def _build_explainer(self, X_sample: np.ndarray):
        if self._explainer is not None:
            return
        try:
            # Try TreeExplainer first (XGBoost / RF)
            inner = self.model
            if hasattr(self.model, "named_steps"):
                inner = self.model.named_steps.get("clf", self.model)
            self._explainer = shap.TreeExplainer(inner)
        except Exception:
            try:
                # Fallback: generic Explainer
                self._explainer = shap.Explainer(self.model, X_sample)
            except Exception as e:
                logger.warning("Could not build SHAP explainer: %s", e)
                self._explainer = None

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        self._build_explainer(X)
        if self._explainer is None:
            return np.zeros_like(X)
        raw = self._explainer(X)
        return _extract_shap_array(raw)

    def _compute_raw(self, X: np.ndarray):
        """Return the raw SHAP result (Explanation or ndarray) for base-value extraction."""
        self._build_explainer(X)
        if self._explainer is None:
            return None
        return self._explainer(X)

    # ── Public API ───────────────────────────────────────────────────────────

    def global_importance_df(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        X = df[self.feature_cols].fillna(0).values
        sv = self.compute_shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        return pd.DataFrame({
            "feature":       self.feature_cols,
            "label":         [get_label(f) for f in self.feature_cols],
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    def local_explanation(self, feature_dict: dict) -> dict:
        x = np.array([[feature_dict.get(c, 0.0) for c in self.feature_cols]])
        raw = self._compute_raw(x)
        sv = _extract_shap_array(raw)[0] if raw is not None else np.zeros(len(self.feature_cols))
        base = _get_base_value(self._explainer, raw)
        return {
            "base_value":    base,
            "shap_values":   sv.tolist(),
            "feature_values":[feature_dict.get(c, 0.0) for c in self.feature_cols],
            "feature_names": self.feature_cols,
            "labels":        [get_label(f) for f in self.feature_cols],
        }

    # ── Matplotlib figures ────────────────────────────────────────────────────

    def plot_global_importance(
        self,
        df: pd.DataFrame,
        top_n: int = 15,
        title: str = "Global Feature Importance (mean |SHAP|)",
    ) -> plt.Figure:
        imp = self.global_importance_df(df, top_n=top_n)
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = [
            "#ef4444" if any(k in f for k in ("uncertainty", "negation", "velocity"))
            else "#6366f1"
            for f in imp["feature"]
        ]
        ax.barh(imp["label"][::-1], imp["mean_abs_shap"][::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    def plot_local_waterfall(
        self,
        feature_dict: dict,
        top_n: int = 12,
        title: str = "Local Explanation — Which features drove this prediction?",
    ) -> plt.Figure:
        expl = self.local_explanation(feature_dict)
        sv     = np.array(expl["shap_values"])
        labels = expl["labels"]
        vals   = expl["feature_values"]

        idx       = np.argsort(np.abs(sv))[-top_n:][::-1]
        sel_sv    = sv[idx]
        sel_labels= [labels[i] for i in idx]
        sel_vals  = [vals[i]   for i in idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = ["#ef4444" if s > 0 else "#22c55e" for s in sel_sv]
        y_pos   = np.arange(len(sel_sv))

        ax.barh(y_pos, sel_sv[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{l}  [{v:.2f}]" for l, v in zip(sel_labels[::-1], sel_vals[::-1])],
            fontsize=9,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on risk score)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        red_patch   = mpatches.Patch(color="#ef4444", label="↑ Risk increasing")
        green_patch = mpatches.Patch(color="#22c55e", label="↓ Risk reducing")
        ax.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig
