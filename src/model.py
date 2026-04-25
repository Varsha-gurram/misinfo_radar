"""
model.py
========
Three classifiers (Logistic Regression, Random Forest, XGBoost) with
leave-one-event-out cross-validation and model serialisation via joblib.
"""

import os
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier

from src.feature_extractor import get_feature_cols

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

EVENTS = [
    "charliehebdo",
    "ferguson",
    "germanwings-crash",
    "ottawashooting",
    "sydneysiege",
    "ebola-essien",
    "gurlitt",
    "prince-toronto",
    "putinmissing",
]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────── builders ──────────────────────────────────────

def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
        )),
    ])


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


# ─────────────────────────── LOEO CV ───────────────────────────────────────

def leave_one_event_out_cv(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    label_col: str = "label_binary",
) -> dict:
    """
    Leave-one-event-out cross-validation.
    Returns per-fold and aggregated metrics.
    """
    feature_cols = get_feature_cols(df)
    results = []

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for test_event in EVENTS:
        train_df = df[df["event"] != test_event].copy()
        test_df = df[df["event"] == test_event].copy()

        if len(test_df) == 0:
            logger.warning("No samples for event: %s", test_event)
            continue

        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df[label_col].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df[label_col].values

        # handle class imbalance for XGB
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        spw = neg / max(1, pos)

        if model_type == "lr":
            model = build_logistic_regression()
        elif model_type == "rf":
            model = build_random_forest()
        else:
            model = build_xgboost(scale_pos_weight=spw)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "named_steps"):
            y_prob = model.named_steps["clf"].predict_proba(
                model.named_steps["scaler"].transform(X_test)
            )[:, 1]
        else:
            y_prob = y_pred.astype(float)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        fold_result = {
            "event": test_event,
            "n_test": len(y_test),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
        results.append(fold_result)
        logger.info(
            "Event %-20s | Acc %.3f | F1 %.3f | F1-macro %.3f",
            test_event,
            fold_result["accuracy"],
            fold_result["f1"],
            fold_result["f1_macro"],
        )

    overall = {
        "accuracy": accuracy_score(all_y_true, all_y_pred),
        "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        "recall": recall_score(all_y_true, all_y_pred, zero_division=0),
        "f1": f1_score(all_y_true, all_y_pred, zero_division=0),
        "f1_macro": f1_score(all_y_true, all_y_pred, average="macro", zero_division=0),
    }
    return {"folds": results, "overall": overall, "y_true": all_y_true, "y_pred": all_y_pred, "y_prob": all_y_prob}


# ─────────────────────────── full train ────────────────────────────────────

def train_final_model(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    label_col: str = "label_binary",
    save_path: Optional[str] = None,
) -> tuple:
    """Train on the full dataset and return (model, feature_cols)."""
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].fillna(0).values
    y = df[label_col].values

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    spw = neg / max(1, pos)

    if model_type == "lr":
        model = build_logistic_regression()
    elif model_type == "rf":
        model = build_random_forest()
    else:
        model = build_xgboost(scale_pos_weight=spw)

    model.fit(X, y)
    logger.info("Trained %s on %d samples", model_type, len(y))

    if save_path:
        joblib.dump({"model": model, "feature_cols": feature_cols}, save_path)
        logger.info("Model saved → %s", save_path)

    return model, feature_cols


# ─────────────────────────── loader ────────────────────────────────────────

def load_model(path: str) -> tuple:
    """Load (model, feature_cols) from joblib file."""
    obj = joblib.load(path)
    return obj["model"], obj["feature_cols"]


# ─────────────────────────── prediction ────────────────────────────────────

def predict_thread(
    model,
    feature_cols: list[str],
    feature_dict: dict,
) -> tuple[int, float]:
    """
    Predict label and risk score for a single thread.
    Returns (label: int, risk_score: float 0-1).
    """
    x = np.array([[feature_dict.get(c, 0.0) for c in feature_cols]])
    label = int(model.predict(x)[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x)[0, 1])
    elif hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
        prob = float(clf.predict_proba(scaler.transform(x))[0, 1])
    else:
        prob = float(label)
    return label, prob
