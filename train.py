"""
train.py
========
Offline training script.
Run ONCE to build the feature matrix and train all models.

Usage:
    python train.py --data data/pheme-rnr-annotation-v1.0

This script will:
  1. Parse the PHEME dataset and save pheme_processed.pkl
  2. Build feature matrices for N=5, 10, 20 and full thread
  3. Run leave-one-event-out CV for all three classifiers
  4. Train final XGBoost model on the full dataset
  5. Save models/xgb_model_N*.joblib and results/cv_results.pkl
"""

import os
import sys
import argparse
import pickle
import logging
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/pheme-rnr-annotation-v1.0",
        help="Path to PHEME root directory",
    )
    parser.add_argument(
        "--processed",
        default="data/pheme_processed.pkl",
        help="Path to save/load processed threads",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip parsing if pheme_processed.pkl already exists",
    )
    args = parser.parse_args()

    # ── 1. Load / parse dataset ───────────────────────────────────────────
    from src.data_loader import load_pheme_dataset, save_processed, load_processed

    if args.skip_parse and os.path.exists(args.processed):
        logger.info("Loading pre-parsed threads from %s", args.processed)
        threads = load_processed(args.processed)
    else:
        logger.info("Parsing PHEME dataset from %s", args.data)
        threads = load_pheme_dataset(args.data)
        save_processed(threads, args.processed)

    logger.info("Total threads: %d", len(threads))

    # ── 2. Build feature matrices ─────────────────────────────────────────
    from src.feature_extractor import build_feature_matrix

    all_results = {}
    feature_dfs = {}

    for n in [5, 10, 20]:
        logger.info("Building feature matrix for N=%d", n)
        df = build_feature_matrix(threads, n=n, strategy="n")
        feature_dfs[n] = df
        logger.info("  Shape: %s | Class dist: %s",
                    df.shape, df["label_binary"].value_counts().to_dict())

    # full thread (use all replies)
    max_replies = max(len(t["reactions"]) for t in threads)
    logger.info("Building feature matrix for full thread (N=%d)", max_replies)
    df_full = build_feature_matrix(threads, n=max_replies, strategy="n")
    feature_dfs["full"] = df_full

    # ── 3. Leave-one-event-out CV ─────────────────────────────────────────
    from src.model import leave_one_event_out_cv

    model_types = ["lr", "rf", "xgboost"]

    for n_key, df in feature_dfs.items():
        all_results[n_key] = {}
        for mt in model_types:
            logger.info("--- LOEO CV | N=%s | model=%s ---", n_key, mt)
            res = leave_one_event_out_cv(df, model_type=mt)
            all_results[n_key][mt] = res
            logger.info(
                "  Overall → Acc %.3f | P %.3f | R %.3f | F1 %.3f | F1-macro %.3f",
                res["overall"]["accuracy"],
                res["overall"]["precision"],
                res["overall"]["recall"],
                res["overall"]["f1"],
                res["overall"]["f1_macro"],
            )

    # save results
    with open("results/cv_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    logger.info("CV results saved → results/cv_results.pkl")

    # ── 4. Train final XGBoost models ─────────────────────────────────────
    from src.model import train_final_model

    for n in [5, 10, 20]:
        logger.info("Training final XGBoost for N=%d", n)
        train_final_model(
            feature_dfs[n],
            model_type="xgboost",
            save_path=f"models/xgb_model_N{n}.joblib",
        )

    # primary model is N=10
    logger.info("Primary model is N=10 → models/xgb_model_N10.joblib")

    # also save feature column names for N=10 separately
    from src.feature_extractor import get_feature_cols
    feat_cols = get_feature_cols(feature_dfs[10])
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(feat_cols, f)

    # ── 5. Print summary table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE — XGBoost, all early-window sizes")
    print("=" * 70)
    header = f"{'N':>6} | {'Event':<22} | {'Acc':>6} | {'P':>6} | {'R':>6} | {'F1':>6} | {'F1-mac':>7}"
    print(header)
    print("-" * 70)
    for n_key in [5, 10, 20, "full"]:
        res = all_results[n_key]["xgboost"]
        for fold in res["folds"]:
            print(
                f"{str(n_key):>6} | {fold['event']:<22} | "
                f"{fold['accuracy']:>6.3f} | {fold['precision']:>6.3f} | "
                f"{fold['recall']:>6.3f} | {fold['f1']:>6.3f} | "
                f"{fold['f1_macro']:>7.3f}"
            )
        ov = res["overall"]
        print(
            f"{'':>6}   {'OVERALL':<22}   "
            f"{ov['accuracy']:>6.3f}   {ov['precision']:>6.3f}   "
            f"{ov['recall']:>6.3f}   {ov['f1']:>6.3f}   "
            f"{ov['f1_macro']:>7.3f}"
        )
        print("-" * 70)

    print("\nDone. Models saved to models/")


if __name__ == "__main__":
    main()
