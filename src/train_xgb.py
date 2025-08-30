"""Train an XGBoost model on Criteo-format data.

Saves JSON metrics to outputs/metrics and ROC/PR to outputs/figs and a model to outputs/models.

Usage: python -m src.train_xgb --data_path data/criteo_day_0.csv
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

from src.data import load_criteo_csv, split_df
from src.feature_engineering import fit_fe, apply_fe


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_matrix(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, fe=None, use_hashing: bool = False, n_features: int = 65536):
    """Return X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays.

    If `fe` is provided (from fit_fe), append engineered features; otherwise only use integer columns and simple label encoding for cats.
    """
    int_cols = [f"I{i}" for i in range(1, 14)]
    cat_cols = [f"C{i}" for i in range(1, 27)]

    def basic_cat_encode(df: pd.DataFrame) -> pd.DataFrame:
        # for XGBoost we can label-encode categorical columns to integers
        out = pd.DataFrame(index=df.index)
        for c in cat_cols:
            out[c] = pd.factorize(df[c].fillna("__MISSING__"))[0].astype(int)
        return out

    X_train_num = train_df[int_cols].fillna(0).astype(float)
    X_val_num = val_df[int_cols].fillna(0).astype(float)
    X_test_num = test_df[int_cols].fillna(0).astype(float)

    X_train = X_train_num.copy()
    X_val = X_val_num.copy()
    X_test = X_test_num.copy()

    # add simple encoded categorical features
    X_train = pd.concat([X_train, basic_cat_encode(train_df)], axis=1)
    X_val = pd.concat([X_val, basic_cat_encode(val_df)], axis=1)
    X_test = pd.concat([X_test, basic_cat_encode(test_df)], axis=1)

    # append engineered dense features if provided
    if fe is not None:
        X_train_fe = apply_fe(train_df, fe)
        X_val_fe = apply_fe(val_df, fe)
        X_test_fe = apply_fe(test_df, fe)
        # align columns
        X_train_fe = X_train_fe.fillna(0)
        X_val_fe = X_val_fe.reindex(columns=X_train_fe.columns, fill_value=0).fillna(0)
        X_test_fe = X_test_fe.reindex(columns=X_train_fe.columns, fill_value=0).fillna(0)

        X_train = pd.concat([X_train, X_train_fe.reset_index(drop=True)], axis=1)
        X_val = pd.concat([X_val, X_val_fe.reset_index(drop=True)], axis=1)
        X_test = pd.concat([X_test, X_test_fe.reset_index(drop=True)], axis=1)

    y_train = train_df["label"].astype(int).values if "label" in train_df.columns else train_df["clicked"].astype(int).values
    y_val = val_df["label"].astype(int).values if "label" in val_df.columns else val_df["clicked"].astype(int).values
    y_test = test_df["label"].astype(int).values if "label" in test_df.columns else test_df["clicked"].astype(int).values

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgb(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, params: dict, num_boost_round: int = 500, early_stopping_rounds: int = 50, out_dir: str = "outputs"):
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError("xgboost is required to run this script. Install with `pip install xgboost`") from e
    # optionally import mlflow for logging
    try:
        import mlflow
        MLFLOW = True
    except Exception:
        mlflow = None
        MLFLOW = False

    dtrain = xgb.DMatrix(X_train.values, label=y_train)
    dval = xgb.DMatrix(X_val.values, label=y_val)
    dtest = xgb.DMatrix(X_test.values, label=y_test)

    evals = [(dtrain, "train"), (dval, "val")]

    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

    # predict on test
    # Predict: newer xgboost versions removed the `ntree_limit` kw argument.
    # Prefer iteration_range with best_iteration when available, fall back gracefully.
    probs = None
    if hasattr(bst, "best_iteration") and getattr(bst, "best_iteration") is not None:
        # iteration_range is (start, end) where end is exclusive, so use best_iteration+1
        probs = bst.predict(dtest, iteration_range=(0, int(bst.best_iteration) + 1))
    else:
        # older xgboost might have best_ntree_limit
        best_ntree = getattr(bst, "best_ntree_limit", None)
        try:
            if best_ntree is not None:
                probs = bst.predict(dtest, ntree_limit=best_ntree)
            else:
                probs = bst.predict(dtest)
        except TypeError:
            # fallback for versions that removed ntree_limit
            probs = bst.predict(dtest)

    auc = float(roc_auc_score(y_test, probs))
    ap = float(average_precision_score(y_test, probs))

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "name": "xgb_hist",
        "params": params,
        "num_boost_round": int(bst.best_ntree_limit) if hasattr(bst, 'best_ntree_limit') else num_boost_round,
        "auc_test": auc,
        "ap_test": ap,
        "ts": ts,
    }

    _ensure_dir(os.path.join(out_dir, "metrics"))
    _ensure_dir(os.path.join(out_dir, "figs"))
    _ensure_dir(os.path.join(out_dir, "models"))

    metrics_path = os.path.join(out_dir, "metrics", f"xgb_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save model
    model_path = os.path.join(out_dir, "models", f"xgb_{ts}.json")
    bst.save_model(model_path)

    # MLflow logging (best-effort)
    if MLFLOW:
        try:
            mlflow.start_run(run_name=f"xgb_{ts}")
            mlflow.log_params(params)
            mlflow.log_metric('auc_test', auc)
            mlflow.log_metric('ap_test', ap)
            mlflow.log_artifact(metrics_path, artifact_path='metrics')
            mlflow.log_artifact(model_path, artifact_path='models')
            mlflow.end_run()
        except Exception:
            pass

    # plots
    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)

    fig_roc = os.path.join(out_dir, "figs", f"xgb_roc_{ts}.png")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"XGB (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend()
    plt.savefig(fig_roc)
    plt.close()

    fig_pr = os.path.join(out_dir, "figs", f"xgb_pr_{ts}.png")
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"XGB (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    plt.savefig(fig_pr)
    plt.close()

    return metrics, model_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--sample_frac", type=float, default=0.2)
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--top_k_te", type=int, default=3)
    p.add_argument("--smoothing", type=float, default=20.0)
    p.add_argument("--num_boost_round", type=int, default=500)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--eta", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.out_dir)
    _ensure_dir(os.path.join(args.out_dir, "metrics"))
    _ensure_dir(os.path.join(args.out_dir, "figs"))
    _ensure_dir(os.path.join(args.out_dir, "models"))

    df = load_criteo_csv(args.data_path, sample_frac=args.sample_frac, nrows=args.nrows, seed=args.random_state)
    label_col = "label" if "label" in df.columns else ("clicked" if "clicked" in df.columns else None)
    if label_col is None:
        raise ValueError("No label column found")

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    train_df, val_df, test_df = split_df(df, stratify_col=label_col, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, random_state=args.random_state)

    # fit FE for dense engineered features
    fe = fit_fe(train_df, label_col=label_col, top_k_te=args.top_k_te, smoothing=args.smoothing, n_bins=args.n_bins)
    # persist FE for later serving
    try:
        joblib.dump(fe, os.path.join(args.out_dir, 'fe.joblib'))
    except Exception:
        pass

    X_train, X_val, X_test, y_train, y_val, y_test = build_matrix(train_df, val_df, test_df, fe=fe)

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "eval_metric": "auc",
        "verbosity": 0,
    }

    metrics, model_path = train_xgb(X_train, X_val, X_test, y_train, y_val, y_test, params, num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping_rounds, out_dir=args.out_dir)

    print("Saved model to", model_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
