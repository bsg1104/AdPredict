"""Train a hashed logistic baseline (SGDClassifier) for Criteo-format data.

Usage example:
    python -m src.train_lr --data_path data/criteo_day_0.csv --sample_frac 0.2 --out_dir outputs

Features
- integer features I1..I13: QuantileTransformer (fit on train)
- categorical C1..C26: FeatureHasher into a large sparse vector

Saves JSON metrics to outputs/metrics and ROC/PR plots to outputs/figs.
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
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve, precision_recall_curve)
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_extraction import FeatureHasher

from src.data import load_criteo_csv, split_df
from src.feature_engineering import fit_fe


INT_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _hash_cats(df: pd.DataFrame, n_features: int) -> sparse.spmatrix:
    """Hash categorical columns into a sparse matrix using FeatureHasher.

    We encode each categorical column as separate tokens like 'C1=value'.
    """
    records = []
    for _, row in df[CAT_COLS].iterrows():
        d = {}
        for c in CAT_COLS:
            v = row[c]
            if pd.isna(v):
                # represent missing explicitly
                key = f"{c}=__MISSING__"
            else:
                key = f"{c}={v}"
            d[key] = 1
        records.append(d)

    hasher = FeatureHasher(n_features=n_features, input_type="dict", alternate_sign=False)
    X_cat = hasher.transform(records)
    return X_cat


def _quantile_transform(train: pd.DataFrame, parts: List[pd.DataFrame]) -> List[np.ndarray]:
    """Fit QuantileTransformer on train int cols and transform list of parts.

    Returns list of numpy arrays corresponding to transformed integer features.
    """
    qt = QuantileTransformer(output_distribution="normal", copy=True)
    # fit on train ints (fillna with 0)
    X_train_int = train[INT_COLS].fillna(0).astype(float).values
    # limit n_quantiles to reasonable number depending on sample size
    n_samples = max(100, X_train_int.shape[0])
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, n_samples), copy=True)
    qt.fit(X_train_int)

    transformed = []
    for part in parts:
        X_int = part[INT_COLS].fillna(0).astype(float).values
        X_t = qt.transform(X_int)
        transformed.append(X_t)
    return transformed


def run(args: argparse.Namespace) -> int:
    _ensure_dir(args.out_dir)
    _ensure_dir(os.path.join(args.out_dir, "metrics"))
    _ensure_dir(os.path.join(args.out_dir, "figs"))
    _ensure_dir(os.path.join(args.out_dir, "models"))

    print("Loading data...")
    df = load_criteo_csv(args.data_path, sample_frac=args.sample_frac, nrows=args.nrows, seed=args.random_state)

    # pick label column
    label_col = "label" if "label" in df.columns else ("clicked" if "clicked" in df.columns else None)
    if label_col is None:
        raise ValueError("No label column found ('label' or 'clicked')")

    # ensure label numeric
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    print("Splitting data...")
    train_df, val_df, test_df = split_df(df, stratify_col=label_col, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, random_state=args.random_state)

    print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # Quantile-transform integer features
    print("Fitting quantile transformer on integer features...")
    X_train_int, X_val_int, X_test_int = _quantile_transform(train_df, [train_df, val_df, test_df])

    # fit and save FE for parity (optional)
    try:
        fe = fit_fe(train_df, label_col=label_col)
        joblib.dump(fe, os.path.join(args.out_dir, 'fe.joblib'))
    except Exception:
        fe = None

    # Hash categorical features
    print(f"Hashing categorical features into {args.n_features} dims (may be large)...")
    X_train_cat = _hash_cats(train_df, n_features=args.n_features)
    X_val_cat = _hash_cats(val_df, n_features=args.n_features)
    X_test_cat = _hash_cats(test_df, n_features=args.n_features)

    # Convert numeric to sparse and hstack
    X_train_num_sp = sparse.csr_matrix(X_train_int)
    X_val_num_sp = sparse.csr_matrix(X_val_int)
    X_test_num_sp = sparse.csr_matrix(X_test_int)

    X_train = sparse.hstack([X_train_num_sp, X_train_cat], format="csr")
    X_val = sparse.hstack([X_val_num_sp, X_val_cat], format="csr")
    X_test = sparse.hstack([X_test_num_sp, X_test_cat], format="csr")

    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values

    print("Training SGDClassifier (logistic loss)...")
    clf = SGDClassifier(loss="log_loss", max_iter=args.max_iter, tol=args.tol, random_state=args.random_state)
    clf.fit(X_train, y_train)

    # predict probabilities (use decision_function + sigmoid if predict_proba unavailable)
    def get_probs(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            return expit(model.decision_function(X))

    probs_val = get_probs(clf, X_val)
    probs_test = get_probs(clf, X_test)

    auc_val = float(roc_auc_score(y_val, probs_val))
    ap_val = float(average_precision_score(y_val, probs_val))
    auc_test = float(roc_auc_score(y_test, probs_test))
    ap_test = float(average_precision_score(y_test, probs_test))

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "name": "lr_hash_sgd",
        "n_features_hashed": args.n_features,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "auc_val": auc_val,
        "ap_val": ap_val,
        "auc_test": auc_test,
        "ap_test": ap_test,
        "random_state": args.random_state,
    }

    metrics_path = os.path.join(args.out_dir, "metrics", f"lr_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # ROC and PR plots
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    precision, recall, _ = precision_recall_curve(y_test, probs_test)

    fig_roc = os.path.join(args.out_dir, "figs", f"lr_roc_{ts}.png")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"LR (AUC={auc_test:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend()
    plt.savefig(fig_roc)
    plt.close()

    fig_pr = os.path.join(args.out_dir, "figs", f"lr_pr_{ts}.png")
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"LR (AP={ap_test:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    plt.savefig(fig_pr)
    plt.close()

    print(f"Saved ROC to {fig_roc} and PR to {fig_pr}")

    # save model
    model_path = os.path.join(args.out_dir, "models", f"lr_{ts}.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hashed logistic baseline (SGDClassifier)")
    p.add_argument("--data_path", type=str, required=True, help="path to criteo-style TSV")
    p.add_argument("--sample_frac", type=float, default=None, help="optional fraction to sample for speed")
    p.add_argument("--nrows", type=int, default=None, help="optional nrows to read")
    p.add_argument("--out_dir", type=str, default="outputs", help="output directory")
    p.add_argument("--n_features", type=int, default=(1 << 20), help="number of hashed dims (default 2^20)")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args))
