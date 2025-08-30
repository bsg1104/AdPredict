"""Train and evaluate baseline vs feature-engineered logistic (SGD) models.

This script runs two models on the same train/val/test splits:
- baseline: quantile-transformed ints + hashed categorical features
- fe: baseline + engineered features from `src.feature_engineering`

Saves metrics JSONs, ROC/PR plots, and a short markdown report explaining which
features moved the needle.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

from src.data import load_criteo_csv, split_df
from src.feature_engineering import fit_fe, apply_fe, explain_features
from src.train_lr import _hash_cats, _quantile_transform


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    _ensure_dir(args.out_dir)
    metrics_dir = os.path.join(args.out_dir, "metrics")
    figs_dir = os.path.join(args.out_dir, "figs")
    _ensure_dir(metrics_dir)
    _ensure_dir(figs_dir)

    print("Loading data...")
    df = load_criteo_csv(args.data_path, sample_frac=args.sample_frac, nrows=args.nrows, seed=args.random_state)

    label_col = "label" if "label" in df.columns else ("clicked" if "clicked" in df.columns else None)
    if label_col is None:
        raise ValueError("No label column found ('label' or 'clicked')")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    train_df, val_df, test_df = split_df(df, stratify_col=label_col, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, random_state=args.random_state)

    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # baseline numeric transform
    X_train_int, X_val_int, X_test_int = _quantile_transform(train_df, [train_df, val_df, test_df])

    # hash categorical
    X_train_cat = _hash_cats(train_df, n_features=args.n_features)
    X_val_cat = _hash_cats(val_df, n_features=args.n_features)
    X_test_cat = _hash_cats(test_df, n_features=args.n_features)

    # sparse combine baseline
    X_train_base = sparse.hstack([sparse.csr_matrix(X_train_int), X_train_cat], format="csr")
    X_val_base = sparse.hstack([sparse.csr_matrix(X_val_int), X_val_cat], format="csr")
    X_test_base = sparse.hstack([sparse.csr_matrix(X_test_int), X_test_cat], format="csr")

    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values

    # Train baseline
    print("Training baseline model...")
    base_clf = SGDClassifier(loss="log_loss", max_iter=args.max_iter, tol=args.tol, random_state=args.random_state)
    base_clf.fit(X_train_base, y_train)

    def get_probs(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            from scipy.special import expit
            return expit(model.decision_function(X))

    probs_base_val = get_probs(base_clf, X_val_base)
    probs_base_test = get_probs(base_clf, X_test_base)

    auc_base = float(roc_auc_score(y_test, probs_base_test))
    ap_base = float(average_precision_score(y_test, probs_base_test))

    # Feature engineering
    print("Fitting feature engineering artifacts...")
    fe = fit_fe(train_df, label_col=label_col, cat_cols=None, int_cols=None, top_k_te=args.top_k_te, smoothing=args.smoothing, n_bins=args.n_bins)

    print("Applying engineered features...")
    X_train_fe_df = apply_fe(train_df, fe)
    X_val_fe_df = apply_fe(val_df, fe)
    X_test_fe_df = apply_fe(test_df, fe)

    # combine engineered features with baseline (dense -> sparse)
    X_train_fe = sparse.hstack([X_train_base, sparse.csr_matrix(X_train_fe_df.values)], format="csr")
    X_val_fe = sparse.hstack([X_val_base, sparse.csr_matrix(X_val_fe_df.values)], format="csr")
    X_test_fe = sparse.hstack([X_test_base, sparse.csr_matrix(X_test_fe_df.values)], format="csr")

    print("Training FE-enhanced model...")
    fe_clf = SGDClassifier(loss="log_loss", max_iter=args.max_iter, tol=args.tol, random_state=args.random_state)
    fe_clf.fit(X_train_fe, y_train)

    probs_fe_test = get_probs(fe_clf, X_test_fe)
    auc_fe = float(roc_auc_score(y_test, probs_fe_test))
    ap_fe = float(average_precision_score(y_test, probs_fe_test))

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    metrics_base = {"name": "lr_baseline", "auc_test": auc_base, "ap_test": ap_base, "ts": ts}
    metrics_fe = {"name": "lr_fe", "auc_test": auc_fe, "ap_test": ap_fe, "ts": ts}

    base_metrics_path = os.path.join(metrics_dir, f"lr_baseline_{ts}.json")
    fe_metrics_path = os.path.join(metrics_dir, f"lr_fe_{ts}.json")
    with open(base_metrics_path, "w") as f:
        json.dump(metrics_base, f, indent=2)
    with open(fe_metrics_path, "w") as f:
        json.dump(metrics_fe, f, indent=2)
    print(f"Saved metrics: {base_metrics_path}, {fe_metrics_path}")

    # plots
    fpr_b, tpr_b, _ = roc_curve(y_test, probs_base_test)
    fpr_f, tpr_f, _ = roc_curve(y_test, probs_fe_test)

    fig_roc = os.path.join(figs_dir, f"compare_roc_{ts}.png")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_b, tpr_b, label=f"Baseline AUC={auc_base:.3f}")
    plt.plot(fpr_f, tpr_f, label=f"FE AUC={auc_fe:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC: Baseline vs FE")
    plt.savefig(fig_roc)
    plt.close()

    fig_pr = os.path.join(figs_dir, f"compare_pr_{ts}.png")
    precision_b, recall_b, _ = precision_recall_curve(y_test, probs_base_test)
    precision_f, recall_f, _ = precision_recall_curve(y_test, probs_fe_test)
    plt.figure(figsize=(6, 6))
    plt.plot(recall_b, precision_b, label=f"Baseline AP={ap_base:.3f}")
    plt.plot(recall_f, precision_f, label=f"FE AP={ap_fe:.3f}")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR: Baseline vs FE")
    plt.savefig(fig_pr)
    plt.close()

    print(f"Saved figs: {fig_roc}, {fig_pr}")

    # small report
    report = []
    report.append(f"# LR baseline vs FE report ({ts})\n")
    report.append(f"Baseline AUC: {auc_base:.4f}, AP: {ap_base:.4f}\n")
    report.append(f"FE AUC: {auc_fe:.4f}, AP: {ap_fe:.4f}\n")
    report.append("\n## FE summary\n")
    report.append(explain_features(fe))
    report_path = os.path.join(args.out_dir, f"report_lr_fe_{ts}.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"Saved report to {report_path}")

    # save models
    joblib.dump(base_clf, os.path.join(args.out_dir, "models", f"base_{ts}.joblib"))
    joblib.dump(fe_clf, os.path.join(args.out_dir, "models", f"fe_{ts}.joblib"))

    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline vs FE logistic models")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--sample_frac", type=float, default=None)
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--n_features", type=int, default=(1 << 18))
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--top_k_te", type=int, default=3)
    p.add_argument("--smoothing", type=float, default=20.0)
    p.add_argument("--n_bins", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
    raise SystemExit(run(args))
