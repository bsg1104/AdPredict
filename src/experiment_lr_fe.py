"""Run a suite of experiments: baseline vs several FE ablations.

Variants:
- baseline: original hashed + quantile ints
- fe_all: all FE features
- fe_no_te: all FE except target encoding
- fe_only_te: only target encodings (with hashed + ints)
- fe_only_counts: only count encodings
- tuned: FE with tuned SGD params (higher alpha via different solver emulation)

Saves metrics JSONs under outputs/experiments and a summary CSV.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data import load_criteo_csv, split_df
from src.feature_engineering import fit_fe, apply_fe, explain_features
from src.train_lr import _hash_cats, _quantile_transform


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def train_and_eval(X_train, X_test, y_train, y_test, max_iter=200, tol=1e-3, random_state=42):
    clf = SGDClassifier(loss="log_loss", max_iter=max_iter, tol=tol, random_state=random_state)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    if probs is None:
        from scipy.special import expit
        probs = expit(clf.decision_function(X_test))
    auc = float(roc_auc_score(y_test, probs))
    ap = float(average_precision_score(y_test, probs))
    return clf, auc, ap


def run_variant(name: str, train_df, val_df, test_df, label_col: str, n_features: int, fe_params: Dict, out_dir: str):
    # baseline numeric transform
    X_train_int, X_val_int, X_test_int = _quantile_transform(train_df, [train_df, val_df, test_df])
    X_train_cat = _hash_cats(train_df, n_features)
    X_val_cat = _hash_cats(val_df, n_features)
    X_test_cat = _hash_cats(test_df, n_features)

    X_train_base = sparse.hstack([sparse.csr_matrix(X_train_int), X_train_cat], format="csr")
    X_test_base = sparse.hstack([sparse.csr_matrix(X_test_int), X_test_cat], format="csr")
    y_train = train_df[label_col].values
    y_test = test_df[label_col].values

    # if no FE, just train baseline
    if name == "baseline":
        clf, auc, ap = train_and_eval(X_train_base, X_test_base, y_train, y_test)
        return {"name": name, "auc": auc, "ap": ap}

    # fit fe according to fe_params
    fe = fit_fe(train_df, label_col=label_col, cat_cols=fe_params.get("cat_cols", None), int_cols=fe_params.get("int_cols", None), top_k_te=fe_params.get("top_k_te", 3), smoothing=fe_params.get("smoothing", 20.0), n_bins=fe_params.get("n_bins", 10))

    # optionally zero-out parts when doing ablations
    # build engineered feature DataFrames
    X_train_fe_df = apply_fe(train_df, fe)
    X_val_fe_df = apply_fe(val_df, fe)
    X_test_fe_df = apply_fe(test_df, fe)

    # depending on ablation flags, keep only certain columns
    if fe_params.get("only_te"):
        cols = [c for c in X_train_fe_df.columns if c.endswith("_te")]
        X_train_fe_df = X_train_fe_df.reindex(columns=cols, fill_value=0)
        X_test_fe_df = X_test_fe_df.reindex(columns=cols, fill_value=0)
    if fe_params.get("only_counts"):
        cols = [c for c in X_train_fe_df.columns if c.endswith("_count")]
        X_train_fe_df = X_train_fe_df.reindex(columns=cols, fill_value=0)
        X_test_fe_df = X_test_fe_df.reindex(columns=cols, fill_value=0)
    if fe_params.get("no_te"):
        cols = [c for c in X_train_fe_df.columns if not c.endswith("_te")]
        X_train_fe_df = X_train_fe_df.reindex(columns=cols, fill_value=0)
        X_test_fe_df = X_test_fe_df.reindex(columns=cols, fill_value=0)

    X_train = sparse.hstack([X_train_base, sparse.csr_matrix(X_train_fe_df.values)], format="csr")
    X_test = sparse.hstack([X_test_base, sparse.csr_matrix(X_test_fe_df.values)], format="csr")

    clf, auc, ap = train_and_eval(X_train, X_test, y_train, y_test, max_iter=fe_params.get("max_iter", 200), tol=fe_params.get("tol", 1e-3))

    # save model
    joblib.dump(clf, os.path.join(out_dir, f"models/{name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.joblib"))

    return {"name": name, "auc": auc, "ap": ap}


def run(args: argparse.Namespace):
    _ensure_dir(args.out_dir)
    _ensure_dir(os.path.join(args.out_dir, "metrics"))
    _ensure_dir(os.path.join(args.out_dir, "models"))
    _ensure_dir(os.path.join(args.out_dir, "figs"))

    # If data path is missing, try to generate a small injected dataset for experiments.
    if not os.path.exists(args.data_path):
        try:
            from src.data import generate_injected_criteo

            print(f"Data file {args.data_path} not found — generating a small injected dataset for experiments.")
            generate_injected_criteo(args.data_path, nrows=20000, seed=args.random_state)
        except Exception:
            # If generation fails, let the subsequent load raise the original error
            pass

    df = load_criteo_csv(args.data_path, sample_frac=args.sample_frac, nrows=args.nrows, seed=args.random_state)
    label_col = "label" if "label" in df.columns else ("clicked" if "clicked" in df.columns else None)
    if label_col is None:
        raise ValueError("No label column found")

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    train_df, val_df, test_df = split_df(df, stratify_col=label_col, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, random_state=args.random_state)

    variants = []
    # baseline
    variants.append(("baseline", {}))
    # full FE
    variants.append(("fe_all", {"top_k_te": args.top_k_te, "smoothing": args.smoothing, "n_bins": args.n_bins}))
    # no target encoding
    variants.append(("fe_no_te", {"top_k_te": 0, "smoothing": args.smoothing, "n_bins": args.n_bins, "no_te": True}))
    # only target encoding
    variants.append(("fe_only_te", {"top_k_te": args.top_k_te, "only_te": True}))
    # only counts
    variants.append(("fe_only_counts", {"top_k_te": 0, "only_counts": True}))
    # tuned model (increase iterations and lower tol)
    variants.append(("tuned_fe", {"top_k_te": args.top_k_te, "smoothing": args.smoothing, "n_bins": args.n_bins, "max_iter": 500, "tol": 1e-4}))

    results = []
    for name, params in variants:
        print(f"Running variant: {name} with params {params}")
        res = run_variant(name, train_df, val_df, test_df, label_col, args.n_features, params, args.out_dir)
        print(f"Result: {res}")
        results.append(res)
        # save metrics
        metrics_path = os.path.join(args.out_dir, "metrics", f"exp_{name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
        with open(metrics_path, "w") as f:
            json.dump(res, f, indent=2)

    # summary CSV
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, "experiments_summary.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved experiment summary to {csv_path}")
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--sample_frac", type=float, default=0.2)
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/experiments")
    p.add_argument("--n_features", type=int, default=65536)
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
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
    raise SystemExit(run(args))
