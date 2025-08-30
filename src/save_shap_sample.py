"""Compute SHAP values for X_test and save a sampled CSV of shap values and raw feature values.

Usage:
  python -m src.save_shap_sample --run_dir <run_dir> [--model <model_path>] [--n_samples 1000]
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

try:
    import shap
    import xgboost as xgb
except Exception:
    shap = None

from src.data import load_criteo_csv, split_df
from src.feature_engineering import fit_fe
import src.train_xgb as train_xgb_module


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--data_path", default="data/criteo_injected.csv")
    p.add_argument("--sample_frac", type=float, default=0.2)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if shap is None:
        raise SystemExit("Please install shap and xgboost to run this script")

    run_dir = args.run_dir
    analysis = os.path.join(run_dir, "analysis")
    os.makedirs(os.path.join(analysis, "shap"), exist_ok=True)

    # locate model if not provided
    model_path = args.model
    if model_path is None:
        models_dir = os.path.join(run_dir, "models")
        if not os.path.exists(models_dir):
            raise SystemExit(f"No models dir at {models_dir} and no --model provided")
        found = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith('xgb_') and f.endswith('.json')]
        if not found:
            raise SystemExit(f"No xgb model JSON found in {models_dir}")
        model_path = sorted(found)[-1]

    # load data and make FE + X_test
    df = load_criteo_csv(args.data_path, sample_frac=args.sample_frac, seed=args.random_state)
    label_col = 'label' if 'label' in df.columns else ('clicked' if 'clicked' in df.columns else None)
    train_df, val_df, test_df = split_df(df, stratify_col=label_col, train_frac=0.7, val_frac=0.1, test_frac=0.2, random_state=args.random_state)
    fe = fit_fe(train_df, label_col=label_col)

    X_train, X_val, X_test, y_train, y_val, y_test = train_xgb_module.build_matrix(train_df, val_df, test_df, fe=fe)

    # load model and compute shap values
    bst = xgb.Booster()
    bst.load_model(model_path)

    explainer = shap.TreeExplainer(bst)
    # shap_values shape: (n_samples, n_features)
    shap_vals = explainer.shap_values(X_test)

    # ensure 2D array
    shap_arr = np.array(shap_vals)

    n_rows = shap_arr.shape[0]
    n_samples = min(args.n_samples, n_rows)
    rng = np.random.RandomState(args.random_state)
    idx = rng.choice(n_rows, size=n_samples, replace=False)

    Xs = X_test.reset_index(drop=True).iloc[idx]
    S = pd.DataFrame(shap_arr[idx, :], columns=X_test.columns)

    # combine: for each column, write shap value column named as feature and a companion raw value column feature + '_value'
    out_df = pd.DataFrame()
    for col in X_test.columns:
        out_df[col] = S[col]
        out_df[col + '_value'] = Xs[col].values

    out_csv = os.path.join(analysis, 'shap', 'shap_values_sample.csv')
    out_df.to_csv(out_csv, index=False)
    print(f'Wrote shap sample to {out_csv} (n={n_samples})')


if __name__ == '__main__':
    main()
