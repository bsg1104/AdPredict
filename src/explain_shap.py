"""Generate SHAP summary plots for a saved XGBoost model.

Usage:
  python -m src.explain_shap --model outputs/xgb_sweep_full/.../models/xgb_*.json --run_dir outputs/xgb_sweep_full/... --out_dir outputs/xgb_sweep_full/.../analysis/shap

Note: shap package is optional; if not installed the script will exit with a helpful message.
"""
from __future__ import annotations

import argparse
import os
import joblib
import pandas as pd

try:
    import shap
    import xgboost as xgb
    import matplotlib.pyplot as plt
except Exception:
    shap = None
import numpy as np

from src.data import load_criteo_csv, split_df
from src.feature_engineering import fit_fe
import src.train_xgb as train_xgb_module


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_path", default="data/criteo_injected.csv")
    p.add_argument("--sample_frac", type=float, default=0.2)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if shap is None:
        raise SystemExit("Please install shap and xgboost to run this script: pip install shap xgboost")

    # load model
    bst = xgb.Booster()
    bst.load_model(args.model)

    # load data and fit FE
    df = load_criteo_csv(args.data_path, sample_frac=args.sample_frac)
    label_col = "label" if "label" in df.columns else ("clicked" if "clicked" in df.columns else None)
    train_df, val_df, test_df = split_df(df, stratify_col=label_col, train_frac=0.7, val_frac=0.1, test_frac=0.2)
    fe = fit_fe(train_df, label_col=label_col)

    X_train, X_val, X_test, y_train, y_val, y_test = train_xgb_module.build_matrix(train_df, val_df, test_df, fe=fe)

    # compute SHAP values (may be slow)
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_test)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.model), "shap")
    os.makedirs(out_dir, exist_ok=True)

    # summary plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    summary_png = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(summary_png)
    plt.close()

    # write top mean |shap| values
    mean_abs = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": np.mean(np.abs(shap_values), axis=0)})
    mean_abs = mean_abs.sort_values(by="mean_abs_shap", ascending=False)
    mean_abs.to_csv(os.path.join(out_dir, "shap_top.csv"), index=False)
    print(f"Wrote SHAP summary to {summary_png} and CSV to {os.path.join(out_dir, 'shap_top.csv')}")


if __name__ == "__main__":
    main()
