"""Align XGBoost feature names (f0,f1,...) to DataFrame column names and merge with importances CSV.

Usage:
  python -m src.align_importances --run_dir outputs/xgb_sweep_full/eta0.01_md8_n300_20250829T223703Z

The script will: rebuild the FE and feature matrix for the run (same defaults), read the importances CSV produced by `src.inspect_xgb`, and write `top_features_readable.csv` in the run analysis folder.
"""
from __future__ import annotations

import argparse
import os
import pandas as pd

from src.data import load_criteo_csv, split_df
from src.feature_engineering import fit_fe
import src.train_xgb as train_xgb_module


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="path to the sweep run directory containing run_config.json and analysis folder")
    p.add_argument("--data_path", default="data/criteo_injected.csv")
    p.add_argument("--sample_frac", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    analysis_dir = os.path.join(run_dir, "analysis")

    # find the importance CSV produced earlier (search recursively)
    imp_csv = None
    for root, dirs, files in os.walk(analysis_dir):
        for fn in files:
            if fn.endswith("xgb_feature_importances.csv"):
                imp_csv = os.path.join(root, fn)
                break
        if imp_csv:
            break

    if not imp_csv:
        raise SystemExit(f"No importance CSV found under {analysis_dir}")

    # load data and rebuild FE and X matrix
    df = load_criteo_csv(args.data_path, sample_frac=args.sample_frac, seed=args.random_state)
    label_col = "label" if "label" in df.columns else ("clicked" if "clicked" in df.columns else None)
    train_df, val_df, test_df = split_df(df, stratify_col=label_col, train_frac=0.7, val_frac=0.1, test_frac=0.2, random_state=args.random_state)
    fe = fit_fe(train_df, label_col=label_col)

    X_train, X_val, X_test, y_train, y_val, y_test = train_xgb_module.build_matrix(train_df, val_df, test_df, fe=fe)

    cols = list(X_train.columns)

    imp = pd.read_csv(imp_csv)
    # map feature names (f0 -> cols[0])
    def map_feat(f):
        if isinstance(f, str) and f.startswith("f") and f[1:].isdigit():
            idx = int(f[1:])
            if 0 <= idx < len(cols):
                return cols[idx]
        return f

    imp["mapped_feature"] = imp["feature"].apply(map_feat)

    out_csv = os.path.join(analysis_dir, "top_features_readable.csv")
    imp.to_csv(out_csv, index=False)
    print(f"Wrote readable feature importances to {out_csv}")


if __name__ == "__main__":
    main()
