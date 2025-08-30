"""Generate SHAP dependence plots using SHAP's plotting utilities.

This expects a previously created `shap_values_sample.csv` under
`<run_dir>/analysis/shap/` with columns:
 - for each feature: a shap column named exactly as the feature
 - for each feature: a raw value column named `<feature>_value`

It reads the top features from `merged_shap_xgb.csv` and produces
`dependence_<feature>.png` files in the analysis folder.
"""
from __future__ import annotations

import argparse
import os
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--top_n", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    analysis = os.path.join(args.run_dir, "analysis")
    merged_csv = os.path.join(analysis, "merged_shap_xgb.csv")
    sample_csv = os.path.join(analysis, "shap", "shap_values_sample.csv")

    if not os.path.exists(merged_csv):
        raise SystemExit("Please run src.merge_shap_importances first")
    if not os.path.exists(sample_csv):
        raise SystemExit(f"Missing shap sample CSV: {sample_csv}")

    merged = pd.read_csv(merged_csv)
    top_features = merged.head(args.top_n)['mapped_feature'].tolist()

    samp = pd.read_csv(sample_csv)
    # rebuild shap matrix and feature matrix from sample: shap cols are those without _value suffix
    shap_cols = [c for c in samp.columns if not c.endswith('_value')]
    val_cols = [c for c in samp.columns if c.endswith('_value')]
    # feature names by stripping _value
    X = pd.DataFrame({c[:-6]: samp[c].values for c in val_cols})
    shap_vals = samp[shap_cols].values

    # ensure columns align
    # shap_cols should match X.columns order; we built X from val_cols which correspond

    for feat in top_features:
        if feat not in X.columns:
            print('Skipping dependence for missing feature:', feat)
            continue
        plt.figure(figsize=(6,4))
        try:
            shap.dependence_plot(feat, shap_vals, X, show=False, interaction_index=None)
        except Exception as e:
            # fallback: simple scatter
            plt.scatter(X[feat], shap_vals[:, X.columns.get_loc(feat)])
            plt.xlabel(feat)
            plt.ylabel('SHAP value')
            plt.title(f'Dependence: {feat} (fallback)')

        out = os.path.join(analysis, f'dependence_{feat}.png')
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print('Wrote', out)


if __name__ == '__main__':
    main()
