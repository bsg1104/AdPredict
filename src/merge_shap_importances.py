"""Merge SHAP top CSV with readable XGBoost feature importances for a run.

Usage:
  python -m src.merge_shap_importances --run_dir outputs/xgb_sweep_full/eta0.01_md8_n300_20250829T223703Z

This looks for:
 - <run_dir>/analysis/shap/shap_top.csv
 - <run_dir>/analysis/top_features_readable.csv

and writes merged CSV to <run_dir>/analysis/merged_shap_xgb.csv
"""
from __future__ import annotations

import argparse
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    analysis = os.path.join(run_dir, "analysis")

    shap_csv = os.path.join(analysis, "shap", "shap_top.csv")
    xgb_csv = os.path.join(analysis, "top_features_readable.csv")

    if not os.path.exists(shap_csv):
        raise SystemExit(f"Missing SHAP CSV: {shap_csv}")
    if not os.path.exists(xgb_csv):
        raise SystemExit(f"Missing XGBoost readable CSV: {xgb_csv}")

    shap_df = pd.read_csv(shap_csv)
    xgb_df = pd.read_csv(xgb_csv)

    # shap_top has columns: feature, mean_abs_shap
    # xgb_df has: feature,gain,weight,mapped_feature
    # map f# -> mapped_feature where possible
    xgb_map = xgb_df.set_index('feature')['mapped_feature'].to_dict()

    def map_feat(f):
        return xgb_map.get(f, f)

    shap_df['mapped_feature'] = shap_df['feature'].apply(map_feat)

    # aggregate by mapped_feature (some shap cols may map to same readable feature)
    merged = shap_df.groupby('mapped_feature', as_index=False)['mean_abs_shap'].sum()
    merged = merged.sort_values('mean_abs_shap', ascending=False)

    out_csv = os.path.join(analysis, 'merged_shap_xgb.csv')
    merged.to_csv(out_csv, index=False)
    print(f"Wrote merged SHAP+XGB importances to {out_csv}")


if __name__ == '__main__':
    main()
