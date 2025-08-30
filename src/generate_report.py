"""Generate a small report (plots + CSV) combining XGBoost gains, SHAP, and merged rankings for a run.

Usage:
  python -m src.generate_report --run_dir <run_dir>
"""
from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    analysis = os.path.join(args.run_dir, "analysis")
    shap_top = os.path.join(analysis, "shap", "shap_top.csv")
    xgb_read = os.path.join(analysis, "top_features_readable.csv")
    merged = os.path.join(analysis, "merged_shap_xgb.csv")

    if not os.path.exists(merged):
        raise SystemExit("Please run src.merge_shap_importances first")

    mdf = pd.read_csv(merged)
    xdf = pd.read_csv(xgb_read) if os.path.exists(xgb_read) else pd.DataFrame()
    sdf = pd.read_csv(shap_top) if os.path.exists(shap_top) else pd.DataFrame()

    # small plot: merged SHAP rank bar
    topn = mdf.head(15)
    plt.figure(figsize=(6,4))
    plt.barh(topn['mapped_feature'][::-1], topn['mean_abs_shap'][::-1])
    plt.xlabel('sum mean_abs_shap')
    plt.tight_layout()
    out = os.path.join(analysis, 'merged_shap_top15.png')
    plt.savefig(out)
    print('Wrote', out)

    # write a combined CSV with xgb gain if available
    if not xdf.empty:
        comb = mdf.merge(xdf[['mapped_feature','gain','weight']], left_on='mapped_feature', right_on='mapped_feature', how='left')
    else:
        comb = mdf

    outcsv = os.path.join(analysis, 'report_combined.csv')
    comb.to_csv(outcsv, index=False)
    print('Wrote', outcsv)


if __name__ == '__main__':
    main()
