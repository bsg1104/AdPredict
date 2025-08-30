"""Inspect an XGBoost model and write feature importances.

Saves CSV and a bar plot of top features by gain and weight.

Usage:
    python -m src.inspect_xgb --model outputs/xgb_full/models/xgb_*.json --out_dir outputs/xgb_full/analysis --top_n 30
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


def load_booster(model_path: str):
    # Try common loaders: joblib (sklearn wrapper), xgboost.Booster
    if joblib:
        try:
            m = joblib.load(model_path)
            # xgboost sklearn wrapper
            if hasattr(m, "get_booster"):
                return m.get_booster()
            # direct booster object saved with joblib
            if isinstance(m, (xgb.Booster,)):
                return m
        except Exception:
            pass

    if xgb:
        try:
            b = xgb.Booster()
            b.load_model(model_path)
            return b
        except Exception:
            pass

    raise RuntimeError(f"Could not load XGBoost model from {model_path}")


def extract_importances(booster) -> pd.DataFrame:
    # importance types available: weight, gain, cover, total_gain, total_cover
    types = ["gain", "weight"]
    records = {}
    for t in types:
        try:
            sc = booster.get_score(importance_type=t)
        except Exception:
            sc = {}
        records[t] = sc

    keys = set().union(*[set(d.keys()) for d in records.values()])
    rows = []
    for k in keys:
        rows.append({"feature": k, "gain": float(records.get("gain", {}).get(k, 0)), "weight": float(records.get("weight", {}).get(k, 0))})
    df = pd.DataFrame(rows)
    # sort by gain then weight
    df = df.sort_values(by=["gain", "weight"], ascending=False).reset_index(drop=True)
    return df


def plot_top(df: pd.DataFrame, out_path: str, top_n: int = 30):
    df = df.head(top_n).set_index("feature")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.2 * len(df))))
    df["gain"].plot(kind="barh", ax=ax, color="#2c7fb8")
    ax.invert_yaxis()
    ax.set_title("Top features by gain")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, nargs='+', help="one or more xgboost model files (json or joblib); globs expanded by the shell")
    p.add_argument("--out_dir", default="outputs/xgb_analysis", help="directory to write CSV and plots")
    p.add_argument("--top_n", type=int, default=30)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for model_path in args.model:
        try:
            booster = load_booster(model_path)
        except Exception as e:
            print(f"Skipping {model_path}: {e}")
            continue

        df = extract_importances(booster)
        base = os.path.splitext(os.path.basename(model_path))[0]
        model_out = os.path.join(args.out_dir, base)
        os.makedirs(model_out, exist_ok=True)
        csv_path = os.path.join(model_out, "xgb_feature_importances.csv")
        df.to_csv(csv_path, index=False)
        plot_top(df, os.path.join(model_out, "xgb_top_gain.png"), top_n=args.top_n)
        print(f"Wrote importances to {csv_path} and plot to {os.path.join(model_out, 'xgb_top_gain.png')}")


if __name__ == "__main__":
    raise SystemExit(main())
