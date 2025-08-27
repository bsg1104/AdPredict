"""Train a logistic regression baseline on the synthetic AdPredict data.
Saves ROC plot to outputs/roc.png and prints ROC-AUC.
"""
import argparse
import os
import sys

# Ensure the repository root is on sys.path so `from src import ...` works
# when this script is run as `python3 src/train.py`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.data import generate_sample_csv, split_df
from src.features import featurize


def main(nrows: int, test_splits: bool = False):
    """Train a quick logistic-regression baseline on the demo data.

    If `data/sample.csv` is missing the script will generate a small
    synthetic file for fast iteration. Use `--test-splits` to run and
    print the stratified split diagnostics instead of training.
    """
    data_path = "data/sample.csv"
    if not os.path.exists(data_path):
        print(f"No data found at {data_path}. Generating {nrows} synthetic rows so you can try things out.")
        generate_sample_csv(data_path, nrows=nrows)

    df = pd.read_csv(data_path)

    if test_splits:
        print("Running stratified split diagnostics (train/val/test)...")
        train, val, test = split_df(df)
        print('\nSplit sizes:', len(train), len(val), len(test))
        return

    # Prepare features and labels
    y = df["clicked"].values
    X = featurize(df)

    # quick train/test split for a baseline evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"ROC AUC (baseline logistic): {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, probs)
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"LogReg (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend()
    out_path = "outputs/roc.png"
    plt.savefig(out_path)
    print(f"Saved ROC plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdPredict demo training script")
    parser.add_argument("--nrows", type=int, default=10000, help="number of synthetic rows to generate if data is missing")
    parser.add_argument("--test-splits", action="store_true", help="run stratified split diagnostics and exit")
    args = parser.parse_args()
    main(args.nrows, test_splits=args.test_splits)
