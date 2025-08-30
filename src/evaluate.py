"""Aggregate experiment metrics and produce comparison plots.

Searches recursively for `*/metrics/*.json` files, builds a ranked CSV, and
produces comparison plots (AUC/AP bar chart) and combined ROC/PR image grids
if per-run ROC/PR PNGs are present under corresponding `figs/` folders.

Usage:
    python -m src.evaluate --root outputs --summary_csv outputs/eval_summary.csv --figs_dir outputs/eval_figs

The script is forgiving: if you pass `--runs_dir outputs/metrics` and that
path doesn't exist it will search `--root` recursively for metrics JSONs.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def find_metrics(runs_dir: str, root: str) -> List[str]:
    if runs_dir and os.path.isdir(runs_dir):
        pattern = os.path.join(runs_dir, "*.json")
        files = glob.glob(pattern)
        # if no direct files, try recursive
        if not files:
            files = glob.glob(os.path.join(runs_dir, "**", "*.json"), recursive=True)
        return files

    # fallback: search root recursively for */metrics/*.json
    files = glob.glob(os.path.join(root, "**", "metrics", "*.json"), recursive=True)
    return files


def load_metrics(files: List[str]) -> pd.DataFrame:
    rows = []
    for f in sorted(files):
        try:
            with open(f, "r") as fh:
                j = json.load(fh)
        except Exception:
            continue
    name = j.get("name") or os.path.basename(os.path.dirname(os.path.dirname(f)))
    auc = j.get("auc_test") or j.get("auc") or j.get("auc_val")
    ap = j.get("ap_test") or j.get("ap") or j.get("ap_val")
    ts = j.get("ts") or j.get("timestamp") or os.path.splitext(os.path.basename(f))[0]

    # Create a deterministic, filesystem-safe unique run label by combining
    # the provided name and a short timestamp or the metrics filename. This
    # avoids duplicate indistinguishable labels when multiple runs share the
    # same "name" (common for repeated experiments).
    safe_name = str(name).replace(' ', '_')
    safe_ts = str(ts).replace(' ', '_')
    run_label = f"{safe_name}_{safe_ts}"

    rows.append({"file": f, "run": run_label, "auc_test": auc, "ap_test": ap, "ts": ts})

    df = pd.DataFrame(rows)
    # numeric safe
    df["auc_test"] = pd.to_numeric(df["auc_test"], errors="coerce")
    df["ap_test"] = pd.to_numeric(df["ap_test"], errors="coerce")
    df = df.sort_values(by=["auc_test", "ap_test"], ascending=False).reset_index(drop=True)
    return df


def save_summary(df: pd.DataFrame, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def plot_auc_ap(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Shorten labels for plotting so they fit on the x-axis. Keep them
    # deterministic and human-readable (truncate long names).
    def _shorten(s: str, maxlen: int = 24) -> str:
        s = str(s)
        return s if len(s) <= maxlen else s[: maxlen - 3] + "..."

    labels = df["run"].astype(str).apply(lambda s: _shorten(s, 24))
    x = np.arange(len(labels))

    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    ax.bar(x - width/2, df["auc_test"].fillna(0), width, label="AUC")
    ax.bar(x + width/2, df["ap_test"].fillna(0), width, label="AP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison: AUC and AP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def collect_figs_for_metric(metrics_file: str, kind: str) -> str:
    # kind is 'roc' or 'pr'
    # look for a sibling figs dir with a file containing kind
    metrics_dir = os.path.dirname(metrics_file)
    run_dir = os.path.dirname(metrics_dir)
    figs_dir = os.path.join(run_dir, "figs")
    if not os.path.isdir(figs_dir):
        return None
    candidates = glob.glob(os.path.join(figs_dir, f"*{kind}*.png")) + glob.glob(os.path.join(figs_dir, f"*{kind}*.jpg"))
    return candidates[0] if candidates else None


def make_image_grid(image_paths: List[str], out_path: str, cols: int = 3) -> None:
    if not image_paths:
        return
    imgs = [mpimg.imread(p) for p in image_paths]
    n = len(imgs)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig_w = cols * 3
    fig_h = rows * 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis('off')
            if i < n:
                ax.imshow(imgs[i])
                title = os.path.basename(image_paths[i])
                ax.set_title(title, fontsize=8)
            i += 1

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs", help="root to search for runs")
    p.add_argument("--runs_dir", default="", help="explicit runs/metrics dir (optional)")
    p.add_argument("--summary_csv", default="outputs/eval_summary.csv")
    p.add_argument("--figs_dir", default="outputs/eval_figs")
    args = p.parse_args()

    metrics_files = find_metrics(args.runs_dir, args.root)
    print(f"Found {len(metrics_files)} metrics files")
    if not metrics_files:
        print("No metrics JSON files found. Exiting.")
        return 1

    df = load_metrics(metrics_files)
    print(df[["run", "auc_test", "ap_test"]])

    save_summary(df, args.summary_csv)
    print(f"Saved summary CSV to {args.summary_csv}")

    # plot AUC/AP comparison
    auc_ap_path = os.path.join(args.figs_dir, "auc_ap_comparison.png")
    plot_auc_ap(df, auc_ap_path)
    print(f"Saved AUC/AP comparison to {auc_ap_path}")

    # collect ROC/PR images
    roc_imgs = []
    pr_imgs = []
    labels = []
    for f in df["file"]:
        roc = collect_figs_for_metric(f, "roc")
        pr = collect_figs_for_metric(f, "pr")
        if roc:
            roc_imgs.append(roc)
            labels.append(os.path.basename(os.path.dirname(os.path.dirname(f))))
        if pr:
            pr_imgs.append(pr)

    # make grids
    make_image_grid(roc_imgs, os.path.join(args.figs_dir, "roc_grid.png"), cols=3)
    make_image_grid(pr_imgs, os.path.join(args.figs_dir, "pr_grid.png"), cols=3)
    print(f"Saved ROC/PR grids to {args.figs_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
