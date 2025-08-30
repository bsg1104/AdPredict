"""Hyperparameter sweep harness for XGBoost.

Runs a small grid of (eta, max_depth, num_boost_round) and stores per-run
artifacts under `out_dir/<run_label>/` (metrics, models, figs). A summary CSV
is written to `out_dir/summary.csv`.

Usage (smoke):
    python -m src.xgb_sweep --data_path data/criteo_injected.csv --sample_frac 0.05 --out_dir outputs/xgb_sweep --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime
from typing import List

import pandas as pd
import joblib

from src.data import load_criteo_csv, split_df
from src.feature_engineering import fit_fe
import src.train_xgb as train_xgb_module


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_grid(etas: List[float], max_depths: List[int], nrounds: List[int]):
    for e, md, nr in itertools.product(etas, max_depths, nrounds):
        yield {"eta": float(e), "max_depth": int(md), "num_boost_round": int(nr)}


def _run_single(cfg, args, ts):
    """Run a single config, return a record dict."""
    label = f"eta{cfg['eta']}_md{cfg['max_depth']}_n{cfg['num_boost_round']}_{ts}"
    run_dir = os.path.join(args.out_dir, label)
    _ensure_dir(run_dir)
    print(f"Running {label} -> {run_dir}")

    # load and split inside worker to avoid IPC of large dfs
    df = load_criteo_csv(args.data_path, sample_frac=args.sample_frac, nrows=args.nrows, seed=args.random_state)
    label_col = "label" if "label" in df.columns else ("clicked" if "clicked" in df.columns else None)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    train_df, val_df, test_df = split_df(df, stratify_col=label_col, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, random_state=args.random_state)

    fe = fit_fe(train_df, label_col=label_col, top_k_te=args.top_k_te, smoothing=args.smoothing, n_bins=args.n_bins)
    # persist FE so serving code can load it later
    try:
        joblib.dump(fe, os.path.join(run_dir, 'fe.joblib'))
    except Exception:
        pass
    X_train, X_val, X_test, y_train, y_val, y_test = train_xgb_module.build_matrix(train_df, val_df, test_df, fe=fe)

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": int(cfg["max_depth"]),
        "eta": float(cfg["eta"]),
        "eval_metric": "auc",
        "verbosity": 0,
    }

    metrics, model_path = train_xgb_module.train_xgb(X_train, X_val, X_test, y_train, y_val, y_test, params, num_boost_round=cfg["num_boost_round"], early_stopping_rounds=args.early_stopping_rounds, out_dir=run_dir)

    rec = {"label": label, "eta": cfg["eta"], "max_depth": cfg["max_depth"], "num_boost_round": cfg["num_boost_round"], "auc_test": metrics.get("auc_test"), "ap_test": metrics.get("ap_test"), "model_path": model_path}
    # write per-run metadata
    with open(os.path.join(run_dir, "run_config.json"), "w") as fh:
        json.dump({"cfg": cfg, "params": params, "metrics": metrics}, fh, indent=2)
    return rec


def run_sweep(args):
    # choose a default small grid for smoke runs
    if args.smoke:
        etas = [0.1, 0.05]
        depths = [6, 8]
        nrs = [50]
    else:
        etas = [float(x) for x in args.etas.split(",")]
        depths = [int(x) for x in args.max_depths.split(",")]
        nrs = [int(x) for x in args.num_boost_rounds.split(",")]

    grid = list(make_grid(etas, depths, nrs))
    print(f"Running grid with {len(grid)} configurations")

    _ensure_dir(args.out_dir)
    records = []

    # We'll build/run per-config; timestamp for labels
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if args.parallel and args.n_jobs != 1:
        # simple parallel map: each worker loads its own subset and runs
        from multiprocessing import Pool, cpu_count

        n_jobs = args.n_jobs if args.n_jobs and args.n_jobs > 0 else max(1, cpu_count() - 1)
        print(f"Running sweep in parallel with {n_jobs} workers")
        with Pool(n_jobs) as p:
            results = p.starmap(_run_single, [(cfg, args, ts) for cfg in grid])
        records.extend(results)
    else:
        for cfg in grid:
            rec = _run_single(cfg, args, ts)
            records.append(rec)

    # summary
    summary_df = pd.DataFrame(records).sort_values(by=["auc_test", "ap_test"], ascending=False).reset_index(drop=True)
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote sweep summary to {summary_csv}")

    # produce an aggregation plot (AUC vs AP), colored by eta
    try:
        import matplotlib.pyplot as plt

        fig_path = os.path.join(args.out_dir, "auc_ap_grid.png")
        plt.figure(figsize=(6, 6))
        sc = plt.scatter(summary_df["auc_test"], summary_df["ap_test"], c=summary_df["eta"].astype(float), cmap="viridis", s=80)
        for i, r in summary_df.iterrows():
            plt.text(r["auc_test"], r["ap_test"], r["label"].split("_")[0], fontsize=7)
        plt.colorbar(sc, label="eta")
        plt.xlabel("AUC")
        plt.ylabel("AP")
        plt.title("Sweep: AUC vs AP by config (color=eta)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"Wrote sweep aggregation plot to {fig_path}")
    except Exception as e:
        print(f"Could not write aggregation plot: {e}")

    # append best run to global eval summary if requested
    if args.append_eval:
        eval_csv = args.eval_csv or os.path.join("outputs", "eval_summary.csv")
        try:
            existing = pd.read_csv(eval_csv) if os.path.exists(eval_csv) else pd.DataFrame(columns=["run", "auc_test", "ap_test"])
            best = summary_df.iloc[0]
            newrow = {"run": best["label"], "auc_test": best["auc_test"], "ap_test": best["ap_test"]}
            existing = pd.concat([existing, pd.DataFrame([newrow])], ignore_index=True)
            existing.to_csv(eval_csv, index=False)
            print(f"Appended best run to {eval_csv}")
        except Exception as e:
            print(f"Failed to append to eval summary: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--sample_frac", type=float, default=0.2)
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--out_dir", default="outputs/xgb_sweep")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--top_k_te", type=int, default=3)
    p.add_argument("--smoothing", type=float, default=20.0)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--etas", default="0.01,0.03,0.05")
    p.add_argument("--max_depths", default="6,8")
    p.add_argument("--num_boost_rounds", default="300,500")
    p.add_argument("--smoke", action="store_true", help="run a small smoke grid")
    return p.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.out_dir)
    run_sweep(args)


if __name__ == "__main__":
    main()
