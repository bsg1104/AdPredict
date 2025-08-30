"""Run the reporting pipeline for a given sweep out_dir containing summary.csv.

Usage:
  python -m src.run_report_for_outdir --out_dir outputs/xgb_sweep_smoke
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd):
    print('RUN:', cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f'Command failed: {cmd}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True, help='path to sweep out_dir containing summary.csv and per-run folders')
    p.add_argument('--n_samples', type=int, default=200)
    p.add_argument('--top_n', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    summary = os.path.join(args.out_dir, 'summary.csv')
    if not os.path.exists(summary):
        raise SystemExit(f'Missing summary.csv under {args.out_dir}')
    # pick best row
    import pandas as pd
    df = pd.read_csv(summary)
    best = df.sort_values(['auc_test', 'ap_test'], ascending=False).iloc[0]
    run_dir = os.path.join(args.out_dir, best['label'])
    if not os.path.exists(run_dir):
        raise SystemExit(f'Run dir not found: {run_dir}')

    py = sys.executable
    cmds = [
        f'{py} -m src.merge_shap_importances --run_dir "{run_dir}"',
        f'{py} -m src.save_shap_sample --run_dir "{run_dir}" --n_samples {args.n_samples}',
        f'{py} -m src.generate_report --run_dir "{run_dir}"',
        f'{py} -m src.shap_dependence_plots --run_dir "{run_dir}" --top_n {args.top_n}',
        f'{py} -m src.build_pdf_report --run_dir "{run_dir}"'
    ]
    for c in cmds:
        run(c)


if __name__ == '__main__':
    main()
