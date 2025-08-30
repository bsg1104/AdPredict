"""Find best sweep run and execute the reporting pipeline for it.

This script reads `outputs/xgb_sweep_full/summary.csv`, selects the row with
highest `auc_test` (tie-breaker `ap_test`), and runs the following steps:
 - merge_shap_importances
 - save_shap_sample
 - generate_report
 - shap_dependence_plots
 - build_pdf_report

It calls the module entry points so we keep shell logic in Python for reliability.
"""
from __future__ import annotations

import subprocess
import os
import pandas as pd
import sys


def run(cmd):
    print('RUN:', cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f'Command failed: {cmd}')


def main():
    summary = 'outputs/xgb_sweep_full/summary.csv'
    if not os.path.exists(summary):
        raise SystemExit('Missing summary.csv for sweep')
    df = pd.read_csv(summary)
    # choose best by auc_test then ap_test
    best = df.sort_values(['auc_test', 'ap_test'], ascending=False).iloc[0]
    run_dir = os.path.join('outputs', 'xgb_sweep_full', best['label'])
    if not os.path.exists(run_dir):
        raise SystemExit(f'Run dir not found: {run_dir}')

    py = sys.executable
    cmds = [
        f'{py} -m src.merge_shap_importances --run_dir "{run_dir}"',
        f'{py} -m src.save_shap_sample --run_dir "{run_dir}" --n_samples 500',
        f'{py} -m src.generate_report --run_dir "{run_dir}"',
        f'{py} -m src.shap_dependence_plots --run_dir "{run_dir}" --top_n 10',
        f'{py} -m src.build_pdf_report --run_dir "{run_dir}"'
    ]

    for c in cmds:
        run(c)


if __name__ == '__main__':
    main()
