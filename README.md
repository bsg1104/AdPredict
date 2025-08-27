# AdPredict — Click-Through Rate Prediction for Ads

This repository contains a minimal, reproducible skeleton for AdPredict: a CTR prediction project.

What is included
- `src/data.py` — generates a small synthetic sample dataset (fast, reproducible), includes a Criteo TSV loader and a stratified split helper.
- `src/features.py` — simple preprocessing and hashing for categorical features.
- `src/train.py` — trains a logistic regression baseline, prints metrics, and saves an ROC plot. Use `--test-splits` to run split diagnostics.
- `requirements.txt` — Python deps for the demo.

Quick start (macOS / sh):

1. Create and activate a virtualenv (optional):

```sh
python3 -m venv .venv
. .venv/bin/activate
```

2. Install dependencies:

```sh
pip install -r requirements.txt
```

3. Run the demo training script:

```sh
python3 src/train.py --nrows 20000
```

The script will generate `data/sample.csv`, train a logistic regression baseline, print ROC-AUC, and save `outputs/roc.png`.

Notes
- This is a small, local demo. To scale to Criteo-size data, swap the data loader for streaming reading and switch to scalable featurizers and model training.
- Next steps: add XGBoost training script, hyperparameter tuning, and evaluation notebooks.
