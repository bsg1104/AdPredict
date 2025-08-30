# AdPredict — lightweight Criteo-style CTR pipeline

What this repo contains
- `src/`: data loader, feature engineering, LR and XGBoost trainers, experiments, and evaluation tools
- `data/`: helper synthetic/injected datasets (generated when missing)
- `outputs/`: models, metrics, and figures produced by runs

Quickstart

1) Create a virtualenv and install dependencies

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Generate a small injected dataset (if you don't have real Criteo data)

```py
python -c "from src.data import generate_injected_criteo; generate_injected_criteo('data/criteo_injected.csv', nrows=20000)"
```

3) Run a smoke LR run

```sh
python -m src.train_lr --data_path data/criteo_injected.csv --sample_frac 0.05 --out_dir outputs/lr_smoke
```

4) Run a smoke XGBoost run

```sh
python -m src.train_xgb --data_path data/criteo_injected.csv --sample_frac 0.05 --out_dir outputs/xgb_smoke --num_boost_round 100 --early_stopping_rounds 20
```

5) Aggregate evaluation results

```sh
python -m src.evaluate --root outputs --summary_csv outputs/eval_summary.csv --figs_dir outputs/eval_figs
```

Next steps
- Run a hyperparameter sweep (`src/train_xgb.py`) and collect results under `outputs/xgb_sweep/`
- Extract XGBoost feature importances from `outputs/xgb_full/models`
- Improve CI and add unit tests for core functions in `src/`

License: MIT
