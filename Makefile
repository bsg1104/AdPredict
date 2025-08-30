# Makefile for common tasks
.PHONY: venv install run-lr run-xgb eval test inspect-xgb clean
 .PHONY: xgb-sweep-full sweep-agg append-eval explain-shap

report:
	. .venv/bin/activate && python -m src.run_full_report || true

report-smoke:
	. .venv/bin/activate && python -m src.xgb_sweep --data_path data/criteo_injected.csv --sample_frac 0.02 --out_dir outputs/xgb_sweep_smoke --smoke ; \
	. .venv/bin/activate && python -m src.run_report_for_outdir --out_dir outputs/xgb_sweep_smoke --n_samples 100 --top_n 6 || true

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run-lr:
	. .venv/bin/activate && python -m src.train_lr --data_path data/criteo_injected.csv --sample_frac 0.2 --out_dir outputs/lr_full

run-xgb:
	. .venv/bin/activate && python -m src.train_xgb --data_path data/criteo_injected.csv --sample_frac 0.2 --out_dir outputs/xgb_full --num_boost_round 300 --early_stopping_rounds 50 --max_depth 8 --eta 0.05

eval:
	. .venv/bin/activate && python -m src.evaluate --root outputs --summary_csv outputs/eval_summary.csv --figs_dir outputs/eval_figs

inspect-xgb:
	. .venv/bin/activate && python -m src.inspect_xgb --model outputs/xgb_full/models/xgb_*.json --out_dir outputs/xgb_full/analysis --top_n 30

xgb-sweep:
	. .venv/bin/activate && python -m src.xgb_sweep --data_path data/criteo_injected.csv --sample_frac 0.05 --out_dir outputs/xgb_sweep --smoke

xgb-sweep-full:
	. .venv/bin/activate && python -m src.xgb_sweep --data_path data/criteo_injected.csv --sample_frac 0.2 --out_dir outputs/xgb_sweep_full --etas 0.01,0.03,0.05 --max_depths 6,8,10 --num_boost_rounds 300 --early_stopping_rounds 50

sweep-agg:
	. .venv/bin/activate && python -c "import pandas as pd; df=pd.read_csv('outputs/xgb_sweep_full/summary.csv'); df.plot.scatter(x='auc_test', y='ap_test', c='eta', colormap='viridis', figsize=(6,6)); import matplotlib.pyplot as plt; plt.savefig('outputs/xgb_sweep_full/auc_ap_grid.png')"

append-eval:
	. .venv/bin/activate && python -c "import pandas as pd; s=pd.read_csv('outputs/xgb_sweep_full/summary.csv'); best=s.sort_values(['auc_test','ap_test'], ascending=False).iloc[0]; df=pd.read_csv('outputs/eval_summary.csv') if os.path.exists('outputs/eval_summary.csv') else pd.DataFrame(columns=['run','auc_test','ap_test']); df=df.append({'run':best['label'],'auc_test':best['auc_test'],'ap_test':best['ap_test']}, ignore_index=True); df.to_csv('outputs/eval_summary.csv', index=False); print('Appended best run to outputs/eval_summary.csv')"

explain-shap:
	. .venv/bin/activate && pip install shap xgboost && python -m src.explain_shap --model $(shell ls outputs/xgb_sweep_full/*/models/xgb_*.json | head -n1) --out_dir outputs/shap

test:
	. .venv/bin/activate && pip install pytest && pytest -q

clean:
	rm -rf .venv __pycache__ *.pyc outputs/*
