"""Small FastAPI service to serve predictions from the best XGBoost model.

Endpoints:
 - GET /health
 - POST /predict  {"rows": [ {col: val, ...}, ... ] }

The service will load the best model under `outputs/xgb_sweep_full/summary.csv` and
the FE from the corresponding run's `run_config.json` where available.
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi import Header
from pydantic import BaseModel

from src.feature_engineering import fit_fe, apply_fe
from src.data import load_criteo_csv, split_df


app = FastAPI()


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]


def _find_best_model():
    summary = 'outputs/xgb_sweep_full/summary.csv'
    if not os.path.exists(summary):
        return None, None
    df = pd.read_csv(summary)
    best = df.sort_values(['auc_test', 'ap_test'], ascending=False).iloc[0]
    model_path = best['model_path']
    run_dir = os.path.dirname(os.path.dirname(model_path))
    return model_path, run_dir


MODEL = None
FE = None
FEATURE_COLS = None


def load_model_and_fe():
    global MODEL, FE, FEATURE_COLS
    model_path, run_dir = _find_best_model()
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError('No trained model found; run sweep first')

    # attempt to load xgboost model
    bst = xgb.Booster()
    bst.load_model(model_path)
    MODEL = bst

    # try to load FE config from run_config.json
    rc = os.path.join(run_dir, 'run_config.json')
    if os.path.exists(rc):
        cfg = json.load(open(rc))
        # FE can't be fully reconstructed from json; attempt to load fitted FE if present
        fe_path = os.path.join(run_dir, 'fe.joblib')
        if os.path.exists(fe_path):
            FE = joblib.load(fe_path)
        else:
            FE = None
    else:
        FE = None


@app.on_event('startup')
def startup_event():
    try:
        load_model_and_fe()
        print('Model and FE loaded')
    except Exception as e:
        print('Warning: model/FE not loaded on startup:', e)


def _pred_from_rows(rows: List[Dict[str, Any]]):
    if FE is None:
        # fallback: try to guess numeric columns and run predict on raw values
        df = pd.DataFrame(rows)
        try:
            dmat = xgb.DMatrix(df)
            preds = MODEL.predict(dmat)
            return preds.tolist()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')

    # apply FE (assumes FE has transform_dataframe method or similar)
    df = pd.DataFrame(rows)
    try:
        X = apply_fe(df, fe=FE)
    except Exception:
        # try fit_fe quickly using a tiny sample from data
        raise HTTPException(status_code=500, detail='FE application failed')

    dmat = xgb.DMatrix(X)
    preds = MODEL.predict(dmat)
    return preds.tolist()


@app.get('/health')
def health():
    ok = MODEL is not None
    return {'ok': ok}


@app.post('/predict')
def predict(req: PredictRequest, x_api_token: str | None = Header(None)):
    # simple token auth
    expected = os.environ.get('API_TOKEN')
    if expected and x_api_token != expected:
        raise HTTPException(status_code=401, detail='Invalid API token')
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    preds = _pred_from_rows(req.rows)
    return {'predictions': preds}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('src.predict_api:app', host='127.0.0.1', port=8000, reload=False)
