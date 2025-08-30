"""Lightweight feature engineering utilities for AdPredict (Milestone 3).

This module implements a small, easy-to-use pipeline that can be fit on a
training split and applied to val/test. It focuses on practical encodings
that often move the needle quickly:

- time/device features (hour, is_mobile)
- count encodings (frequency of category in train)
- smoothed target encodings for top-k categorical columns
- bucketed continuous features (quantile bins) and log1p transforms

Usage example:
    from src.feature_engineering import fit_fe, apply_fe, explain_features

    fe = fit_fe(train_df, label_col='label', cat_cols=CAT_COLS, int_cols=INT_COLS)
    X_train = apply_fe(train_df, fe)
    X_val = apply_fe(val_df, fe)

The returned `fe` object is a dictionary containing the fitted mappings and
metadata; it's safe to pickle if you want to reuse it later.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# sensible defaults matching our loader
INT_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]


def _safe_str(x):
    if pd.isna(x):
        return "__MISSING__"
    return str(x)


def time_and_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple time and device-derived features.

    - If `hour` exists, create cyclical hour features (sin/cos) and hour bucket.
    - If `device` exists, create is_mobile/is_tablet/is_desktop flags.

    Returns a small DataFrame with new columns.
    """
    out = pd.DataFrame(index=df.index)

    # hour features
    if "hour" in df.columns:
        try:
            hour = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int) % 24
            out["hour"] = hour
            # cyclical encoding
            out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        except Exception:
            pass

    # device features
    if "device" in df.columns:
        dev = df["device"].fillna("").astype(str).str.lower()
        out["is_mobile"] = dev.str.contains("mobile").astype(int)
        out["is_tablet"] = dev.str.contains("tablet").astype(int)
        out["is_desktop"] = dev.str.contains("desktop").astype(int)

    return out


def fit_count_encodings(train: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, int]]:
    """Compute frequency (count) maps for categorical columns based on training data."""
    maps = {}
    for c in cols:
        if c in train.columns:
            counts = train[c].fillna("__MISSING__").astype(str).value_counts().to_dict()
            maps[c] = counts
    return maps


def fit_target_encodings(train: pd.DataFrame, label_col: str, cols: List[str], top_k: int = 3, smoothing: float = 20.0) -> Dict[str, Dict[str, float]]:
    """Fit smoothed target mean encodings for top_k categorical cols by cardinality.

    smoothing: higher -> stronger pull to global mean for low-count categories.
    Returns dict: col -> {category: encoded_value}
    """
    global_mean = train[label_col].mean()
    # pick top_k columns by unique values intersection with provided cols
    cand = [(c, train[c].nunique() if c in train.columns else 0) for c in cols]
    cand = sorted(cand, key=lambda x: x[1], reverse=True)[:top_k]
    out = {}
    for c, _ in cand:
        if c not in train.columns:
            continue
        grp = train.groupby(train[c].fillna("__MISSING__").astype(str))[label_col].agg(["sum", "count"]).rename(columns={"sum": "pos", "count": "n"})
        # smoothed mean = (pos + m * global_mean) / (n + m)
        m = smoothing
        grp["te"] = (grp["pos"] + m * global_mean) / (grp["n"] + m)
        out[c] = grp["te"].to_dict()
    return out


def fit_bucketed_continuous(train: pd.DataFrame, cols: List[str], n_bins: int = 10) -> Dict[str, List[float]]:
    """Compute quantile-based bin edges for integer/continuous columns.

    Returns dict: col -> bin_edges list (length n_bins+1)
    """
    edges = {}
    for c in cols:
        if c in train.columns:
            try:
                arr = pd.to_numeric(train[c], errors="coerce").dropna().values
                if len(arr) < 10:
                    continue
                # use empirical quantiles
                quants = np.linspace(0, 1, n_bins + 1)
                bins = np.unique(np.quantile(arr, quants)).tolist()
                if len(bins) > 1:
                    edges[c] = bins
            except Exception:
                continue
    return edges


def fit_fe(train: pd.DataFrame, label_col: str, cat_cols: Optional[List[str]] = None, int_cols: Optional[List[str]] = None, top_k_te: int = 3, smoothing: float = 20.0, n_bins: int = 10) -> Dict:
    """Fit feature-engineering artifacts from training data.

    Returns a dictionary (`fe`) containing count maps, target encodings, bucket edges and metadata.
    """
    cat_cols = cat_cols or CAT_COLS
    int_cols = int_cols or INT_COLS

    fe = {}
    fe["meta"] = {"cat_cols": cat_cols, "int_cols": int_cols, "top_k_te": top_k_te}

    fe["time_device_cols"] = list(time_and_device_features(train).columns)

    fe["count_maps"] = fit_count_encodings(train, cat_cols)
    fe["target_maps"] = fit_target_encodings(train, label_col, cat_cols, top_k=top_k_te, smoothing=smoothing)
    fe["bucket_edges"] = fit_bucketed_continuous(train, int_cols, n_bins=n_bins)

    return fe


def apply_fe(df: pd.DataFrame, fe: Dict) -> pd.DataFrame:
    """Apply fitted feature-engineering `fe` to df and return a DataFrame of new features.

    The function does not mutate the input `df` and returns a new DataFrame with:
    - time/device features
    - count encodings: col_count
    - target encodings (smoothed) for fitted cols: col_te
    - bucketed continuous features: col_bin_k (as integer bin index)
    - log1p transformed ints: col_log1p
    """
    parts = []
    # time & device
    td = time_and_device_features(df)
    parts.append(td)

    # count encodings
    for c, cmap in fe.get("count_maps", {}).items():
        if c in df.columns:
            s = df[c].fillna("__MISSING__").astype(str).map(cmap).fillna(0).astype(int)
            parts.append(s.rename(f"{c}_count"))

    # target encodings
    for c, tmap in fe.get("target_maps", {}).items():
        if c in df.columns:
            s = df[c].fillna("__MISSING__").astype(str).map(tmap)
            # fallback to global mean if unseen
            global_mean = np.mean(list(tmap.values())) if len(tmap) > 0 else 0.0
            s = s.fillna(global_mean)
            parts.append(s.rename(f"{c}_te"))

    # bucketed continuous and log1p
    for c, edges in fe.get("bucket_edges", {}).items():
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors="coerce").fillna(0)
            parts.append(np.log1p(arr).rename(f"{c}_log1p"))
            try:
                # pandas.cut with right=False so bins represent [edge_i, edge_i+1)
                bins = pd.cut(arr, bins=edges, include_lowest=True, labels=False, duplicates="drop")
                parts.append(bins.rename(f"{c}_bin"))
            except Exception:
                # fallback: skip binning
                continue

    # combine
    if len(parts) == 0:
        return pd.DataFrame(index=df.index)

    # convert Series to DataFrame properly
    df_parts = []
    for p in parts:
        if isinstance(p, pd.Series):
            df_parts.append(p)
        elif isinstance(p, pd.DataFrame):
            df_parts.append(p)
        else:
            # numpy array -> Series
            df_parts.append(pd.Series(p, index=df.index))

    out = pd.concat(df_parts, axis=1)
    # fill missing numeric values
    out = out.fillna(0)
    return out


def explain_features(fe: Dict) -> str:
    """Return a human-readable summary of the fitted feature engineering artifacts."""
    lines = []
    meta = fe.get("meta", {})
    lines.append(f"Fitted feature engineering: cat_cols={len(meta.get('cat_cols', []))}, int_cols={len(meta.get('int_cols', []))}")
    lines.append(f"Time/device features: {fe.get('time_device_cols', [])}")
    lines.append(f"Count-encoded cols: {list(fe.get('count_maps', {}).keys())}")
    lines.append(f"Target-encoded cols: {list(fe.get('target_maps', {}).keys())}")
    lines.append(f"Bucketed continuous cols: {list(fe.get('bucket_edges', {}).keys())}")
    return "\n".join(lines)


__all__ = [
    "fit_fe",
    "apply_fe",
    "explain_features",
    "time_and_device_features",
]
