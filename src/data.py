"""Data utilities for the AdPredict demo.

Includes small helpers for fast iteration during development:
- generate_sample_csv: quick synthetic data generator
- load_criteo_csv: read Criteo-format TSV (headerless) with optional sampling
- split_df: stratified train/val/test split with printed diagnostics
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_sample_csv(path: str, nrows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic dataset and save it to `path`.

    The dataset contains numeric, categorical and time features plus a
    binary `clicked` target. It's intentionally simple so you can iterate
    quickly while developing features and models.
    """
    np.random.seed(seed)

    num_1 = np.random.randn(nrows) * 2 + 1
    num_2 = np.random.exponential(scale=1.0, size=nrows)

    slots = [f"slot_{i}" for i in range(10)]
    devices = ["mobile", "desktop", "tablet"]

    slot = np.random.choice(slots, size=nrows)
    device = np.random.choice(devices, size=nrows)
    hour = np.random.randint(0, 24, size=nrows)

    base_rate = 0.01
    score = (
        base_rate
        + 0.02 * (device == "mobile").astype(float)
        + 0.01 * (slot == "slot_3").astype(float)
        + 0.001 * (hour >= 18)
        + 0.05 * (num_1 > 2)
    )

    probs = 1 / (1 + np.exp(- (score - 0.03) * 10))
    clicked = np.random.binomial(1, probs)

    df = pd.DataFrame(
        {
            "num_1": num_1,
            "num_2": num_2,
            "slot": slot,
            "device": device,
            "hour": hour,
            "clicked": clicked,
        }
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def generate_injected_criteo(path: str, nrows: int = 10000, seed: int = 42, inject_cols=None, signal_strength: float = 1.0) -> pd.DataFrame:
    """Generate a Criteo-like TSV with an injected predictive signal.

    The output is a headerless TSV matching columns expected by
    `load_criteo_csv`: label, I1..I13, C1..C26. Injection is added to a
    small set of columns (categorical or integer) to make FE experiments
    reproducible during development.
    """
    np.random.seed(seed)

    if inject_cols is None:
        inject_cols = ["C1", "C2", "I1"]

    # integer features I1..I13
    int_cols = [f"I{i}" for i in range(1, 14)]
    X_int = {c: np.random.poisson(lam=2 + (i % 3), size=nrows) for i, c in enumerate(int_cols)}

    # categorical features C1..C26 with moderate cardinality
    cat_cols = [f"C{i}" for i in range(1, 27)]
    categories = [f"cat_{j}" for j in range(50)]
    X_cat = {c: np.random.choice(categories, size=nrows) for c in cat_cols}

    # build dataframe
    df = pd.DataFrame({**X_int, **X_cat})

    # create a base score and then inject signal from chosen cols
    score = np.zeros(nrows, dtype=float)
    for c in inject_cols:
        if c in int_cols:
            score += signal_strength * (df[c] > df[c].mean()).astype(float)
        elif c in cat_cols:
            # make certain categories positively predictive
            top = df[c].value_counts().index[:3].tolist()
            score += signal_strength * df[c].isin(top).astype(float)

    # convert score to probability and sample labels
    probs = 1 / (1 + np.exp(-(score - 0.5)))
    labels = np.random.binomial(1, probs)

    # final TSV requires label first
    cols = ["label"] + int_cols + cat_cols
    out_df = pd.DataFrame(columns=cols)
    out_df["label"] = labels
    for c in int_cols:
        out_df[c] = df[c]
    for c in cat_cols:
        out_df[c] = df[c]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # write as TSV without header to match load_criteo_csv expectations
    out_df.to_csv(path, sep="\t", header=False, index=False)
    return out_df


def load_criteo_csv(path: str, sample_frac: float = None, nrows: int = None, seed: int = 42) -> pd.DataFrame:
    """Load a Criteo-format TSV into a DataFrame with columns:
    label, I1..I13, C1..C26.

    Args:
        path: path to TSV file
        sample_frac: optional fraction to sample for quicker iteration
        nrows: optional number of rows to read (pandas `nrows`)
        seed: random seed used when sampling

    Returns:
        DataFrame with columns [label, I1..I13, C1..C26]; integer fields are coerced to numeric.
    """
    int_cols = [f"I{i}" for i in range(1, 14)]
    cat_cols = [f"C{i}" for i in range(1, 27)]
    cols = ["label"] + int_cols + cat_cols

    df = pd.read_csv(path, sep="\t", header=None, names=cols, nrows=nrows, dtype=str)

    # convert numeric-ish columns to numbers; keep NaN for missing
    for c in int_cols + ["label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if sample_frac is not None and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)

    return df


def split_df(df: pd.DataFrame, stratify_col: str = "label", train_frac: float = 0.7, val_frac: float = 0.1, test_frac: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train/val/test and print simple stats.

    The function prints rows, CTR (mean of `stratify_col`) and label counts for
    the full dataset and each split so you can quickly verify splits are balanced.
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "fractions must sum to 1"
    if stratify_col not in df.columns:
        # friendly fallback for the demo synthetic generator which uses 'clicked'
        alt = None
        if 'clicked' in df.columns:
            alt = 'clicked'
        elif 'label' in df.columns:
            alt = 'label'

        if alt is not None:
            print(f"Note: stratify column '{stratify_col}' not found; using '{alt}' instead.")
            stratify_col = alt
        else:
            raise ValueError(f"Stratify column '{stratify_col}' not in dataframe and no fallback found (expected 'clicked' or 'label')")

    # first split out train
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        stratify=df[stratify_col],
        random_state=random_state,
    )

    # split the remainder into val/test
    val_relative = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative),
        stratify=temp_df[stratify_col],
        random_state=random_state,
    )

    def _print_stats(name: str, d: pd.DataFrame):
        total = len(d)
        ctr = d[stratify_col].mean() if total > 0 else float("nan")
        counts = d[stratify_col].value_counts(dropna=False).to_dict()
        print(f"{name}: rows={total}, CTR={ctr:.6f}, label_counts={counts}")

    print("Dataset summary:")
    _print_stats("All", df)
    print("Split summary:")
    _print_stats("Train", train_df)
    _print_stats("Val", val_df)
    _print_stats("Test", test_df)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
