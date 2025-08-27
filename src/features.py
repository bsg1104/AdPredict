"""Friendly, minimal featurizer for the demo.

This function keeps numeric features as floats and hashes categorical
features into a fixed-size vector so models can run quickly without an
explosion of one-hot columns.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher


def featurize(df: pd.DataFrame, n_features: int = 2 ** 10) -> np.ndarray:
    """Convert a DataFrame into a numeric feature matrix.

    Keeps `num_1` and `num_2` as continuous features, and hashes the
    categorical fields `slot`, `device`, and `hour` into a dense array.
    """
    # numeric columns (fill missing values with 0)
    X_num = df[["num_1", "num_2"]].fillna(0).astype(float).values

    # turn categorical columns into a list of dicts for FeatureHasher
    cats = df[["slot", "device", "hour"]].astype(str).to_dict(orient="records")
    hasher = FeatureHasher(n_features=n_features, input_type="dict")
    X_cat = hasher.transform(cats).toarray()

    # join numeric and hashed categorical features
    X = np.hstack([X_num, X_cat])
    return X
