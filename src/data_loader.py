"""
Data Loader & Preprocessing Pipeline
====================================
Loads, inspects, engineers features, standardizes, and splits datasets for Gradient Descent.
Handles:
- D:\\gradient_descent\\train.csv
- D:\\gradient_descent\\test.csv
- D:\\gradient_descent\\sample_submission.csv
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


DEFAULT_PATHS = {
    "train": r"D:\gradient_descent\train.csv",
    "test": r"D:\gradient_descent\test.csv",
    "sample_submission": r"D:\gradient_descent\sample_submission.csv"
}


def load_raw_datasets(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    sample_sub_path: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Loads raw CSV files from disk.
    """
    t_path = train_path or DEFAULT_PATHS["train"]
    te_path = test_path or DEFAULT_PATHS["test"]
    s_path = sample_sub_path or DEFAULT_PATHS["sample_submission"]

    datasets = {}
    if os.path.exists(t_path):
        datasets["train"] = pd.read_csv(t_path)
    else:
        raise FileNotFoundError(f"Train dataset not found at {t_path}")

    if os.path.exists(te_path):
        datasets["test"] = pd.read_csv(te_path)
    else:
        raise FileNotFoundError(f"Test dataset not found at {te_path}")

    if os.path.exists(s_path):
        datasets["sample_submission"] = pd.read_csv(s_path)
    else:
        datasets["sample_submission"] = None

    return datasets


def build_feature_pipeline(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    target_col: str = "label",
    id_cols: Optional[list] = None
) -> Dict[str, Any]:
    """
    Extracts features, engineers interaction & cyclical terms, applies StandardScaler,
    and returns matrix arrays ready for Gradient Descent optimization.
    """
    if id_cols is None:
        id_cols = ["transaction_id"]

    df_tr = df_train.copy()
    df_te = df_test.copy() if df_test is not None else None

    # Separate target
    y_train = None
    if target_col in df_tr.columns:
        y_train = df_tr[target_col].to_numpy().astype(float)
        feature_cols = [c for c in df_tr.columns if c not in id_cols + [target_col]]
    else:
        feature_cols = [c for c in df_tr.columns if c not in id_cols]

    # Feature Engineering function
    def transform_features(df: pd.DataFrame) -> pd.DataFrame:
        df_feat = pd.DataFrame(index=df.index)

        # 1. Base numerical features
        if "amount" in df.columns:
            df_feat["amount"] = df["amount"]
            df_feat["log_amount"] = np.log1p(np.maximum(0, df["amount"]))

        if "hours_since_prev_txn" in df.columns:
            df_feat["hours_since_prev_txn"] = df["hours_since_prev_txn"]
            df_feat["log_hours_since_prev"] = np.log1p(np.maximum(0, df["hours_since_prev_txn"]))

        # 2. Time-based features
        if "timestamp" in df.columns:
            # Cyclical encoding
            t = df["timestamp"]
            df_feat["timestamp_norm"] = (t - t.min()) / (t.max() - t.min() + 1e-9)
            df_feat["time_sin_24h"] = np.sin(2 * np.pi * (t % 86400) / 86400.0)
            df_feat["time_cos_24h"] = np.cos(2 * np.pi * (t % 86400) / 86400.0)

        # 3. Categorical frequency / count features
        for cat_col in ["merchant_category", "country", "channel", "device_id", "user_id"]:
            if cat_col in df.columns:
                df_feat[cat_col] = df[cat_col].astype(float)

        return df_feat

    X_tr_feat = transform_features(df_tr)
    feature_names = list(X_tr_feat.columns)

    # Impute NaNs with median if any exist
    medians = X_tr_feat.median()
    X_tr_imputed = X_tr_feat.fillna(medians).to_numpy()

    # Feature Standardization (Crucial for Gradient Descent convergence and condition number)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_tr_imputed)

    X_test_scaled = None
    if df_te is not None:
        X_te_feat = transform_features(df_te)
        # Align columns
        for c in feature_names:
            if c not in X_te_feat.columns:
                X_te_feat[c] = medians[c]
        X_te_feat = X_te_feat[feature_names]
        X_te_imputed = X_te_feat.fillna(medians).to_numpy()
        X_test_scaled = scaler.transform(X_te_imputed)

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "feature_names": feature_names,
        "scaler": scaler,
        "medians": medians
    }


def load_and_preprocess_data(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    val_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    High-level end-to-end data loading and preparation function.
    """
    raw = load_raw_datasets(train_path, test_path)
    df_train = raw["train"]
    df_test = raw["test"]
    df_sub = raw.get("sample_submission")

    pipeline_res = build_feature_pipeline(df_train, df_test)
    X = pipeline_res["X_train"]
    y = pipeline_res["y_train"]

    # Stratified train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )

    return {
        "raw_train": df_train,
        "raw_test": df_test,
        "raw_sample_submission": df_sub,
        "X_train": X_tr,
        "y_train": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": pipeline_res["X_test"],
        "feature_names": pipeline_res["feature_names"],
        "scaler": pipeline_res["scaler"]
    }
