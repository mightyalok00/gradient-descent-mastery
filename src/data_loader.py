"""Data loading, validation, feature engineering and preprocessing."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import DataConfig
from .features import engineer_features
from .validation import validate_train_test


DEFAULT_PATHS = {
    "train": r"D:\gradient_descent\train.csv",
    "test": r"D:\gradient_descent\test.csv",
    "sample_submission": r"D:\gradient_descent\sample_submission.csv",
}


def _resolve_default_path(kind: str) -> str:
    """Resolve a dataset path using the central DataConfig."""

    config = DataConfig().resolve()
    return {
        "train": config.train_path,
        "test": config.test_path,
        "sample_submission": config.sample_submission_path,
    }[kind]


def load_raw_datasets(
    train_path: str | None = None,
    test_path: str | None = None,
    sample_sub_path: str | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Load train, test and optional sample-submission CSV files."""

    config = DataConfig(
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_sub_path,
    ).resolve()

    if not config.train_path or not os.path.isfile(config.train_path):
        raise FileNotFoundError(f"Train dataset not found at {config.train_path}")
    if not config.test_path or not os.path.isfile(config.test_path):
        raise FileNotFoundError(f"Test dataset not found at {config.test_path}")

    datasets: dict[str, pd.DataFrame | None] = {
        "train": pd.read_csv(config.train_path),
        "test": pd.read_csv(config.test_path),
        "sample_submission": None,
    }

    if config.sample_submission_path and os.path.isfile(config.sample_submission_path):
        datasets["sample_submission"] = pd.read_csv(config.sample_submission_path)

    return datasets


def build_feature_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
    target_col: str = "label",
    id_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Engineer features, fit preprocessing on train data, and transform test data."""

    id_cols = id_cols or ["transaction_id"]
    df_tr = df_train.copy()
    df_te = df_test.copy() if df_test is not None else None

    y_train = None
    if target_col in df_tr.columns:
        y_train = pd.to_numeric(df_tr[target_col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y_train).any():
            raise ValueError(f"Target column '{target_col}' contains missing/non-numeric values.")

    timestamp_min = None
    timestamp_max = None
    if "timestamp" in df_tr.columns:
        train_timestamp = pd.to_numeric(df_tr["timestamp"], errors="coerce")
        timestamp_min = train_timestamp.min()
        timestamp_max = train_timestamp.max()

    X_tr_feat = engineer_features(
        df_tr,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
    )
    feature_names = list(X_tr_feat.columns)

    medians = X_tr_feat.median()
    X_train_imputed = X_tr_feat.fillna(medians).to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    X_test_scaled = None
    if df_te is not None:
        X_te_feat = engineer_features(
            df_te,
            timestamp_min=timestamp_min,
            timestamp_max=timestamp_max,
        )
        for column in feature_names:
            if column not in X_te_feat.columns:
                X_te_feat[column] = medians[column]
        X_te_feat = X_te_feat[feature_names]
        X_test_imputed = X_te_feat.fillna(medians).to_numpy(dtype=float)
        X_test_scaled = scaler.transform(X_test_imputed)

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "feature_names": feature_names,
        "scaler": scaler,
        "medians": medians,
        "id_columns": id_cols,
    }


def load_and_preprocess_data(
    train_path: str | None = None,
    test_path: str | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load, validate, engineer and split the transaction dataset."""

    raw = load_raw_datasets(train_path, test_path)
    df_train = raw["train"]
    df_test = raw["test"]

    if df_train is None or df_test is None:
        raise ValueError("Train and test datasets are required.")

    validate_train_test(df_train, df_test)

    pipeline = build_feature_pipeline(df_train, df_test)
    X = pipeline["X_train"]
    y = pipeline["y_train"]

    if y is None:
        raise ValueError("Training target is required.")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        "raw_train": df_train,
        "raw_test": df_test,
        "raw_sample_submission": raw.get("sample_submission"),
        "X_train": X_tr,
        "y_train": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": pipeline["X_test"],
        "feature_names": pipeline["feature_names"],
        "scaler": pipeline["scaler"],
        "medians": pipeline["medians"],
    }
