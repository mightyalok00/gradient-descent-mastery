"""Dataset schema and integrity validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TRAIN_COLUMNS = {
    "transaction_id",
    "amount",
    "timestamp",
    "hours_since_prev_txn",
    "merchant_category",
    "country",
    "channel",
    "device_id",
    "user_id",
    "label",
}

DEFAULT_TEST_COLUMNS = DEFAULT_TRAIN_COLUMNS - {"label"}


def validate_schema(
    df: pd.DataFrame,
    required_columns: set[str],
    *,
    dataset_name: str = "dataset",
) -> None:
    """Validate required columns and reject empty input."""

    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {sorted(missing)}"
        )


def validate_finite_numeric(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    dataset_name: str = "dataset",
) -> None:
    """Reject infinite values in numeric columns."""

    numeric_columns = columns or df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        return

    values = df[numeric_columns].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset_name} contains non-finite numeric values.")


def validate_binary_target(
    df: pd.DataFrame,
    target_column: str = "label",
) -> None:
    """Validate that a classification target contains only binary values."""

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")

    values = pd.to_numeric(df[target_column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Target column '{target_column}' contains non-numeric values.")

    unique_values = set(values.dropna().unique())
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_column}' must contain only 0/1 values; "
            f"found {sorted(unique_values)}."
        )


def validate_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str = "label",
    id_columns: tuple[str, ...] = ("transaction_id",),
) -> None:
    """Validate train/test schema, IDs, target and numeric integrity."""

    validate_schema(
        train,
        DEFAULT_TRAIN_COLUMNS,
        dataset_name="train dataset",
    )
    validate_schema(
        test,
        DEFAULT_TEST_COLUMNS,
        dataset_name="test dataset",
    )
    validate_binary_target(train, target_column)
    validate_finite_numeric(train, dataset_name="train dataset")
    validate_finite_numeric(test, dataset_name="test dataset")

    for id_column in id_columns:
        if id_column in train.columns and train[id_column].duplicated().any():
            raise ValueError(f"train dataset contains duplicate '{id_column}' values.")
        if id_column in test.columns and test[id_column].duplicated().any():
            raise ValueError(f"test dataset contains duplicate '{id_column}' values.")

    train_features = set(train.columns) - {target_column, *id_columns}
    test_features = set(test.columns) - set(id_columns)
    if train_features != test_features:
        raise ValueError(
            "Train/test feature columns do not match after removing target and ID columns."
        )
