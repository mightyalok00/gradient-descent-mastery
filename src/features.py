"""Feature engineering utilities for the transaction dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_NUMERIC_COLUMNS = (
    "merchant_category",
    "country",
    "channel",
    "device_id",
    "user_id",
)


def engineer_features(
    df: pd.DataFrame,
    *,
    timestamp_min: float | None = None,
    timestamp_max: float | None = None,
) -> pd.DataFrame:
    """Create deterministic features using optional train-fitted time bounds."""

    features = pd.DataFrame(index=df.index)

    if "amount" in df.columns:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        features["amount"] = amount
        features["log_amount"] = np.log1p(np.maximum(0, amount))

    if "hours_since_prev_txn" in df.columns:
        hours = pd.to_numeric(df["hours_since_prev_txn"], errors="coerce")
        features["hours_since_prev_txn"] = hours
        features["log_hours_since_prev"] = np.log1p(np.maximum(0, hours))

    if "timestamp" in df.columns:
        timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
        lower = timestamp.min() if timestamp_min is None else timestamp_min
        upper = timestamp.max() if timestamp_max is None else timestamp_max
        denominator = upper - lower + 1e-9
        features["timestamp_norm"] = (timestamp - lower) / denominator
        features["time_sin_24h"] = np.sin(2 * np.pi * (timestamp % 86400) / 86400.0)
        features["time_cos_24h"] = np.cos(2 * np.pi * (timestamp % 86400) / 86400.0)

    for column in CATEGORICAL_NUMERIC_COLUMNS:
        if column in df.columns:
            features[column] = pd.to_numeric(df[column], errors="coerce")

    return features
