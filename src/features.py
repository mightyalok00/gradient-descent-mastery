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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic model features without fitting preprocessing state."""

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
        denominator = timestamp.max() - timestamp.min() + 1e-9
        features["timestamp_norm"] = (timestamp - timestamp.min()) / denominator
        features["time_sin_24h"] = np.sin(2 * np.pi * (timestamp % 86400) / 86400.0)
        features["time_cos_24h"] = np.cos(2 * np.pi * (timestamp % 86400) / 86400.0)

    for column in CATEGORICAL_NUMERIC_COLUMNS:
        if column in df.columns:
            features[column] = pd.to_numeric(df[column], errors="coerce")

    return features
