import numpy as np
import pandas as pd
import pytest

from src.validation import (
    validate_binary_target,
    validate_finite_numeric,
    validate_schema,
    validate_train_test,
)


def _valid_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "amount": [10.0, 20.0],
            "timestamp": [100, 200],
            "hours_since_prev_txn": [1.0, 2.0],
            "merchant_category": [1, 2],
            "country": [1, 2],
            "channel": [1, 2],
            "device_id": [10, 11],
            "user_id": [20, 21],
            "label": [0, 1],
        }
    )


def test_validate_schema_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(pd.DataFrame({"amount": [1]}), {"amount", "label"})


def test_validate_binary_target_rejects_non_binary_values():
    df = _valid_train().assign(label=[0, 2])
    with pytest.raises(ValueError, match="only 0/1"):
        validate_binary_target(df)


def test_validate_finite_numeric_rejects_infinity():
    df = pd.DataFrame({"amount": [1.0, np.inf]})
    with pytest.raises(ValueError, match="non-finite"):
        validate_finite_numeric(df)


def test_validate_train_test_accepts_matching_schema():
    train = _valid_train()
    test = train.drop(columns="label").copy()
    validate_train_test(train, test)


def test_validate_train_test_rejects_duplicate_ids():
    train = _valid_train().assign(transaction_id=[1, 1])
    test = _valid_train().drop(columns="label")
    with pytest.raises(ValueError, match="duplicate"):
        validate_train_test(train, test)
