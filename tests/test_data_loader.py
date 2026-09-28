import pandas as pd

from src.data_loader import build_feature_pipeline


def test_feature_pipeline_returns_scaled_features():
    train = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4],
            "amount": [100.0, 200.0, 150.0, 300.0],
            "hours_since_prev_txn": [1.0, 2.0, 4.0, 8.0],
            "timestamp": [0, 3600, 7200, 10800],
            "label": [0, 1, 0, 1],
        }
    )
    test = train.drop(columns="label").copy()

    result = build_feature_pipeline(train, test)

    assert result["X_train"].shape[0] == 4
    assert result["X_test"].shape[0] == 4
    assert result["y_train"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result["X_train"].shape[1] == len(result["feature_names"])
