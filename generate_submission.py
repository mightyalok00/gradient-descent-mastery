"""
Script to generate test predictions and verify against sample_submission.csv
"""

import pandas as pd
import numpy as np
import os
from src.data_loader import load_and_preprocess_data, build_feature_pipeline, load_raw_datasets
from src.gradient_descent import LogisticRegressionGD, OptimizerType

def generate_predictions():
    print("Loading datasets...")
    raw = load_raw_datasets()
    df_train = raw["train"]
    df_test = raw["test"]
    df_sample = raw["sample_submission"]

    print(f"Train Shape: {df_train.shape} | Test Shape: {df_test.shape}")
    
    print("Preprocessing and standardizing features...")
    pipeline_data = build_feature_pipeline(df_train, df_test)
    X_train = pipeline_data["X_train"]
    y_train = pipeline_data["y_train"]
    X_test = pipeline_data["X_test"]

    print(f"Features: {pipeline_data['feature_names']}")

    print("Training high-performance AdamW Logistic Regression model...")
    model = LogisticRegressionGD(
        learning_rate=0.01,
        max_epochs=35,
        batch_size=128,
        optimizer=OptimizerType.ADAMW,
        l2_lambda=0.01,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Generating predictions on test set...")
    test_probs = model.predict_proba(X_test)

    submission_df = pd.DataFrame({
        "transaction_id": df_test["transaction_id"],
        "label": np.round(test_probs, 6)
    })

    sub_path = "submission.csv"
    submission_df.to_csv(sub_path, index=False)
    print(f"Saved {sub_path} with {len(submission_df)} rows.")

    if df_sample is not None:
        assert len(submission_df) == len(df_sample), f"Row count mismatch: {len(submission_df)} vs {len(df_sample)}"
        assert list(submission_df.columns) == list(df_sample.columns), "Column mismatch!"
        print("Verification SUCCESS: Matches sample_submission.csv format perfectly!")

    print("\nFirst 5 rows of submission:")
    print(submission_df.head(5))

if __name__ == "__main__":
    generate_predictions()
