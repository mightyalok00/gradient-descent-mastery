import numpy as np

from src.evaluation import ConvergenceDiagnosisSystem, evaluate_classification


def test_classification_metrics_are_finite():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.2, 0.8, 0.95])

    metrics = evaluate_classification(y_true, y_proba)

    assert metrics["roc_auc"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_convergence_diagnosis_handles_short_history():
    result = ConvergenceDiagnosisSystem.diagnose(
        {"epoch": [0, 1], "train_loss": [1.0, 0.8]}
    )

    assert result["status"] == "INSUFFICIENT_DATA"
