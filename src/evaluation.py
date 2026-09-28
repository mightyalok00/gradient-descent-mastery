"""
Model Evaluation, Benchmark Comparison, and Convergence Diagnostics
===================================================================
Provides comprehensive evaluation metrics (ROC-AUC, PR-AUC, Confusion Matrix, Brier score),
automated optimization diagnosis, and ablation study frameworks.
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.linear_model import SGDClassifier, LogisticRegression, Ridge


def evaluate_classification(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics tailored for imbalanced datasets.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        roc_auc = 0.5

    # PR-AUC
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_pred_proba))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "brier_score": brier,
        "confusion_matrix": cm,
        "pos_prevalence": float(np.mean(y_true))
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes regression evaluation metrics.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    }


class ConvergenceDiagnosisSystem:
    """
    Automated diagnosis engine as designed in Question 15:
    Receives training telemetry: epoch, train_loss, val_loss, grad_norm, lr, param_norm
    and evaluates:
    - Underfitting
    - Overfitting
    - Learning-rate explosion
    - Vanishing gradients
    - Gradient explosion
    - Poor conditioning
    - Plateauing
    - Healthy Convergence
    """
    @staticmethod
    def diagnose(history: Dict[str, List[float]]) -> Dict[str, Any]:
        epochs = history.get("epoch", [])
        train_loss = np.array(history.get("train_loss", []))
        val_loss = np.array(history.get("val_loss", []))
        grad_norm = np.array(history.get("gradient_norm", []))
        param_norm = np.array(history.get("parameter_norm", []))
        lr = np.array(history.get("learning_rate", []))

        n_steps = len(train_loss)
        if n_steps < 3:
            return {"status": "INSUFFICIENT_DATA", "diagnoses": ["Need at least 3 epochs for diagnosis."]}

        diagnoses = []
        severity = "NORMAL"

        # 1. Check for NaN / Inf (Explosion / Numerical Instability)
        if np.any(np.isnan(train_loss)) or np.any(np.isinf(train_loss)):
            diagnoses.append("CRITICAL: Loss exploded to NaN/Inf. Learning rate is excessively high or loss is unstable.")
            return {"status": "LR_EXPLOSION", "severity": "CRITICAL", "diagnoses": diagnoses}

        # 2. Gradient Explosion Check
        if len(grad_norm) > 0 and (np.max(grad_norm) > 1e4 or np.any(np.isnan(grad_norm))):
            diagnoses.append("GRADIENT EXPLOSION: Gradient norm exceeded 10,000. Apply gradient clipping and reduce learning rate.")
            severity = "HIGH"

        # 3. Vanishing Gradient Check
        if len(grad_norm) > 5 and np.mean(grad_norm[-5:]) < 1e-6:
            diagnoses.append("VANISHING GRADIENTS: Gradient norm is near zero (< 1e-6). Optimization has stalled.")
            severity = "HIGH"

        # 4. Overfitting Check
        if len(val_loss) > 5 and not np.all(np.isnan(val_loss)):
            valid_val = val_loss[~np.isnan(val_loss)]
            if len(valid_val) >= 5:
                val_trend = valid_val[-1] - np.min(valid_val)
                train_trend = train_loss[-1] - np.min(train_loss)
                # Validation loss increasing while training loss decreases
                if valid_val[-1] > 1.25 * np.min(valid_val) and train_loss[-1] <= np.min(train_loss[:len(valid_val)]):
                    diagnoses.append("OVERFITTING DETECTED: Validation loss is rising while training loss is decreasing.")
                    severity = "WARNING"

        # 5. Underfitting Check
        if train_loss[-1] > 0.6 and (train_loss[0] - train_loss[-1]) / max(1e-9, train_loss[0]) < 0.05:
            diagnoses.append("UNDERFITTING: Training loss remains high and has barely decreased from initialization.")
            severity = "WARNING"

        # 6. Poor Conditioning / High Oscillations
        diffs = np.diff(train_loss)
        sign_flips = np.sum(diffs[:-1] * diffs[1:] < 0)
        oscillation_rate = sign_flips / max(1, len(diffs) - 1)
        if oscillation_rate > 0.6 and np.std(train_loss[-min(10, n_steps):]) > 0.05:
            diagnoses.append("POOR CONDITIONING / HIGH OSCILLATION: Loss is zig-zagging erratically across steps. Use Momentum/Adam or decrease LR.")
            severity = "WARNING"

        # 7. Plateauing Check
        if len(train_loss) >= 10:
            recent_delta = abs(train_loss[-1] - train_loss[-10]) / max(1e-9, train_loss[-10])
            if recent_delta < 1e-4:
                diagnoses.append("PLATEAUING: Loss change over past 10 epochs is under 0.01%. Optimization has converged or is in a saddle region.")

        if not diagnoses:
            diagnoses.append("HEALTHY OPTIMIZATION: Loss is decreasing stably with controlled gradient norms and parameter bounds.")
            severity = "OPTIMAL"

        return {
            "status": "HEALTHY" if severity == "OPTIMAL" else "ATTENTION_REQUIRED",
            "severity": severity,
            "diagnoses": diagnoses,
            "final_train_loss": float(train_loss[-1]),
            "final_val_loss": float(val_loss[-1]) if len(val_loss) > 0 and not np.isnan(val_loss[-1]) else None,
            "final_grad_norm": float(grad_norm[-1]) if len(grad_norm) > 0 else None,
            "final_param_norm": float(param_norm[-1]) if len(param_norm) > 0 else None
        }


def run_ablation_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40
) -> pd.DataFrame:
    """
    Executes an ablation study comparing optimization configurations.
    """
    from .gradient_descent import LogisticRegressionGD, OptimizerType

    configs = [
        {"name": "Adam (Standard)", "opt": OptimizerType.ADAM, "lr": 0.01, "l2": 0.0},
        {"name": "AdamW (Decoupled L2=0.01)", "opt": OptimizerType.ADAMW, "lr": 0.01, "l2": 0.01},
        {"name": "SGD + Momentum (beta=0.9)", "opt": OptimizerType.MOMENTUM, "lr": 0.05, "l2": 0.0},
        {"name": "RMSProp (beta2=0.99)", "opt": OptimizerType.RMSPROP, "lr": 0.01, "l2": 0.0},
        {"name": "Batch Gradient Descent", "opt": OptimizerType.BATCH_GD, "lr": 0.1, "l2": 0.0},
        {"name": "Mini-Batch GD (B=64)", "opt": OptimizerType.MINI_BATCH_GD, "lr": 0.05, "l2": 0.0}
    ]

    records = []
    for cfg in configs:
        model = LogisticRegressionGD(
            learning_rate=cfg["lr"],
            max_epochs=epochs,
            batch_size=64,
            optimizer=cfg["opt"],
            l2_lambda=cfg["l2"]
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        y_val_proba = model.predict_proba(X_val)
        metrics = evaluate_classification(y_val, y_val_proba)

        records.append({
            "Configuration": cfg["name"],
            "Val ROC-AUC": round(metrics["roc_auc"], 4),
            "Val PR-AUC": round(metrics["pr_auc"], 4),
            "Val F1-Score": round(metrics["f1_score"], 4),
            "Val Brier Score": round(metrics["brier_score"], 4),
            "Final Val Loss": round(model.history["val_loss"][-1], 4)
        })

    return pd.DataFrame(records)
