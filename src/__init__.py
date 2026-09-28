"""Public API for the Gradient Descent ML Framework."""

from .config import DataConfig
from .data_loader import build_feature_pipeline, load_and_preprocess_data, load_raw_datasets
from .evaluation import (
    ConvergenceDiagnosisSystem,
    evaluate_classification,
    evaluate_regression,
    run_ablation_study,
)
from .features import engineer_features
from .gradient_descent import LinearRegressionGD, LogisticRegressionGD, LRScheduler, OptimizerType
from .missing_analysis import MissingDataFormulaAnalyzer
from .profiling import ComprehensiveDataProfiler
from .questions_solutions import QUESTIONS_AND_SOLUTIONS
from .validation import validate_binary_target, validate_finite_numeric, validate_schema, validate_train_test

__all__ = [
    "DataConfig",
    "load_raw_datasets",
    "load_and_preprocess_data",
    "build_feature_pipeline",
    "engineer_features",
    "validate_schema",
    "validate_finite_numeric",
    "validate_binary_target",
    "validate_train_test",
    "MissingDataFormulaAnalyzer",
    "LinearRegressionGD",
    "LogisticRegressionGD",
    "OptimizerType",
    "LRScheduler",
    "ComprehensiveDataProfiler",
    "evaluate_classification",
    "evaluate_regression",
    "ConvergenceDiagnosisSystem",
    "run_ablation_study",
    "QUESTIONS_AND_SOLUTIONS",
]
