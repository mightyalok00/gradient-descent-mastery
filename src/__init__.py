# Package initialization for Gradient Descent ML Mastery Framework
"""
Gradient Descent ML Framework
=============================
A comprehensive suite for Gradient Descent optimization, formulaic missing data analysis,
automated data profiling, model benchmarking, and diagnostics.
"""

from .data_loader import load_and_preprocess_data, build_feature_pipeline
from .missing_analysis import MissingDataFormulaAnalyzer
from .gradient_descent import (
    LinearRegressionGD,
    LogisticRegressionGD,
    OptimizerType,
    LRScheduler
)
from .profiling import ComprehensiveDataProfiler
from .evaluation import (
    evaluate_classification,
    evaluate_regression,
    ConvergenceDiagnosisSystem,
    run_ablation_study
)
from .questions_solutions import QUESTIONS_AND_SOLUTIONS

__all__ = [
    "load_and_preprocess_data",
    "build_feature_pipeline",
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
    "QUESTIONS_AND_SOLUTIONS"
]
