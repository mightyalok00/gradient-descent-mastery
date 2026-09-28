import numpy as np
import pandas as pd

from src.missing_analysis import MissingDataFormulaAnalyzer


def test_missingness_formulas():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, 2.0, np.nan],
        }
    )

    analyzer = MissingDataFormulaAnalyzer(df)

    assert analyzer.row_missing_counts.tolist() == [1.0, 1.0, 1.0]
    assert analyzer.col_missing_counts.tolist() == [1.0, 2.0]
    assert analyzer.total_missing_cells == 3
    assert analyzer.global_missing_rate == 0.5


def test_missingness_thresholds():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.nan, np.nan],
            "b": [1.0, 2.0, 3.0, np.nan],
        }
    )

    analyzer = MissingDataFormulaAnalyzer(df)
    result = analyzer.filter_by_threshold(row_tau=0.5, col_tau=0.5)

    # The implementation uses >= for threshold comparisons.
    assert result["rows_above_threshold_indices"] == [1, 2, 3]
    assert result["cols_above_threshold_names"] == ["a"]
