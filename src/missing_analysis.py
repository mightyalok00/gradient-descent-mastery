r"""
Missing Data Formula Analyzer Module
====================================
This module provides mathematical and algorithmic methods for finding missing rows and columns
in tabular datasets using rigorous mathematical formulations and linear algebra representations.

Mathematical Definitions:
-------------------------
Let $X \in \mathbb{R}^{N \times d}$ be a data matrix with $N$ rows (observations) and $d$ columns (features).
Let $M \in \{0, 1\}^{N \times d}$ be the Missingness Indicator Matrix:

    $$M_{i,j} = \begin{cases} 1 & \text{if } X_{i,j} \text{ is missing (NaN / null / empty)} \\ 0 & \text{if } X_{i,j} \text{ is observed} \end{cases}$$

1. Row Missingness:
   - Row Missing Count: $c_{\text{row}}(i) = \sum_{j=1}^d M_{i,j} = (M \mathbf{1}_d)_i$
   - Row Missing Fraction / Rate: $\rho_i = \frac{1}{d} \sum_{j=1}^d M_{i,j} = \frac{(M \mathbf{1}_d)_i}{d}$
   - Set of Incomplete Rows: $\mathcal{I}_{\text{incomplete}} = \{ i \in \{1, \dots, N\} \mid \rho_i > 0 \}$
   - Set of Fully Missing Rows: $\mathcal{I}_{\text{empty}} = \{ i \in \{1, \dots, N\} \mid \rho_i = 1 \}$

2. Column Missingness:
   - Column Missing Count: $c_{\text{col}}(j) = \sum_{i=1}^N M_{i,j} = (\mathbf{1}_N^T M)_j$
   - Column Missing Fraction / Rate: $\gamma_j = \frac{1}{N} \sum_{i=1}^N M_{i,j} = \frac{(\mathbf{1}_N^T M)_j}{N}$
   - Set of Incomplete Columns: $\mathcal{J}_{\text{incomplete}} = \{ j \in \{1, \dots, d\} \mid \gamma_j > 0 \}$
   - Set of Fully Missing Columns: $\mathcal{J}_{\text{empty}} = \{ j \in \{1, \dots, d\} \mid \gamma_j = 1 \}$

3. Global Missingness:
   - Total Missing Elements: $S_{\text{total}} = \sum_{i=1}^N \sum_{j=1}^d M_{i,j} = \mathbf{1}_N^T M \mathbf{1}_d$
   - Global Missingness Rate: $\Omega = \frac{S_{\text{total}}}{N \cdot d} = \frac{\mathbf{1}_N^T M \mathbf{1}_d}{N \cdot d}$

4. Missing Data Mechanisms:
   - MCAR (Missing Completely at Random): $P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) = P(M \mid \psi)$
   - MAR (Missing at Random): $P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) = P(M \mid X_{\text{obs}}, \psi)$
   - MNAR (Missing Not at Random): $P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi)$ depends on $X_{\text{mis}}$.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MissingDataFormulaAnalyzer:
    """
    Mathematical and programmatic engine for analyzing missingness in datasets.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a Pandas DataFrame.
        
        Args:
            df: Input pandas DataFrame
        """
        self.df = df
        self.N, self.d = df.shape
        self.feature_names = list(df.columns)
        
        # Construct the binary Missingness Indicator Matrix M \in {0, 1}^{N x d}
        # M[i, j] = 1 if df.iloc[i, j] is null, 0 otherwise
        self.M = df.isna().astype(int).to_numpy()
        
        # 1-vectors for matrix multiplications
        self.ones_d = np.ones((self.d, 1))
        self.ones_N = np.ones((self.N, 1))
        
        # Compute exact mathematical metrics via linear algebra
        self._compute_formula_metrics()

    def _compute_formula_metrics(self) -> None:
        """
        Executes matrix formula calculations for missing counts and rates.
        Formula:
            c_row = M * 1_d  (shape: N x 1)
            c_col = 1_N^T * M (shape: 1 x d)
            S_total = 1_N^T * M * 1_d (scalar)
        """
        # Row-level calculations
        self.row_missing_counts = np.dot(self.M, self.ones_d).flatten()  # c_row(i)
        self.row_missing_rates = self.row_missing_counts / float(self.d)   # rho_i
        
        # Column-level calculations
        self.col_missing_counts = np.dot(self.ones_N.T, self.M).flatten() # c_col(j)
        self.col_missing_rates = self.col_missing_counts / float(self.N)   # gamma_j
        
        # Global calculations
        self.total_missing_cells = int(np.dot(np.dot(self.ones_N.T, self.M), self.ones_d)[0, 0])
        self.total_possible_cells = self.N * self.d
        self.global_missing_rate = float(self.total_missing_cells) / float(self.total_possible_cells)

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive dictionary summary of missingness.
        """
        incomplete_rows_idx = np.where(self.row_missing_counts > 0)[0]
        empty_rows_idx = np.where(self.row_missing_counts == self.d)[0]
        
        incomplete_cols_idx = np.where(self.col_missing_counts > 0)[0]
        empty_cols_idx = np.where(self.col_missing_counts == self.N)[0]
        
        col_summary = pd.DataFrame({
            "Column_Name": self.feature_names,
            "Missing_Count_Formula": self.col_missing_counts.astype(int),
            "Missing_Rate_Formula": self.col_missing_rates,
            "Missing_Percentage": self.col_missing_rates * 100.0,
            "Observed_Count": self.N - self.col_missing_counts.astype(int),
            "Data_Type": [str(self.df[col].dtype) for col in self.feature_names]
        }).sort_values(by="Missing_Count_Formula", ascending=False).reset_index(drop=True)
        
        return {
            "matrix_dimensions": {"N_rows": self.N, "d_cols": self.d},
            "global_missing_cells": self.total_missing_cells,
            "global_missing_rate": self.global_missing_rate,
            "global_missing_percentage": self.global_missing_rate * 100.0,
            "incomplete_rows_count": len(incomplete_rows_idx),
            "incomplete_rows_percentage": (len(incomplete_rows_idx) / self.N) * 100.0 if self.N > 0 else 0.0,
            "completely_empty_rows_count": len(empty_rows_idx),
            "incomplete_cols_count": len(incomplete_cols_idx),
            "incomplete_cols_percentage": (len(incomplete_cols_idx) / self.d) * 100.0 if self.d > 0 else 0.0,
            "completely_empty_cols_count": len(empty_cols_idx),
            "column_breakdown": col_summary,
            "incomplete_row_indices": incomplete_rows_idx.tolist(),
            "incomplete_col_names": [self.feature_names[j] for j in incomplete_cols_idx]
        }

    def filter_by_threshold(self, row_tau: float = 0.5, col_tau: float = 0.5) -> Dict[str, Any]:
        """
        Finds rows and columns exceeding a missingness threshold tau in [0, 1].
        
        Mathematical criteria:
            R_tau = { i | rho_i >= row_tau }
            C_tau = { j | gamma_j >= col_tau }
        """
        rows_above_tau = np.where(self.row_missing_rates >= row_tau)[0]
        cols_above_tau = np.where(self.col_missing_rates >= col_tau)[0]
        
        return {
            "row_threshold_tau": row_tau,
            "rows_above_threshold_count": len(rows_above_tau),
            "rows_above_threshold_indices": rows_above_tau.tolist(),
            "col_threshold_tau": col_tau,
            "cols_above_threshold_count": len(cols_above_tau),
            "cols_above_threshold_names": [self.feature_names[j] for j in cols_above_tau]
        }

    def get_latex_formulations(self) -> Dict[str, str]:
        """
        Returns LaTeX representation of all formulas for interactive UI and reports.
        """
        return {
            "indicator_matrix": r"M_{i,j} = \mathbb{I}(X_{i,j} \in \{\text{NaN}, \text{null}\}) = \begin{cases} 1 & \text{if missing} \\ 0 & \text{if observed} \end{cases}",
            "row_missing_count": r"c_{\text{row}}(i) = \sum_{j=1}^d M_{i,j} = (M \mathbf{1}_d)_i",
            "row_missing_rate": r"\rho_i = \frac{1}{d} \sum_{j=1}^d M_{i,j} = \frac{(M \mathbf{1}_d)_i}{d} \in [0, 1]",
            "col_missing_count": r"c_{\text{col}}(j) = \sum_{i=1}^N M_{i,j} = (\mathbf{1}_N^T M)_j",
            "col_missing_rate": r"\gamma_j = \frac{1}{N} \sum_{i=1}^N M_{i,j} = \frac{(\mathbf{1}_N^T M)_j}{N} \in [0, 1]",
            "global_missing_rate": r"\Omega = \frac{\mathbf{1}_N^T M \mathbf{1}_d}{N \times d} \times 100\%",
            "row_set_filter": r"\mathcal{I}_{\tau_{\text{row}}} = \{ i \in \{1, \dots, N\} \mid \rho_i \ge \tau_{\text{row}} \}",
            "col_set_filter": r"\mathcal{J}_{\tau_{\text{col}}} = \{ j \in \{1, \dots, d\} \mid \gamma_j \ge \tau_{\text{col}} \}"
        }

    @staticmethod
    def create_synthetic_missing_benchmark(
        df_clean: pd.DataFrame,
        mcar_rate: float = 0.05,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Injects controlled MCAR missingness into a clean DataFrame to benchmark and validate
        formula-based missing value detection.
        
        Args:
            df_clean: Clean baseline DataFrame
            mcar_rate: Fraction of entries to artificially set to NaN
            random_seed: Seed for reproducible Bernoulli mask generation
        
        Returns:
            Tuple of (corrupted DataFrame, ground truth injection metadata)
        """
        np.random.seed(random_seed)
        df_corrupted = df_clean.copy()
        N, d = df_corrupted.shape
        
        # Generate Bernoulli trial mask: B_{i,j} ~ Bernoulli(p = mcar_rate)
        bernoulli_mask = np.random.rand(N, d) < mcar_rate
        
        # Ensure we do not overwrite targets or key IDs if requested, or corrupt all features
        for col_idx, col_name in enumerate(df_corrupted.columns):
            if col_name not in ['label', 'transaction_id']:
                mask_col = bernoulli_mask[:, col_idx]
                df_corrupted.loc[mask_col, col_name] = np.nan
        
        actual_injected = int(df_corrupted.isna().sum().sum())
        
        ground_truth = {
            "injected_rate_target": mcar_rate,
            "actual_injected_cells": actual_injected,
            "actual_injected_fraction": actual_injected / float(N * d),
            "injected_mask_shape": bernoulli_mask.shape
        }
        
        return df_corrupted, ground_truth
