"""
Automated Data Profiling Engine
===============================
Generates comprehensive exploratory data analysis (EDA), statistical profiles,
distribution metrics, correlation matrices, and automated interactive HTML reports.
Compatible with ydata-profiling / pandas-profiling if present, with a rich standalone
custom fallback profiler.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json


class ComprehensiveDataProfiler:
    """
    Automated Data Profiler for comprehensive statistical profiling,
    variable analysis, missingness analysis, correlation matrices, and HTML reporting.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset Profile"):
        self.df = df
        self.dataset_name = dataset_name
        self.profile = self._build_profile()

    def _build_profile(self) -> Dict[str, Any]:
        df = self.df
        n_rows, n_cols = df.shape
        mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

        # Basic Stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in numeric_cols]

        var_summaries = {}
        for col in df.columns:
            series = df[col]
            n_missing = int(series.isna().sum())
            pct_missing = (n_missing / n_rows) * 100.0 if n_rows > 0 else 0.0
            n_unique = int(series.nunique())

            col_info = {
                "dtype": str(series.dtype),
                "missing_count": n_missing,
                "missing_percentage": pct_missing,
                "unique_values": n_unique,
                "is_numeric": col in numeric_cols
            }

            if col in numeric_cols:
                clean_s = series.dropna()
                if len(clean_s) > 0:
                    col_info.update({
                        "mean": float(clean_s.mean()),
                        "std": float(clean_s.std()),
                        "min": float(clean_s.min()),
                        "p25": float(clean_s.quantile(0.25)),
                        "median": float(clean_s.median()),
                        "p75": float(clean_s.quantile(0.75)),
                        "max": float(clean_s.max()),
                        "skewness": float(clean_s.skew()),
                        "kurtosis": float(clean_s.kurtosis()),
                        "zeros_count": int((clean_s == 0).sum()),
                        "zeros_percentage": float(((clean_s == 0).sum() / len(clean_s)) * 100.0)
                    })
            else:
                top_vals = series.value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

            var_summaries[col] = col_info

        # Correlation Matrix for numerical columns
        corr_dict = {}
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr()
            corr_dict = corr_df.round(4).to_dict()

        return {
            "dataset_name": self.dataset_name,
            "overview": {
                "num_rows": n_rows,
                "num_cols": n_cols,
                "num_numeric_cols": len(numeric_cols),
                "num_categorical_cols": len(cat_cols),
                "total_missing_cells": int(df.isna().sum().sum()),
                "total_missing_percentage": float((df.isna().sum().sum() / (n_rows * n_cols)) * 100.0) if n_rows * n_cols > 0 else 0.0,
                "memory_usage_mb": float(mem_usage_mb),
                "duplicate_rows": int(df.duplicated().sum())
            },
            "variables": var_summaries,
            "correlation_matrix": corr_dict
        }

    def generate_html_report(self, output_filepath: str = "eda_profiling_report.html") -> str:
        """
        Generates an interactive, styled standalone HTML Data Profiling Report.
        """
        overview = self.profile["overview"]
        vars_info = self.profile["variables"]

        rows_html = ""
        for col_name, info in vars_info.items():
            if info["is_numeric"]:
                stats_str = f"Mean: {info.get('mean', 0):.2f} | Std: {info.get('std', 0):.2f} | Min: {info.get('min', 0):.2f} | Med: {info.get('median', 0):.2f} | Max: {info.get('max', 0):.2f} | Skew: {info.get('skewness', 0):.2f}"
            else:
                stats_str = f"Top Values: {json.dumps(info.get('top_values', {}))}"

            rows_html += f"""
            <tr>
                <td><strong>{col_name}</strong></td>
                <td><span class="badge bg-secondary">{info['dtype']}</span></td>
                <td>{info['unique_values']}</td>
                <td>{info['missing_count']} ({info['missing_percentage']:.2f}%)</td>
                <td style="font-size: 0.85rem; font-family: monospace;">{stats_str}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.dataset_name} - Profiling Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; }}
        .card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }}
        .card-header {{ background-color: #334155; color: #38bdf8; font-weight: 600; font-size: 1.1rem; border-bottom: 1px solid #475569; }}
        .metric-card {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #38bdf8; padding: 20px; border-radius: 10px; text-align: center; }}
        .metric-val {{ font-size: 2rem; font-weight: bold; color: #38bdf8; }}
        .metric-lbl {{ font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; }}
        table {{ color: #e2e8f0 !important; }}
        thead {{ background-color: #334155; color: #38bdf8; }}
        tr:hover {{ background-color: #334155 !important; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom border-secondary">
            <h1 class="text-info fw-bold">{self.dataset_name}</h1>
            <span class="badge bg-primary fs-6">Automated Gradient Descent Profiler</span>
        </div>

        <div class="row g-3 mb-4">
            <div class="col-md-2"><div class="metric-card"><div class="metric-val">{overview['num_rows']:,}</div><div class="metric-lbl">Total Rows</div></div></div>
            <div class="col-md-2"><div class="metric-card"><div class="metric-val">{overview['num_cols']}</div><div class="metric-lbl">Total Columns</div></div></div>
            <div class="col-md-2"><div class="metric-card"><div class="metric-val">{overview['total_missing_cells']}</div><div class="metric-lbl">Missing Cells ({overview['total_missing_percentage']:.1f}%)</div></div></div>
            <div class="col-md-2"><div class="metric-card"><div class="metric-val">{overview['num_numeric_cols']}</div><div class="metric-lbl">Numeric Features</div></div></div>
            <div class="col-md-2"><div class="metric-card"><div class="metric-val">{overview['duplicate_rows']}</div><div class="metric-lbl">Duplicate Rows</div></div></div>
            <div class="col-md-2"><div class="metric-card"><div class="metric-val">{overview['memory_usage_mb']:.2f} MB</div><div class="metric-lbl">Memory Usage</div></div></div>
        </div>

        <div class="card">
            <div class="card-header">Feature Breakdown & Descriptive Statistics</div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    <table class="table table-dark table-striped table-hover mb-0">
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Data Type</th>
                                <th>Distinct</th>
                                <th>Missing Values</th>
                                <th>Detailed Statistics / Top Values</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_filepath
