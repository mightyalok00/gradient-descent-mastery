"""
Gradient Descent Mastery — Streamlit Application
=================================================
Production-oriented interactive dashboard for:
- 16 theory modules with LaTeX derivations
- Formula-based missingness analysis
- Automated EDA/profile
- Interactive 2D gradient-descent laboratory
- From-scratch logistic-regression optimization
- Convergence diagnosis and optimizer benchmarking
- Test prediction + submission generation

Dataset resolution is delegated to src.data_loader.py, which supports:
1) environment variables,
2) repository data/ files,
3) repository-root CSVs,
4) the original local Windows paths.
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import build_feature_pipeline, load_and_preprocess_data
from src.evaluation import ConvergenceDiagnosisSystem, evaluate_classification
from src.gradient_descent import LogisticRegressionGD, LRScheduler, OptimizerType
from src.missing_analysis import MissingDataFormulaAnalyzer
from src.profiling import ComprehensiveDataProfiler
from src.questions_solutions import QUESTIONS_AND_SOLUTIONS


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Gradient Descent Mastery",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
    --bg: #07111d;
    --panel: #0c1b2b;
    --panel2: #10263b;
    --border: #1f405b;
    --cyan: #55d6ff;
    --violet: #9d8cff;
    --text: #e8f2fb;
    --muted: #91a7bb;
    --good: #35d39b;
    --warn: #f7c948;
}
.stApp {
    background:
        radial-gradient(circle at 10% -5%, rgba(85,214,255,.14), transparent 26%),
        radial-gradient(circle at 95% 0%, rgba(157,140,255,.13), transparent 28%),
        linear-gradient(180deg, #07111d 0%, #081421 55%, #07111d 100%);
}
.block-container { max-width: 1540px; padding: 1.2rem 2rem 3rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050c15, #091624);
    border-right: 1px solid #17344c;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(16,38,59,.95), rgba(7,20,34,.95));
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 14px;
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { font-weight: 850; }
.stButton > button {
    border-radius: 11px;
    font-weight: 750;
    border: 1px solid #244862;
}
.stButton > button:hover {
    border-color: var(--cyan);
    box-shadow: 0 0 18px rgba(85,214,255,.12);
}
div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: rgba(8,20,33,.58);
}
.hero {
    border: 1px solid #234963;
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, rgba(18,45,69,.97), rgba(8,20,34,.97));
    box-shadow: 0 24px 80px rgba(0,0,0,.25);
}
.hero h1 { margin: 0; font-size: 2.7rem; letter-spacing: -.05em; }
.hero p { color: #9eb2c5; margin: 8px 0 0; }
.badge {
    display: inline-block;
    margin: 14px 6px 0 0;
    padding: 6px 11px;
    border-radius: 999px;
    color: var(--cyan);
    background: rgba(85,214,255,.08);
    border: 1px solid rgba(85,214,255,.22);
    font-size: .72rem;
    font-weight: 850;
    letter-spacing: .06em;
}
.section-label {
    margin: 24px 0 9px;
    color: var(--cyan);
    font-size: .72rem;
    font-weight: 900;
    letter-spacing: .13em;
    text-transform: uppercase;
}
.card {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    background: linear-gradient(145deg, rgba(13,31,49,.92), rgba(7,18,31,.92));
}
.muted { color: var(--muted); font-size: .85rem; }
.theory-header {
    border: 1px solid #234963;
    border-radius: 19px;
    padding: 23px 26px;
    margin: 18px 0;
    background: linear-gradient(135deg, rgba(18,45,69,.96), rgba(9,21,36,.96));
}
.theory-index {
    color: var(--cyan);
    font-size: .72rem;
    font-weight: 900;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-bottom: 7px;
}
.progress {
    height: 8px;
    border-radius: 99px;
    background: #142a3f;
    overflow: hidden;
    margin: 9px 0 5px;
}
.progress > div {
    height: 100%;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_repository_dataset() -> dict[str, Any] | None:
    """Load the repository dataset; return None if deployment has no CSVs."""
    try:
        return load_and_preprocess_data(val_size=0.20, random_state=42)
    except (FileNotFoundError, OSError, ValueError):
        return None


def clear_dataset_cache() -> None:
    load_repository_dataset.clear()


def get_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, str]:
    """Return train, test, sample-submission and source label."""
    train_file = st.session_state.get("uploaded_train")
    test_file = st.session_state.get("uploaded_test")
    sample_file = st.session_state.get("uploaded_sample")

    if train_file is not None:
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file) if test_file is not None else None
        sample = pd.read_csv(sample_file) if sample_file is not None else None
        return train, test, sample, "Uploaded CSV"

    data = load_repository_dataset()
    if data is None:
        return None, None, None, "No dataset available"

    return (
        data["raw_train"],
        data["raw_test"],
        data["raw_sample_submission"],
        "Repository data/",
    )


def require_dataset() -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, str]:
    train, test, sample, source = get_data()
    if train is None:
        st.warning(
            "No dataset is available. Add the CSV files under data/ or upload "
            "train/test CSVs from the sidebar."
        )
        st.stop()
    return train, test, sample, source


def premium_chart(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#dbeafe"),
        title=dict(text=title, font=dict(size=18, color="#e8f1fb")),
        margin=dict(l=18, r=18, t=58, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.10)", zeroline=False)
    return fig


def hero(title: str, subtitle: str, badges: list[str]) -> None:
    badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(
        f"""
        <div class="hero">
            <div style="font-size:.72rem;color:#55d6ff;font-weight:850;letter-spacing:.16em;">
                OPTIMIZATION INTELLIGENCE PLATFORM
            </div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_numeric_sample(df: pd.DataFrame, n: int = 12000) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=42)


def build_training_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    max_rows: int,
) -> dict[str, Any]:
    """Build the same feature pipeline used by the project, with an optional row cap."""
    working = train_df
    if len(working) > max_rows:
        if "label" in working.columns:
            positive = working[working["label"] == 1]
            negative = working[working["label"] == 0]
            pos_n = min(len(positive), max(1, int(max_rows * len(positive) / len(working))))
            neg_n = min(len(negative), max_rows - pos_n)
            working = pd.concat(
                [
                    positive.sample(pos_n, random_state=42),
                    negative.sample(neg_n, random_state=42),
                ]
            ).sample(frac=1.0, random_state=42)
        else:
            working = working.sample(max_rows, random_state=42)

    pipeline = build_feature_pipeline(working, test_df)
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        pipeline["X_train"],
        pipeline["y_train"],
        test_size=0.20,
        random_state=42,
        stratify=pipeline["y_train"],
    )
    return {
        **pipeline,
        "X_train_split": X_train,
        "X_val": X_val,
        "y_train_split": y_train,
        "y_val": y_val,
        "rows_used": len(working),
    }


def train_gd_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimizer: OptimizerType,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2_lambda: float,
    clip_norm: float | None,
    scheduler: str,
) -> LogisticRegressionGD:
    model = LogisticRegressionGD(
        learning_rate=learning_rate,
        max_epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        l2_lambda=l2_lambda,
        gradient_clip_norm=clip_norm,
        lr_scheduler_mode=scheduler,
        early_stopping_patience=10,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return model


def store_experiment(record: dict[str, Any]) -> None:
    history = st.session_state.setdefault("experiment_history", [])
    history.append(record)
    st.session_state["experiment_history"] = history[-12:]


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.markdown(
    '<div style="font-size:1.25rem;font-weight:850;">⚡ GRADIENT DESCENT</div>'
    '<div style="color:#55d6ff;font-weight:750;letter-spacing:.08em;">MASTERY LAB</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Optimization • Data Quality • Diagnostics")

with st.sidebar.expander("Dataset source", expanded=True):
    st.caption("Repository data/ is used automatically. Uploads override it for this session.")
    train_upload = st.file_uploader("Train CSV", type="csv", key="train_csv")
    test_upload = st.file_uploader("Test CSV", type="csv", key="test_csv")
    sample_upload = st.file_uploader("Sample submission CSV", type="csv", key="sample_csv")

    if st.button("Use uploaded files", use_container_width=True):
        if train_upload is None:
            st.error("Train CSV is required.")
        else:
            st.session_state["uploaded_train"] = train_upload
            st.session_state["uploaded_test"] = test_upload
            st.session_state["uploaded_sample"] = sample_upload
            clear_dataset_cache()
            st.rerun()

    if st.button("Clear uploads", use_container_width=True):
        for key in ("uploaded_train", "uploaded_test", "uploaded_sample"):
            st.session_state.pop(key, None)
        clear_dataset_cache()
        st.rerun()

app_mode = st.sidebar.radio(
    "WORKSPACE",
    [
        "🏠 Executive Overview",
        "📘 16 Tough Questions & Theory",
        "🔍 Formula-Based Missingness Analyzer",
        "📊 Automated Data Profiler (EDA)",
        "🧪 Interactive Gradient Descent Lab",
        "⚡ Model Training & Diagnostics",
        "🏆 Optimizer Benchmark",
        "🚀 Test Predictions & Submissions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Gradient Descent Mastery • reproducible NumPy optimization")


# ---------------------------------------------------------------------
# Executive Overview
# ---------------------------------------------------------------------

if app_mode == "🏠 Executive Overview":
    train, test, sample, source = get_data()

    hero(
        "Gradient Descent <span style='color:#55d6ff;'>Mastery</span>",
        "Experiment, visualize and diagnose gradient-based learning from raw data to validated submission.",
        ["9 OPTIMIZERS", "16 THEORY MODULES", "LIVE TELEMETRY", "FORMULA ENGINE"],
    )

    if train is None:
        st.warning("Dataset not found. Your repository should contain data/train.csv, data/test.csv and data/sample_submission.csv.")
        st.info("The theory workspace remains available without a dataset.")
    else:
        profile = ComprehensiveDataProfiler(train, "Gradient Descent Dataset").profile["overview"]
        positive_rate = float(train["label"].mean() * 100) if "label" in train else float("nan")
        health = max(0.0, 100.0 - min(100.0, profile["total_missing_percentage"] * 4))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("OBSERVATIONS", f"{profile['num_rows']:,}")
        c2.metric("FEATURES", profile["num_cols"])
        c3.metric("MISSING", f"{profile['total_missing_percentage']:.2f}%")
        c4.metric("DUPLICATES", f"{profile['duplicate_rows']:,}")
        c5.metric("POSITIVE CLASS", f"{positive_rate:.2f}%")
        c6.metric("DATA HEALTH", f"{health:.1f}/100")

        st.markdown('<div class="section-label">DATASET SNAPSHOT</div>', unsafe_allow_html=True)
        left, right = st.columns([1.35, 1])

        with left:
            st.dataframe(train.head(12), use_container_width=True, hide_index=True)

        with right:
            if "label" in train:
                counts = train["label"].value_counts().rename_axis("label").reset_index(name="count")
                fig = px.bar(counts, x="label", y="count", text_auto=True, title="Target distribution")
                st.plotly_chart(premium_chart(fig), use_container_width=True)

        st.markdown('<div class="section-label">NUMERIC FEATURE EXPLORER</div>', unsafe_allow_html=True)
        numeric = train.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            feature = st.selectbox("Feature", numeric)
            sample_df = safe_numeric_sample(train)
            fig = px.histogram(
                sample_df,
                x=feature,
                color="label" if "label" in sample_df else None,
                marginal="box",
                title=f"{feature} distribution",
            )
            st.plotly_chart(premium_chart(fig), use_container_width=True)

        st.caption(f"Data source: {source}")


# ---------------------------------------------------------------------
# Theory workspace
# ---------------------------------------------------------------------

elif app_mode == "📘 16 Tough Questions & Theory":
    sections = {
        "Part A · Foundations": [1, 2, 3, 4],
        "Part B · Optimization Mechanics": [5, 6, 7, 8, 9, 10],
        "Part C · Convergence & Generalization": [11, 12],
        "Part D · Production & Advanced Systems": [13, 14, 15, 16],
    }
    icons = dict(zip(range(1, 17), ["📐","🔬","⚖️","📦","🚨","📏","🚀","🧭","⚙️","🧮","〰️","📊","🏭","🧠","🩺","🏗️"]))

    missing = [q for q in range(1, 17) if q not in QUESTIONS_AND_SOLUTIONS]
    hero(
        "Theory <span style='color:#55d6ff;'>Workspace</span>",
        "Sixteen rigorous modules covering gradient descent mathematics, optimizer mechanics, convergence and production ML.",
        ["16 QUESTIONS", "LATEX DERIVATIONS", "OPTIMIZATION", "DIAGNOSTICS"],
    )

    if missing:
        st.error("Missing theory modules: " + ", ".join(f"Q{x:02d}" for x in missing))
        st.stop()

    all_q = list(range(1, 17))

    # Navigation buttons cannot mutate the state of an already-instantiated
    # Streamlit widget. Store navigation requests first, then apply them on
    # the next script run before creating the widgets.
    pending_section = st.session_state.pop("pending_theory_section", None)
    pending_question = st.session_state.pop("pending_theory_question", None)
    if pending_section is not None:
        st.session_state["theory_section"] = pending_section
    if pending_question is not None:
        st.session_state["theory_question"] = pending_question

    section = st.selectbox("Theory section", list(sections), key="theory_section")
    valid_q = sections[section]

    if st.session_state.get("theory_question") not in valid_q:
        st.session_state["theory_question"] = valid_q[0]

    q_number = st.selectbox(
        "Question",
        valid_q,
        format_func=lambda q: f"Q{q:02d} · {QUESTIONS_AND_SOLUTIONS[q]['title']}",
        key="theory_question",
    )

    idx = all_q.index(q_number)
    q = QUESTIONS_AND_SOLUTIONS[q_number]

    st.markdown(
        f"""
        <div class="theory-header">
            <div class="theory-index">{icons[q_number]} Q{q_number:02d} / 16</div>
            <h2 style="margin:0;">{q['title']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress((q_number) / 16, text=f"Module {q_number} of 16")

    prev_q = all_q[idx - 1] if idx > 0 else None
    next_q = all_q[idx + 1] if idx < 15 else None
    a, b, c = st.columns([1, 2, 1])
    with a:
        if st.button("← Previous", disabled=prev_q is None, use_container_width=True):
            prev_section = next(name for name, nums in sections.items() if prev_q in nums)
            st.session_state["pending_theory_section"] = prev_section
            st.session_state["pending_theory_question"] = prev_q
            st.rerun()
    with b:
        st.markdown(
            f"<div style='text-align:center;color:#91a7bb;padding:8px;font-weight:800;'>Q{q_number:02d} · {q['title']}</div>",
            unsafe_allow_html=True,
        )
    with c:
        if st.button("Next →", disabled=next_q is None, use_container_width=True):
            next_section = next(name for name, nums in sections.items() if next_q in nums)
            st.session_state["pending_theory_section"] = next_section
            st.session_state["pending_theory_question"] = next_q
            st.rerun()

    st.markdown('<div class="section-label">Problem statement</div>', unsafe_allow_html=True)
    st.info(q["question"])

    st.markdown('<div class="section-label">Mathematical solution</div>', unsafe_allow_html=True)
    st.markdown(q["latex_derivation"])

    st.markdown('<div class="section-label">Interactive convergence intuition</div>', unsafe_allow_html=True)
    if q_number in (2, 11):
        lr = st.slider("Learning rate", 0.001, 0.05, 0.018, 0.001, key=f"lr_{q_number}")
        kappa = st.slider("Condition number", 1, 200, 100, 1, key=f"kappa_{q_number}")
        theta = np.array([2.5, 2.0], dtype=float)
        path = [theta.copy()]
        for _ in range(80):
            grad = np.array([theta[0], kappa * theta[1]])
            theta = theta - lr * grad
            path.append(theta.copy())
        path = np.asarray(path)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers", name="GD path"))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=12), name="Minimum"))
        fig.update_layout(
            title="2D quadratic optimization trajectory",
            xaxis_title="x",
            yaxis_title="y",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Use the derivation above as the primary mathematical reference for this module.")


# ---------------------------------------------------------------------
# Missingness analyzer
# ---------------------------------------------------------------------

elif app_mode == "🔍 Formula-Based Missingness Analyzer":
    train, _, _, source = require_dataset()
    hero(
        "Formula-Based <span style='color:#55d6ff;'>Missingness</span>",
        "Compute row, column and global missingness directly from the binary indicator matrix.",
        ["LINEAR ALGEBRA", "ROW FORMULAS", "COLUMN FORMULAS", "THRESHOLDS"],
    )

    analyzer = MissingDataFormulaAnalyzer(train)
    report = analyzer.get_summary_report()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MISSING CELLS", f"{report['global_missing_cells']:,}")
    c2.metric("GLOBAL RATE", f"{report['global_missing_percentage']:.4f}%")
    c3.metric("INCOMPLETE ROWS", f"{report['incomplete_rows_count']:,}")
    c4.metric("INCOMPLETE COLS", report["incomplete_cols_count"])

    st.markdown('<div class="section-label">Core formulas</div>', unsafe_allow_html=True)
    formulas = analyzer.get_latex_formulations()
    f1, f2, f3 = st.columns(3)
    with f1:
        st.latex(formulas["row_missing_count"])
        st.latex(formulas["col_missing_count"])
    with f2:
        st.latex(formulas["row_missing_rate"])
        st.latex(formulas["col_missing_rate"])
    with f3:
        st.latex(formulas["global_missing_rate"])

    row_tau, col_tau = st.columns(2)
    with row_tau:
        row_threshold = st.slider("Row missingness threshold", 0.0, 1.0, 0.5, 0.05)
    with col_tau:
        col_threshold = st.slider("Column missingness threshold", 0.0, 1.0, 0.5, 0.05)

    filtered = analyzer.filter_by_threshold(row_threshold, col_threshold)

    c1, c2 = st.columns(2)
    c1.metric("Rows above threshold", filtered["rows_above_threshold_count"])
    c2.metric("Columns above threshold", filtered["cols_above_threshold_count"])

    st.dataframe(report["column_breakdown"], use_container_width=True, hide_index=True)

    st.caption(f"Data source: {source}")


# ---------------------------------------------------------------------
# Automated EDA
# ---------------------------------------------------------------------

elif app_mode == "📊 Automated Data Profiler (EDA)":
    train, _, _, source = require_dataset()
    hero(
        "Automated Data <span style='color:#55d6ff;'>Profiler</span>",
        "Schema, distributions, descriptive statistics, missingness and correlations.",
        ["EDA", "STATISTICS", "CORRELATION", "DATA QUALITY"],
    )

    profiler = ComprehensiveDataProfiler(train, "Gradient Descent Dataset")
    profile = profiler.profile
    overview = profile["overview"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROWS", f"{overview['num_rows']:,}")
    c2.metric("COLUMNS", overview["num_cols"])
    c3.metric("NUMERIC", overview["num_numeric_cols"])
    c4.metric("CATEGORICAL", overview["num_categorical_cols"])
    c5.metric("MEMORY", f"{overview['memory_usage_mb']:.2f} MB")

    variable_rows = []
    for name, info in profile["variables"].items():
        variable_rows.append(
            {
                "Feature": name,
                "Dtype": info["dtype"],
                "Unique": info["unique_values"],
                "Missing": info["missing_count"],
                "Missing %": round(info["missing_percentage"], 4),
                "Mean": info.get("mean"),
                "Std": info.get("std"),
                "Min": info.get("min"),
                "Median": info.get("median"),
                "Max": info.get("max"),
            }
        )
    st.dataframe(pd.DataFrame(variable_rows), use_container_width=True, hide_index=True)

    numeric = train.select_dtypes(include=np.number).columns.tolist()
    if len(numeric) >= 2:
        corr = train[numeric].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Numeric correlation matrix")
        st.plotly_chart(premium_chart(fig), use_container_width=True)

    feature = st.selectbox("Distribution feature", numeric if numeric else train.columns.tolist())
    plot_df = safe_numeric_sample(train)
    if pd.api.types.is_numeric_dtype(plot_df[feature]):
        fig = px.histogram(plot_df, x=feature, marginal="box", title=f"{feature} distribution")
    else:
        counts = plot_df[feature].value_counts().head(20).reset_index()
        counts.columns = [feature, "count"]
        fig = px.bar(counts, x=feature, y="count", title=f"Top {feature} values")
    st.plotly_chart(premium_chart(fig), use_container_width=True)

    st.caption(f"Data source: {source}")


# ---------------------------------------------------------------------
# 2D Gradient Descent Lab
# ---------------------------------------------------------------------

elif app_mode == "🧪 Interactive Gradient Descent Lab":
    hero(
        "Interactive <span style='color:#55d6ff;'>Gradient Descent Lab</span>",
        "Explore learning-rate stability, curvature and optimizer trajectories on a 2D quadratic.",
        ["2D OBJECTIVE", "TRAJECTORY", "LR CONTROL", "CURVATURE"],
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        learning_rate = st.slider("Learning rate", 0.001, 0.10, 0.018, 0.001)
    with col2:
        curvature = st.slider("Curvature ratio", 1, 200, 100)
    with col3:
        steps = st.slider("Iterations", 20, 300, 100, 10)

    optimizer = st.selectbox(
        "Optimizer",
        [x.value for x in OptimizerType],
        index=0,
    )
    theta = np.array([2.5, 2.0], dtype=float)
    velocity = np.zeros(2)
    s = np.zeros(2)
    m = np.zeros(2)
    v = np.zeros(2)
    beta1, beta2 = 0.9, 0.999
    path = [theta.copy()]

    for t in range(1, steps + 1):
        grad = np.array([theta[0], curvature * theta[1]])

        if optimizer == OptimizerType.MOMENTUM.value:
            velocity = 0.9 * velocity + learning_rate * grad
            theta -= velocity
        elif optimizer == OptimizerType.NAG.value:
            look = theta - 0.9 * velocity
            grad = np.array([look[0], curvature * look[1]])
            velocity = 0.9 * velocity + learning_rate * grad
            theta -= velocity
        elif optimizer == OptimizerType.ADAGRAD.value:
            s += grad ** 2
            theta -= learning_rate * grad / (np.sqrt(s) + 1e-8)
        elif optimizer == OptimizerType.RMSPROP.value:
            s = 0.99 * s + 0.01 * grad ** 2
            theta -= learning_rate * grad / (np.sqrt(s) + 1e-8)
        elif optimizer in (OptimizerType.ADAM.value, OptimizerType.ADAMW.value):
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            mh = m / (1 - beta1 ** t)
            vh = v / (1 - beta2 ** t)
            theta -= learning_rate * mh / (np.sqrt(vh) + 1e-8)
            if optimizer == OptimizerType.ADAMW.value:
                theta *= 1 - learning_rate * 0.01
        else:
            theta -= learning_rate * grad

        path.append(theta.copy())

    path = np.asarray(path)
    objective = 0.5 * (path[:, 0] ** 2 + curvature * path[:, 1] ** 2)

    left, right = st.columns([1.35, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers", name="trajectory"))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=12), name="minimum"))
        fig.update_layout(
            title="Optimization trajectory",
            xaxis_title="x",
            yaxis_title="y",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        loss_fig = px.line(
            x=np.arange(len(objective)),
            y=objective,
            labels={"x": "Iteration", "y": "Objective"},
            title="Objective decay",
        )
        st.plotly_chart(premium_chart(loss_fig), use_container_width=True)

    st.metric("Final objective", f"{objective[-1]:.6g}")


# ---------------------------------------------------------------------
# Model training and diagnostics
# ---------------------------------------------------------------------

elif app_mode == "⚡ Model Training & Diagnostics":
    train, test, _, source = require_dataset()
    hero(
        "Model Training <span style='color:#55d6ff;'>& Diagnostics</span>",
        "Train the project's from-scratch logistic regression with configurable gradient optimizers.",
        ["FROM SCRATCH", "TELEMETRY", "CONVERGENCE", "IMBALANCE-AWARE METRICS"],
    )

    with st.expander("Training configuration", expanded=True):
        a, b, c, d = st.columns(4)
        optimizer_label = a.selectbox("Optimizer", [x.value for x in OptimizerType], index=6)
        lr = b.number_input("Learning rate", min_value=1e-5, max_value=1.0, value=0.01, format="%.5f")
        epochs = c.slider("Epochs", 5, 100, 30)
        batch_size = d.selectbox("Batch size", [32, 64, 128, 256, 512, 2048], index=1)

        e, f, g = st.columns(3)
        max_rows = e.select_slider("Maximum training rows", options=[10000, 20000, 50000, 100000, 182125], value=50000)
        l2 = f.number_input("L2 / AdamW weight decay", min_value=0.0, max_value=1.0, value=0.0, format="%.5f")
        scheduler = g.selectbox("LR scheduler", ["constant", "step", "exponential", "cosine", "inverse_time"])

        clip = st.checkbox("Gradient clipping", value=True)
        clip_norm = st.number_input("Clip norm", min_value=0.1, max_value=1000.0, value=5.0) if clip else None

    if st.button("🚀 Train model", type="primary", use_container_width=True):
        arrays = build_training_arrays(train, test, max_rows)
        selected_opt = OptimizerType(optimizer_label)

        start = time.perf_counter()
        with st.spinner("Training from scratch..."):
            model = train_gd_model(
                arrays["X_train_split"],
                arrays["y_train_split"],
                arrays["X_val"],
                arrays["y_val"],
                selected_opt,
                lr,
                epochs,
                batch_size,
                l2,
                clip_norm,
                scheduler,
            )
        elapsed = time.perf_counter() - start

        val_proba = model.predict_proba(arrays["X_val"])
        metrics = evaluate_classification(arrays["y_val"], val_proba)
        diagnosis = ConvergenceDiagnosisSystem.diagnose(model.history)

        st.session_state["last_model"] = model
        st.session_state["last_arrays"] = arrays
        st.session_state["last_metrics"] = metrics
        st.session_state["last_source"] = source

        store_experiment(
            {
                "run": len(st.session_state.get("experiment_history", [])) + 1,
                "optimizer": optimizer_label,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1_score"],
            }
        )

        st.success(f"Training completed in {elapsed:.2f}s using {arrays['rows_used']:,} rows.")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        m2.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
        m3.metric("F1", f"{metrics['f1_score']:.4f}")
        m4.metric("Brier", f"{metrics['brier_score']:.4f}")
        m5.metric("Final val loss", f"{model.history['val_loss'][-1]:.5f}")

        hist = pd.DataFrame(model.history)
        l1, l2 = st.columns(2)
        with l1:
            fig = px.line(hist, x="epoch", y=["train_loss", "val_loss"], markers=True, title="Loss telemetry")
            st.plotly_chart(premium_chart(fig), use_container_width=True)
        with l2:
            fig = px.line(hist, x="epoch", y=["gradient_norm", "parameter_norm"], markers=True, title="Optimization telemetry")
            st.plotly_chart(premium_chart(fig), use_container_width=True)

        st.markdown('<div class="section-label">Automated diagnosis</div>', unsafe_allow_html=True)
        st.write(f"**Status:** {diagnosis.get('status', 'UNKNOWN')}  ·  **Severity:** {diagnosis.get('severity', 'UNKNOWN')}")
        for item in diagnosis.get("diagnoses", []):
            st.info(item)

        cm = np.asarray(metrics["confusion_matrix"])
        cm_fig = px.imshow(cm, text_auto=True, title="Validation confusion matrix", labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(premium_chart(cm_fig), use_container_width=True)

    elif "last_model" in st.session_state:
        st.info("A previous training run is stored in this session. Train again to replace it.")


# ---------------------------------------------------------------------
# Optimizer benchmark
# ---------------------------------------------------------------------

elif app_mode == "🏆 Optimizer Benchmark":
    train, test, _, source = require_dataset()
    hero(
        "Optimizer <span style='color:#55d6ff;'>Benchmark</span>",
        "Compare multiple optimization strategies under the same train/validation pipeline.",
        ["CONTROLLED EXPERIMENT", "ROC-AUC", "PR-AUC", "CONVERGENCE"],
    )

    a, b, c = st.columns(3)
    max_rows = a.select_slider("Rows", options=[10000, 20000, 50000], value=20000)
    epochs = b.slider("Epochs per optimizer", 5, 50, 15)
    batch = c.selectbox("Batch size", [32, 64, 128, 256], index=1)

    selected = st.multiselect(
        "Optimizers",
        [x.value for x in OptimizerType],
        default=[
            OptimizerType.BATCH_GD.value,
            OptimizerType.MOMENTUM.value,
            OptimizerType.RMSPROP.value,
            OptimizerType.ADAM.value,
            OptimizerType.ADAMW.value,
        ],
    )

    if st.button("🏁 Run benchmark", type="primary", use_container_width=True):
        arrays = build_training_arrays(train, test, max_rows)
        records = []

        progress = st.progress(0, text="Starting benchmark...")
        for i, label in enumerate(selected, start=1):
            opt = OptimizerType(label)
            default_lr = 0.05 if opt in (OptimizerType.MOMENTUM, OptimizerType.BATCH_GD) else 0.01
            start = time.perf_counter()
            model = train_gd_model(
                arrays["X_train_split"],
                arrays["y_train_split"],
                arrays["X_val"],
                arrays["y_val"],
                opt,
                default_lr,
                epochs,
                batch,
                0.01 if opt == OptimizerType.ADAMW else 0.0,
                5.0,
                "constant",
            )
            elapsed = time.perf_counter() - start
            metrics = evaluate_classification(arrays["y_val"], model.predict_proba(arrays["X_val"]))
            records.append(
                {
                    "Optimizer": label,
                    "ROC-AUC": metrics["roc_auc"],
                    "PR-AUC": metrics["pr_auc"],
                    "F1": metrics["f1_score"],
                    "Brier": metrics["brier_score"],
                    "Final Val Loss": model.history["val_loss"][-1],
                    "Seconds": elapsed,
                }
            )
            progress.progress(i / len(selected), text=f"Completed {label}")

        result = pd.DataFrame(records)
        st.session_state["benchmark_result"] = result

    result = st.session_state.get("benchmark_result")
    if result is not None:
        st.dataframe(result.round(5), use_container_width=True, hide_index=True)
        fig = px.bar(result, x="Optimizer", y=["ROC-AUC", "PR-AUC"], barmode="group", title="Validation ranking by metric")
        st.plotly_chart(premium_chart(fig), use_container_width=True)
        st.caption(f"Data source: {source}")


# ---------------------------------------------------------------------
# Predictions and submissions
# ---------------------------------------------------------------------

elif app_mode == "🚀 Test Predictions & Submissions":
    train, test, sample, source = require_dataset()
    hero(
        "Test Predictions <span style='color:#55d6ff;'>& Submission</span>",
        "Train a final model on the available training data and export test-set probabilities.",
        ["TEST SET", "PROBABILITIES", "SUBMISSION CSV", "DOWNLOAD"],
    )

    if test is None:
        st.error("Test CSV is required for submission generation.")
        st.stop()

    if sample is None:
        st.warning("sample_submission.csv is not available. A submission template will be generated from test IDs.")

    a, b, c = st.columns(3)
    optimizer_label = a.selectbox(
        "Optimizer",
        [x.value for x in OptimizerType],
        index=6,
        key="submission_optimizer",
    )
    lr = b.number_input("Learning rate", 1e-5, 1.0, 0.01, format="%.5f", key="submission_lr")
    epochs = c.slider("Epochs", 5, 100, 30, key="submission_epochs")

    max_rows = st.select_slider(
        "Training rows",
        options=[10000, 20000, 50000, 100000, 182125],
        value=50000,
        key="submission_rows",
    )

    if st.button("Generate test predictions", type="primary", use_container_width=True):
        arrays = build_training_arrays(train, test, max_rows)
        model = train_gd_model(
            arrays["X_train_split"],
            arrays["y_train_split"],
            arrays["X_val"],
            arrays["y_val"],
            OptimizerType(optimizer_label),
            lr,
            epochs,
            64,
            0.01 if optimizer_label == OptimizerType.ADAMW.value else 0.0,
            5.0,
            "constant",
        )

        if arrays["X_test"] is None:
            st.error("The feature pipeline could not construct test features.")
            st.stop()

        proba = model.predict_proba(arrays["X_test"])
        if sample is not None and "label" in sample.columns:
            submission = sample.copy()
            submission["label"] = proba
        else:
            id_col = "transaction_id" if "transaction_id" in test.columns else test.columns[0]
            submission = pd.DataFrame({id_col: test[id_col], "label": proba})

        st.session_state["submission_df"] = submission

    submission = st.session_state.get("submission_df")
    if submission is not None:
        c1, c2 = st.columns(2)
        c1.metric("PREDICTIONS", f"{len(submission):,}")
        c2.metric("MEAN PROBABILITY", f"{submission['label'].mean():.5f}")

        st.dataframe(submission.head(20), use_container_width=True, hide_index=True)

        csv_bytes = submission.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(f"Data source: {source}")


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#91a7bb;font-size:.82rem;'>"
    "⚡ Gradient Descent Mastery · NumPy Optimization · Streamlit · "
    "Data Quality · Convergence Diagnostics"
    "</div>",
    unsafe_allow_html=True,
)
