"""
Gradient Descent Mastery & Machine Learning Pipeline Dashboard
==============================================================
Interactive Streamlit Application featuring:
- Solutions & LaTeX Derivations for all 16 Tough Questions
- Formula-Based Missing Rows and Columns Analyzer
- Automated Data Profiler & Exploratory Data Analysis
- Interactive 2D Gradient Descent Optimization Lab
- Dataset Model Training, Telemetry & Automated Convergence Diagnostics
- Test Set Prediction & Submission Generator
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time

from src.missing_analysis import MissingDataFormulaAnalyzer
from src.gradient_descent import (
    LogisticRegressionGD,
    LinearRegressionGD,
    OptimizerType,
    LRScheduler
)
from src.data_loader import load_raw_datasets, build_feature_pipeline, load_and_preprocess_data
from src.profiling import ComprehensiveDataProfiler
from src.evaluation import evaluate_classification, ConvergenceDiagnosisSystem, run_ablation_study
from src.questions_solutions import QUESTIONS_AND_SOLUTIONS

# Page configuration
st.set_page_config(
    page_title="Gradient Descent Optimization & ML Framework",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 8% 0%,rgba(56,189,248,.10),transparent 26%),radial-gradient(circle at 92% 8%,rgba(139,92,246,.10),transparent 25%),#070d18;color:#e5edf7}
.block-container{max-width:1500px;padding-top:1.2rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07101d,#0a1625);border-right:1px solid #1c3550}
[data-testid="stMetric"]{background:linear-gradient(145deg,#102238,#0b1828);border:1px solid #1d3a56;border-radius:14px;padding:14px}
.hero-v2{border:1px solid #1d3a56;border-radius:22px;padding:28px 30px;margin-bottom:18px;background:linear-gradient(135deg,#122b43,#0b1727);box-shadow:0 18px 60px rgba(0,0,0,.25)}
.hero-v2 h1{margin:0;font-size:2.5rem;letter-spacing:-.04em}.hero-v2 p{color:#8ea3b8;margin:8px 0 0}
.badge-v2{display:inline-block;padding:5px 10px;margin:12px 5px 0 0;border-radius:999px;background:rgba(56,189,248,.10);color:#55d6ff;border:1px solid rgba(56,189,248,.22);font-size:.76rem;font-weight:750}
.card-v2{border:1px solid #1d3a56;border-radius:15px;padding:16px;background:rgba(13,27,43,.76);min-height:110px}
.card-v2 .num{font-size:1.45rem;font-weight:800;color:#55d6ff}.card-v2 .lbl{font-size:.78rem;color:#8ea3b8;margin-top:5px}
.stButton>button{border-radius:10px;font-weight:700}
.theorem-card{background:#0d1b2b;border-left:4px solid #55d6ff;padding:16px;border-radius:0 12px 12px 0}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset():
    return load_and_preprocess_data(val_size=0.2, random_state=42)

def get_uploaded_or_local_data():
    train_file = st.session_state.get("uploaded_train")
    test_file = st.session_state.get("uploaded_test")
    sample_file = st.session_state.get("uploaded_sample")

    if train_file is not None:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file) if test_file is not None else None
        sample_df = pd.read_csv(sample_file) if sample_file is not None else None
        return train_df, test_df, sample_df, "Uploaded CSV"

    data = get_dataset()
    return (
        data["raw_train"],
        data["raw_test"],
        data["raw_sample_submission"],
        "Repository dataset",
    )


# Sidebar Navigation
st.sidebar.markdown(
    '<div style="font-size:1.25rem;font-weight:850;">⚡ GRADIENT DESCENT</div>'
    '<div style="color:#55d6ff;font-weight:750;letter-spacing:.08em;">MASTERY LAB</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Optimization • Data Quality • Diagnostics")

with st.sidebar.expander("Dataset source", expanded=False):
    st.caption("Use local repository paths or upload CSVs for cloud deployment.")
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
            st.cache_data.clear()
            st.rerun()
    if st.button("Clear uploads", use_container_width=True):
        for key in ("uploaded_train", "uploaded_test", "uploaded_sample"):
            st.session_state.pop(key, None)
        st.cache_data.clear()
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
        "🚀 Test Predictions & Submissions"
    ]
)

# -------------------------------------------------------------
# 0. Executive Overview
# -------------------------------------------------------------
if app_mode == "🏠 Executive Overview":
    st.markdown(
        """
        <div class="hero-v2">
            <h1>Gradient Descent <span style="color:#55d6ff;">Mastery</span></h1>
            <p>Optimization intelligence for data quality, gradient dynamics,
            convergence and production-style model evaluation.</p>
            <span class="badge-v2">9 OPTIMIZERS</span>
            <span class="badge-v2">16 THEORY MODULES</span>
            <span class="badge-v2">FORMULA ENGINE</span>
            <span class="badge-v2">IMBALANCED ML</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_train, raw_test, raw_sample, data_source = get_uploaded_or_local_data()

    if raw_train is None:
        st.info("Load the repository dataset or upload train.csv from the sidebar.")
    else:
        profiler = ComprehensiveDataProfiler(raw_train, dataset_name="Gradient Descent Dataset")
        overview = profiler.profile["overview"]
        positive_rate = raw_train["label"].mean() * 100 if "label" in raw_train else float("nan")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Observations", f"{overview['num_rows']:,}")
        c2.metric("Features", overview["num_cols"])
        c3.metric("Missing cells", f"{overview['total_missing_cells']:,}")
        c4.metric("Duplicates", f"{overview['duplicate_rows']:,}")
        c5.metric("Positive class", f"{positive_rate:.2f}%")

        st.markdown("### Production workflow")
        stages = [
            ("01", "Validate", "Schema + nulls"),
            ("02", "Engineer", "Features + scaling"),
            ("03", "Optimize", "BGD → AdamW"),
            ("04", "Diagnose", "Loss + gradients"),
            ("05", "Evaluate", "ROC / PR / F1"),
            ("06", "Export", "Validated submission"),
        ]
        cols = st.columns(6)
        for col, (num, title, desc) in zip(cols, stages):
            col.markdown(
                f'<div class="card-v2"><div class="num">{num}</div>'
                f'<b>{title}</b><div class="lbl">{desc}</div></div>',
                unsafe_allow_html=True,
            )

        left, right = st.columns(2)
        with left:
            if "label" in raw_train:
                counts = raw_train["label"].value_counts().rename_axis("label").reset_index(name="count")
                fig = px.bar(
                    counts,
                    x="label",
                    y="count",
                    text_auto=True,
                    template="plotly_dark",
                    title="Target distribution",
                )
                st.plotly_chart(fig, use_container_width=True)
        with right:
            numeric = raw_train.select_dtypes(include=[np.number]).columns.tolist()
            if numeric:
                feature = st.selectbox("Feature distribution", numeric)
                fig = px.histogram(
                    raw_train.sample(min(12000, len(raw_train)), random_state=42),
                    x=feature,
                    color="label" if "label" in raw_train else None,
                    marginal="box",
                    template="plotly_dark",
                    title=f"Distribution — {feature}",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.caption("Data source: " + data_source)

# -------------------------------------------------------------
# 1. 16 Tough Questions & Theory Solver
# -------------------------------------------------------------
if app_mode == "📘 16 Tough Questions & Theory":
    st.title("📘 Solutions to 16 Advanced Optimization Questions")
    st.markdown("Rigorous mathematical derivations, proofs, system designs, and empirical diagnostic protocols.")

    part_filter = st.selectbox("Select Section", ["All Questions", "Part A: Questions 1 - 12 (Core Theory)", "Part B: Questions 13 - 15 (System Architecture)", "Part C: Question 16 (Deep Analysis)"])

    selected_q = st.selectbox(
        "Choose Question to Inspect",
        options=list(QUESTIONS_AND_SOLUTIONS.keys()),
        format_func=lambda q: f"Q{q}: {QUESTIONS_AND_SOLUTIONS[q]['title']}"
    )

    q_data = QUESTIONS_AND_SOLUTIONS[selected_q]

    st.markdown(f"### Question {selected_q}: {q_data['title']}")
    st.info(f"**Question Statement:**\n\n{q_data['question']}")

    st.markdown(q_data["latex_derivation"])

    # Visual demonstration for Q11 (Zig-zagging)
    if selected_q == 11 or selected_q == 2:
        st.subheader("Interactive 2D Ill-Conditioned Quadratic Visualization")
        col1, col2 = st.columns(2)
        with col1:
            eta = st.slider("Learning Rate (η)", 0.001, 0.025, 0.018, 0.001, key="q11_lr")
        with col2:
            n_iters = st.slider("Iterations", 5, 50, 20, 1, key="q11_iter")

        # Simulate trajectory
        traj = [[-8.0, 1.0]]
        pos = np.array([-8.0, 1.0])
        for _ in range(n_iters):
            grad = np.array([pos[0], 100.0 * pos[1]])
            pos = pos - eta * grad
            traj.append(pos.tolist())
        traj = np.array(traj)

        # Plot contour
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-2, 2, 200)
        X, Y = np.meshgrid(x, y)
        Z = 0.5 * (X**2 + 100 * Y**2)

        fig = go.Figure()
        fig.add_trace(go.Contour(x=x, y=y, z=Z, contours_coloring='lines', line_width=1.5, colorscale='Viridis', showscale=False))
        fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], mode='lines+markers', marker=dict(color='red', size=6), line=dict(color='red', width=2), name='GD Path'))
        fig.update_layout(title=f"Ill-Conditioned Quadratic Optimization Path (κ=100, η={eta})", template="plotly_dark", xaxis_title="w1 (slow direction λ=1)", yaxis_title="w2 (steep direction λ=100)")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 2. Formula-Based Missing Data Analyzer
# -------------------------------------------------------------
elif app_mode == "🔍 Formula-Based Missingness Analyzer":
    st.title("🔍 Missing Rows and Columns Analysis by Mathematical Formula")
    st.markdown(r"""
    This module uses linear algebra formulations ($M \in \{0,1\}^{N \times d}$, $\rho_i = (M \mathbf{1})_i / d$, $\gamma_j = (\mathbf{1}^T M)_j / N$) 
    to detect missing rows and columns with exact mathematical precision.
    """)

    raw_train, raw_test, raw_sample, data_source = get_uploaded_or_local_data()

    tab1, tab2, tab3 = st.tabs(["📐 Mathematical Formulations", "📊 Real Dataset Analysis", "🧪 Synthetic MCAR Injection Benchmark"])

    analyzer = MissingDataFormulaAnalyzer(raw_train)
    summary = analyzer.get_summary_report()

    with tab1:
        st.subheader("Mathematical Formulations")
        formulas = analyzer.get_latex_formulations()
        for k, form in formulas.items():
            st.markdown(f"**{k.replace('_', ' ').title()}:**")
            st.latex(form)

    with tab2:
        st.subheader("Real Dataset Missingness Inspection (`train.csv`)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows (N)", f"{summary['matrix_dimensions']['N_rows']:,}")
        c2.metric("Total Columns (d)", summary["matrix_dimensions"]["d_cols"])
        c3.metric("Missing Cells (Total)", f"{summary['global_missing_cells']}")
        c4.metric("Global Missingness (Ω)", f"{summary['global_missing_percentage']:.2f}%")

        st.markdown("#### Column-by-Column Missingness Breakdown")
        st.dataframe(summary["column_breakdown"], use_container_width=True)

        st.markdown("#### Threshold Row/Column Extractor")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            row_tau = st.slider("Row Missing Threshold (τ_row)", 0.0, 1.0, 0.5, 0.05)
        with col_t2:
            col_tau = st.slider("Column Missing Threshold (τ_col)", 0.0, 1.0, 0.5, 0.05)

        filtered = analyzer.filter_by_threshold(row_tau=row_tau, col_tau=col_tau)
        st.write(f"Rows with $\\rho_i \\ge {row_tau}$: **{filtered['rows_above_threshold_count']}**")
        st.write(f"Columns with $\\gamma_j \\ge {col_tau}$: **{filtered['cols_above_threshold_count']}** ({filtered['cols_above_threshold_names']})")

    with tab3:
        st.subheader("Synthetic Injection Testing (Validating Formula Engine)")
        st.markdown("Inject controlled Missing Completely at Random (MCAR) nulls into a sample to verify exact formulaic recovery.")

        inject_rate = st.slider("Injection Missing Rate (p)", 0.01, 0.20, 0.05, 0.01)
        sample_df = raw_train.head(1000).copy()

        df_injected, ground_truth = MissingDataFormulaAnalyzer.create_synthetic_missing_benchmark(
            sample_df, mcar_rate=inject_rate, random_seed=42
        )
        injected_analyzer = MissingDataFormulaAnalyzer(df_injected)
        injected_summary = injected_analyzer.get_summary_report()

        st.success(f"Injected Target Rate: **{inject_rate*100:.1f}%** | Formula Computed Missing Rate: **{injected_summary['global_missing_percentage']:.2f}%**")
        st.write(f"Incomplete Rows Detected by Formula: **{injected_summary['incomplete_rows_count']} / 1,000**")
        st.dataframe(injected_summary["column_breakdown"], use_container_width=True)

# -------------------------------------------------------------
# 3. Automated Data Profiler (EDA)
# -------------------------------------------------------------
elif app_mode == "📊 Automated Data Profiler (EDA)":
    st.title("📊 Automated Data Profiling & Exploratory Data Analysis")
    st.markdown("Statistical distributions, feature correlation matrices, skewness, kurtosis, and HTML report export.")

    df_train, _, _, data_source = get_uploaded_or_local_data()

    profiler = ComprehensiveDataProfiler(df_train, dataset_name="Transaction Fraud Dataset")
    profile = profiler.profile

    overview = profile["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{overview['num_rows']:,}")
    c2.metric("Features", overview["num_cols"])
    c3.metric("Numeric Features", overview["num_numeric_cols"])
    c4.metric("Memory Footprint", f"{overview['memory_usage_mb']:.1f} MB")

    st.markdown("### Interactive Correlation Heatmap")
    num_cols = df_train.select_dtypes(include=[np.number]).columns
    corr_df = df_train[num_cols].corr()

    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Feature Correlation Matrix (Pearson)"
    )
    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Feature Distribution Visualizer")
    selected_feature = st.selectbox("Select Feature to Plot", options=num_cols)
    fig_dist = px.histogram(
        df_train.sample(min(10000, len(df_train))),
        x=selected_feature,
        color="label" if "label" in df_train.columns else None,
        marginal="box",
        title=f"Distribution of {selected_feature} by Class Label",
        template="plotly_dark",
        opacity=0.75
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    if st.button("Generate & Download HTML Profiling Report"):
        html_path = profiler.generate_html_report("dataset_profiling_report.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html_bytes = f.read()
        st.download_button(
            label="📥 Download HTML Profiling Report",
            data=html_bytes,
            file_name="dataset_profiling_report.html",
            mime="text/html"
        )

# -------------------------------------------------------------
# 4. Interactive Gradient Descent Lab
# -------------------------------------------------------------
elif app_mode == "🧪 Interactive Gradient Descent Lab":
    st.title("🧪 Interactive 2D Gradient Descent Optimization Lab")
    st.markdown("Live visual optimization testbed comparing convergence paths of all Gradient Descent variants.")

    c1, c2, c3 = st.columns(3)
    with c1:
        loss_surface = st.selectbox("Loss Surface", ["Ill-Conditioned Quadratic (Canyon)", "Rosenbrock (Banana Function)", "Beale's Function"])
    with c2:
        optimizer_choice = st.selectbox("Optimizer", [
            "Batch Gradient Descent", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam"
        ])
    with c3:
        learning_rate = st.slider("Learning Rate (η)", 0.0001, 0.5, 0.01, 0.0005)

    c4, c5, c6 = st.columns(3)
    with c4:
        momentum_val = st.slider("Momentum / Beta1", 0.0, 0.99, 0.9, 0.01)
    with c5:
        max_steps = st.slider("Max Iterations", 10, 200, 50, 5)
    with c6:
        start_x = st.slider("Start X", -5.0, 5.0, -3.5, 0.5)

    # Define loss function
    def compute_loss_and_grad(x, y, surface):
        if surface == "Ill-Conditioned Quadratic (Canyon)":
            loss = 0.5 * (x**2 + 50 * y**2)
            gx = x
            gy = 50 * y
        elif surface == "Rosenbrock (Banana Function)":
            loss = (1 - x)**2 + 100 * (y - x**2)**2
            gx = -2 * (1 - x) - 400 * x * (y - x**2)
            gy = 200 * (y - x**2)
        else:
            loss = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
            gx = 2*(1.5 - x + x*y)*(-1 + y) + 2*(2.25 - x + x*y**2)*(-1 + y**2) + 2*(2.625 - x + x*y**3)*(-1 + y**3)
            gy = 2*(1.5 - x + x*y)*x + 2*(2.25 - x + x*y**2)*(2*x*y) + 2*(2.625 - x + x*y**3)*(3*x*y**2)
        return loss, np.array([gx, gy])

    # Run optimization
    pos = np.array([start_x, 2.5])
    traj = [pos.copy()]
    losses = []

    v = np.zeros(2)
    s = np.zeros(2)

    for step in range(1, max_steps + 1):
        loss, g = compute_loss_and_grad(pos[0], pos[1], loss_surface)
        losses.append(loss)

        if optimizer_choice == "Batch Gradient Descent":
            pos = pos - learning_rate * g
        elif optimizer_choice == "Momentum":
            v = momentum_val * v + learning_rate * g
            pos = pos - v
        elif optimizer_choice == "NAG":
            pos_ahead = pos - momentum_val * v
            _, g_ahead = compute_loss_and_grad(pos_ahead[0], pos_ahead[1], loss_surface)
            v = momentum_val * v + learning_rate * g_ahead
            pos = pos - v
        elif optimizer_choice == "AdaGrad":
            s += g**2
            pos = pos - (learning_rate / (np.sqrt(s) + 1e-8)) * g
        elif optimizer_choice == "RMSProp":
            s = 0.99 * s + 0.01 * (g**2)
            pos = pos - (learning_rate / (np.sqrt(s) + 1e-8)) * g
        elif optimizer_choice == "Adam":
            v = momentum_val * v + (1 - momentum_val) * g
            s = 0.999 * s + (1 - 0.999) * (g**2)
            v_hat = v / (1 - momentum_val**step)
            s_hat = s / (1 - 0.999**step)
            pos = pos - (learning_rate / (np.sqrt(s_hat) + 1e-8)) * v_hat

        traj.append(pos.copy())

    traj = np.array(traj)

    # 2D Contour Plot
    x_grid = np.linspace(-6, 6, 150)
    y_grid = np.linspace(-4, 4, 150)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            Z[j, i], _ = compute_loss_and_grad(X[j, i], Y[j, i], loss_surface)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Contour(x=x_grid, y=y_grid, z=Z, contours_coloring='lines', colorscale='Viridis', showscale=False))
        fig_opt.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], mode='lines+markers', marker=dict(color='orange', size=6), line=dict(color='orange', width=2), name=optimizer_choice))
        fig_opt.update_layout(title=f"Optimization Trajectory ({optimizer_choice})", template="plotly_dark", xaxis_title="w1", yaxis_title="w2")
        st.plotly_chart(fig_opt, use_container_width=True)

    with col_fig2:
        fig_loss = px.line(x=list(range(len(losses))), y=losses, log_y=True, title="Loss vs Iteration (Log Scale)", labels={"x": "Iteration", "y": "Loss"}, template="plotly_dark")
        st.plotly_chart(fig_loss, use_container_width=True)

# -------------------------------------------------------------
# 5. Model Training & Diagnostics
# -------------------------------------------------------------
elif app_mode == "⚡ Model Training & Diagnostics":
    st.title("⚡ Model Training & Autonomous Convergence Diagnostics")
    st.markdown("Train custom Gradient Descent models on the Transaction dataset and trigger automated health diagnosis.")

    raw_train, raw_test, raw_sample, data_source = get_uploaded_or_local_data()
    prepared = build_feature_pipeline(raw_train, raw_test)
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        prepared["X_train"],
        prepared["y_train"],
        test_size=0.2,
        random_state=42,
        stratify=prepared["y_train"],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        opt_choice = st.selectbox("Optimizer Type", [
            OptimizerType.ADAM,
            OptimizerType.ADAMW,
            OptimizerType.MOMENTUM,
            OptimizerType.NAG,
            OptimizerType.RMSPROP,
            OptimizerType.MINI_BATCH_GD,
            OptimizerType.BATCH_GD
        ])
    with c2:
        lr = st.number_input("Learning Rate", value=0.01, format="%.4f")
    with c3:
        epochs = st.slider("Max Epochs", 10, 100, 30, 5)
    with c4:
        batch_sz = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256, 512, 1024], value=64)

    if st.button("🚀 Train Model via Gradient Descent"):
        with st.spinner("Training Gradient Descent Model..."):
            model = LogisticRegressionGD(
                learning_rate=lr,
                max_epochs=epochs,
                batch_size=batch_sz,
                optimizer=opt_choice,
                l2_lambda=0.001
            )
            model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        st.success("Training completed successfully!")

        # Performance evaluation
        y_val_proba = model.predict_proba(X_va)
        metrics = evaluate_classification(y_va, y_val_proba)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Validation ROC-AUC", f"{metrics['roc_auc']:.4f}")
        m2.metric("Validation PR-AUC", f"{metrics['pr_auc']:.4f}")
        m3.metric("Validation Accuracy", f"{metrics['accuracy']*100:.2f}%")
        m4.metric("Brier Loss Score", f"{metrics['brier_score']:.4f}")

        # Training history plots
        hist = model.history
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], name="Train Loss", line=dict(color="#38bdf8", width=2)))
        fig_hist.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="Val Loss", line=dict(color="#f43f5e", width=2)))
        fig_hist.update_layout(title="Learning Curve (Loss Convergence)", template="plotly_dark", xaxis_title="Epoch", yaxis_title="Binary Cross-Entropy")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Automated Diagnostic Engine (Question 15)
        st.subheader("Autonomous Convergence Diagnosis Report")
        diagnostic_history = {
            "epoch": hist.get("epoch", []),
            "train_loss": hist.get("train_loss", []),
            "val_loss": hist.get("val_loss", []),
            "gradient_norm": hist.get("gradient_norm", hist.get("grad_norm", [])),
            "parameter_norm": hist.get("parameter_norm", hist.get("param_norm", [])),
            "learning_rate": hist.get("learning_rate", hist.get("lr", [])),
        }
        diag = ConvergenceDiagnosisSystem.diagnose(diagnostic_history)
        
        status_color = "🟢" if diag["severity"] == "OPTIMAL" else "🟡" if diag["severity"] == "WARNING" else "🔴"
        st.markdown(f"**Status:** {status_color} `{diag['status']}` (Severity: `{diag['severity']}`)")
        for d in diag["diagnoses"]:
            st.write(f"- {d}")

# -------------------------------------------------------------
# 6. Test Predictions & Submissions
# -------------------------------------------------------------
elif app_mode == "🚀 Test Predictions & Submissions":
    st.title("🚀 Test Predictions & Submission File Generator")
    st.markdown("Generates predictions for `test.csv` in the exact format required by `sample_submission.csv`.")

    raw_train, raw_test, raw_sample, data_source = get_uploaded_or_local_data()
    prepared = build_feature_pipeline(raw_train, raw_test)
    X_test = prepared["X_test"]
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]

    st.write(f"Test Set Records: **{len(raw_test):,}**")

    if st.button("Generate Submission Predictions"):
        with st.spinner("Training production model on full data and predicting test set..."):
            prod_model = LogisticRegressionGD(
                learning_rate=0.01,
                max_epochs=40,
                batch_size=128,
                optimizer=OptimizerType.ADAMW,
                l2_lambda=0.01
            )
            prod_model.fit(X_train, y_train)
            test_proba = prod_model.predict_proba(X_test)

            sub_df = pd.DataFrame({
                "transaction_id": raw_test["transaction_id"],
                "label": np.round(test_proba, 6)
            })

            sub_df.to_csv("submission.csv", index=False)
            st.success("Predictions generated successfully and saved to `submission.csv`!")

            st.dataframe(sub_df.head(10), use_container_width=True)

            csv_bytes = sub_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download submission.csv",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv"
            )
