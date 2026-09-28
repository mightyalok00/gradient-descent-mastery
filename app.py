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
    .main { background-color: #0b0f19; color: #f1f5f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
    .metric-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-val { font-size: 1.8rem; font-weight: bold; color: #38bdf8; }
    .metric-lbl { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }
    .theorem-card {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset():
    data_dict = load_and_preprocess_data(val_size=0.2, random_state=42)
    return data_dict


# Sidebar Navigation
st.sidebar.title("⚡ Gradient Descent AI")
st.sidebar.caption("Optimization Engine, Missing Formula Analyzer & ML Suite")

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "📘 16 Tough Questions & Theory",
        "🔍 Formula-Based Missingness Analyzer",
        "📊 Automated Data Profiler (EDA)",
        "🧪 Interactive Gradient Descent Lab",
        "⚡ Model Training & Diagnostics",
        "🚀 Test Predictions & Submissions"
    ]
)

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
    st.markdown("""
    This module uses linear algebra formulations ($M \in \{0,1\}^{N \times d}$, $\rho_i = (M \mathbf{1})_i / d$, $\gamma_j = (\mathbf{1}^T M)_j / N$) 
    to detect missing rows and columns with exact mathematical precision.
    """)

    data = get_dataset()
    raw_train = data["raw_train"]

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
        st.write(f"Rows with $\\rho_i \ge {row_tau}$: **{filtered['rows_above_threshold_count']}**")
        st.write(f"Columns with $\\gamma_j \ge {col_tau}$: **{filtered['cols_above_threshold_count']}** ({filtered['cols_above_threshold_names']})")

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

    data = get_dataset()
    df_train = data["raw_train"]

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

    data = get_dataset()
    X_tr = data["X_train"]
    y_tr = data["y_train"]
    X_va = data["X_val"]
    y_va = data["y_val"]

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
        diag = ConvergenceDiagnosisSystem.diagnose(hist)
        
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

    data = get_dataset()
    raw_test = data["raw_test"]
    X_test = data["X_test"]
    X_train = data["X_train"]
    y_train = data["y_train"]

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
