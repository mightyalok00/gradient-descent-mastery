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
:root{--cyan:#55d6ff;--violet:#9b8cff;--bg:#060b14;--panel:#0b1626;--border:#1d3853;--muted:#8ea3b8;--good:#34d399;--warn:#fbbf24;--bad:#fb7185}
.stApp{background:radial-gradient(circle at 10% -5%,rgba(85,214,255,.13),transparent 25%),radial-gradient(circle at 92% 2%,rgba(155,140,255,.13),transparent 27%),linear-gradient(180deg,#060b14,#08111e 55%,#060b14);color:#e8f1fb}
.block-container{max-width:1520px;padding:1.1rem 2rem 2.5rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#050c16,#091522);border-right:1px solid #17324b}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(16,37,59,.92),rgba(7,20,34,.94));border:1px solid var(--border);border-radius:15px;padding:15px;box-shadow:0 10px 35px rgba(0,0,0,.16)}
[data-testid="stMetricLabel"]{color:var(--muted)}
[data-testid="stMetricValue"]{font-weight:800}
.hero-v2{position:relative;overflow:hidden;border:1px solid #234663;border-radius:24px;padding:32px;margin:0 0 20px;background:linear-gradient(135deg,rgba(18,45,69,.96),rgba(9,21,36,.96));box-shadow:0 24px 80px rgba(0,0,0,.28)}
.hero-v2:after{content:"";position:absolute;width:280px;height:280px;right:-90px;top:-110px;border-radius:50%;background:radial-gradient(circle,rgba(85,214,255,.18),transparent 65%);pointer-events:none}
.hero-v2 h1{margin:0;font-size:2.7rem;letter-spacing:-.055em}.hero-v2 p{color:#9bb0c5;margin:8px 0 0;font-size:1.02rem}
.badge-v2{display:inline-block;padding:6px 11px;margin:14px 5px 0 0;border-radius:999px;background:rgba(85,214,255,.09);color:var(--cyan);border:1px solid rgba(85,214,255,.22);font-size:.73rem;font-weight:800;letter-spacing:.05em}
.card-v2{border:1px solid var(--border);border-radius:16px;padding:17px;background:linear-gradient(145deg,rgba(13,29,46,.90),rgba(7,18,31,.90));min-height:110px;box-shadow:0 12px 40px rgba(0,0,0,.12)}
.card-v2 .num{font-size:1.5rem;font-weight:850;color:var(--cyan)}.card-v2 .lbl{font-size:.77rem;color:var(--muted);margin-top:5px;text-transform:uppercase;letter-spacing:.06em}
.section-v2{font-size:.74rem;font-weight:850;letter-spacing:.12em;color:var(--cyan);text-transform:uppercase;margin:24px 0 9px}
.status-card{border:1px solid var(--border);border-radius:16px;padding:18px;background:rgba(10,24,39,.86)}
.status-dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 14px rgba(52,211,153,.7);margin-right:7px}
.score-ring{font-size:2.1rem;font-weight:900;color:var(--cyan)}
.stButton>button{border-radius:11px;font-weight:750;border:1px solid #244762}
.stButton>button:hover{border-color:var(--cyan);box-shadow:0 0 20px rgba(85,214,255,.10)}
div[data-testid="stExpander"]{border:1px solid var(--border);border-radius:13px;background:rgba(8,20,33,.55)}
div[data-baseweb="tab-list"]{gap:8px}
button[data-baseweb="tab"]{font-weight:700}
.theorem-card{background:#0d1b2b;border-left:4px solid var(--cyan);padding:16px;border-radius:0 12px 12px 0}
.small-muted{color:var(--muted);font-size:.82rem}
hr{border-color:#183149}

.question-header{
    border:1px solid #234663;
    border-radius:20px;
    padding:24px 26px;
    margin:20px 0 16px;
    background:linear-gradient(135deg,rgba(18,45,69,.96),rgba(9,21,36,.96));
    box-shadow:0 14px 45px rgba(0,0,0,.18);
}
.question-index{
    color:var(--cyan);
    font-size:.72rem;
    font-weight:900;
    letter-spacing:.14em;
    text-transform:uppercase;
    margin-bottom:7px;
}
.question-header h2{
    margin:0;
    font-size:1.65rem;
    line-height:1.25;
}
.theory-section-label{
    color:var(--cyan);
    font-size:.72rem;
    font-weight:900;
    letter-spacing:.12em;
    text-transform:uppercase;
    margin:20px 0 9px;
}
.theory-nav{
    border:1px solid var(--border);
    border-radius:16px;
    padding:16px;
    background:rgba(10,24,39,.72);
}
.theory-progress{
    height:8px;
    border-radius:99px;
    background:#13273b;
    overflow:hidden;
    margin:8px 0 4px;
}
.theory-progress > div{
    height:100%;
    border-radius:99px;
    background:linear-gradient(90deg,var(--cyan),var(--violet));
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset():
    return load_and_preprocess_data(val_size=0.2, random_state=42)

def plotly_premium(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply one consistent visual language to every chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#dbeafe"),
        title=dict(text=title, font=dict(size=18, color="#e8f1fb")),
        margin=dict(l=18, r=18, t=58, b=18),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.10)", zeroline=False)
    return fig


def store_experiment(record: dict) -> None:
    history = st.session_state.setdefault("experiment_history", [])
    history.append(record)
    st.session_state["experiment_history"] = history[-12:]


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
        "🏆 Optimizer Benchmark",
        "🚀 Test Predictions & Submissions"
    ]
)

# -------------------------------------------------------------
# 0. Executive Overview
# -------------------------------------------------------------
if app_mode == "🏠 Executive Overview":
    raw_train, raw_test, raw_sample, data_source = get_uploaded_or_local_data()

    st.markdown(
        """
        <div class="hero-v2">
            <div style="font-size:.72rem;color:#55d6ff;font-weight:850;letter-spacing:.16em;">
                OPTIMIZATION INTELLIGENCE PLATFORM
            </div>
            <h1>Gradient Descent <span style="color:#55d6ff;">Mastery</span></h1>
            <p>Experiment, visualize and diagnose gradient-based learning from raw data to validated submission.</p>
            <span class="badge-v2">9 OPTIMIZERS</span>
            <span class="badge-v2">16 THEORY MODULES</span>
            <span class="badge-v2">LIVE TELEMETRY</span>
            <span class="badge-v2">FORMULA ENGINE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if raw_train is None:
        st.warning("No dataset is available. Upload train/test CSVs from the sidebar.")
    else:
        profile = ComprehensiveDataProfiler(raw_train, "Gradient Descent Dataset").profile["overview"]
        positive_rate = raw_train["label"].mean() * 100 if "label" in raw_train else float("nan")
        missing_rate = profile["total_missing_percentage"]
        health = max(0.0, 100.0 - min(100.0, missing_rate * 4.0))

        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("OBSERVATIONS", f"{profile['num_rows']:,}")
        c2.metric("FEATURES", profile["num_cols"])
        c3.metric("MISSING", f"{missing_rate:.2f}%")
        c4.metric("DUPLICATES", f"{profile['duplicate_rows']:,}")
        c5.metric("POSITIVE CLASS", f"{positive_rate:.2f}%")
        c6.metric("DATA HEALTH", f"{health:.1f}/100")

        st.markdown('<div class="section-v2">SYSTEM STATUS</div>', unsafe_allow_html=True)
        s1,s2,s3,s4=st.columns(4)
        for col,label,detail in [
            (s1,"DATA ENGINE","Schema + profiling ready"),
            (s2,"OPTIMIZATION","9 gradient methods"),
            (s3,"DIAGNOSTICS","Convergence engine ready"),
            (s4,"EXPORT","Submission pipeline ready"),
        ]:
            col.markdown(
                f'<div class="status-card"><span class="status-dot"></span><b>{label}</b>'
                f'<div class="small-muted" style="margin-top:7px;">{detail}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-v2">OPTIMIZATION COMMAND CENTER</div>', unsafe_allow_html=True)
        left,right=st.columns([1.45,1])

        with left:
            if "label" in raw_train:
                counts=raw_train["label"].value_counts().rename_axis("label").reset_index(name="count")
                fig=px.bar(counts,x="label",y="count",text_auto=True,title="Target distribution")
                st.plotly_chart(plotly_premium(fig),use_container_width=True)

        with right:
            experiment_history=st.session_state.get("experiment_history",[])
            if experiment_history:
                exp_df=pd.DataFrame(experiment_history)
                fig=px.line(exp_df,x="run",y="roc_auc",markers=True,title="Experiment ROC-AUC history")
                st.plotly_chart(plotly_premium(fig),use_container_width=True)
            else:
                st.markdown(
                    '<div class="card-v2" style="min-height:250px;">'
                    '<div class="score-ring">READY</div>'
                    '<h3>Experiment telemetry</h3>'
                    '<div class="small-muted">Train a model from Model Training to populate live experiment history.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-v2">FEATURE EXPLORER</div>', unsafe_allow_html=True)
        numeric=raw_train.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            feature=st.selectbox("Select feature",numeric,key="overview_feature")
            sample=raw_train.sample(min(12000,len(raw_train)),random_state=42)
            fig=px.histogram(sample,x=feature,color="label" if "label" in sample else None,
                             marginal="box",title=f"{feature} distribution")
            st.plotly_chart(plotly_premium(fig),use_container_width=True)

        st.caption(f"Data source: {data_source}")

# -------------------------------------------------------------
# 1. 16 Tough Questions & Theory Solver
# -------------------------------------------------------------
if app_mode == "📘 16 Tough Questions & Theory":

    question_sections = {
        "Part A · Foundations": [1, 2, 3, 4],
        "Part B · Optimization Mechanics": [5, 6, 7, 8, 9, 10],
        "Part C · Convergence & Generalization": [11, 12],
        "Part D · Production & Advanced Systems": [13, 14, 15, 16],
    }

    question_icons = {
        1: "📐",
        2: "🔬",
        3: "⚖️",
        4: "📦",
        5: "🚨",
        6: "📏",
        7: "🚀",
        8: "🧭",
        9: "⚙️",
        10: "🧮",
        11: "〰️",
        12: "📊",
        13: "🏭",
        14: "🧠",
        15: "🩺",
        16: "🏗️",
    }

    all_questions = list(range(1, 17))
    available_questions = [
        number
        for number in all_questions
        if number in QUESTIONS_AND_SOLUTIONS
    ]
    missing_questions = [
        number
        for number in all_questions
        if number not in QUESTIONS_AND_SOLUTIONS
    ]

    st.markdown(
        """
        <div class="hero-v2">
            <div style="font-size:.72rem;color:#55d6ff;font-weight:850;
                        letter-spacing:.16em;">
                THEORY WORKSPACE · 16 MODULES
            </div>
            <h1>Gradient Descent <span style="color:#55d6ff;">Mastery</span></h1>
            <p>
                Rigorous derivations, optimizer mechanics, convergence analysis,
                generalization, and production ML system design.
            </p>
            <span class="badge-v2">16 QUESTIONS</span>
            <span class="badge-v2">MATHEMATICAL DERIVATIONS</span>
            <span class="badge-v2">OPTIMIZATION</span>
            <span class="badge-v2">DIAGNOSTICS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if missing_questions:
        st.error(
            "The theory source is missing: "
            + ", ".join(f"Q{number:02d}" for number in missing_questions)
        )
        st.stop()

    completed_count = len(available_questions)
    progress_ratio = completed_count / len(all_questions)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("QUESTIONS", f"{completed_count}/16")
    p2.metric("SECTIONS", "4")
    p3.metric("DERIVATIONS", "16")
    p4.metric("WORKSPACE", "READY")

    st.markdown(
        f"""
        <div class="theory-progress">
            <div style="width:{progress_ratio * 100:.0f}%;"></div>
        </div>
        <div class="small-muted">
            {completed_count} of 16 theory modules available
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="theory-section-label">01 · SECTION NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    section_names = list(question_sections.keys())
    selected_section = st.selectbox(
        "Select a theory section",
        section_names,
        key="theory_section_selector",
    )

    section_questions = [
        number
        for number in question_sections[selected_section]
        if number in QUESTIONS_AND_SOLUTIONS
    ]

    # Keep the question widget valid when the user changes sections.
    if st.session_state.get("theory_question_selector") not in section_questions:
        st.session_state["theory_question_selector"] = section_questions[0]

    st.markdown(
        '<div class="theory-section-label">02 · QUESTION NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    selected_question = st.selectbox(
        "Choose a question to inspect",
        section_questions,
        format_func=lambda number: (
            f"Q{number:02d} · "
            f"{QUESTIONS_AND_SOLUTIONS[number]['title']}"
        ),
        key="theory_question_selector",
    )

    current_index = all_questions.index(selected_question)
    current_data = QUESTIONS_AND_SOLUTIONS[selected_question]
    icon = question_icons.get(selected_question, "📘")

    nav_left, nav_center, nav_right = st.columns([1, 2, 1])

    with nav_left:
        previous_question = (
            all_questions[current_index - 1]
            if current_index > 0
            else None
        )
        if st.button(
            "← Previous",
            use_container_width=True,
            disabled=previous_question is None,
            key="theory_previous",
        ):
            st.session_state["theory_question_selector"] = previous_question
            previous_section = next(
                name
                for name, numbers in question_sections.items()
                if previous_question in numbers
            )
            st.session_state["theory_section_selector"] = previous_section
            st.rerun()

    with nav_center:
        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding:8px;
                color:#8ea3b8;
                font-weight:800;
            ">
                {icon} Q{selected_question:02d} / 16
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_right:
        next_question = (
            all_questions[current_index + 1]
            if current_index < len(all_questions) - 1
            else None
        )
        if st.button(
            "Next →",
            use_container_width=True,
            disabled=next_question is None,
            key="theory_next",
        ):
            st.session_state["theory_question_selector"] = next_question
            next_section = next(
                name
                for name, numbers in question_sections.items()
                if next_question in numbers
            )
            st.session_state["theory_section_selector"] = next_section
            st.rerun()

    st.markdown(
        f"""
        <div class="question-header">
            <div class="question-index">
                QUESTION {selected_question:02d} / 16
            </div>
            <h2>{icon} {current_data['title']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="theory-section-label">03 · PROBLEM STATEMENT</div>',
        unsafe_allow_html=True,
    )
    st.info(current_data["question"])

    st.markdown(
        '<div class="theory-section-label">04 · MATHEMATICAL SOLUTION</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(current_data["latex_derivation"])

    with st.expander("💡 Intuition & Practical Interpretation", expanded=False):
        st.markdown(
            """
            Use the derivation above as the mathematical foundation.
            The interactive labs elsewhere in this application demonstrate
            how the corresponding optimization behavior appears numerically.
            """
        )

    if selected_question in (2, 11):
        st.markdown(
            '<div class="theory-section-label">05 · INTERACTIVE CONVERGENCE LAB</div>',
            unsafe_allow_html=True,
        )

        diagnostic_left, diagnostic_right = st.columns(2)

        with diagnostic_left:
            eta = st.slider(
                "Learning Rate (η)",
                0.001,
                0.025,
                0.018,
                0.001,
                key=f"theory_lr_{selected_question}",
            )

        with diagnostic_right:
            n_iters = st.slider(
                "Iterations",
                5,
                50,
                20,
                1,
                key=f"theory_iter_{selected_question}",
            )

        trajectory = [[-8.0, 1.0]]
        position = np.array([-8.0, 1.0], dtype=float)

        for _ in range(n_iters):
            gradient = np.array(
                [position[0], 100.0 * position[1]],
                dtype=float,
            )
            position -= eta * gradient
            trajectory.append(position.tolist())

        trajectory = np.asarray(trajectory)

        x_grid = np.linspace(-10, 10, 200)
        y_grid = np.linspace(-2, 2, 200)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        z_mesh = 0.5 * (x_mesh**2 + 100 * y_mesh**2)

        fig = go.Figure()

        fig.add_trace(
            go.Contour(
                x=x_grid,
                y=y_grid,
                z=z_mesh,
                contours_coloring="lines",
                line_width=1.2,
                colorscale="Viridis",
                showscale=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode="lines+markers",
                marker={"size": 6},
                name="Gradient Descent Path",
            )
        )

        fig.update_layout(
            title=(
                "Ill-Conditioned Quadratic Path "
                f"(κ=100, η={eta})"
            ),
            xaxis_title="w₁ · slow direction (λ=1)",
            yaxis_title="w₂ · steep direction (λ=100)",
        )

        st.plotly_chart(
            plotly_premium(fig),
            use_container_width=True,
        )

    st.markdown(
        '<div class="theory-section-label">06 · MODULE MAP</div>',
        unsafe_allow_html=True,
    )

    module_cols = st.columns(4)
    for column, (section_name, numbers) in zip(
        module_cols,
        question_sections.items(),
    ):
        with column:
            section_available = [
                number for number in numbers
                if number in QUESTIONS_AND_SOLUTIONS
            ]
            st.markdown(
                f"**{section_name}**  \n"
                f"{len(section_available)} modules"
            )
            for number in section_available:
                marker = "●" if number == selected_question else "○"
                st.caption(
                    f"{marker} Q{number:02d} · "
                    f"{QUESTIONS_AND_SOLUTIONS[number]['title']}"
                )

    st.markdown("---")

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
    st.plotly_chart(plotly_premium(fig_corr), use_container_width=True)

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
    st.plotly_chart(plotly_premium(fig_dist), use_container_width=True)

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
        st.plotly_chart(plotly_premium(fig_opt), use_container_width=True)

    with col_fig2:
        fig_loss = px.line(x=list(range(len(losses))), y=losses, log_y=True, title="Loss vs Iteration (Log Scale)", labels={"x": "Iteration", "y": "Loss"}, template="plotly_dark")
        st.plotly_chart(plotly_premium(fig_loss), use_container_width=True)

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

        # Evaluate before storing experiment telemetry.
        y_val_proba = model.predict_proba(X_va)
        metrics = evaluate_classification(y_va, y_val_proba)

        # Persist a compact experiment record for the command center.
        run_number = len(st.session_state.get("experiment_history", [])) + 1
        store_experiment({
            "run": run_number,
            "optimizer": str(opt_choice).split(".")[-1],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1_score"],
            "final_loss": model.history["val_loss"][-1],
            "epochs": len(model.history["epoch"]),
        })

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
        st.plotly_chart(plotly_premium(fig_hist), use_container_width=True)
        hist_df=pd.DataFrame(hist)
        gkey="gradient_norm" if "gradient_norm" in hist_df else ("grad_norm" if "grad_norm" in hist_df else None)
        if gkey:
            grad_fig=px.line(hist_df,x="epoch",y=gkey,title="Gradient norm telemetry",log_y=True)
            st.plotly_chart(plotly_premium(grad_fig),use_container_width=True)



        if st.session_state.get("experiment_history"):
            st.markdown("### Experiment history")
            st.dataframe(
                pd.DataFrame(st.session_state["experiment_history"]).round(5),
                use_container_width=True,
                hide_index=True,
            )

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
# 6. Optimizer Benchmark
# -------------------------------------------------------------
elif app_mode == "🏆 Optimizer Benchmark":
    st.markdown(
        """
        <div class="hero-v2">
            <div style="font-size:.72rem;color:#55d6ff;font-weight:850;letter-spacing:.16em;">BENCHMARK ENGINE</div>
            <h1>Optimizer <span style="color:#55d6ff;">Arena</span></h1>
            <p>Run the same validation protocol across gradient-based optimizers and inspect convergence behavior.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_train, raw_test, raw_sample, data_source = get_uploaded_or_local_data()
    prepared=build_feature_pipeline(raw_train,raw_test)
    from sklearn.model_selection import train_test_split
    X_tr,X_va,y_tr,y_va=train_test_split(
        prepared["X_train"],prepared["y_train"],test_size=.2,random_state=42,stratify=prepared["y_train"]
    )

    b1,b2,b3=st.columns(3)
    bench_epochs=b1.slider("Benchmark epochs",5,60,20,5)
    bench_batch=b2.select_slider("Batch size",[32,64,128,256,512],value=128)
    bench_lr=b3.number_input("Benchmark learning rate",0.0001,0.5,0.01,0.005,format="%.4f")
    selected_opts=st.multiselect(
        "Optimizers to compare",
        [o.value for o in OptimizerType],
        default=[OptimizerType.ADAM.value,OptimizerType.ADAMW.value,OptimizerType.RMSPROP.value,OptimizerType.MOMENTUM.value],
    )

    if st.button("🏁 Run benchmark",type="primary",use_container_width=True):
        if not selected_opts:
            st.warning("Select at least one optimizer before running the benchmark.")
            st.stop()

        records=[]
        progress=st.progress(0.0)
        for i,name in enumerate(selected_opts,1):
            opt=next(o for o in OptimizerType if o.value==name)
            model=LogisticRegressionGD(
                learning_rate=bench_lr,max_epochs=bench_epochs,batch_size=bench_batch,
                optimizer=opt,l2_lambda=.01 if name==OptimizerType.ADAMW.value else .001
            )
            t0=time.perf_counter()
            model.fit(X_tr,y_tr,X_val=X_va,y_val=y_va)
            elapsed=time.perf_counter()-t0
            score=evaluate_classification(y_va,model.predict_proba(X_va))
            history=model.history
            records.append({
                "Optimizer":name,
                "ROC-AUC":score["roc_auc"],
                "PR-AUC":score["pr_auc"],
                "F1":score["f1_score"],
                "Final Loss":history["val_loss"][-1],
                "Seconds":elapsed,
            })
            progress.progress(i/len(selected_opts))
        bench_df=pd.DataFrame(records).sort_values("ROC-AUC",ascending=False)
        st.session_state["benchmark_results"]=bench_df

    if "benchmark_results" in st.session_state:
        bench_df=st.session_state["benchmark_results"]
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Runs",len(bench_df))
        m2.metric("Best ROC-AUC",f"{bench_df['ROC-AUC'].max():.4f}")
        m3.metric("Best PR-AUC",f"{bench_df['PR-AUC'].max():.4f}")
        m4.metric("Fastest",f"{bench_df['Seconds'].min():.2f}s")

        tab1,tab2=st.tabs(["Leaderboard","Metrics"])
        with tab1:
            st.dataframe(bench_df.style.format({
                "ROC-AUC":"{:.4f}","PR-AUC":"{:.4f}","F1":"{:.4f}",
                "Final Loss":"{:.5f}","Seconds":"{:.2f}"
            }),use_container_width=True,hide_index=True)
        with tab2:
            metric_df=bench_df.melt(
                id_vars="Optimizer",
                value_vars=["ROC-AUC","PR-AUC","F1"],
                var_name="Metric",value_name="Score"
            )
            fig=px.bar(metric_df,x="Optimizer",y="Score",color="Metric",barmode="group",
                       title="Optimizer metric comparison")
            st.plotly_chart(plotly_premium(fig),use_container_width=True)

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
