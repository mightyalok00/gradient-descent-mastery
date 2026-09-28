# Gradient Descent Optimization, Theory & ML Framework

A comprehensive, production-grade Machine Learning pipeline, mathematical theorem repository, and interactive Streamlit application solving 16 advanced optimization challenges, formula-based missing data analysis, and automated profiling.

---

## 📂 Project Architecture

```
gradient_descent_mastery/
├── app.py                                   # Interactive Streamlit Web Application
├── gradient_descent_pipeline.ipynb          # Comprehensive Jupyter Notebook (All 16 Qs + Models + EDA)
├── submission.csv                           # Test set predictions matching sample_submission.csv
├── requirements.txt                         # Python package requirements
├── README.md                                # Full documentation and user guide
├── build_notebook.py                        # Automated notebook compilation script
├── generate_submission.py                   # Automated submission generation pipeline
└── src/
    ├── __init__.py                          # Package initialization
    ├── data_loader.py                       # Preprocessing, cyclical features & scalers
    ├── missing_analysis.py                  # Linear-algebra missing rows & cols formula engine
    ├── gradient_descent.py                  # Vectorized BGD, SGD, MBGD, Momentum, NAG, Adam, AdamW
    ├── profiling.py                         # Automated statistical profiling & HTML generator
    ├── questions_solutions.py               # Theoretical proofs & derivations for all 16 questions
    └── evaluation.py                        # Metrics, automated diagnosis & ablation studies
```

---

## 🚀 Key Features

### 1. 📐 Mathematical Missing Rows & Columns Formula Engine
- Formulates missingness through binary indicator matrices $M \in \{0, 1\}^{N \times d}$.
- Row missing rate vector: $\boldsymbol{\rho} = \frac{1}{d} M \mathbf{1}_d$.
- Column missing rate vector: $\boldsymbol{\gamma} = \frac{1}{N} M^T \mathbf{1}_N$.
- Global missingness: $\Omega = \frac{\mathbf{1}_N^T M \mathbf{1}_d}{N \times d}$.
- Set filter predicates: $\mathcal{R}_{\tau} = \{ i \mid \rho_i \ge \tau \}$, $\mathcal{C}_{\tau} = \{ j \mid \gamma_j \ge \tau \}$.
- Validated with synthetic Bernoulli MCAR injection experiments.

### 2. ⚡ Pure NumPy Vectorized Gradient Descent Optimizers
- **Batch Gradient Descent (BGD)**: Exact empirical risk gradient $\nabla J(\theta)$.
- **Stochastic Gradient Descent (SGD)**: Unbiased single-sample stochastic estimate.
- **Mini-Batch Gradient Descent (MBGD)**: Configurable mini-batch size ($B \in [16, 1024]$).
- **Polyak Momentum**: Exponential moving velocity $v_{t+1} = \beta v_t + \eta \nabla J(\theta_t)$.
- **Nesterov Accelerated Gradient (NAG)**: Look-ahead braking mechanism $\nabla J(\theta_t - \beta v_t)$.
- **AdaGrad**: Parameter-wise adaptive learning rate with accumulated squared gradients.
- **RMSProp**: Exponentially moving second moment $v_{t+1} = \beta v_t + (1-\beta) g_t^2$.
- **Adam**: First and second moments with analytical bias correction $\hat{m}_t, \hat{v}_t$.
- **AdamW**: Decoupled weight decay regularization.

### 3. 📘 16 Advanced Question Solutions & Derivations
- **Part A (Q1–Q12)**: First-principles MSE gradient derivation, spectral stability bound $0 < \eta < \frac{2}{\lambda_{\max}}$, condition number analysis ($\kappa = 500$), unbiased stochastic gradient proof, mini-batch variance trade-off, loss oscillation diagnosis, feature scaling curvature derivation, Adam bias correction proof, and the flat vs sharp generalization gap.
- **Part B (Q13–Q15)**: 14-stage leak-free production ML pipeline, optimizer selection framework, and real-time quantitative convergence diagnosis system.
- **Part C (Q16)**: Deep analysis of 10M parameter network under severe class imbalance, ROC-AUC vs PR-AUC divergence, Adam generalization bias, and ablation protocols.

### 4. 📊 Automated Data Profiling & EDA
- Detailed summary statistics: mean, std, quantiles, skewness, kurtosis, missing counts.
- Feature correlation matrices and distribution visualizations.
- Standalone HTML report generator.

### 5. 🌐 Interactive Streamlit Web Application (`app.py`)
- Multi-tab UI featuring an interactive 2D loss landscape contour simulator, dataset explorer, live training loss monitoring, convergence diagnosis, and 1-click submission exporter.

---

## 🛠️ Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```

3. **Open & Run Jupyter Notebook:**
   ```bash
   jupyter notebook gradient_descent_pipeline.ipynb
   ```

4. **Generate Submissions Directly via CLI:**
   ```bash
   python generate_submission.py
   ```

---

## 📊 Dataset Reference
- Train Dataset: `D:\gradient_descent\train.csv` (182,125 rows, 10 columns)
- Test Dataset: `D:\gradient_descent\test.csv` (45,531 rows, 9 columns)
- Sample Submission: `D:\gradient_descent\sample_submission.csv` (45,531 rows, 2 columns)
