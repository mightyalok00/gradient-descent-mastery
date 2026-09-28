# Gradient Descent Mastery

### Theory • From-Scratch Optimization • ML Diagnostics • Interactive Experiments

A professional end-to-end machine learning project focused on **Gradient Descent, optimization algorithms, mathematical missing-data analysis, model evaluation, convergence diagnostics, and interactive experimentation**.

The repository combines mathematical foundations with practical Python/NumPy implementations and a Streamlit research dashboard.

---

## 🎯 Project at a Glance

> **Gradient Descent Mastery is a machine-learning laboratory for understanding how models learn, why optimization works, and how different optimizers behave during training.**

The project follows the complete workflow:

```text
Raw Data
   ↓
Validation & Missing-Data Analysis
   ↓
Preprocessing & Feature Engineering
   ↓
Train / Validation Split
   ↓
Gradient Computation
   ↓
Optimizer
   ↓
Learning-Rate / Training Controls
   ↓
Convergence Diagnostics
   ↓
Model Evaluation
   ↓
Optimizer Benchmarking
   ↓
Interactive Visualization
   ↓
Submission Generation
```

---

## 🧠 What Is Gradient Descent?

Gradient Descent is an optimization algorithm that helps a machine-learning model **reduce its loss (error)** by repeatedly updating its parameters in the direction that lowers the loss.

The core update rule is:

```text
θ_new = θ_old − η ∇J(θ)
```

Where:

- **θ** = model parameters
- **η** = learning rate
- **∇J(θ)** = gradient of the loss
- **J(θ)** = loss function

In simple words:

> **Calculate the error → find the direction that reduces the error → take a step → repeat.**

---

## 📉 Gradient Descent & the Parabola

A simple loss function can be visualized as a **parabola**:

```text
Loss
  ↑
  │        ╲       ╱
  │         ╲     ╱
  │          ╲   ╱
  │           ╲_╱
  │            ●  ← Minimum Loss
  │           ↙
  │      Gradient Descent
  │
  └────────────────────────→ Model Parameter
```

For a simple quadratic example:

```text
J(θ) = θ²
```

The lowest point of the parabola represents the **minimum loss**.

Gradient Descent tries to move the model parameters toward this minimum.

### Learning-rate intuition

```text
Learning rate too small  →  very slow convergence
Learning rate appropriate →  smooth convergence
Learning rate too large  →  overshooting / possible divergence
```

This relationship between the **loss surface, gradient, learning rate, and convergence** is one of the central concepts explored throughout this repository.

---

## ⚡ Optimization Algorithms

The project implements **9 optimization strategies**:

| Optimizer | Core idea |
|---|---|
| Batch Gradient Descent | Uses the complete training dataset for each update |
| Stochastic Gradient Descent | Updates using individual samples |
| Mini-Batch Gradient Descent | Updates using small batches |
| Momentum | Uses accumulated velocity to accelerate learning |
| Nesterov Accelerated Gradient | Uses a look-ahead gradient |
| AdaGrad | Adapts the learning rate per parameter |
| RMSProp | Uses a moving average of squared gradients |
| Adam | Combines first and second moment estimates |
| AdamW | Adam with decoupled weight decay |

This makes it possible to study not only **whether a model learns**, but also **how different optimization strategies affect learning behavior**.

---

## 📐 Mathematical Missing-Data Engine

The repository includes a formula-based missing-data analyzer.

Missingness is represented using a binary indicator matrix:

```text
M ∈ {0, 1}^{N × d}
```

The framework calculates:

- Row missingness
- Column missingness
- Global missingness
- Threshold-based missing-row detection
- Threshold-based missing-column detection

It also supports synthetic MCAR-style validation experiments.

---

## 📚 16 Advanced Optimization Questions

The project includes mathematical explanations, derivations, and engineering solutions for **16 advanced questions**.

### Part A — Optimization Fundamentals

Topics include:

- MSE gradient derivation
- Learning-rate stability
- Spectral analysis
- Condition numbers
- Stochastic-gradient unbiasedness
- Mini-batch variance
- Loss oscillation
- Feature scaling and curvature
- Adam bias correction
- Flat vs. sharp minima

### Part B — Production Optimization

Topics include:

- Leak-free ML pipelines
- Optimizer selection
- Quantitative convergence diagnosis

### Part C — Advanced ML Behavior

Topics include:

- Severe class imbalance
- ROC-AUC vs PR-AUC
- Adam and generalization
- Ablation-study methodology
- Large-parameter neural-network analysis

---

## 🌐 Interactive Streamlit Dashboard

The `app.py` application turns the project into an interactive **Gradient Descent Research & Optimization Command Center**.

### Dashboard capabilities

- 📊 Executive overview
- 🔍 Dataset explorer
- 📐 Missing-data analysis
- 📋 Automated profiling
- ⚡ Gradient Descent training
- 📈 Training and validation loss
- 📡 Gradient-norm telemetry
- 🧪 Optimization experiments
- 🏆 Optimizer benchmarking
- 📊 ROC-AUC, PR-AUC and F1 evaluation
- 🔬 Convergence diagnostics
- 🧾 Experiment history
- 📤 Submission generation

---

## 🏆 Optimizer Benchmarking

The dashboard can compare multiple optimizers under a common experimental setup.

Example workflow:

```text
Same Dataset
     │
     ├── Batch GD
     ├── SGD
     ├── Mini-Batch GD
     ├── Momentum
     ├── NAG
     ├── AdaGrad
     ├── RMSProp
     ├── Adam
     └── AdamW
             ↓
      Compare Results
             ↓
 ROC-AUC • PR-AUC • F1
 Final Loss • Training Time
```

This provides a practical way to study optimization rather than relying only on theoretical formulas.

---

## 📊 Dataset

The current project is designed around a transaction-style binary classification dataset.

The feature pipeline supports fields such as:

- `transaction_id`
- `amount`
- `timestamp`
- `hours_since_prev_txn`
- `merchant_category`
- `country`
- `channel`
- `device_id`
- `user_id`
- `label`

The target column is `label`.

The repository documentation specifies:

| Dataset | Size |
|---|---:|
| Training data | 182,125 × 10 |
| Test data | 45,531 × 9 |
| Sample submission | 45,531 × 2 |

> **Note:** The repository defines the target as `label`, but does not currently document an authoritative external dataset source or a precise semantic definition of each label value. The README therefore avoids claiming a specific dataset provider or fraud definition.

---

## 🏗️ Project Architecture

```text
gradient-descent-mastery/
│
├── app.py
├── gradient_descent_pipeline.ipynb
├── generate_submission.py
├── build_notebook.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── gradient_descent.py
│   ├── missing_analysis.py
│   ├── profiling.py
│   ├── evaluation.py
│   └── questions_solutions.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

### Core modules

| Module | Responsibility |
|---|---|
| `data_loader.py` | Loading, preprocessing, feature engineering and scaling |
| `gradient_descent.py` | Gradient Descent models and optimizers |
| `missing_analysis.py` | Mathematical missing-data analysis |
| `profiling.py` | Statistical profiling and EDA |
| `evaluation.py` | Metrics, diagnostics and ablation analysis |
| `questions_solutions.py` | Theory, proofs and derivations |
| `app.py` | Interactive Streamlit application |

---

## 🔄 Data & ML Pipeline

```text
CSV Data
   ↓
Data Validation
   ↓
Missing / Duplicate Analysis
   ↓
Feature Engineering
   ↓
Median Imputation
   ↓
Standard Scaling
   ↓
Stratified Train / Validation Split
   ↓
Gradient Descent Training
   ↓
Validation Prediction
   ↓
Classification Metrics
   ↓
Convergence Diagnosis
   ↓
Optimizer Comparison
```

Feature engineering includes transformations such as:

- Log-transformed transaction amounts
- Log-transformed time intervals
- Normalized timestamps
- Cyclical time features
- Numerical representations of supported categorical fields

---

## 🧪 Evaluation

The project evaluates classification models using metrics including:

- **ROC-AUC**
- **PR-AUC**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Brier Score**
- **Confusion Matrix**
- **Positive-class prevalence**

The project also records optimization telemetry such as loss history and gradient behavior.

---

## ✅ GitHub Actions CI

The repository includes an automated CI workflow for:

- Python 3.10
- Python 3.11
- Python 3.12
- Dependency installation
- `app.py` syntax validation
- `src/` compilation
- Core module import validation
- Notebook build validation
- Notebook format validation

Workflow:

```text
Push / Pull Request
        ↓
Install Dependencies
        ↓
Python Syntax Check
        ↓
Compile Source
        ↓
Import Core Modules
        ↓
Validate Notebook
        ↓
Validate Notebook Format
```

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/mightyalok00/gradient-descent-mastery.git
cd gradient-descent-mastery
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open the Jupyter notebook:

```bash
jupyter notebook gradient_descent_pipeline.ipynb
```

Generate the submission:

```bash
python generate_submission.py
```

---

## 💡 Why This Project?

Most ML projects stop at:

```text
Load Data → Train Model → Calculate Score
```

This project goes deeper:

```text
Mathematics
    ↓
Optimization Algorithms
    ↓
Data Engineering
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Convergence Analysis
    ↓
Experimentation
    ↓
Visualization
    ↓
Evaluation
    ↓
CI/CD
```

The goal is to understand **what happens inside the learning process**, not simply call a pre-built training function.

---

## 🎓 Skills Demonstrated

**Machine Learning**

- Gradient Descent
- Binary classification
- Feature engineering
- Model evaluation
- Optimization
- Convergence analysis

**Mathematics**

- Gradients
- Optimization
- Matrix operations
- Missing-data formulas
- Curvature
- Stability analysis

**Python / Data Science**

- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Jupyter

**Engineering**

- Modular Python architecture
- Reproducible experiments
- Automated validation
- GitHub Actions
- CLI workflows

---

## 📌 Project Summary

**Gradient Descent Mastery** is an end-to-end project for learning and experimenting with machine-learning optimization.

It connects:

> **Mathematical theory → From-scratch optimizers → Data pipeline → Model training → Convergence diagnostics → Interactive dashboard → Automated CI**

The central idea is simple:

> **A machine-learning model learns by reducing its error, and Gradient Descent provides the mechanism for moving toward lower loss.**

---

## 👨‍💻 Author

**Alok Agarwal**

Data Science • Machine Learning • Python • Optimization

---

## ⭐ Repository

If this project is useful for learning about Gradient Descent and machine-learning optimization, consider starring the repository.

**Repository:** https://github.com/mightyalok00/gradient-descent-mastery
