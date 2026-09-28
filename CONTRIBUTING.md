# Contributing to Gradient Descent Mastery Framework

Thank you for your interest in contributing! This project contains pure Python/NumPy implementations of Gradient Descent algorithms, missing data mathematical formulas, automated EDA profiling, and interactive Streamlit tools.

## Development Workflow

1. **Fork the repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/gradient-descent-mastery.git
   cd gradient-descent-mastery
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run tests & verification**:
   ```bash
   python build_notebook.py
   python generate_submission.py
   ```
6. **Submit a Pull Request** with a detailed summary of changes.

## Code Standards
- Keep mathematical docstrings formatted in LaTeX notation.
- Ensure all new methods in `src/gradient_descent.py` maintain fully vectorized NumPy implementations.
- Keep Jupyter notebook cells documented and thoroughly commented.
