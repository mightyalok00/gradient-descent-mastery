"""
Solutions and Theoretical Foundations for the 16 Tough Gradient Descent Questions
==================================================================================
Comprehensive mathematical derivations, physical interpretations, architecture designs,
optimizer taxonomy, and deep statistical analysis for all 16 questions.
"""

from typing import Dict, Any

QUESTIONS_AND_SOLUTIONS: Dict[int, Dict[str, Any]] = {
    1: {
        "title": "Derivation of Gradient Descent & Quadratic Convergence Condition",
        "question": "Derive the gradient-descent update rule from first principles for multivariate linear regression using MSE. Explain exactly how the learning rate affects convergence and derive the condition under which gradient descent converges for a quadratic loss.",
        "latex_derivation": r"""
### 1. First-Principles Derivation for Multivariate Linear Regression
Given training matrix $X \in \mathbb{R}^{m \times d}$, targets $y \in \mathbb{R}^m$, parameter vector $w \in \mathbb{R}^d$, and bias $b \in \mathbb{R}$.
The Mean Squared Error (MSE) objective function is:
$$J(w, b) = \frac{1}{2m} \|Xw + b\mathbf{1} - y\|_2^2 = \frac{1}{2m} (Xw + b\mathbf{1} - y)^T (Xw + b\mathbf{1} - y)$$

Expanding the quadratic form with residual $e = Xw + b\mathbf{1} - y$:
$$\nabla_w J(w, b) = \frac{1}{m} X^T (Xw + b\mathbf{1} - y) = \frac{1}{m} X^T e$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \mathbf{1}^T (Xw + b\mathbf{1} - y) = \frac{1}{m} \sum_{i=1}^m e_i$$

Gradient Descent updates with step size $\eta > 0$:
$$w^{(t+1)} = w^{(t)} - \eta \nabla_w J(w^{(t)}, b^{(t)}) = w^{(t)} - \frac{\eta}{m} X^T (X w^{(t)} + b^{(t)}\mathbf{1} - y)$$
$$b^{(t+1)} = b^{(t)} - \eta \frac{\partial J}{\partial b} = b^{(t)} - \frac{\eta}{m} \sum_{i=1}^m (x_i^T w^{(t)} + b^{(t)} - y_i)$$

### 2. Convergence Condition for Quadratic Loss via Spectral Analysis
Let a general quadratic loss be $f(\theta) = \frac{1}{2} \theta^T H \theta - b^T \theta + c$, where $H = \nabla^2 f(\theta)$ is the symmetric positive definite Hessian matrix with eigenvalues $0 < \lambda_{\min} \le \dots \le \lambda_{\max}$.
The gradient is $\nabla f(\theta) = H\theta - b$. At the global optimum $\theta^*$, we have $H\theta^* = b$.
The error vector at iteration $t$ is $e_t = \theta^{(t)} - \theta^*$.

Substitute into the GD update rule:
$$\theta^{(t+1)} = \theta^{(t)} - \eta (H\theta^{(t)} - b)$$
$$e_{t+1} + \theta^* = e_t + \theta^* - \eta (H(e_t + \theta^*) - b)$$
$$e_{t+1} = (I - \eta H) e_t = (I - \eta H)^{t+1} e_0$$

Using the spectral decomposition $H = Q \Lambda Q^T$:
$$\|e_{t+1}\|_2 \le \rho(I - \eta H)^{t+1} \|e_0\|_2 = \max_i |1 - \eta \lambda_i|^{t+1} \|e_0\|_2$$

For linear convergence $\|e_t\|_2 \to 0$ as $t \to \infty$, we require the spectral radius $\rho(I - \eta H) < 1$:
$$-1 < 1 - \eta \lambda_i < 1 \quad \forall i \implies 0 < \eta < \frac{2}{\lambda_{\max}(H)}$$

The optimal learning rate minimizing the contraction factor $\max_i |1 - \eta \lambda_i|$ is:
$$\eta^* = \frac{2}{\lambda_{\max} + \lambda_{\min}}, \quad \text{yielding contraction factor } \rho^* = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\kappa - 1}{\kappa + 1}$$
where $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ is the condition number of the Hessian.
        """
    },
    2: {
        "title": "Hessian Eigenvalues & Convergence Speed Analysis",
        "question": "A loss function has a Hessian with eigenvalues λ₁ = 0.5, λ₂ = 10, λ₃ = 250. Determine the theoretical range of learning rates for convergence of vanilla gradient descent. Then explain why convergence can be extremely slow despite using a theoretically valid learning rate.",
        "latex_derivation": r"""
### 1. Theoretical Range of Learning Rates
Given eigenvalues $\lambda_1 = 0.5$, $\lambda_2 = 10$, $\lambda_3 = 250$:
- $\lambda_{\max} = 250$
- $\lambda_{\min} = 0.5$

From the quadratic stability condition $\eta < \frac{2}{\lambda_{\max}}$:
$$\eta \in \left(0, \frac{2}{250}\right) = (0, 0.008)$$

### 2. Condition Number and Convergence Rate
Condition number of the Hessian:
$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{250}{0.5} = 500$$

At the optimal learning rate $\eta^* = \frac{2}{\lambda_{\max} + \lambda_{\min}} = \frac{2}{250.5} \approx 0.007984$:
The contraction factor for error along the slowest eigenvector ($\lambda_{\min} = 0.5$) is:
$$\rho_1 = 1 - \eta^* \lambda_1 = 1 - (0.007984)(0.5) = 1 - 0.003992 = 0.996008$$

The number of iterations $k$ needed to reduce the error along $\lambda_{\min}$ by a factor of $\epsilon = 10^{-4}$:
$$\rho_1^k \le 10^{-4} \implies k \approx \frac{\ln(10^{-4})}{\ln(0.996008)} \approx \frac{-9.2103}{-0.004000} \approx 2,302 \text{ iterations}$$

**Physical Explanation:**
The gradient is dominated by the steep direction ($\lambda_3 = 250$), forcing the learning rate to be minuscule ($\eta < 0.008$) to avoid divergence. However, along the flat valley ($\lambda_1 = 0.5$), this tiny step size results in microscopic progress per iteration, causing severe zig-zagging perpendicular to the ravine and agonizingly slow progress along the valley floor.
        """
    },
    3: {
        "title": "BGD vs SGD: Mathematical Comparison & Unbiased Variance Proof",
        "question": "Batch Gradient Descent vs Stochastic Gradient Descent: mathematically compare their gradient estimates. Prove why the stochastic estimate is unbiased under standard sampling assumptions, and explain why its variance can prevent convergence to an exact minimum.",
        "latex_derivation": r"""
### 1. Mathematical Definitions of Gradient Estimates
- **True Empirical Risk Gradient (BGD):**
$$g_{\text{BGD}}(\theta) = \nabla J(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(\theta)$$
- **Stochastic Gradient (SGD):** Sample index $i_t \sim \text{Uniform}(\{1, \dots, N\})$:
$$g_{\text{SGD}}(\theta) = \nabla f_{i_t}(\theta)$$

### 2. Proof of Unbiasedness
Taking the expectation with respect to the uniform sampling distribution $P(i_t = k) = \frac{1}{N}$:
$$\mathbb{E}_{i_t}[g_{\text{SGD}}(\theta)] = \sum_{k=1}^N P(i_t = k) \nabla f_k(\theta) = \sum_{k=1}^N \frac{1}{N} \nabla f_k(\theta) = \frac{1}{N} \sum_{k=1}^N \nabla f_k(\theta) = \nabla J(\theta) = g_{\text{BGD}}(\theta)$$
Hence, $\mathbb{E}[g_{\text{SGD}}(\theta)] = g_{\text{BGD}}(\theta)$ is strictly unbiased. $\blacksquare$

### 3. Gradient Variance & Convergence to a Noise Ball
The variance of the stochastic gradient estimate is:
$$\sigma_{\text{SGD}}^2(\theta) = \mathbb{E}\left[ \|g_{\text{SGD}}(\theta) - \nabla J(\theta)\|^2 \right] = \frac{1}{N} \sum_{i=1}^N \|\nabla f_i(\theta) - \nabla J(\theta)\|^2$$
At the exact global minimum $\theta^*$, $\nabla J(\theta^*) = 0$, but individual sample gradients $\nabla f_i(\theta^*) \ne 0$.
Thus, $\sigma^2(\theta^*) = \frac{1}{N} \sum_{i=1}^N \|\nabla f_i(\theta^*)\|^2 = \sigma_*^2 > 0$.

Under a constant learning rate $\eta$, the parameter vector does not converge to $\theta^*$, but instead oscillates in a limit-cycle / noise ball around $\theta^*$ with steady-state variance:
$$\mathbb{E}[\|\theta_t - \theta^*\|^2] \approx \frac{\eta \sigma_*^2}{2 \mu}$$
To guarantee asymptotic convergence to the exact point $\theta^*$, $\eta_t$ must satisfy the Robbins-Monro conditions:
$$\sum_{t=1}^\infty \eta_t = \infty \quad \text{and} \quad \sum_{t=1}^\infty \eta_t^2 < \infty$$
        """
    },
    4: {
        "title": "Mini-Batch Trade-Off Analysis (B = 1, 32, 256, N)",
        "question": "Consider Mini-Batch Gradient Descent with batch sizes 1, 32, 256, and N. Analyze the trade-off between gradient variance, GPU utilization, memory consumption, generalization, and convergence stability.",
        "latex_derivation": r"""
### Rigorous Multi-Dimensional Trade-Off Matrix

| Metric / Dimension | $B = 1$ (Pure SGD) | $B = 32$ (Small Mini-Batch) | $B = 256$ (Optimal Mini-Batch) | $B = N$ (Full Batch GD) |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Variance** $\operatorname{Var}(g_B)$ | $\sigma^2$ (Maximal) | $\frac{\sigma^2}{32}$ (Reduced 96.9%) | $\frac{\sigma^2}{256}$ (Reduced 99.6%) | $0$ (Exact deterministic gradient) |
| **Vectorization / GPU Efficiency** | Poor (Memory bandwidth bound) | Moderate (Tensor cores partially loaded) | High (Optimal SIMD/SIMT warp occupancy) | High (Subject to OOM memory ceiling) |
| **Memory Footprint** $\mathcal{O}(B \cdot d)$ | $\mathcal{O}(d)$ (Minimal) | $\mathcal{O}(32d)$ (Very low) | $\mathcal{O}(256d)$ (Low/Moderate) | $\mathcal{O}(Nd)$ (Can exceed VRAM) |
| **Generalization / Flat Minima** | Highest (Escapes sharp saddle points) | Excellent (Sufficient noise regularization) | Good (Balanced exploration/drift) | Poor (Often trapped in sharp local minima) |
| **Step Stability** | Highly erratic random walk | Smooth with slight stochastic jitter | Highly stable gradient trajectory | Deterministically monotonic in loss |

**Mathematical Justification:**
The variance of a mini-batch gradient sampled with replacement is $\operatorname{Var}\left(\frac{1}{B} \sum_{i=1}^B \nabla f_i\right) = \frac{\sigma^2}{B}$.
As $B$ scales, the computation scales linearly in FLOPs per step, but the standard error of the gradient only decreases at rate $\mathcal{O}(1/\sqrt{B})$. Beyond a critical batch size $B^*$, gradient signal-to-noise ratio yields diminishing returns in convergence rate per sample while degrading generalization performance (Keskar et al., 2016).
        """
    },
    5: {
        "title": "Diagnosis of Oscillating and Exploding Training Loss",
        "question": "A model's training loss oscillates dramatically between iterations and occasionally increases by several orders of magnitude. Diagnose the possible causes involving the learning rate, feature scaling, curvature, gradient explosion, and numerical instability. Explain how you would distinguish each cause experimentally.",
        "latex_derivation": r"""
### 5 Root Causes and Experimental Disambiguation Protocol

1. **Excessive Learning Rate ($\eta > 2/\lambda_{\max}$):**
   - *Mechanism:* Gradient steps overshoot the paraboloid walls, leading to exponential geometric divergence.
   - *Experiment:* Halve $\eta$ across orders of magnitude ($10^{-1} \to 10^{-2} \to 10^{-3} \to 10^{-4}$). If oscillations vanish monotonically with smaller $\eta$, learning rate overshoot is confirmed.

2. **Unscaled Features / Ill-Conditioned Curvature ($\kappa \gg 10^3$):**
   - *Mechanism:* Gradient vectors align with the steep canyon walls rather than the minimum.
   - *Experiment:* Compute condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ of $X^T X$. Apply `StandardScaler`. If oscillations disappear without lowering $\eta$, poor feature scaling was the culprit.

3. **Gradient Explosion in Deep Architectures:**
   - *Mechanism:* Backpropagated product of weight matrices $\prod W_l > 1$ amplifies gradient magnitudes exponentially.
   - *Experiment:* Log $\|\nabla_\theta J\|_2$ at each layer before update. Implement gradient clipping: $g \leftarrow g \cdot \min(1, c / \|g\|_2)$ with $c=1.0$. If clipping eliminates loss spikes while maintaining progress, gradient explosion is confirmed.

4. **Numerical Underflow / Overflow in Softmax / Log-Loss:**
   - *Mechanism:* $\ln(\hat{y}) \to -\infty$ when $\hat{y} \to 0$ or $\exp(z) \to \infty$ for large $z$.
   - *Experiment:* Replace raw logits with Log-Sum-Exp trick: $\ln \sum e^{z_i} = c + \ln \sum e^{z_i - c}$ where $c = \max_i z_i$, and clip $\hat{y} \in [\epsilon, 1-\epsilon]$. Check if NaNs vanish.

5. **Mini-batch Noise / Outlier Samples:**
   - *Mechanism:* High-loss outliers in small mini-batches pull the parameter vector far off course.
   - *Experiment:* Increase batch size $B$ from 16 to 256. If oscillations reduce proportionally to $1/\sqrt{B}$, stochastic mini-batch noise was the driver.
        """
    },
    6: {
        "title": "Geometry of Feature Scaling & Hessian Curvature Derivation",
        "question": "Explain why feature scaling changes the geometry of the optimization problem. Derive how scaling one feature by a constant changes the corresponding curvature and explain why StandardScaler can dramatically improve gradient-descent convergence.",
        "latex_derivation": r"""
### Mathematical Derivation of Feature Scaling on Curvature
Consider linear regression with loss $J(w) = \frac{1}{2m} \|Xw - y\|_2^2$.
The Hessian with respect to weights $w$ is:
$$H_w = \nabla^2_w J(w) = \frac{1}{m} X^T X$$

Suppose feature $j$ is transformed by scaling factor $c > 0$: $\tilde{x}_{ij} = c \cdot x_{ij}$.
Let $S = \operatorname{diag}(1, \dots, c, \dots, 1)$. Then $\tilde{X} = X S$.
The new Hessian with respect to transformed weights $\tilde{w}$ is:
$$\tilde{H} = \frac{1}{m} \tilde{X}^T \tilde{X} = \frac{1}{m} S X^T X S = S H_w S$$

The $(j, j)$-th diagonal entry of the Hessian scales quadratically:
$$\tilde{H}_{jj} = c^2 H_{jj}$$
The off-diagonal elements scale linearly: $\tilde{H}_{jk} = c H_{jk}$.

### Impact on Condition Number and Loss Contours
If $x_1 \in [0, 1]$ and $x_2 \in [0, 1000]$, then $H_{22} \approx 10^6 H_{11}$.
The condition number becomes:
$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \approx 10^6$$
The loss contours become eccentric ellipses (flattened canyons). Gradient descent steps oscillate violently across $w_2$ while crawling along $w_1$.
Applying `StandardScaler` ($\tilde{X} = (X - \mu) / \sigma$) enforces $\operatorname{Var}(\tilde{x}_j) = 1$, spherifying the loss surface ($\tilde{H} \approx I$), reducing $\kappa \to 1$, allowing maximal step sizes $\eta^* \approx 1$ and converging in minimal iterations.
        """
    },
    7: {
        "title": "Momentum Gradient Descent: Derivation & Velocity Interpretation",
        "question": "Derive the update equations for Momentum Gradient Descent from the standard gradient-descent rule. Explain the physical interpretation of velocity, and analyze what happens when the momentum coefficient approaches 0 and 1.",
        "latex_derivation": r"""
### 1. Derivation and Physics Analogy
Consider a particle of mass $m$ rolling down a potential energy landscape $J(\theta)$ under friction coefficient $\gamma$ and force $F(\theta) = -\nabla J(\theta)$.
From Newton's second law:
$$m \frac{d^2 \theta}{dt^2} + \gamma \frac{d\theta}{dt} = -\nabla J(\theta)$$
Discretizing via Euler integration with velocity $v_t = \frac{\Delta \theta}{\Delta t}$:
$$v_{t+1} = \beta v_t + \eta \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$
where $\beta = 1 - \frac{\gamma \Delta t}{m} \in [0, 1)$ is the momentum decay factor and $\eta$ is the effective learning rate.

### 2. Exponential Moving Average of Past Gradients
Expanding $v_{t+1}$ recursively:
$$v_{t+1} = \eta \sum_{k=0}^t \beta^{t-k} \nabla J(\theta_k)$$
Effective velocity is an exponentially weighted moving average with time constant $\tau = \frac{1}{1 - \beta}$.

### 3. Asymptotic Analysis as $\beta \to 0$ and $\beta \to 1$
- **Case $\beta \to 0$:**
  $$v_{t+1} = \eta \nabla J(\theta_t) \implies \theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$$
  Momentum collapses to vanilla Gradient Descent (no inertia, memoryless).
- **Case $\beta \to 1$:**
  Effective step size along persistent gradient direction accelerates by a factor of $\frac{1}{1 - \beta}$:
  $$v_{\text{terminal}} = \frac{\eta}{1 - \beta} \nabla J$$
  When $\beta \to 1$, $\frac{1}{1 - \beta} \to \infty$. The system has zero friction and acts as a lossless undamped harmonic oscillator that overshoots the minimum and oscillates indefinitely without stopping.
        """
    },
    8: {
        "title": "Nesterov Accelerated Gradient (NAG): Derivation & Look-Ahead Computation Graph",
        "question": "Nesterov Accelerated Gradient (NAG): derive its update equations and explain why evaluating the gradient at the look-ahead position can provide better optimization behavior than classical momentum. Compare the computational graphs of Momentum and NAG.",
        "latex_derivation": r"""
### 1. NAG Mathematical Derivation
In classical momentum, the step is $v_{t+1} = \beta v_t + \eta \nabla J(\theta_t)$. The gradient is evaluated at the current position $\theta_t$ before applying momentum.
NAG realizes that the parameter will move by approximately $\beta v_t$ regardless. Hence, it evaluates the gradient at the predicted look-ahead position $\theta_t - \beta v_t$:
$$v_{t+1} = \beta v_t + \eta \nabla J(\theta_t - \beta v_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

### 2. Look-Ahead Braking Mechanism
Let $\theta_{\text{ahead}} = \theta_t - \beta v_t$.
If momentum is carrying the parameters uphill towards an overshoot, $\nabla J(\theta_{\text{ahead}})$ points strongly backward *before* the parameter actually reaches the crest, acting as an adaptive predictive brake.

### 3. Theoretical Convergence Rate
For convex $L$-smooth functions:
- Vanilla GD: $\mathcal{O}(1/k)$
- Polyak Momentum: $\mathcal{O}(1/k)$
- Nesterov NAG: $\mathcal{O}(1/k^2)$ (matches Nesterov's optimal theoretical lower bound for first-order black-box optimization).
        """
    },
    9: {
        "title": "Comparative Analysis of AdaGrad, RMSProp, and Adam",
        "question": "Compare AdaGrad, RMSProp, and Adam mathematically. Derive their parameter-wise adaptive learning-rate mechanisms and explain why AdaGrad can eventually stop learning while RMSProp and Adam generally avoid this behavior.",
        "latex_derivation": r"""
### 1. Mathematical Formulas for Adaptive Optimizers

| Algorithm | Accumulated Second Moment $G_t$ / $v_t$ | Parameter Update Rule |
| :--- | :--- | :--- |
| **AdaGrad** | $G_t = G_{t-1} + g_t \odot g_t = \sum_{\tau=1}^t g_\tau^2$ | $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t$ |
| **RMSProp** | $v_t = \beta v_{t-1} + (1 - \beta) g_t^2$ | $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t$ |
| **Adam** | $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$<br>$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ | $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \ \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$<br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$ |

### 2. Why AdaGrad Stops Learning (The Diminishing Gradient Problem)
In AdaGrad, $G_t = \sum_{\tau=1}^t g_\tau^2$ is a monotonically non-decreasing sum of positive numbers:
$$G_t \ge G_{t-1} \ge \dots \ge G_1 \implies \lim_{t \to \infty} G_t = \infty$$
Consequently, the effective learning rate along every coordinate satisfies:
$$\lim_{t \to \infty} \eta_{\text{eff}}(t) = \lim_{t \to \infty} \frac{\eta}{\sqrt{G_t} + \epsilon} = 0$$
The step size decays to zero prematurely long before reaching the global minimum, freezing learning permanently.

### 3. How RMSProp and Adam Fix AdaGrad
RMSProp and Adam replace the monotonic sum with an Exponential Moving Average (EMA) with forgetting factor $\beta \in (0, 1)$:
$$v_t = (1 - \beta) \sum_{\tau=1}^t \beta^{t-\tau} g_\tau^2$$
The effective window size is $\approx \frac{1}{1 - \beta}$. Old historical gradients are exponentially discounted, bounding $v_t \approx \mathbb{E}[g^2]$ and preventing the denominator from growing unbounded.
        """
    },
    10: {
        "title": "Derivation of Adam's Bias Correction Terms",
        "question": "Adam uses first- and second-moment estimates with bias correction. Derive the bias-correction terms from the expected value of the exponential moving averages and explain what would happen if the correction were removed.",
        "latex_derivation": r"""
### 1. Derivation of First-Moment Bias Correction
Let $m_0 = 0$. The first moment recursion is $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$.
Unrolling recursively:
$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$

Taking mathematical expectation $\mathbb{E}[m_t]$ assuming true gradient distribution $\mathbb{E}[g_i] \approx \mathbb{E}[g_t]$:
$$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] = \mathbb{E}[g_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$
The finite geometric sum is:
$$\sum_{i=1}^t \beta_1^{t-i} = \sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$$

Substituting back:
$$\mathbb{E}[m_t] = \mathbb{E}[g_t] (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g_t] (1 - \beta_1^t)$$
Therefore, $m_t$ is biased towards 0 by factor $(1 - \beta_1^t)$.
Dividing by $(1 - \beta_1^t)$ yields the strictly unbiased estimator $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ where $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]$.

### 2. Derivation of Second-Moment Bias Correction
Analogously for $v_0 = 0$ with $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$:
$$\mathbb{E}[v_t] = \mathbb{E}[g_t^2] (1 - \beta_2^t) \implies \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### 3. What Happens if Bias Correction is Removed?
For initial steps ($t=1, \beta_2 = 0.999$):
$$1 - \beta_2^1 = 0.001 \implies v_1 \approx 0.001 g_1^2$$
Without bias correction, the effective step size is:
$$\Delta \theta_1 \approx \frac{\eta}{\sqrt{v_1} + \epsilon} m_1 = \frac{\eta}{\sqrt{0.001 g_1^2}} (0.1 g_1) = \frac{0.1}{\sqrt{0.001}} \eta = \sqrt{10} \eta \approx 3.16 \eta$$
More dangerously, if $\beta_1 = 0.9$ and $\beta_2 = 0.999$, $\frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}} = \frac{0.1}{\sqrt{0.001}} \approx 3.16$. In high dimensions with small gradients, $\sqrt{v_t} \approx 0$ causes the denominator to collapse to $\epsilon$, triggering massive gradient explosion during the first 10-50 iterations and causing parameters to diverge instantly.
        """
    },
    11: {
        "title": "2D Convex Quadratic Zig-Zagging & Condition Number",
        "question": "Construct a 2D convex quadratic function for which gradient descent exhibits severe zig-zagging. Explain the relationship between the Hessian's condition number and the shape of the optimization trajectory.",
        "latex_derivation": r"""
### 1. Construction of Ill-Conditioned 2D Quadratic Function
Define $f(x, y) = \frac{1}{2} (x^2 + 100 y^2)$.
Matrix form: $f(\theta) = \frac{1}{2} \theta^T H \theta$ with $\theta = \begin{bmatrix} x \\ y \end{bmatrix}$ and Hessian:
$$H = \begin{bmatrix} 1 & 0 \\ 0 & 100 \end{bmatrix}$$
Eigenvalues: $\lambda_1 = 1$, $\lambda_2 = 100$. Condition number:
$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{100}{1} = 100$$

### 2. Gradient and Trajectory Dynamics
Gradient: $\nabla f(x, y) = \begin{bmatrix} x \\ 100 y \end{bmatrix}$.
Update rule:
$$x^{(t+1)} = (1 - \eta) x^{(t)}$$
$$y^{(t+1)} = (1 - 100\eta) y^{(t)}$$

For stability along $y$, we require $|1 - 100\eta| < 1 \implies \eta < 0.02$.
Setting $\eta = 0.018$:
- Along $y$: $y^{(t+1)} = (1 - 1.8) y^{(t)} = -0.8 y^{(t)}$ (Oscillates sign every iteration with slow decay 0.8)
- Along $x$: $x^{(t+1)} = (1 - 0.018) x^{(t)} = 0.982 x^{(t)}$ (Creeps forward at 1.8% per step)

**Trajectory Shape:**
The parameter trajectory bounces back and forth across the steep $y$-axis walls while making agonizingly slow horizontal progress towards $x=0$, producing severe zig-zagging.
        """
    },
    12: {
        "title": "Generalization Gap: Adam vs SGD + Momentum",
        "question": "A neural network reaches almost zero training loss using Adam but performs significantly worse on unseen data than the same architecture trained with SGD + momentum. Give a technically rigorous explanation involving optimization dynamics, implicit regularization, noise in gradient estimates, and generalization.",
        "latex_derivation": r"""
### 1. Flat vs Sharp Minima (Hochreiter & Schmidhuber, Keskar et al.)
- **SGD + Momentum:** The isotropic gradient noise covariance matrix $\Sigma = \mathbb{E}[(g - \nabla J)(g - \nabla J)^T]$ acts as Brownian motion with temperature $T = \frac{\eta}{2B}$. This noise kicks parameters out of narrow, sharp valleys with large eigenvalues ($\operatorname{Tr}(H) \gg 0$) into broad, flat basins. Flat minima have low curvature, so domain shifts between train and test distributions $\Delta x$ cause negligible increase in test loss: $J_{\text{test}} \approx J_{\text{train}} + \frac{1}{2} \Delta\theta^T H \Delta\theta$.
- **Adam:** Coordinate-wise scaling by $\frac{1}{\sqrt{v_t}}$ rescales the geometry, dividing each coordinate's gradient by its individual standard deviation. This suppresses the natural escaping noise along sharp directions, allowing Adam to settle into sharp, non-generalizing local minima with high test error.

### 2. Non-Uniform Implicit Regularization
Wilson et al. (2017) proved that on overparameterized linear models, SGD finds the minimum $\ell_2$-norm solution ($\min \|w\|_2$), which maximizes margins. In contrast, adaptive methods like Adam find solutions with minimal $\ell_\infty$-norm, which is known to generalize poorly in presence of uninformative features.

### 3. Non-Decaying Second Moment Memory
In late training stages, Adam's $v_t$ retains historical gradient magnitudes, causing inappropriate scaling on sparse or rare gradient updates.
        """
    },
    13: {
        "title": "Production ML Optimization Pipeline & Data Leakage Prevention",
        "question": "Design the complete optimization pipeline for a production ML system (Raw Data -> Preprocessing -> Split -> Initialization -> Gradients -> Optimizer -> Scheduler -> Clipping -> Validation -> Early Stopping -> Final Evaluation).",
        "latex_derivation": r"""
### Complete Production Architecture & Leakage Prevention Framework

```
[ Raw Data Ingestion ]
         │
         ▼
[ Validation & Schema Check ] ── (Assert nulls, dtypes, range constraints)
         │
         ▼
[ Train / Validation / Test Split ] ── (CRITICAL: MUST split BEFORE any transform)
         │
 ┌───────┴───────────────────────┐
 ▼                               ▼
[ Fit Transformers on Train ]  [ Transform Val & Test via Train Params ]
 (Compute μ, σ, medians, encodings)
         │
         ▼
[ Feature Matrix Construction ]
         │
         ▼
[ Parameter Initialization (He/Xavier) ]
         │
         ▼ ◄────────────────────────────────────────┐ (Epoch Loop)
[ Forward Pass & Loss J(θ) ]                        │
         │                                          │
         ▼                                          │
[ Backward Pass: Gradient Computation ∇J ]          │
         │                                          │
         ▼                                          │
[ Gradient Clipping: min(1, c / ||g||) ]            │
         │                                          │
         ▼                                          │
[ Optimizer Step (AdamW / SGD+M) ]                  │
         │                                          │
         ▼                                          │
[ Learning Rate Scheduler Update η(t) ]             │
         │                                          │
         ▼                                          │
[ Validation Evaluation J_val(θ) ] ─────────────────┘
         │
         ▼
[ Early Stopping & Checkpoint Restore ]
         │
         ▼
[ Final Test Evaluation & Calibration ]
```

### Critical Data Leakage Traps:
1. **Target Leakage:** Fitting scalers, imputers, or encoders on the combined dataset before splitting.
2. **Temporal Leakage:** Random shuffling on time-series/transaction data instead of time-based splits.
3. **Data Snooping:** Tuning hyperparameters directly on the test set rather than hold-out validation folds.
        """
    },
    14: {
        "title": "Comprehensive Optimizer-Selection Decision Framework",
        "question": "Design an optimizer-selection framework for a new ML problem (Batch GD, SGD, Mini-Batch GD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW).",
        "latex_derivation": r"""
### Optimizer Selection Decision Matrix

```
                          [ New ML Optimization Problem ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         [ Small Dataset ]                               [ Large Dataset ]
         (N < 10,000, Fits in RAM)                       (N > 10,000, High Dimensional)
                 │                                               │
        ┌────────┴────────┐                              ┌───────┴───────┐
        ▼                 ▼                              ▼               ▼
[ Convex Quadratic ]  [ Highly Ill-Conditioned ]   [ Sparse NLP/Tabular ] [ Dense Vision/Deep Net ]
   Batch GD / L-BFGS     Momentum / NAG                AdaGrad / Adam           │
                                                                         ┌───────┴───────┐
                                                                         ▼               ▼
                                                                   [ Fast Prototyping ] [ Max Generalization ]
                                                                       Adam / AdamW       SGD + Momentum
```

### Mathematical Selection Criteria:
- **Batch GD / L-BFGS:** Best when $N$ is small, exact gradient is fast, and second-order Hessian approximation is tractable.
- **SGD with Momentum:** Best for vision and general deep networks where generalization and flat minima are top priority.
- **AdamW:** Universal default for Transformers, NLP, and complex loss landscapes with non-convex saddle points.
- **AdaGrad:** Sparse features (word embeddings, bag-of-words) where rare features need larger step sizes.
        """
    },
    15: {
        "title": "Autonomous Convergence-Diagnosis System Specification",
        "question": "Design a complete convergence-diagnosis system that receives only epoch, training_loss, validation_loss, gradient_norm, learning_rate, parameter_norm, and automatically identifies failure modes.",
        "latex_derivation": r"""
### Quantitative Threshold Decision Logic

| Detected Failure Mode | Quantitative Mathematical Signal Criteria | Actionable Remediation |
| :--- | :--- | :--- |
| **Learning-Rate Explosion** | $J_{\text{train}}^{(t)} \in \{\text{NaN}, \infty\} \lor J_{\text{train}}^{(t)} > 10^4 \cdot J_{\text{train}}^{(0)}$ | Reduce learning rate by $\times 10^{-1}$ to $\times 10^{-2}$. |
| **Gradient Explosion** | $\|\nabla_\theta J\|_2 > 10^3 \lor \frac{\|\nabla_\theta J\|_2^{(t)}}{\|\nabla_\theta J\|_2^{(t-1)}} > 10^2$ | Apply gradient clipping ($\|g\|_2 \le 1.0$) and weight decay. |
| **Vanishing Gradients** | $\|\nabla_\theta J\|_2 < 10^{-7} \land J_{\text{train}} > J_{\text{target}}$ | Switch activation to ReLU/GELU, use Residuals or Xavier init. |
| **Overfitting** | $J_{\text{val}}^{(t)} > 1.25 \min_{\tau \le t} J_{\text{val}}^{(\tau)} \land J_{\text{train}}^{(t)} \le \min_{\tau} J_{\text{train}}^{(\tau)}$ | Early stopping, increase L2 weight decay, dropout, or augment. |
| **Underfitting** | $\frac{J_{\text{train}}^{(0)} - J_{\text{train}}^{(t)}}{J_{\text{train}}^{(0)}} < 0.05$ after $t \ge 20$ | Increase model capacity, reduce regularization, increase LR. |
| **Poor Conditioning / Oscillation** | $\operatorname{SignFlipRate}(\Delta J) > 0.6 \land \operatorname{Std}(J_{\text{last 10}}) > 0.05$ | Apply feature normalization (StandardScaler), use Momentum. |
| **Plateauing** | $\frac{|J_{\text{train}}^{(t)} - J_{\text{train}}^{(t-10)}|}{J_{\text{train}}^{(t-10)}} < 10^{-4}$ | Learning rate decay (ReduceLROnPlateau / Cosine). |
        """
    },
    16: {
        "title": "Deep Analysis — The 10M Parameter Imbalanced Optimization Challenge",
        "question": "Deep Analysis of 10M parameter network on imbalanced data with Train Loss 0.018, Val Loss 0.092, ROC-AUC (Tr 0.998, Val 0.871), PR-AUC (Tr 0.991, Val 0.643), Grad Norm 0.0007, LR 1e-4, Param Norm 147.2.",
        "latex_derivation": r"""
### 1. Quantitative Metric Interpretation
- **Loss Gap ($0.018 \to 0.092$):** Train loss is $5.1\times$ lower than validation loss, signaling generalization divergence.
- **ROC-AUC (0.871) vs PR-AUC (0.643):** ROC-AUC is misleadingly inflated by the overwhelming majority negative class (True Negative Rate remains high even with false positives). PR-AUC evaluates Precision against Recall directly, exposing severe positive-class false discovery rate ($35.7\%$ precision deficit).
- **Gradient Norm (0.0007):** Indicates that the optimization is near a local stationary point ($\nabla J \approx 0$). It does **not** imply the optimum is flat or generalizing.
- **Parameter Norm (147.2):** Indicates moderate weight growth; combined with Adam, this confirms weights have settled without explosive drift.

### 2. Diagnosis: Primary Problem is Combination (G: C + E + F)
- **C (Overfitting):** Late-stage PR-AUC deterioration while training loss continues falling is textbook overfitting.
- **E (Optimizer-induced Generalization Bias):** Adam in late epochs settles into a sharp local minimum with non-uniform coordinate updates.
- **F (Severe Class Imbalance):** The massive divergence between ROC-AUC and PR-AUC is caused by unweighted loss prioritizing the negative majority.

### 3. Revised Optimization & Ablation Protocol
1. Replace Adam with **AdamW** (decoupled weight decay $\lambda = 0.01$) or **SGD + Momentum** ($\beta=0.9, \eta_0=0.01$ with Cosine Annealing).
2. Incorporate **Focal Loss** ($\gamma=2.0, \alpha=0.25$) or Cost-Sensitive Effective Class Weighting ($w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$).
3. Evaluate statistical significance via 5-Fold Stratified Cross-Validation paired with Wilcoxon Signed-Rank Test ($p < 0.01$).
        """
    }
}
