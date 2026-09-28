"""
Gradient Descent Optimization Engine from Scratch
=================================================
This module contains pure NumPy vectorized implementations of:
1. Batch Gradient Descent (BGD)
2. Stochastic Gradient Descent (SGD)
3. Mini-Batch Gradient Descent (MBGD)
4. Momentum Gradient Descent
5. Nesterov Accelerated Gradient (NAG)
6. AdaGrad (Adaptive Gradient Algorithm)
7. RMSProp (Root Mean Square Propagation)
8. Adam (Adaptive Moment Estimation with Bias Correction)
9. AdamW (Decoupled Weight Decay Adam)

Supported Models:
- Linear Regression (Mean Squared Error Loss)
- Logistic Regression (Binary Cross-Entropy / Log Loss) with L1/L2 Regularization

Mathematical Derivations:
------------------------
For Logistic Regression with parameter vector $w \in \mathbb{R}^d$, bias $b \in \mathbb{R}$,
hypothesis $\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-(w^T x_i + b)}}$:

Cost Function (Binary Cross-Entropy with L2 regularization $\frac{\lambda}{2} \|w\|_2^2$):
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right] + \frac{\lambda}{2m} \|w\|_2^2$$

Gradients:
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)$$
"""

import numpy as np
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable


class OptimizerType(str, Enum):
    BATCH_GD = "Batch Gradient Descent (BGD)"
    SGD = "Stochastic Gradient Descent (SGD)"
    MINI_BATCH_GD = "Mini-Batch Gradient Descent (MBGD)"
    MOMENTUM = "Momentum"
    NAG = "Nesterov Accelerated Gradient (NAG)"
    ADAGRAD = "AdaGrad"
    RMSPROP = "RMSProp"
    ADAM = "Adam"
    ADAMW = "AdamW"


class LRScheduler:
    """
    Learning Rate Scheduling strategies.
    """
    def __init__(self, mode: str = "constant", initial_lr: float = 0.01, decay_rate: float = 0.95, step_size: int = 10, T_max: int = 100):
        self.mode = mode.lower()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.step_size = step_size
        self.T_max = T_max

    def get_lr(self, epoch: int) -> float:
        if self.mode == "constant":
            return self.initial_lr
        elif self.mode == "step":
            return self.initial_lr * (self.decay_rate ** (epoch // self.step_size))
        elif self.mode == "exponential":
            return self.initial_lr * (self.decay_rate ** epoch)
        elif self.mode == "cosine":
            return 0.5 * self.initial_lr * (1.0 + np.cos(np.pi * epoch / max(1, self.T_max)))
        elif self.mode == "inverse_time":
            return self.initial_lr / (1.0 + self.decay_rate * epoch)
        else:
            return self.initial_lr


class BaseGradientDescent:
    """
    Base class providing optimizer update mechanisms and training infrastructure.
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 100,
        batch_size: int = 32,
        optimizer: OptimizerType = OptimizerType.ADAM,
        l2_lambda: float = 0.0,
        l1_lambda: float = 0.0,
        momentum_beta: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        gradient_clip_norm: Optional[float] = None,
        lr_scheduler_mode: str = "constant",
        early_stopping_patience: Optional[int] = 10,
        early_stopping_min_delta: float = 1e-5,
        random_state: int = 42
    ):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.momentum_beta = momentum_beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gradient_clip_norm = gradient_clip_norm
        self.lr_scheduler = LRScheduler(mode=lr_scheduler_mode, initial_lr=learning_rate, T_max=max_epochs)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.random_state = random_state

        # Parameters
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

        # Optimization states
        self.v_w = None  # Velocity / first moment
        self.v_b = 0.0
        self.s_w = None  # Second moment / squared gradients
        self.s_b = 0.0
        self.t_step = 0  # Global optimizer step counter

        # History tracking
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "gradient_norm": [],
            "parameter_norm": [],
            "learning_rate": [],
            "epoch_time_ms": []
        }
        self.trajectory: List[Tuple[np.ndarray, float]] = []

    def _initialize_optimizer_state(self, n_features: int) -> None:
        np.random.seed(self.random_state)
        # Xavier/He normal initialization
        limit = np.sqrt(2.0 / n_features) if n_features > 0 else 0.01
        self.w = np.random.randn(n_features) * limit
        self.b = 0.0

        self.v_w = np.zeros(n_features)
        self.v_b = 0.0
        self.s_w = np.zeros(n_features)
        self.s_b = 0.0
        self.t_step = 0

    def _clip_gradients(self, grad_w: np.ndarray, grad_b: float) -> Tuple[np.ndarray, float, float]:
        grad_norm = np.sqrt(np.sum(grad_w ** 2) + grad_b ** 2)
        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            if grad_norm > self.gradient_clip_norm:
                scale = self.gradient_clip_norm / (grad_norm + 1e-12)
                grad_w = grad_w * scale
                grad_b = grad_b * scale
        return grad_w, grad_b, float(grad_norm)

    def _apply_optimizer_step(self, grad_w: np.ndarray, grad_b: float, current_lr: float) -> None:
        """
        Executes parameter updates according to the selected mathematical optimization algorithm.
        """
        self.t_step += 1
        opt = self.optimizer

        if opt == OptimizerType.BATCH_GD or opt == OptimizerType.SGD or opt == OptimizerType.MINI_BATCH_GD:
            # Vanilla Gradient Descent update: theta_{t+1} = theta_t - eta * grad
            self.w -= current_lr * grad_w
            self.b -= current_lr * grad_b

        elif opt == OptimizerType.MOMENTUM:
            # Momentum update:
            # v_{t+1} = beta * v_t + eta * grad
            # theta_{t+1} = theta_t - v_{t+1}
            self.v_w = self.momentum_beta * self.v_w + current_lr * grad_w
            self.v_b = self.momentum_beta * self.v_b + current_lr * grad_b
            self.w -= self.v_w
            self.b -= self.v_b

        elif opt == OptimizerType.NAG:
            # Nesterov update evaluated at lookahead
            self.v_w = self.momentum_beta * self.v_w + current_lr * grad_w
            self.v_b = self.momentum_beta * self.v_b + current_lr * grad_b
            self.w -= self.v_w
            self.b -= self.v_b

        elif opt == OptimizerType.ADAGRAD:
            # AdaGrad update:
            # G_{t+1} = G_t + grad^2
            # theta_{t+1} = theta_t - (eta / (sqrt(G_{t+1}) + eps)) * grad
            self.s_w += grad_w ** 2
            self.s_b += grad_b ** 2
            self.w -= (current_lr / (np.sqrt(self.s_w) + self.epsilon)) * grad_w
            self.b -= (current_lr / (np.sqrt(self.s_b) + self.epsilon)) * grad_b

        elif opt == OptimizerType.RMSPROP:
            # RMSProp update:
            # v_{t+1} = beta * v_t + (1 - beta) * grad^2
            # theta_{t+1} = theta_t - (eta / (sqrt(v_{t+1}) + eps)) * grad
            self.s_w = self.beta2 * self.s_w + (1.0 - self.beta2) * (grad_w ** 2)
            self.s_b = self.beta2 * self.s_b + (1.0 - self.beta2) * (grad_b ** 2)
            self.w -= (current_lr / (np.sqrt(self.s_w) + self.epsilon)) * grad_w
            self.b -= (current_lr / (np.sqrt(self.s_b) + self.epsilon)) * grad_b

        elif opt == OptimizerType.ADAM:
            # Adam update with first/second moment bias corrections:
            # m_{t+1} = beta1 * m_t + (1 - beta1) * grad
            # v_{t+1} = beta2 * v_t + (1 - beta2) * grad^2
            # m_hat = m_{t+1} / (1 - beta1^t)
            # v_hat = v_{t+1} / (1 - beta2^t)
            # theta_{t+1} = theta_t - (eta / (sqrt(v_hat) + eps)) * m_hat
            self.v_w = self.beta1 * self.v_w + (1.0 - self.beta1) * grad_w
            self.v_b = self.beta1 * self.v_b + (1.0 - self.beta1) * grad_b
            self.s_w = self.beta2 * self.s_w + (1.0 - self.beta2) * (grad_w ** 2)
            self.s_b = self.beta2 * self.s_b + (1.0 - self.beta2) * (grad_b ** 2)

            m_hat_w = self.v_w / (1.0 - self.beta1 ** self.t_step)
            m_hat_b = self.v_b / (1.0 - self.beta1 ** self.t_step)
            v_hat_w = self.s_w / (1.0 - self.beta2 ** self.t_step)
            v_hat_b = self.s_b / (1.0 - self.beta2 ** self.t_step)

            self.w -= (current_lr / (np.sqrt(v_hat_w) + self.epsilon)) * m_hat_w
            self.b -= (current_lr / (np.sqrt(v_hat_b) + self.epsilon)) * m_hat_b

        elif opt == OptimizerType.ADAMW:
            # AdamW: Decoupled weight decay:
            # w_{t+1} = w_t - eta * lambda * w_t - (eta / (sqrt(v_hat) + eps)) * m_hat
            self.v_w = self.beta1 * self.v_w + (1.0 - self.beta1) * grad_w
            self.v_b = self.beta1 * self.v_b + (1.0 - self.beta1) * grad_b
            self.s_w = self.beta2 * self.s_w + (1.0 - self.beta2) * (grad_w ** 2)
            self.s_b = self.beta2 * self.s_b + (1.0 - self.beta2) * (grad_b ** 2)

            m_hat_w = self.v_w / (1.0 - self.beta1 ** self.t_step)
            m_hat_b = self.v_b / (1.0 - self.beta1 ** self.t_step)
            v_hat_w = self.s_w / (1.0 - self.beta2 ** self.t_step)
            v_hat_b = self.s_b / (1.0 - self.beta2 ** self.t_step)

            # Decoupled decay on weights
            self.w = self.w * (1.0 - current_lr * self.l2_lambda) - (current_lr / (np.sqrt(v_hat_w) + self.epsilon)) * m_hat_w
            self.b -= (current_lr / (np.sqrt(v_hat_b) + self.epsilon)) * m_hat_b


class LogisticRegressionGD(BaseGradientDescent):
    """
    Binary Logistic Regression trained with arbitrary Gradient Descent Optimizers.
    Target y \in {0, 1}.
    """
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid function
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def _compute_loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, float]:
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        y_hat = self._sigmoid(z)
        
        # Binary Cross-Entropy with clipping to avoid log(0)
        eps = 1e-15
        y_hat_safe = np.clip(y_hat, eps, 1.0 - eps)
        bce_loss = -np.mean(y * np.log(y_hat_safe) + (1.0 - y) * np.log(1.0 - y_hat_safe))
        
        # Regularization penalties
        reg_loss = 0.0
        reg_grad_w = np.zeros_like(self.w)
        if self.l2_lambda > 0 and self.optimizer != OptimizerType.ADAMW:
            reg_loss += (0.5 * self.l2_lambda / m) * np.sum(self.w ** 2)
            reg_grad_w += (self.l2_lambda / m) * self.w
        if self.l1_lambda > 0:
            reg_loss += (self.l1_lambda / m) * np.sum(np.abs(self.w))
            reg_grad_w += (self.l1_lambda / m) * np.sign(self.w)

        total_loss = float(bce_loss + reg_loss)
        
        # Gradients: dJ/dw = (1/m) X^T (y_hat - y) + reg
        error = y_hat - y
        grad_w = (np.dot(X.T, error) / m) + reg_grad_w
        grad_b = float(np.mean(error))
        
        return total_loss, grad_w, grad_b

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose_interval: int = 10
    ) -> "LogisticRegressionGD":
        N, d = X_train.shape
        self._initialize_optimizer_state(d)
        
        best_val_loss = float("inf")
        patience_counter = 0

        effective_batch_size = (
            N if self.optimizer == OptimizerType.BATCH_GD else
            1 if self.optimizer == OptimizerType.SGD else
            min(self.batch_size, N)
        )

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            current_lr = self.lr_scheduler.get_lr(epoch)
            
            # Shuffle training data
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_grad_norms = []

            # Mini-batch loop
            for start_idx in range(0, N, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                if self.optimizer == OptimizerType.NAG:
                    # Lookahead position: w_lookahead = w - beta * v_w
                    w_orig = self.w.copy()
                    b_orig = self.b
                    self.w -= self.momentum_beta * self.v_w
                    self.b -= self.momentum_beta * self.v_b
                    _, grad_w, grad_b = self._compute_loss_and_grads(X_batch, y_batch)
                    self.w = w_orig
                    self.b = b_orig
                else:
                    _, grad_w, grad_b = self._compute_loss_and_grads(X_batch, y_batch)

                grad_w, grad_b, g_norm = self._clip_gradients(grad_w, grad_b)
                epoch_grad_norms.append(g_norm)
                self._apply_optimizer_step(grad_w, grad_b, current_lr)

            t1 = time.perf_counter()
            epoch_time_ms = (t1 - t0) * 1000.0

            # Evaluate epoch training & validation loss
            train_loss, _, _ = self._compute_loss_and_grads(X_train, y_train)
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss, _, _ = self._compute_loss_and_grads(X_val, y_val)

            param_norm = float(np.sqrt(np.sum(self.w ** 2) + self.b ** 2))
            mean_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss if val_loss is not None else np.nan)
            self.history["gradient_norm"].append(mean_grad_norm)
            self.history["parameter_norm"].append(param_norm)
            self.history["learning_rate"].append(current_lr)
            self.history["epoch_time_ms"].append(epoch_time_ms)
            self.trajectory.append((self.w.copy(), self.b))

            # Early stopping check
            if val_loss is not None and self.early_stopping_patience is not None:
                if val_loss < best_val_loss - self.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


class LinearRegressionGD(BaseGradientDescent):
    """
    Multivariate Linear Regression trained with Gradient Descent.
    Cost Function: Mean Squared Error (MSE):
    J(w, b) = (1 / 2m) || X w + b 1 - y ||_2^2
    """
    def _compute_loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, float]:
        m = X.shape[0]
        y_hat = np.dot(X, self.w) + self.b
        error = y_hat - y
        
        mse_loss = float(0.5 * np.mean(error ** 2))
        
        reg_loss = 0.0
        reg_grad_w = np.zeros_like(self.w)
        if self.l2_lambda > 0 and self.optimizer != OptimizerType.ADAMW:
            reg_loss += (0.5 * self.l2_lambda / m) * np.sum(self.w ** 2)
            reg_grad_w += (self.l2_lambda / m) * self.w

        total_loss = mse_loss + reg_loss
        grad_w = (np.dot(X.T, error) / m) + reg_grad_w
        grad_b = float(np.mean(error))
        
        return total_loss, grad_w, grad_b

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "LinearRegressionGD":
        N, d = X_train.shape
        self._initialize_optimizer_state(d)
        effective_batch_size = (
            N if self.optimizer == OptimizerType.BATCH_GD else
            1 if self.optimizer == OptimizerType.SGD else
            min(self.batch_size, N)
        )

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            current_lr = self.lr_scheduler.get_lr(epoch)
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_grad_norms = []
            for start_idx in range(0, N, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                _, grad_w, grad_b = self._compute_loss_and_grads(X_batch, y_batch)
                grad_w, grad_b, g_norm = self._clip_gradients(grad_w, grad_b)
                epoch_grad_norms.append(g_norm)
                self._apply_optimizer_step(grad_w, grad_b, current_lr)

            t1 = time.perf_counter()
            train_loss, _, _ = self._compute_loss_and_grads(X_train, y_train)
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss, _, _ = self._compute_loss_and_grads(X_val, y_val)

            param_norm = float(np.sqrt(np.sum(self.w ** 2) + self.b ** 2))
            mean_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss if val_loss is not None else np.nan)
            self.history["gradient_norm"].append(mean_grad_norm)
            self.history["parameter_norm"].append(param_norm)
            self.history["learning_rate"].append(current_lr)
            self.history["epoch_time_ms"].append((t1 - t0) * 1000.0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b
