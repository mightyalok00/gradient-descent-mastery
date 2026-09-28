import numpy as np

from src.gradient_descent import LogisticRegressionGD, OptimizerType


def test_logistic_regression_training_reduces_loss():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 3))
    true_w = np.array([1.5, -2.0, 0.8])
    logits = X @ true_w
    y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(float)

    model = LogisticRegressionGD(
        learning_rate=0.03,
        max_epochs=30,
        batch_size=32,
        optimizer=OptimizerType.ADAM,
        random_state=42,
    )
    model.fit(X, y)

    assert len(model.history["train_loss"]) == 30
    assert np.isfinite(model.history["train_loss"]).all()
    assert model.history["train_loss"][-1] < model.history["train_loss"][0]


def test_predict_proba_returns_valid_probabilities():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = LogisticRegressionGD(
        learning_rate=0.05,
        max_epochs=20,
        batch_size=2,
        optimizer=OptimizerType.MINI_BATCH_GD,
        random_state=42,
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (4,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
