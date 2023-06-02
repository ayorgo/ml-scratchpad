from typing import Callable
from functools import partial
import numpy as np


def linear(x: np.ndarray, w: np.float32, b: np.float32, f: Callable) -> np.ndarray:
    return f(x) * w + b


def loss(model_pred: np.ndarray, y: np.ndarray) -> np.float32:
    # MSE
    return ((model_pred - y) ** 2).mean()


def grad(
    model_pred: np.ndarray, x: np.ndarray, y: np.ndarray, f: Callable
) -> np.float32:
    # dl/dw = dl/dz * dz/dw: chain rule
    # d(1/N*sum((f(x)*w + b - y)**2)/dw = 1/N*sum(2*f(x)*(f(x)*w + b - y))
    dw = (2 * f(x) * (model_pred - y)).mean()

    # dl/db = dl/dz * dz/db: chain rule
    # d(1/N*sum((f(x)*w + b - y)**2)/db = 1/N*sum(2*(f(x)*w + b - y))
    db = (2 * (model_pred - y)).mean()
    return dw, db


def descent(
    x: np.ndarray,
    y: np.ndarray,
    model: Callable,
    gradient: Callable,
    loss: Callable,
    learning_rate: np.float32 = 0.1,
    precision: np.float32 = 0.001,
) -> tuple:
    w = 1
    b = 1

    # Forward propagation
    model_pred = model(x, w, b)
    loss_ = loss(model_pred, y)

    epoch = 0
    while loss_ > precision:
        # Backpropagation
        dw, db = gradient(model_pred, x, y)
        w -= learning_rate * dw
        b -= learning_rate * db

        # Forward propagation
        model_pred = model(x, w, b)
        loss_ = loss(model_pred, y)

        # Logging
        epoch += 1
        if epoch % 10 == 0:
            print(
                f"epoch {epoch}, dw: {dw:.3f}, w: {w:.3f}, db: {db:.3f}, b: {b:.3f}, loss: {loss_:.3f}"
            )
    return w, b


# ------
# Linear
# ------

# Data
x = np.array([1, 2, 3], dtype=np.float32)
y = 2 * x + 3

# Model
f = lambda x: x
model = partial(linear, f=f)
gradient = partial(grad, f=f)

# Prediction
w, b = descent(x, y, model, gradient, loss)
print(f"Predicted w: {w:.3f}, b: {b:.3f}\n")


# ---------
# Quadratic
# ---------

# Data
x = np.array([1, 2, 3], dtype=np.float32)
y = 2 * x**2 + 3

# Model
f = lambda x: x**2
model = partial(linear, f=f)
gradient = partial(grad, f=f)

# Prediction
w, b = descent(x, y, model, gradient, loss, learning_rate=0.01)
print(f"Predicted w: {w:.3f}, b: {b:.3f}\n")
