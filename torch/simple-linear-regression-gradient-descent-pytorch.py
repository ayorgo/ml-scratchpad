from typing import Callable
import torch
import torch.nn as nn
from functools import partial


def linear(
    x: torch.FloatTensor, w: torch.float32, b: torch.float32, f: Callable
) -> torch.FloatTensor:
    return f(x) * w + b


def descent(
    x: torch.FloatTensor,
    y: torch.FloatTensor,
    model: Callable,
    loss: Callable,
    learning_rate: torch.float32 = 0.1,
    precision: torch.float32 = 0.001,
) -> tuple:
    w = torch.tensor(1, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(1, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.SGD([w, b], lr=learning_rate)

    # Forward propagation
    model_pred = model(x, w, b)
    loss_ = loss(y, model_pred)

    epoch = 0
    while loss_ > precision:
        # Backpropagation
        loss_.backward()  # calculate gradients
        optimizer.step()  # update weights
        optimizer.zero_grad()

        # Forward propagation
        model_pred = model(x, w, b)
        loss_ = loss(model_pred, y)

        # Logging
        epoch += 1
        if epoch % 10 == 0:
            print(f"epoch {epoch}, w: {w:.3f}, b: {b:.3f}, loss: {loss_:.3f}")
    return w, b


# ------
# Linear
# ------

# Data
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = 2 * x + 3

# Model
f = lambda x: x
model = partial(linear, f=f)
loss = nn.MSELoss()

# Prediction
w, b = descent(x, y, model, loss)
print(f"Predicted w: {w:.3f}, b: {b:.3f}\n")


# ---------
# Quadratic
# ---------

# Data
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = 2 * x**2 + 3

# Model
f = lambda x: x**2
model = partial(linear, f=f)
loss = nn.MSELoss()

# Prediction
w, b = descent(x, y, model, loss, learning_rate=0.01)
print(f"Predicted w: {w:.3f}, b: {b:.3f}\n")
