import numpy as np


def activation_forward(x, activation):
    if activation == "relu":
        return np.maximum(0, x)
    if activation == "gelu":
        return 0.5 * x * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
        )

    raise ValueError("activation must be 'relu' or 'gelu'")


