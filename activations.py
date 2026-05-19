import numpy as np


def activation_forward(x, activation):
    if activation == "relu":
        return np.maximum(0, x)
    if activation == "gelu":
        return 0.5 * x * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
        )

    raise ValueError("activation must be 'relu' or 'gelu'")


def activation_backward(x, activation):
    if activation == "relu":
        return (x > 0).astype(x.dtype)
    if activation == "gelu":
        tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        tanh_val = np.tanh(tanh_arg)
        sech2 = 1.0 - tanh_val**2
        inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
        return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_grad

    raise ValueError("activation must be 'relu' or 'gelu'")
