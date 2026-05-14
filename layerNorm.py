import numpy as np


class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = eps

        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x):
        # x: (..., embed_dim)
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mean) * self.std_inv

        return self.gamma * self.x_norm + self.beta

    def backward(self, d_out):
        # d_out: (..., embed_dim)
        axes = tuple(range(d_out.ndim - 1))

        self.dgamma = np.sum(d_out * self.x_norm, axis=axes)
        self.dbeta = np.sum(d_out, axis=axes)

        dx_norm = d_out * self.gamma
        dim = self.embed_dim

        dx = (
            (1.0 / dim)
            * self.std_inv
            * (
                dim * dx_norm
                - np.sum(dx_norm, axis=-1, keepdims=True)
                - self.x_norm * np.sum(dx_norm * self.x_norm, axis=-1, keepdims=True)
            )
        )

        return dx

    def step(self, lr=1e-3):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta
