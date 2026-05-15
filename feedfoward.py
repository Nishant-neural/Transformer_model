import numpy as np


class FeedForward:
    def __init__(self, embed_dim, hidden_dim, activation="relu"):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.W1 = np.random.randn(embed_dim, hidden_dim) / np.sqrt(embed_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(embed_dim)

        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        if x.shape[-1] != self.embed_dim:
            raise ValueError("last dimension of x must match embed_dim")

        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self._activation_forward(self.z1)
        self.out = self.a1 @ self.W2 + self.b2

        return self.out

    def step(self, lr=1e-3):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def _activation_forward(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        if self.activation == "gelu":
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

        raise ValueError("activation must be 'relu' or 'gelu'")

  