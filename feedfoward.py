import numpy as np
from activations import activation_backward, activation_forward


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
        self.a1 = activation_forward(self.z1, self.activation)
        self.out = self.a1 @ self.W2 + self.b2

        return self.out

    def backward(self, d_out):
        # d_out: (batch, seq_len, embed_dim)
        if d_out.shape[-1] != self.embed_dim:
            raise ValueError("last dimension of d_out must match embed_dim")

        self.dW2 = np.einsum("bth,bte->he", self.a1, d_out)
        self.db2 = np.sum(d_out, axis=(0, 1))

        da1 = d_out @ self.W2.T
        dz1 = da1 * activation_backward(self.z1, self.activation)

        self.dW1 = np.einsum("bte,bth->eh", self.x, dz1)
        self.db1 = np.sum(dz1, axis=(0, 1))

        d_x = dz1 @ self.W1.T
        return d_x

    def step(self, lr=1e-3):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

  
