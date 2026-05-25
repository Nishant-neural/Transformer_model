import numpy as np

from attention import ScaledDotProductAttention
from pos_encoding import InputEmbedding
from softmax import softmax_forward
from transformer_encoder import Encoder


class MLMHead:
    def __init__(self, embed_dim, vocab_size):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.W = np.random.randn(embed_dim, vocab_size) / np.sqrt(embed_dim)
        self.b = np.zeros(vocab_size)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):

        # x: (batch, seq_len, embed_dim)

        self.x = x
        self.logits = x @ self.W + self.b
        return self.logits

    def backward(self, d_logits):
        self.dW = np.einsum("bte,btv->ev", self.x, d_logits)
        self.db = np.sum(d_logits, axis=(0, 1))
        return d_logits @ self.W.T


    def step(self, lr=1e-3):
        self.W -= lr * self.dW
        self.b -= lr * self.db



