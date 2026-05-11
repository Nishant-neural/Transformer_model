from attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention :
    def __init__(self, embed_dim, num_heads, d_k=None, d_v=None):
        if embed_dim % num_heads != 0 and (d_k is None or d_v is None):
            raise ValueError("embed_dim must be divisible by num_heads when d_k/d_v are not provided")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else embed_dim // num_heads
        self.d_v = d_v if d_v is not None else embed_dim // num_heads

        self.heads = [
            ScaledDotProductAttention(embed_dim, self.d_k, self.d_v)
            for _ in range(num_heads)
        ]

        concat_dim = num_heads * self.d_v
        self.WO = np.random.randn(concat_dim, embed_dim) / np.sqrt(concat_dim)
        self.dWO = np.zeros_like(self.WO)

    def forward(self, X_Q, X_K, X_V, mask=None):
        self.X_Q = X_Q
        self.X_K = X_K
        self.X_V = X_V

        self.head_outputs = []
        self.attention_weights = []

        for head in self.heads:
            head_out, weights = head.forward(X_Q, X_K, X_V, mask)
            self.head_outputs.append(head_out)
            self.attention_weights.append(weights)

        self.concat = np.concatenate(self.head_outputs, axis=-1)
        self.out = self.concat @ self.WO

        return self.out, self.attention_weights

   