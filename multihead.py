from attention import ScaledDotProductAttention 
import numpy as np

class MultiHeadAttention :
    def __init__(self, embed_dim, num_heads):
        # error 
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads ")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim / num_heads
        self.d_v =  embed_dim / num_heads

        self.WQ = np.random.randn(num_heads, embed_dim, self.d_k) / np.sqrt(embed_dim)
        self.WK = np.random.randn(num_heads, embed_dim, self.d_k) / np.sqrt(embed_dim)
        self.WV = np.random.randn(num_heads, embed_dim, self.d_v) / np.sqrt(embed_dim)

         # out = weights @v for one head batch ,seq_len , d_v 
         #  out(concatenated) =  batch , seq len , concat_dim

        concat_dim = num_heads * self.d_v
        self.WO = np.random.randn(concat_dim, embed_dim) / np.sqrt(concat_dim)
        self.dWO = np.zeros_like(self.WO)

    def forward(self, X_Q, X_K, X_V, mask=None):
    # X_Q: (B, Tq, E)
    # X_K: (B, Tk, E)
    # X_V: (B, Tk, E)

        Q = np.einsum("bte,hed->bhtd", X_Q, self.WQ)
        K = np.einsum("bte,hed->bhtd", X_K, self.WK)
        V = np.einsum("bte,hed->bhtd", X_V, self.WV)

    # Q, K, V are now:
    # (B, H, T, D)
        att = ScaledDotProductAttention()
        head_out, weights = att.scaled_dot_product_attention(Q, K, V, mask)

    # head_out: (B, H, Tq, d_v)

        head_out = np.transpose(head_out, (0, 2, 1, 3))
    # (B, Tq, H, d_v)

        B, Tq, H, Dv = head_out.shape
        concat = head_out.reshape(B, Tq, H * Dv)

        out = concat @ self.WO
        return out, weights
