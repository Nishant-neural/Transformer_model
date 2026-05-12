from attention import ScaledDotProductAttention 
import numpy as np

class MultiHeadAttention :
    def __init__(self, embed_dim, num_heads):
        # error 
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads ")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.d_v =  embed_dim // num_heads

        self.WQ = np.random.randn(num_heads, embed_dim, self.d_k) / np.sqrt(embed_dim)
        self.WK = np.random.randn(num_heads, embed_dim, self.d_k) / np.sqrt(embed_dim)
        self.WV = np.random.randn(num_heads, embed_dim, self.d_v) / np.sqrt(embed_dim)

         # out = weights @v for one head batch ,seq_len , d_v 
         #  out(concatenated) =  batch , seq len , embed-dim

        self.WO = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.dWO = np.zeros_like(self.WO)

    def forward(self, X_Q, X_K, X_V, mask=None):
    # X_Q: (B, Tq, E) T = seq_len  E = embed_dim
    # X_K: (B, Tk, E)
    # X_V: (B, Tk, E)
        self.X_Q = X_Q
        self.X_K = X_K
        self.X_V = X_V

        self.Q = np.einsum("bte,hed->bhtd", X_Q, self.WQ)
        self.K = np.einsum("bte,hed->bhtd", X_K, self.WK)
        self.V = np.einsum("bte,hed->bhtd", X_V, self.WV)

    # Q, K, V are now:   (B, H, T, D)
   
        self.att = ScaledDotProductAttention()
        self.head_out, self.weights = self.att.forward(self.Q, self.K, self.V, mask)

    # head_out: (B, H, Tq, d_v)

        self.head_out = np.transpose(self.head_out, (0, 2, 1, 3))
    # head _out now:(B, Tq, H, d_v)

        B, Tq, H, Dv = self.head_out.shape
        self.concat = self.head_out.reshape(B, Tq, H * Dv)

        self.out = self.concat @ self.WO
        return self.out, self.weights

    def backward(self , d_out):

        B, Tq, H, Dv = self.head_out.shape

    # out = concat @ WO
        self.dWO = np.einsum("bte,btf->ef", self.concat, d_out)
        self.dconcat = d_out @ self.WO.T

  # concat = B Tq embed_dim
        # WO = embed , embed
        # out = B Tq embed_dim
    # concat = head_out.reshape(B, Tq, H * Dv)

        self.d_head_out = self.dconcat.reshape(B, Tq, H, Dv)
        self.d_att_out = np.transpose(self.d_head_out, (0, 2, 1, 3))

        self.dV ,self.dQ , self.dK = self.att.backward(self.d_att_out)

        # concat = B Tq embed_dim
        # WO = embed , embed
        # out = B Tq embed_dim

         # Q, K, V are now:   (B, H, T, D)
         # X_Q: (B, Tq, E) T = seq_len  E = embed_dim
         # (num_heads, embed_dim, self.d_k = WQ

        self.dWQ = np.einsum("bte,bhtd->hed", self.X_Q, self.dQ)
        self.dWK = np.einsum("bte,bhtd->hed", self.X_K, self.dK)
        self.dWV = np.einsum("bte,bhtd->hed", self.X_V, self.dV)

        self.dX_Q = np.einsum("bhtd,hed->bte", self.dQ, self.WQ)
        self.dX_K = np.einsum("bhtd,hed->bte", self.dK, self.WK)
        self.dX_V = np.einsum("bhtd,hed->bte", self.dV, self.WV)

        return self.dX_Q, self.dX_K, self.dX_V
       

