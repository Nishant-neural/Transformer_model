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

        self.WQ = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.WK = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.WV = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

         # out = weights @v for one head batch ,seq_len , d_v 
         #  out(concatenated) =  batch , seq len , embed-dim

        self.WO = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.dWO = np.zeros_like(self.WO)
        
        self.att = ScaledDotProductAttention()

    def forward(self, X_Q, X_K, X_V, mask=None):
    # X_Q: (B, Tq, E) T = seq_len  E = embed_dim
    # X_K: (B, Tk, E)
    # X_V: (B, Tk, E)
        self.X_Q = X_Q
        self.X_K = X_K
        self.X_V = X_V

        B, Tq, E = X_Q.shape
        _, Tk, _ = X_K.shape

        self.Q_proj = X_Q @ self.WQ
        self.K_proj = X_K @ self.WK
        self.V_proj = X_V @ self.WV

        self.Q = self.Q_proj.reshape(B, Tq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        self.K = self.K_proj.reshape(B, Tk, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        self.V = self.V_proj.reshape(B, Tk, self.num_heads, self.d_v).transpose(0, 2, 1, 3)

    # Q, K, V are now:   (B, H, T, D)
   
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
         # (embed_dim, embed_dim) = WQ

        self.dQ_proj = self.dQ.transpose(0, 2, 1, 3).reshape(self.X_Q.shape[0], self.X_Q.shape[1], self.embed_dim)
        self.dK_proj = self.dK.transpose(0, 2, 1, 3).reshape(self.X_K.shape[0], self.X_K.shape[1], self.embed_dim)
        self.dV_proj = self.dV.transpose(0, 2, 1, 3).reshape(self.X_V.shape[0], self.X_V.shape[1], self.embed_dim)

        self.dWQ = np.einsum("bte,btf->ef", self.X_Q, self.dQ_proj)
        self.dWK = np.einsum("bte,btf->ef", self.X_K, self.dK_proj)
        self.dWV = np.einsum("bte,btf->ef", self.X_V, self.dV_proj)

        self.dX_Q = self.dQ_proj @ self.WQ.T
        self.dX_K = self.dK_proj @ self.WK.T
        self.dX_V = self.dV_proj @ self.WV.T

        return self.dX_Q, self.dX_K, self.dX_V
       
