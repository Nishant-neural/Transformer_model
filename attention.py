from embeddings import Embedding
import numpy as np
from softmax import softmax_forward

class ScaledDotProductAttention:
    def __init__ (self , embed_dim , d_k , d_v) :


        self.WQ = np.random.randn(embed_dim, d_k) / np.sqrt(embed_dim)

        self.dWQ = np.zeros_like(self.WQ)
       

        self.WK = np.random.randn(embed_dim , d_k)/ np.sqrt(embed_dim)
        self.dWK = np.zeros_like(self.WK)
       

        self.WV =  np.random.randn(embed_dim, d_v) / np.sqrt(embed_dim)
        self.dWV = np.zeros_like(self.WV)
       


    def forward(self, X_Q , X_K , mask = None):

        # X_q =  batch , seq_len_q , embed_dim
        # X_K = batch , seq_len_k , embed_dim
        self.K = X_K @ self.WK
        self.V = X_K @self.WV
        self.Q = X_Q @ self.WQ

        batch, seq_len_q, d_k =  self.Q.shape
        self.scores = self.Q @ self.K.transpose(0, 2, 1)
        self.scores = self.scores / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        self.scores =  softmax_forward(self.scores)
        self.attention = self.scores @ self.V


