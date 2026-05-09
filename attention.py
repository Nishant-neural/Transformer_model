from embeddings import Embedding
import numpy as np
from softmax import softmax_forward
class Query_key:
    def __init__ (self , X_Q , X_K, d_k ) :

        # X_q =  batch , seq_len_q , embed_dim
        # X_K + batch , seq_len_k , embed_dim

        batch , seq_len_q , embed_dim = X_Q.shape
        self.WQ = np.random.randn(embed_dim, d_k) / np.sqrt(embed_dim)

        self.dWQ = np.zeros_like(self.WQ)
        self.Q = X_Q @ self.WQ

        self.WK = np.random.randn(embed_dim , d_k)/ np.sqrt(embed_dim)
        self.dWK = np.zeros_like(self.WK)
        self.K = X_K @ self. WK

    def forward(self):
        batch, seq_len_q, d_k =  self.Q.shape
        self.scores = self.Q @ self.K.transpose(0, 2, 1)
        self.scores = self.scores / np.sqrt(d_k)

        self.score =  softmax_forward(self.score)



