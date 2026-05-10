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
       


    def forward(self, X_Q, X_K, X_V, mask=None):

        # X_q =  batch , seq_len_q , embed_dim
        # X_K = batch , seq_len_k , embed_dim
        # X_V = BATCH , seq_len_k , embed_dim

        self.K = X_K @ self.WK
        self.V = X_V @ self.WV
        self.Q = X_Q @ self.WQ

        batch, seq_len_q, d_k =  self.Q.shape
        self.logits = self.Q @ self.K.transpose(0, 2, 1)
        self.logits = self.logits / np.sqrt(d_k)

        if mask is not None:
           self.logits = np.where(mask, self.logits, -1e9)

        self.weights =  softmax_forward(self.logits)
        self.out = self.weights @ self.V


        return self.out, self.weights


    @staticmethod
    def causal_mask(seq_len_q, seq_len_k=None):
        if seq_len_k is None:
            seq_len_k = seq_len_q

        mask = np.tril(np.ones((seq_len_q, seq_len_k), dtype=bool))
        return mask[None, :, :]

    @staticmethod
    def padding_mask(tokens, pad_id=0):
        return (tokens != pad_id)[:, None, :]

