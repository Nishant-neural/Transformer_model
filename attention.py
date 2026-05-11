import numpy as np
from softmax import softmax_forward , softmax_backward

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
        self.X_Q = X_Q
        self.X_K = X_K
        self.X_V = X_V
        self.mask = mask

        self.K = X_K @ self.WK
        self.V = X_V @ self.WV
        self.Q = X_Q @ self.WQ

        batch, seq_len_q, d_k =  self.Q.shape
        self.scores = self.Q @  np.swapaxes(self.K, -1, -2)

        self.scores = self.scores / np.sqrt(d_k)

        if mask is not None:
           self.scores = np.where(mask, self.scores, -1e9)

        self.weights =  softmax_forward(self.scores)
        self.out = self.weights @ self.V

        # weights = batch , seq_lenq , seq_len_k
        # Out = batch , seq_len_q , d_v
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

     
    def backward(self , d_out ):
        batch, seq_len_q, d_k =  self.Q.shape

        self.dV =  self.weights.transpose(0,2,1)  @ d_out

        self.dweights = d_out @ self.V.transpose(0,2,1)

        self.dscaled =  softmax_backward(self.dweights , self.weights)
        if self.mask is not None:
            self.dscaled = np.where(self.mask, self.dscaled, 0)

       # weights = softmax(scores)
       # scores = scores/ scaling factor
        # scores = Q @K.T

        self.dscores = self.dscaled / np.sqrt(d_k)
        self.dQ = self.dscores @ self.K
        self.dK = self.dscores.transpose(0,2,1) @ self.Q

        self.dWQ = np.sum(
        self.X_Q.transpose(0,2,1) @ self.dQ,
        axis=0
        )
        self.dX_Q = self.dQ @ self.WQ.T

        self.dWK = np.sum(
        self.X_K.transpose(0,2,1) @ self.dK,
        axis=0
        )
        self.dX_K = self.dK @ self.WK.T

        self.dWV = np.sum(
        self.X_V.transpose(0,2,1) @ self.dV,
        axis=0
        )
        self.dX_V = self.dV @ self.WV.T

        return self.dX_Q, self.dX_K, self.dX_V


    def scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = Q.shape[-1]

        scores = Q @ np.swapaxes(K, -1, -2)
        scores = scores / np.sqrt(d_k)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None, :, :]  # (B, 1, Tq, Tk)
            scores = np.where(mask, scores, -1e9)

        weights = softmax_forward(scores)
        out = weights @ V

        return out, weights
