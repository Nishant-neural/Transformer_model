import numpy as np
class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Xavier-style init is better than tiny random
        self.W = np.random.randn(vocab_size, d_model) / np.sqrt(vocab_size)
        
        # gradients
        self.dW = np.zeros_like(self.W)

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        self.x = x
        out = self.W[x]  # (batch, seq_len, d_model)
        return out

    def backward(self, d_out):
        """
        d_out: (batch, seq_len, d_model)
        """
        self.dW.fill(0)

        batch, seq_len = self.x.shape

        for b in range(batch):
            for t in range(seq_len):
                idx = self.x[b, t]
                self.dW[idx] += d_out[b, t]

    def step(self, lr=1e-3):
        self.W -= lr * self.dW