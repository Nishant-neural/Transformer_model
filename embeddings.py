import numpy as np
class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.W = np.random.randn(vocab_size, embed_dim) / np.sqrt(vocab_size)

        self.dW = np.zeros_like(self.W)

    def forward(self, x):
      # x : (batch size , seq len)
        self.x = x
        out = self.W[x] 
        return out

    def backward(self, d_out):
        # d_out : (batch, seq_len, embed_dim)
        
        self.dW.fill(0)

        batch, seq_len = self.x.shape

        for b in range(batch):
            for t in range(seq_len):
                idx = self.x[b, t]
                self.dW[idx] += d_out[b, t]

    def step(self, lr=1e-3):
        self.W -= lr * self.dW