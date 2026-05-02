import numpy as np
from embeddings import Embedding

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (i / d_model)

            PE[pos, i] = np.sin(pos / denom)

            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denom)

    return PE

class InputEmbedding:
    def __init__(self, vocab_size, d_model, max_len):
        self.embedding = Embedding(vocab_size, d_model)
        self.PE = positional_encoding(max_len, d_model)

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        emb = self.embedding.forward(x)
        
        seq_len = x.shape[1]
        pos = self.PE[:seq_len]

        return emb + pos  # broadcasting

    def backward(self, d_out):
        self.embedding.backward(d_out)

    def step(self, lr=1e-3):
        self.embedding.step(lr)