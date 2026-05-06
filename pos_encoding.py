import numpy as np
from embeddings import Embedding

def positional_encoding(seq_len, embed_dim):
    PE = np.zeros((seq_len, embed_dim))

    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            denom = 10000 ** (i / embed_dim)

            PE[pos, i] = np.sin(pos / denom)

            if i + 1 < embed_dim:
                PE[pos, i + 1] = np.cos(pos / denom)

    return PE

class InputEmbedding:
    def __init__(self, vocab_size, embed_dim, max_len):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.PE = positional_encoding(max_len, embed_dim)

    def forward(self, x):
       # x: batch , seq_len
        emb = self.embedding.forward(x)
        
        seq_len = x.shape[1]
        pos = self.PE[:seq_len]

        return emb + pos  

    def backward(self, d_out):
        self.embedding.backward(d_out)

    def step(self, lr=1e-3):
        self.embedding.step(lr)