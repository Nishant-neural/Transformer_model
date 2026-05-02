import numpy as np

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int32)

    def decode(self, indices):
        return ''.join([self.itos[int(i)] for i in indices])