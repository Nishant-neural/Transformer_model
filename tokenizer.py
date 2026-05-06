import numpy as np

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.str_to_i = {ch: i for i, ch in enumerate(chars)}
        self.i_to_str = {i: ch for ch, i in self.str_to_i.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return np.array([self.str_to_i[c] for c in s], dtype=np.int32)

    def decode(self, indices):
        return ''.join([self.i_to_str[int(i)] for i in indices])