import numpy as np
from tokenizer import CharTokenizer
from pos_encoding import InputEmbedding

if __name__ == "__main__":
    text = "hello world"

    tokenizer = CharTokenizer(text)
    tokens = tokenizer.encode(text)

    tokens = tokens[None, :]  # (1, seq_len)

    vocab_size = tokenizer.vocab_size
    d_model = 16
    max_len = 50

    model = InputEmbedding(vocab_size, d_model, max_len)

    # forward
    out = model.forward(tokens)

    print("Output shape:", out.shape)  # (1, seq_len, d_model)

    # fake gradient
    d_out = np.random.randn(*out.shape)

    # backward
    model.backward(d_out)

    # update
    model.step(lr=1e-2)

    print("Backward pass successful")