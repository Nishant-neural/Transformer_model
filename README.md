# Transformer Model From Scratch

A small learning project that builds transformer components with NumPy. The code currently covers character tokenization, token embeddings, sinusoidal positional encodings, softmax forward/backward passes, and an early attention score implementation.

The goal of this repository is to make each transformer building block easy to inspect and understand before combining them into a larger model.

## Project Structure

| File | Description |
| --- | --- |
| `tokenizer.py` | Character-level tokenizer with `encode` and `decode` methods. |
| `embeddings.py` | Trainable embedding layer with forward, backward, and update steps. |
| `pos_encoding.py` | Sinusoidal positional encoding and an `InputEmbedding` wrapper. |
| `softmax.py` | Numerically stable softmax and softmax backward function. |
| `attention.py` | Query/key projection and attention score calculation work in progress. |
| `test.py` | Simple smoke test for tokenization, input embeddings, backward pass, and parameter update. |

## How to Run

### Running the Test
Execute the test script to verify all components work together:

```bash
python test.py
```

This will:
- Create a character tokenizer from "hello world"
- Encode the text into tokens
- Create input embeddings with positional encoding
- Run a forward pass
- Simulate a backward pass with random gradients
- Update the model parameters
- Print success confirmation

Expected output:
```
Output shape: (1, 11, 16)
Backward pass successful
```

### Interactive Usage
You can experiment with the components in a Python REPL:

```python
import numpy as np
from tokenizer import CharTokenizer
from pos_encoding import InputEmbedding
from attention import Query_key

# Tokenize text
text = "hello transformer"
tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)[None, :]  # Shape: (1, seq_len)

# Create embeddings
model = InputEmbedding(tokenizer.vocab_size, embed_dim=32, max_len=50)
embeddings = model.forward(tokens)  # Shape: (1, seq_len, embed_dim)

# Test attention (work in progress)
attention = Query_key(embeddings, embeddings, d_k=32)
attention.forward()
print(f"Attention scores shape: {attention.score.shape}")
```

```python
from tokenizer import CharTokenizer
from pos_encoding import InputEmbedding

text = "hello world"

tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)[None, :]

model = InputEmbedding(
    vocab_size=tokenizer.vocab_size,
    embed_dim=16,
    max_len=50,
)

out = model.forward(tokens)
print(out.shape)
```

## Current Features

- Character-level vocabulary creation from input text
- Encoding text into integer token IDs
- Decoding token IDs back into text
- Trainable embedding lookup table
- Sinusoidal positional encodings
- Input embedding layer that combines token and position information
- Softmax forward and backward functions
- Basic gradient accumulation for embedding weights
- Query-Key attention mechanism (work in progress)

## How Components Work

### Tokenization
The `CharTokenizer` creates a vocabulary by sorting unique characters from the input text. Each character gets a unique integer ID, allowing text to be converted to sequences of integers.

### Embeddings
The `Embedding` class maintains a lookup table of shape `(vocab_size, embed_dim)`. During forward pass, it retrieves vectors for each token ID. The backward pass accumulates gradients for each token occurrence.

### Positional Encoding
Uses sinusoidal functions to encode position information:
- Even indices: `sin(pos / 10000^(i/embed_dim))`
- Odd indices: `cos(pos / 10000^(i/embed_dim))`
This allows the model to distinguish token positions.

### Attention Mechanism
The `Query_key` class implements the core attention computation:
- Projects input embeddings to Query and Key matrices
- Computes attention scores as `Q @ K.T / sqrt(seq_len)`
- Applies softmax to get attention weights
(Note: Value projection and weighted sum not yet implemented)
