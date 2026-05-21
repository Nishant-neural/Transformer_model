# Transformer Model From Scratch

A small educational implementation of transformer building blocks using only
Python and NumPy.

This project is meant for learning how the pieces of a transformer work under
the hood. The code favors readable, inspectable implementations over framework
abstractions, so each module can be opened and studied on its own.

## What Is Included

- Character-level tokenization
- Trainable token embeddings
- Sinusoidal positional encodings
- Numerically stable softmax with backward pass
- Scaled dot-product attention with optional causal and padding masks
- Multi-head attention with forward and backward passes
- Feed-forward network with ReLU or GELU activation
- Layer normalization with trainable scale and bias
- Simple smoke test for embedding forward/backward/update flow

## Project Structure

| File | Purpose |
| --- | --- |
| `tokenizer.py` | Character tokenizer with `encode` and `decode` methods. |
| `embeddings.py` | Trainable embedding lookup table with manual gradient updates. |
| `pos_encoding.py` | Sinusoidal positional encoding and input embedding wrapper. |
| `softmax.py` | Softmax forward pass and Jacobian-vector backward pass. |
| `attention.py` | Scaled dot-product attention, masking helpers, and backward pass. |
| `multihead.py` | Multi-head attention built on scaled dot-product attention. |
| `feedfoward.py` | Transformer-style feed-forward block. |
| `activations.py` | ReLU and GELU activation functions and derivatives. |
| `layerNorm.py` | Layer normalization implementation with backward pass. |
| `test.py` | Minimal smoke test for tokenization and input embeddings. |

## Requirements

- Python 3.10+
- NumPy

Install NumPy if it is not already available:

```bash
pip install numpy
```

## Quick Start

Run the included smoke test:

```bash
python test.py
```

Expected output:

```text
Output shape: (1, 11, 16)
Backward pass successful
```

The test script:

1. Builds a character vocabulary from `hello world`.
2. Encodes text into token IDs.
3. Creates token embeddings with positional encodings.
4. Runs a forward pass.
5. Sends a random gradient through the embedding layer.
6. Updates the embedding weights.

## Example Usage

```python
import numpy as np

from tokenizer import CharTokenizer
from pos_encoding import InputEmbedding
from multihead import MultiHeadAttention

text = "hello transformer"

tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)[None, :]  # (batch, seq_len)

embed_dim = 16
model = InputEmbedding(
    vocab_size=tokenizer.vocab_size,
    embed_dim=embed_dim,
    max_len=64,
)

x = model.forward(tokens)  # (batch, seq_len, embed_dim)

attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=4)
out, weights = attention.forward(x, x, x)

print(out.shape)      # (1, seq_len, embed_dim)
print(weights.shape)  # (1, num_heads, seq_len, seq_len)
```

## Attention Masks

`ScaledDotProductAttention` includes helpers for common transformer masks:

```python
from attention import ScaledDotProductAttention

causal_mask = ScaledDotProductAttention.causal_mask(seq_len_q=8)
padding_mask = ScaledDotProductAttention.padding_mask(tokens, pad_id=0)
```

Masks use `True` for positions that attention is allowed to see and `False` for
positions that should be blocked.

## Learning Notes

This repository implements gradients manually instead of using an automatic
differentiation library. That makes it easier to inspect the math behind each
operation:

- `Embedding.backward` accumulates gradients for repeated token IDs.
- `softmax_backward` computes the gradient through the softmax output.
- `ScaledDotProductAttention.backward` propagates gradients to `Q`, `K`, and `V`.
- `MultiHeadAttention.backward` reshapes head gradients and computes projection
  weight gradients.
- `LayerNorm.backward` follows the standard layer normalization derivative.

## Current Limitations

This is not yet a complete trainable transformer model. It does not include:

- A full encoder or decoder block
- Residual connections around attention and feed-forward layers
- A language-model head
- Loss functions
- Dataset loading
- A full training loop
- Optimizers beyond simple SGD-style `step` methods

There are also a few naming inconsistencies kept as-is for now, such as
`feedfoward.py` and `layerNorm.py`.

## Suggested Next Steps

Good next improvements would be:

1. Add a `TransformerBlock` that combines multi-head attention, residual
   connections, layer normalization, and the feed-forward network.
2. Add a cross-entropy loss implementation.
3. Add a tiny training loop for character-level language modeling.
4. Add numerical gradient checks for the backward passes.
5. Add unit tests for masks, attention shapes, and layer normalization.

## License

No license file is currently included. Add one before publishing or reusing this
project outside personal learning.
