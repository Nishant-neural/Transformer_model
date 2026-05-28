# Encoder-Only Transformer From Scratch

A small educational implementation of an encoder-only transformer using only
Python and NumPy.

This project is meant for learning how transformer encoders work under the
hood. The code favors readable, inspectable implementations over framework
abstractions, so each module can be opened and studied on its own.

The current model is closest in spirit to the encoder side of BERT: token
embeddings plus positional encodings, stacked self-attention encoder blocks,
and a masked-language-modeling head.

## What Is Included

- Character-level tokenization
- Trainable token embeddings
- Sinusoidal positional encodings
- Numerically stable softmax with backward pass
- Scaled dot-product self-attention with padding and optional causal masks
- Multi-head attention with forward and backward passes
- Feed-forward network with ReLU or GELU activation
- Layer normalization with trainable scale and bias
- Encoder residual add-and-norm blocks
- Stackable encoder blocks
- Encoder-only transformer wrapper
- Masked language modeling head
- Masked language modeling loss using `-100` as the ignore label
- Basic `train_step` flow with manual forward, backward, and SGD-style updates
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
| `residual_connection.py` | Encoder add-and-norm residual connection. |
| `encoder.py` | Single transformer encoder block. |
| `transformer_encoder.py` | Stack of encoder blocks. |
| `mlm.py` | Masked language model built from embeddings, encoder, and MLM head. |
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

## Encoder Example

```python
import numpy as np

from tokenizer import CharTokenizer
from pos_encoding import InputEmbedding
from transformer_encoder import Encoder

text = "hello transformer"

tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)[None, :]  # (batch, seq_len)

embed_dim = 16
embedding = InputEmbedding(
    vocab_size=tokenizer.vocab_size,
    embed_dim=embed_dim,
    max_len=64,
)

encoder = Encoder(
    num_layers=2,
    embed_dim=embed_dim,
    num_heads=4,
    hidden_dim=64,
    activation="gelu",
)

x = embedding.forward(tokens)
encoded, attention_weights = encoder.forward(x)

print(encoded.shape)                 # (1, seq_len, embed_dim)
print(len(attention_weights))         # 2
print(attention_weights[0].shape)     # (1, num_heads, seq_len, seq_len)
```

## Masked Language Model Example

```python
import numpy as np

from mlm import MaskedLanguageModel
from tokenizer import CharTokenizer

text = "hello transformer"
tokenizer = CharTokenizer(text)
input_ids = tokenizer.encode(text)[None, :]

# Train only one position in this tiny example.
labels = np.full_like(input_ids, -100)
labels[0, 1] = input_ids[0, 1]

model = MaskedLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=16,
    max_len=64,
    num_layers=2,
    num_heads=4,
    hidden_dim=64,
    activation="gelu",
)

loss = model.train_step(input_ids, labels, lr=1e-3)
print(loss)
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

For an encoder-only model, the usual mask is a padding mask. A causal mask is
more useful for decoder-only language modeling, but it is still available for
experiments.

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
- `EncoderBlock.backward` combines gradients from residual and self-attention
  branches.
- `MaskedLanguageModel.backward` only trains positions whose labels are not
  `-100`.

## What To Add Next

To make this a more effective encoder-only transformer implementation, the most
valuable next additions are:

1. **A real MLM data pipeline**: add `[PAD]`, `[MASK]`, `[UNK]`, and maybe
   `[CLS]` tokens, then create batches that randomly mask 15% of non-padding
   tokens.
2. **A tokenizer upgrade**: move from character tokenization to a simple
   WordPiece, BPE, or unigram tokenizer so the model learns more useful text
   units.
3. **Dropout**: add dropout after attention weights, after attention output,
   inside the feed-forward block, and after embeddings.
4. **Pre-norm option**: support `LayerNorm(x)` before each sublayer in addition
   to the current post-norm add-and-norm design. Pre-norm is often easier to
   optimize in deeper stacks.
5. **Optimizer improvements**: add AdamW with gradient clipping and optional
   learning-rate warmup instead of only plain SGD-style updates.
6. **Training script**: create a `train_mlm.py` that loads text, batches
   examples, applies masking, trains for multiple epochs, and logs loss.
7. **Evaluation script**: add masked-token accuracy and loss on a held-out text
   split.
8. **Tests and gradient checks**: add unit tests for masks, shape contracts,
   backward passes, and numerical gradient checks for attention, layer norm,
   feed-forward, and MLM loss.
9. **Model save/load**: add simple checkpointing with `np.savez` and restore
   methods for all trainable parameters.
10. **Sequence classification head**: add a pooling or `[CLS]` head so the
    encoder can be used for classification after pretraining.

Recommended order: build the MLM data pipeline first, then add AdamW plus a
training script, then add tests/gradient checks. Those three steps will make
the project feel like a usable encoder-only transformer instead of just a
collection of working components.

## Current Limitations

This is still a learning implementation, not a production training framework.
Important limitations include:

- No dropout yet
- No AdamW, gradient clipping, or learning-rate scheduling
- No batching/data-loader utilities
- No automatic random masking pipeline
- No checkpoint save/load
- No numerical gradient checks
- No formal unit test suite
- Character-level tokenizer only

There are also a few naming inconsistencies kept as-is for now, such as
`feedfoward.py` and `layerNorm.py`.

## License

No license file is currently included. Add one before publishing or reusing this
project outside personal learning.
