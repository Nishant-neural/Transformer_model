import numpy as np

from attention import ScaledDotProductAttention
from pos_encoding import InputEmbedding
from softmax import softmax_forward
from transformer_encoder import Encoder


class MLMHead:
    def __init__(self, embed_dim, vocab_size):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.W = np.random.randn(embed_dim, vocab_size) / np.sqrt(embed_dim)
        self.b = np.zeros(vocab_size)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):

        # x: (batch, seq_len, embed_dim)

        self.x = x
        self.logits = x @ self.W + self.b
        return self.logits

    def backward(self, d_logits):
        self.dW = np.einsum("bte,btv->ev", self.x, d_logits)
        self.db = np.sum(d_logits, axis=(0, 1))
        return d_logits @ self.W.T


    def step(self, lr=1e-3):
        self.W -= lr * self.dW
        self.b -= lr * self.db



class MaskedLanguageModel:

    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_len,
        num_layers,
        num_heads,
        hidden_dim,
        activation="relu",
        pad_token_id=None,
    ):
        self.pad_token_id = pad_token_id
        self.input_embedding = InputEmbedding(vocab_size, embed_dim, max_len)
        self.encoder = Encoder(
            num_layers,
            embed_dim,
            num_heads,
            hidden_dim,
            activation,
        )
        self.mlm_head = MLMHead(embed_dim, vocab_size)


    def forward(self, input_ids, attention_mask=None):
        
        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = ScaledDotProductAttention.padding_mask(
                input_ids,
                self.pad_token_id,
            )

        x = self.input_embedding.forward(input_ids)
        encoded, attention_weights = self.encoder.forward(x, attention_mask)
        logits = self.mlm_head.forward(encoded)
        return logits, attention_weights


    def loss(self, logits, labels):
        
        self.labels = labels
        self.valid_positions = labels != -100
        valid_count = np.sum(self.valid_positions)

        if valid_count == 0:
            raise ValueError("MLM batch has no masked positions")

        self.probs = softmax_forward(logits)

        batch_idx, time_idx = np.where(self.valid_positions)
        target_ids = labels[batch_idx, time_idx]
        target_probs = self.probs[batch_idx, time_idx, target_ids]

        return -np.mean(np.log(target_probs + 1e-12))


    def backward(self):

        valid_count = np.sum(self.valid_positions)

        d_logits = np.zeros_like(self.probs)
        batch_idx, time_idx = np.where(self.valid_positions)
        target_ids = self.labels[batch_idx, time_idx]

        d_logits[self.valid_positions] = self.probs[self.valid_positions]
        d_logits[batch_idx, time_idx, target_ids] -= 1
        d_logits /= valid_count

        d_encoded = self.mlm_head.backward(d_logits)
        d_embedding = self.encoder.backward(d_encoded)
        self.input_embedding.backward(d_embedding)


    def step(self, lr=1e-3):

        self.input_embedding.step(lr)
        self.encoder.step(lr)
        self.mlm_head.step(lr)


    def train_step(self, input_ids, labels, attention_mask=None, lr=1e-3):

        logits, _ = self.forward(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        self.backward()
        self.step(lr)
        return loss
