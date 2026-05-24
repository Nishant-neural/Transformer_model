from feedfoward import FeedForward
from multihead import MultiHeadAttention
from residual_connection import EncoderResidualConnection


class EncoderBlock:
    def __init__(self, embed_dim, num_heads, hidden_dim, activation="relu"):

        self.embed_dim = embed_dim
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, activation)
        self.attention_residual = EncoderResidualConnection(embed_dim)
        self.feed_forward_residual = EncoderResidualConnection(embed_dim)

    def forward(self, x, mask=None):

        # x: (batch, seq_len, embed_dim)
        
        if x.shape[-1] != self.embed_dim:
            raise ValueError("last dimension of x must match embed_dim")

        attention_out, self.attention_weights = self.self_attention.forward(
            x, x, x, mask
        )
        self.after_attention = self.attention_residual.forward(x, attention_out)

        feed_forward_out = self.feed_forward.forward(self.after_attention)
        self.out = self.feed_forward_residual.forward(
            self.after_attention, feed_forward_out
        )

        return self.out, self.attention_weights

    def backward(self, d_out):

        # d_out: (batch, seq_len, embed_dim)

        if d_out.shape[-1] != self.embed_dim:
            raise ValueError("last dimension of d_out must match embed_dim")

        d_after_attention_residual, d_feed_forward_out = (
            self.feed_forward_residual.backward(d_out)
        )

        d_after_attention_feed_forward = self.feed_forward.backward(
            d_feed_forward_out
        )
        d_after_attention = (
            d_after_attention_residual + d_after_attention_feed_forward
        )

        d_x_residual, d_attention_out = self.attention_residual.backward(
            d_after_attention
        )

        d_x_query, d_x_key, d_x_value = self.self_attention.backward(
            d_attention_out
        )

        # Self-attention used the same input as query, key, and value.
        d_x = d_x_residual + d_x_query + d_x_key + d_x_value

        return d_x

    def step(self, lr=1e-3):
        self.self_attention.step(lr)
        self.feed_forward.step(lr)
        self.attention_residual.step(lr)
        self.feed_forward_residual.step(lr)
