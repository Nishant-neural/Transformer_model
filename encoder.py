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
