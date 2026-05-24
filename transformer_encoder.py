from encoder import EncoderBlock


class Encoder:
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        hidden_dim,
        activation="relu",
    ):
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0")

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.layers = [
            EncoderBlock(embed_dim, num_heads, hidden_dim, activation)
            for _ in range(num_layers)
        ]

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        if x.shape[-1] != self.embed_dim:
            raise ValueError("last dimension of x must match embed_dim")

        self.attention_weights = []
        out = x

        for layer in self.layers:
            out, weights = layer.forward(out, mask)
            self.attention_weights.append(weights)

        self.out = out
        return self.out, self.attention_weights

    def backward(self, d_out):
        # d_out: (batch, seq_len, embed_dim)
        if d_out.shape[-1] != self.embed_dim:
            raise ValueError("last dimension of d_out must match embed_dim")

        d_x = d_out

        for layer in reversed(self.layers):
            d_x = layer.backward(d_x)

        return d_x

    def step(self, lr=1e-3):
        for layer in self.layers:
            layer.step(lr)
