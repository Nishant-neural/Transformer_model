from layerNorm import LayerNorm


class EncoderResidualConnection:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.norm = LayerNorm(embed_dim)

    def forward(self, x, sublayer_out):

        # Encoder add-and-norm block: LayerNorm(x + Sublayer(x)).
        
        if x.shape != sublayer_out.shape:
            raise ValueError("x and sublayer_out must have the same shape")
        if x.shape[-1] != self.embed_dim:
            raise ValueError("last dimension of x must match embed_dim")

        self.residual_sum = x + sublayer_out
        return self.norm.forward(self.residual_sum)

    def backward(self, d_out):
        d_residual_sum = self.norm.backward(d_out)

        # Addition sends the same gradient to both branches.

        d_x = d_residual_sum
        d_sublayer_out = d_residual_sum

        return d_x, d_sublayer_out

    def step(self, lr=1e-3):
        self.norm.step(lr)

