"""The study's Transformer-based SASRec implementation."""
import torch
from torch import nn


def require_finite(tensor, context):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("Non-finite values in " + context)


class SASRec(nn.Module):
    def __init__(self, n_items, max_length=20, dimension=64, layers=2,
                 heads=2, feedforward=256, dropout=0.2):
        super().__init__()
        # Preserve causal/padding behavior outside the optimized eval fastpath.
        torch.backends.mha.set_fastpath_enabled(False)
        self.emb = nn.Embedding(n_items + 1, dimension, padding_idx=0)
        self.pos = nn.Embedding(max_length, dimension)
        layer = nn.TransformerEncoderLayer(
            dimension, heads, feedforward, dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(dimension)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        hidden = self.emb(x) + self.pos(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.ln(self.enc(hidden, mask=causal_mask, src_key_padding_mask=(x == 0)))
