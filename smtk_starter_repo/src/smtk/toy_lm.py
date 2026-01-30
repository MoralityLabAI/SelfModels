from __future__ import annotations
import torch
import torch.nn as nn

class ToyLM(nn.Module):
    """Tiny causal LM that returns next-token logits for ctx."""
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ctx_tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(ctx_tokens.unsqueeze(0))  # (1,T,d)
        T = x.shape[1]
        mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)
        h = self.tr(x, mask=mask)                  # (1,T,d)
        last = h[:, -1, :]                         # (1,d)
        logits = self.lm_head(last).squeeze(0)     # (V,)
        return logits