from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class OutcomeProjector(nn.Module):
    """
    eta(O): fixed-width sketch of O from top-k logits/probs.
    Also computes an expected observation embedding under top-k distribution.
    """
    def __init__(self, vocab_size: int, d_eta: int, d_obs: int, topk: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.topk = topk
        self.proj = nn.Sequential(
            nn.Linear(2 * topk, 256),
            nn.GELU(),
            nn.Linear(256, d_eta),
        )
        self.token_obs_emb = nn.Embedding(vocab_size, d_obs)

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        V = logits.shape[0]
        k = min(self.topk, V)
        vals, idx = torch.topk(logits, k=k, largest=True)  # (k,)
        probs = F.softmax(vals, dim=0)                     # (k,)
        feat = torch.cat([vals, probs], dim=0)             # (2k,)
        eta = self.proj(feat)                              # (d_eta,)
        emb = self.token_obs_emb(idx)                      # (k,d_obs)
        exp = (probs.unsqueeze(-1) * emb).sum(dim=0)       # (d_obs,)
        return eta, exp