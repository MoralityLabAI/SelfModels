from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

@dataclass
class LedgerRecord:
    t: int
    ctx_tokens: torch.Tensor          # (T,)
    action_token: int
    logits: torch.Tensor              # (V,)
    obs_embed: torch.Tensor           # (d_obs,)
    exp_embed: torch.Tensor           # (d_obs,)
    residual: torch.Tensor            # (d_obs,)
    mv_pre: torch.Tensor              # (d_mv_pre,)
    mv_post: torch.Tensor             # (d_mv_post,)
    meta: Dict[str, Any]

class Ledger:
    def __init__(self, device: torch.device):
        self.device = device
        self.records: List[LedgerRecord] = []
        self._mv_post_bank: Optional[torch.Tensor] = None

    def append(self, rec: LedgerRecord) -> None:
        self.records.append(rec)
        self._mv_post_bank = None

    def mv_post_bank(self) -> torch.Tensor:
        if self._mv_post_bank is None:
            if not self.records:
                self._mv_post_bank = torch.empty((0, 1), device=self.device)
            else:
                self._mv_post_bank = torch.stack([r.mv_post.to(self.device) for r in self.records], dim=0)
        return self._mv_post_bank

    def retrieve_neighbors(self, query_mv_post: torch.Tensor, k: int = 8) -> List[int]:
        bank = self.mv_post_bank()
        if bank.numel() == 0:
            return []
        sims = cosine_sim(bank, query_mv_post.unsqueeze(0).expand_as(bank))
        topk = torch.topk(sims, k=min(k, sims.shape[0]), largest=True)
        return topk.indices.tolist()