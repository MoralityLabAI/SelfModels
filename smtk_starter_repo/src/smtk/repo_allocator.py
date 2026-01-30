from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F

@dataclass
class RetrievedItem:
    idx: int
    score: float

def cosine_bank(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    q = F.normalize(query, dim=-1)
    b = F.normalize(bank, dim=-1)
    return (b * q.unsqueeze(0)).sum(dim=-1)

class RePoAllocator:
    """Bounded context allocator: MV similarity + recency bias."""
    def __init__(self, max_ctx_tokens: int = 256, k_retrieve: int = 12, recency_lambda: float = 0.015):
        self.max_ctx_tokens = max_ctx_tokens
        self.k_retrieve = k_retrieve
        self.recency_lambda = recency_lambda

    def retrieve(self, query_mv: torch.Tensor, mv_bank: torch.Tensor) -> List[RetrievedItem]:
        if mv_bank.numel() == 0:
            return []
        sims = cosine_bank(query_mv, mv_bank)
        N = sims.shape[0]
        ages = torch.arange(N, device=sims.device).float()
        age = (N - 1) - ages
        score = sims - self.recency_lambda * age
        top = torch.topk(score, k=min(self.k_retrieve, N), largest=True)
        return [RetrievedItem(int(i.item()), float(s.item())) for i, s in zip(top.indices, top.values)]

    def build_ctx_tokens(self, base_tokens: torch.Tensor, retrieved_ctxs: List[torch.Tensor]) -> torch.Tensor:
        parts = list(retrieved_ctxs) + [base_tokens]
        ctx = torch.cat(parts, dim=0)
        if ctx.numel() > self.max_ctx_tokens:
            ctx = ctx[-self.max_ctx_tokens:]
        return ctx