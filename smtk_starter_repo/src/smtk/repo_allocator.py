from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal
import torch
import torch.nn.functional as F

@dataclass
class RetrievedItem:
    idx: int
    score: float

class RePoAllocator:
    """
    RePo allocator with pluggable similarity:
      - cosine (baseline)
      - spectral_mag (FFT magnitude)
      - spectral_phase (complex phase-sensitive)
    """
    def __init__(
        self,
        max_ctx_tokens: int = 256,
        k_retrieve: int = 12,
        recency_lambda: float = 0.015,
        mode: Literal["cosine", "spectral_mag", "spectral_phase"] = "spectral_mag",
    ):
        self.max_ctx_tokens = max_ctx_tokens
        self.k_retrieve = k_retrieve
        self.recency_lambda = recency_lambda
        self.mode = mode

    # ---------- similarity kernels ----------

    def _cosine(self, q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        qn = F.normalize(q, dim=-1)
        bn = F.normalize(b, dim=-1)
        return (bn * qn.unsqueeze(0)).sum(dim=-1)

    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert x.numel() % 2 == 0, "Vector length must be even for complex view"
        return torch.view_as_complex(x.view(-1, 2))

    def _spectral_mag(self, q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        qc = self._to_complex(q)
        bc = torch.view_as_complex(b.view(b.size(0), -1, 2))
        qf = torch.fft.fft(qc).abs()
        bf = torch.fft.fft(bc, dim=1).abs()
        qf = qf / qf.norm().clamp_min(1e-8)
        bf = bf / bf.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return (bf * qf.unsqueeze(0)).sum(dim=1)

    def _spectral_phase(self, q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        qc = self._to_complex(q)
        bc = torch.view_as_complex(b.view(b.size(0), -1, 2))
        qf = torch.fft.fft(qc)
        bf = torch.fft.fft(bc, dim=1)
        num = (bf * torch.conj(qf).unsqueeze(0)).sum(dim=1)
        den = (bf.abs().norm(dim=1) * qf.abs().norm()).clamp_min(1e-8)
        return (num / den).real

    # ---------- public API ----------

    def retrieve(self, query_mv: torch.Tensor, mv_bank: torch.Tensor) -> List[RetrievedItem]:
        if mv_bank is None or mv_bank.numel() == 0:
            return []

        if self.mode == "cosine":
            sims = self._cosine(query_mv, mv_bank)
        elif self.mode == "spectral_phase":
            sims = self._spectral_phase(query_mv, mv_bank)
        else:
            sims = self._spectral_mag(query_mv, mv_bank)

        # recency bias (older entries penalized)
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
