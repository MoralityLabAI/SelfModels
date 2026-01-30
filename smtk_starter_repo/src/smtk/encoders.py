from __future__ import annotations
import torch
import torch.nn as nn

def l2_clip(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = vec.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (max_norm / norm).clamp_max(1.0)
    return vec * scale

class IdentityKernel(nn.Module):
    """Persistent identity vector with drift-limited explicit updates."""
    def __init__(self, d_I: int):
        super().__init__()
        self.I = nn.Parameter(torch.zeros(d_I))

    def forward(self) -> torch.Tensor:
        return self.I

    @torch.no_grad()
    def apply_delta(self, delta: torch.Tensor, drift_budget: float) -> None:
        self.I.add_(l2_clip(delta, drift_budget))

class MLPEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)