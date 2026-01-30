from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class HRMState:
    h_fast: torch.Tensor
    h_slow: torch.Tensor

class HRMController(nn.Module):
    """Minimal HRM-style controller: fast loop controls decoding; slow loop controls drift."""
    def __init__(self, d_fast: int, d_slow: int, d_in_fast: int, d_in_slow: int):
        super().__init__()
        self.fast = nn.GRUCell(d_in_fast, d_fast)
        self.slow = nn.GRUCell(d_in_slow, d_slow)
        self.to_temp = nn.Sequential(nn.Linear(d_fast, 64), nn.GELU(), nn.Linear(64, 1))
        self.to_drift = nn.Sequential(nn.Linear(d_slow, 64), nn.GELU(), nn.Linear(64, 1))

    def init_state(self, device: torch.device) -> HRMState:
        return HRMState(
            h_fast=torch.zeros(self.fast.hidden_size, device=device),
            h_slow=torch.zeros(self.slow.hidden_size, device=device),
        )

    def step_fast(self, st: HRMState, x: torch.Tensor) -> HRMState:
        return HRMState(h_fast=self.fast(x, st.h_fast), h_slow=st.h_slow)

    def step_slow(self, st: HRMState, x: torch.Tensor) -> HRMState:
        return HRMState(h_fast=st.h_fast, h_slow=self.slow(x, st.h_slow))

    def temperature(self, st: HRMState) -> float:
        t = torch.sigmoid(self.to_temp(st.h_fast)).item()
        return 0.3 + 1.2 * t

    def drift_budget(self, st: HRMState) -> float:
        b = torch.sigmoid(self.to_drift(st.h_slow)).item()
        return 0.005 + 0.075 * b