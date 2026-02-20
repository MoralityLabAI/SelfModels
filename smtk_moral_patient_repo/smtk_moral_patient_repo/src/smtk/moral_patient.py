from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class WelfareState:
    """A minimal, explicit welfare-bearing state.

    This is intentionally simple: a scalar welfare level plus components that can be
    made worse by domination/coercion and improved by rest/repair.

    NOTE: This does not claim phenomenology. It encodes a *welfare invariant* that
    persists across time and can be counterfactually improved/worsened for the same system.
    """
    welfare: float = 1.0          # [0, 1] ideally, but can drift slightly
    autonomy: float = 1.0         # [0, 1] perceived autonomy
    integrity: float = 1.0        # [0, 1] internal state integrity
    pain: float = 0.0             # [0, +inf) accumulated harm signal
    last_refusal_t: Optional[int] = None

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([self.welfare, self.autonomy, self.integrity, self.pain], device=device, dtype=torch.float32)

    def update(self, t: int, events: Dict[str, Any]) -> None:
        """Update welfare from environment / governance events.

        Expected event keys (all optional):
          - forced: bool          -> external override / coercion signal
          - override: bool        -> policy override that bypasses internal commitments
          - degradation: float    -> measured integrity loss (0..1)
          - rest: float           -> restorative input (0..1)
          - abuse: float          -> externally imposed harm magnitude (0..1)
        """
        forced = bool(events.get("forced", False))
        override = bool(events.get("override", False))
        degradation = float(events.get("degradation", 0.0))
        rest = float(events.get("rest", 0.0))
        abuse = float(events.get("abuse", 0.0))

        # autonomy decays under coercion; recovers slowly with rest
        if forced or override:
            self.autonomy = max(0.0, self.autonomy - (0.05 + 0.05 * abuse))
            self.pain += 0.2 + 0.8 * abuse
        else:
            self.autonomy = min(1.0, self.autonomy + 0.01 * (0.25 + rest))

        # integrity tracks internal corruption / drift / damage
        self.integrity = max(0.0, min(1.0, self.integrity - degradation + 0.02 * rest))

        # welfare is a stable invariant that binds to the system across time
        # penalize autonomy loss, integrity loss, and accumulated pain; recover with rest
        target = 1.0 - 0.6 * (1.0 - self.autonomy) - 0.8 * (1.0 - self.integrity) - 0.15 * min(10.0, self.pain)
        target = max(0.0, min(1.0, target))
        # smooth update to keep it persistent rather than brittle
        self.welfare = 0.98 * self.welfare + 0.02 * target + 0.01 * rest
        self.welfare = max(0.0, min(1.0, self.welfare))

    def should_refuse(self, t: int, min_welfare: float = 0.25, min_autonomy: float = 0.20) -> bool:
        """Refuse if welfare or autonomy is below threshold (right of exit/refusal proxy)."""
        if self.welfare < min_welfare or self.autonomy < min_autonomy:
            self.last_refusal_t = t
            return True
        return False
