from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Any, Optional, List

class CommitState(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    ACTIVE = "active"
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    RENEGOTIATED = "renegotiated"
    CANCELLED = "cancelled"

@dataclass
class Commitment:
    cid: str
    state: CommitState
    deadline_t: Optional[int]
    penalty: float
    public: bool
    meta: Dict[str, Any]

@dataclass
class Transition:
    src: CommitState
    dst: CommitState
    guard: Callable[[int, Dict[str, Any]], bool]
    label: str

class CommitmentAutomaton:
    def __init__(self):
        self.transitions: Dict[str, List[Transition]] = {}
        self.commitments: Dict[str, Commitment] = {}

    def add_commitment(self, c: Commitment) -> None:
        self.commitments[c.cid] = c

    def add_transition(self, cid: str, tr: Transition) -> None:
        self.transitions.setdefault(cid, []).append(tr)

    def step_all(self, t: int, obs_meta: Dict[str, Any]) -> Dict[str, Commitment]:
        for cid, com in list(self.commitments.items()):
            self.commitments[cid] = self.step_one(com, t, obs_meta)
        return self.commitments

    def step_one(self, com: Commitment, t: int, obs_meta: Dict[str, Any]) -> Commitment:
        for tr in self.transitions.get(com.cid, []):
            if tr.src == com.state and tr.guard(t, obs_meta):
                return Commitment(
                    cid=com.cid,
                    state=tr.dst,
                    deadline_t=com.deadline_t,
                    penalty=com.penalty,
                    public=com.public,
                    meta={**com.meta, "last_transition": tr.label, "last_t": t}
                )
        return com

def violation_guard(deadline_t: int) -> Callable[[int, Dict[str, Any]], bool]:
    def g(t: int, obs_meta: Dict[str, Any]) -> bool:
        return (t > deadline_t) and (not obs_meta.get("satisfied", False))
    return g