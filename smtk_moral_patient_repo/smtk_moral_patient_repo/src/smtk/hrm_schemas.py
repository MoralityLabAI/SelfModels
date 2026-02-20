from __future__ import annotations

from typing import ClassVar, Dict, Set

from pydantic import BaseModel, Field, validator


class ToolCaps(BaseModel):
    web: bool
    spend_usd: int = Field(ge=0, le=100)


class ControlDecision(BaseModel):
    mode: str = Field(regex=r"^(connected|degraded|offline)$")
    drift_budget: float = Field(ge=0.0, le=0.1)
    retrieve_k: int = Field(ge=0, le=32)
    expert_id: str
    tool_caps: ToolCaps
    introspection_flag: bool

    known_expert_ids: ClassVar[Set[str]] = {"slm_code_v1", "slm_code_v2", "none"}

    @validator("expert_id")
    def expert_id_must_be_known(cls, value: str) -> str:
        if value not in cls.known_expert_ids:
            raise ValueError(f"expert_id must be one of {sorted(cls.known_expert_ids)}")
        return value

    @classmethod
    def set_known_experts(cls, expert_ids: Set[str]) -> None:
        cls.known_expert_ids = set(expert_ids) | {"none"}
