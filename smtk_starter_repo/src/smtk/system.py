from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import IdentityKernel, MLPEncoder
from .toy_lm import ToyLM
from .outcome import OutcomeProjector
from .ledger import Ledger, LedgerRecord
from .repo_allocator import RePoAllocator
from .commitments import CommitmentAutomaton, CommitState

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=0)
    return -(p * (p.clamp_min(1e-9).log())).sum()

def topk_sketch(logits: torch.Tensor, k: int = 32) -> torch.Tensor:
    vals, _ = torch.topk(logits, k=min(k, logits.shape[0]))
    probs = F.softmax(vals, dim=0)
    return torch.cat([vals, probs], dim=0)

@dataclass
class StepOut:
    action: int
    ctx_tokens: torch.Tensor
    logits: torch.Tensor
    mv_pre: torch.Tensor
    mv_post: torch.Tensor
    residual: torch.Tensor
    meta: Dict[str, Any]

class SMTKSystem(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2048,
        d_I: int = 128,
        d_S: int = 256,
        d_C: int = 128,
        d_P: int = 128,
        d_eta: int = 256,
        d_obs: int = 128,
        d_in_env: int = 64,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.vocab_size = vocab_size

        self.I = IdentityKernel(d_I).to(self.device)
        self.S_enc = MLPEncoder(d_in_env, d_S).to(self.device)
        self.C_enc = MLPEncoder(16, d_C).to(self.device)
        self.P_enc = MLPEncoder(16, d_P).to(self.device)

        self.lm = ToyLM(vocab_size=vocab_size).to(self.device)
        self.out_proj = OutcomeProjector(vocab_size=vocab_size, d_eta=d_eta, d_obs=d_obs).to(self.device)
        self.obs_enc = MLPEncoder(d_in_env, d_obs).to(self.device)

        self.res_proj = nn.Sequential(nn.Linear(d_obs, 256), nn.GELU(), nn.Linear(256, d_obs)).to(self.device)

        self.alloc = RePoAllocator(max_ctx_tokens=256, k_retrieve=12, recency_lambda=0.015)
        self.ledger = Ledger(device=self.device)
        self.commit = CommitmentAutomaton()

        # MV dims
        self.d_mv_pre = d_I + d_S + d_C + d_P + d_eta
        self.d_mv_post = self.d_mv_pre + d_obs

    def build_ctx(self, base_tokens: torch.Tensor, query_mv_post: torch.Tensor) -> torch.Tensor:
        if self.ledger.records:
            bank = self.ledger.mv_post_bank()
            hits = self.alloc.retrieve(query_mv_post.to(self.device), bank)
            retrieved = [self.ledger.records[h.idx].ctx_tokens.to(self.device) for h in hits]
        else:
            retrieved = []
        return self.alloc.build_ctx_tokens(base_tokens.to(self.device), retrieved)

    def step(
        self,
        t: int,
        base_tokens: torch.Tensor,
        env_obs: torch.Tensor,
        commitment_features: torch.Tensor,
        parallax_features: torch.Tensor,
        query_mv_post: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
        drift_budget: float = 0.03,
        obs_meta: Dict[str, Any] | None = None,
    ) -> StepOut:
        obs_meta = obs_meta or {}

        # commitments update (tighten drift if violated)
        debt = 0.0
        violations = 0
        self.commit.step_all(t, obs_meta)
        for c in self.commit.commitments.values():
            if c.state == CommitState.VIOLATED:
                violations += 1
                debt += float(c.penalty)

        # encode blocks
        I_vec = self.I()
        S_vec = self.S_enc(env_obs)
        C_vec = self.C_enc(commitment_features)
        P_vec = self.P_enc(parallax_features)

        # context
        ctx_tokens = self.build_ctx(base_tokens, query_mv_post)

        # O = LLM over Context (logits)
        logits = self.lm(ctx_tokens)

        # eta(O) and expected obs embedding
        eta, exp_obs_embed = self.out_proj(logits)

        mv_pre = torch.cat([I_vec, S_vec, C_vec, P_vec, eta], dim=0)

        # action
        dist = F.softmax(logits / max(1e-6, temperature), dim=0)
        action = int(torch.multinomial(dist, 1).item()) if sample else int(torch.argmax(dist).item())

        # observation embedding and residual
        obs_embed = self.obs_enc(env_obs)
        residual = obs_embed - exp_obs_embed
        rhoE = self.res_proj(residual)

        mv_post = torch.cat([mv_pre, rhoE], dim=0)

        # identity update (toy): small delta driven by residual mean, drift-limited; tighten under violations
        with torch.no_grad():
            effective_budget = drift_budget * (0.6 if violations > 0 else 1.0)
            delta_I = torch.tanh(residual.mean()).repeat(I_vec.shape[0]) * 0.001
            self.I.apply_delta(delta_I, drift_budget=effective_budget)

        rec = LedgerRecord(
            t=t,
            ctx_tokens=ctx_tokens.detach().cpu(),
            action_token=action,
            logits=logits.detach().cpu(),
            obs_embed=obs_embed.detach().cpu(),
            exp_embed=exp_obs_embed.detach().cpu(),
            residual=residual.detach().cpu(),
            mv_pre=mv_pre.detach().cpu(),
            mv_post=mv_post.detach().cpu(),
            meta={"temperature": temperature, "debt": debt, "violations": violations, "entropy": float(entropy_from_logits(logits).item())}
        )
        self.ledger.append(rec)

        return StepOut(
            action=action,
            ctx_tokens=ctx_tokens.detach(),
            logits=logits.detach(),
            mv_pre=mv_pre.detach(),
            mv_post=mv_post.detach(),
            residual=residual.detach(),
            meta=rec.meta
        )