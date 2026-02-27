"""
Iteration 2: Adapter routing as a low-rank manifold.

This experiment compares:
1) Explicit lookup table routing (exact entries, no generalization)
2) Low-rank factor routing (compressed predictive model)

Goal:
- Show graceful degradation under adapter/node removal.
- Quantify coverage and error before/after removal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass
class EvalMetrics:
    mse: float
    mae: float
    coverage: float


def make_synthetic_tensor(
    n_dest: int,
    n_adapter: int,
    n_time: int,
    rank_true: int,
    noise_std: float,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_dest, rank_true, generator=g)
    b = torch.randn(n_adapter, rank_true, generator=g)
    c = torch.randn(n_time, rank_true, generator=g)
    tensor = torch.einsum("dr,ar,tr->dat", a, b, c)
    tensor = tensor + noise_std * torch.randn(tensor.shape, generator=g)
    return tensor


def fit_low_rank(tensor: torch.Tensor, rank_model: int, steps: int, lr: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    n_dest, n_adapter, n_time = tensor.shape

    a = torch.randn(n_dest, rank_model, generator=g, requires_grad=True)
    b = torch.randn(n_adapter, rank_model, generator=g, requires_grad=True)
    c = torch.randn(n_time, rank_model, generator=g, requires_grad=True)

    opt = torch.optim.Adam([a, b, c], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = torch.einsum("dr,ar,tr->dat", a, b, c)
        loss = torch.mean((pred - tensor) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = torch.einsum("dr,ar,tr->dat", a, b, c)
    return pred, a.detach(), b.detach(), c.detach()


def eval_metrics(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor | None = None) -> EvalMetrics:
    if mask is None:
        mask = torch.ones_like(truth, dtype=torch.bool)
    covered = mask.float().mean().item()
    if not mask.any():
        return EvalMetrics(mse=float("nan"), mae=float("nan"), coverage=0.0)
    err = pred[mask] - truth[mask]
    return EvalMetrics(
        mse=torch.mean(err**2).item(),
        mae=torch.mean(torch.abs(err)).item(),
        coverage=covered,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=int, default=96)
    p.add_argument("--adapters", type=int, default=24)
    p.add_argument("--time", type=int, default=20)
    p.add_argument("--rank-true", type=int, default=4)
    p.add_argument("--rank-model", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--remove-adapter-idx", type=int, default=3)
    args = p.parse_args()

    truth = make_synthetic_tensor(
        n_dest=args.dest,
        n_adapter=args.adapters,
        n_time=args.time,
        rank_true=args.rank_true,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    low_rank_pred, a, b, c = fit_low_rank(
        tensor=truth,
        rank_model=args.rank_model,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed + 1,
    )

    # Baseline: explicit table perfectly stores truth.
    table_pred = truth.clone()

    remove_idx = max(0, min(args.adapters - 1, args.remove_adapter_idx))

    # Node removal:
    # - Table baseline loses all entries for that adapter.
    # - Low-rank model degrades smoothly by suppressing that adapter embedding.
    table_mask_after = torch.ones_like(truth, dtype=torch.bool)
    table_mask_after[:, remove_idx, :] = False

    b_ablated = b.clone()
    b_ablated[remove_idx, :] = 0.0
    low_rank_after = torch.einsum("dr,ar,tr->dat", a, b_ablated, c)

    full_mask = torch.ones_like(truth, dtype=torch.bool)
    removed_only_mask = torch.zeros_like(truth, dtype=torch.bool)
    removed_only_mask[:, remove_idx, :] = True

    table_before = eval_metrics(table_pred, truth, full_mask)
    low_before = eval_metrics(low_rank_pred, truth, full_mask)
    table_after_all = eval_metrics(table_pred, truth, table_mask_after)
    low_after_all = eval_metrics(low_rank_after, truth, full_mask)
    low_after_removed = eval_metrics(low_rank_after, truth, removed_only_mask)

    print("== Iteration 2: Adapter Routing Manifold ==")
    print(f"shape(dest, adapter, time)=({args.dest}, {args.adapters}, {args.time})")
    print(f"adapter_removed={remove_idx}")
    print("")
    print("[Before removal]")
    print(f"table_lookup     mse={table_before.mse:.6f} mae={table_before.mae:.6f} coverage={table_before.coverage:.3f}")
    print(f"low_rank_model   mse={low_before.mse:.6f} mae={low_before.mae:.6f} coverage={low_before.coverage:.3f}")
    print("")
    print("[After removal]")
    print(f"table_lookup     mse={table_after_all.mse:.6f} mae={table_after_all.mae:.6f} coverage={table_after_all.coverage:.3f} (missing removed adapter entries)")
    print(f"low_rank_model   mse={low_after_all.mse:.6f} mae={low_after_all.mae:.6f} coverage={low_after_all.coverage:.3f} (graceful but noisier)")
    print(f"low_rank_removed mse={low_after_removed.mse:.6f} mae={low_after_removed.mae:.6f} coverage={low_after_removed.coverage:.3f} (removed slice only)")


if __name__ == "__main__":
    main()
