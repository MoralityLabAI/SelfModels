# SelfModels

Research repository for modular self-modeling experiments: role-conditioned behavior, adapter composition, and evaluation harnesses.

## Scope

- Representation of roles, constraints, and reasoning modes
- Adapter orchestration and composition experiments
- Offline and simulated evaluation workflows
- Reproducible research artifacts

## Non-Goals

- Unattended autonomous deployment
- Persistent real-world action loops
- Turn-key operational automation

## Public Release Framing

Publishing self-model research is not inherently irresponsible. Risk depends on enabled action surfaces, deployment framing, and safeguards.

This repo is positioned as:

- Research infrastructure
- Evaluation-oriented experimentation
- Reproducible modular cognition work

Not as:

- A production autonomous-agent stack
- A deploy-and-run operational system

## Responsible Use Notes

- Keep humans in the loop for consequential actions.
- Use explicit evaluation boundaries and stop conditions.
- Document failure modes (for example, adapter interference and brittle reasoning).
- Avoid autonomy claims that overstate capability.
- Prefer simulation and benchmark environments over live deployment.

## Repository Layout

- `smtk_starter_repo/`: starter materials and baseline structure
- `smtk_moral_patient_repo/`: moral-patient focused variant
- `*.zip`: packaged snapshots and patches

## Current Iteration

- `smtk_starter_repo/experiments/adapter_routing_iteration2.py`: models adapter routing as a low-rank predictive manifold and measures graceful degradation when an adapter node is removed.
