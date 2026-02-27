# Experiments

## Iteration 2: Adapter Routing Manifold

`adapter_routing_iteration2.py` models adapter selection as a compressed low-rank manifold over:

- destination/context index
- adapter index
- time index

It compares:

- explicit table lookup (exact but brittle)
- low-rank factor model (compressed, smoother failure behavior)

### Run

```bash
python experiments/adapter_routing_iteration2.py
```

### Expected signal

- Before removal:
  - table lookup has near-zero error and full coverage
  - low-rank has small approximation error and full coverage
- After removing one adapter:
  - table lookup loses coverage for that adapter slice
  - low-rank remains full-coverage but with increased error on/around removed slice

This is the core "failure as deformation, not deletion" hypothesis for adapter warehouses.

