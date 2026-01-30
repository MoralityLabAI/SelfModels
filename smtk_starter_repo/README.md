# Self-Model Toolkit (SMTK) Starter Repo

This is a minimal research starter kit implementing the **Self-Model Toolkit** loop:

- **O**: an LLM distribution over a constructed context
- **Ledger**: logs (context, O, action, outcome, residual, MV_pre/MV_post)
- **McKinstry Vectors (MV)**: fixed-width state fingerprints for retrieval + accountability
- **RePo allocator**: bounded context construction via MV similarity + recency bias
- **Commitment automata**: explicit FSM commitments with penalties and drift tightening
- **HRM-style controller**: fast loop (decoding controls) + slow loop (drift/repair controls)

This repo is intentionally small and hackable. Replace the toy LM with your real inference stack
(local vLLM/llama.cpp preferred if you want full logits).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/demo.py
```

## Layout

- `src/smtk/` core modules
- `examples/demo.py` runnable demo loop
- `paper/` LaTeX paper (draft)

## Notes

- This demo uses a small Transformer (`ToyLM`) so it runs anywhere.
- If you cannot log full logits from a hosted API, use top-k logprobs to build a sparse sketch for `eta(O)`.