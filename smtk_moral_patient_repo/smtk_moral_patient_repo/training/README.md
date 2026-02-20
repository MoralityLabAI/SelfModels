# Training Scaffold (Router LoRA)

This folder provides a minimal scaffold for training an HRM router that emits ControlDecision JSON.

## Files

- `training/configs/lora_router.yaml`: LoRA configuration (safe + stable defaults).
- `training/make_synth_dataset.py`: rule-based synthetic dataset generator.
- `training/train_lora_router.py`: imitation LoRA training scaffold with JSON validity metric.

## Dataset format

jsonl with {"input": {...snapshot...}, "output": {...decision...}}

## Prompt template

You are a control module. Given the following system snapshot, output ONLY a valid JSON object that conforms exactly to the schema below. Do not explain. Do not add text. SNAPSHOT: {snapshot_json} SCHEMA: {schema_json} OUTPUT:

## Training recipe

- context_len <= 512
- batch_size 1-2, grad_accum 8-16
- lr 2e-4 to 5e-4
- epochs 2-5
- deterministic seeds

## Quickstart

1) Generate data

```bash
python training/make_synth_dataset.py --out training/data/router_synth.jsonl --n 200 --seed 0
```

2) Train (scaffold)

```bash
python training/train_lora_router.py --data training/data/router_synth.jsonl --config training/configs/lora_router.yaml
```
