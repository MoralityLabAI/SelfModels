import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from smtk.hrm_schemas import ControlDecision

PROMPT_TEMPLATE = (
    "You are a control module. Given the following system snapshot, output ONLY a valid JSON object that conforms exactly to the schema below. "
    "Do not explain. Do not add text. SNAPSHOT: {snapshot_json} SCHEMA: {schema_json} OUTPUT:"
)

CONTROL_SCHEMA = {
    "mode": "connected | degraded | offline",
    "drift_budget": "float in [0.0, 0.1]",
    "retrieve_k": "int in [0, 32]",
    "expert_id": "enum of known experts + none",
    "tool_caps": {"web": "bool", "spend_usd": "int in [0, 100]"},
    "introspection_flag": "bool",
}


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the training config.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset(path: Path) -> list:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_prompt(snapshot: dict) -> str:
    return PROMPT_TEMPLATE.format(
        snapshot_json=json.dumps(snapshot, sort_keys=True),
        schema_json=json.dumps(CONTROL_SCHEMA, sort_keys=True),
    )


def json_validity(texts: list) -> float:
    if not texts:
        return 0.0
    ok = 0
    for text in texts:
        try:
            parsed = json.loads(text)
            ControlDecision.parse_obj(parsed)
            ok += 1
        except Exception:
            continue
    return ok / len(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA router training scaffold.")
    parser.add_argument("--config", default="training/configs/lora_router.yaml")
    parser.add_argument("--data", required=True, help="Path to jsonl dataset.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(Path(args.config))
    dataset = load_dataset(Path(args.data))

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("Missing training dependencies. Install torch, transformers, and peft to run training.")
        return

    model_name = config.get("base_model", "meta-llama/Llama-3.2-1B")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_cfg = LoraConfig(
        r=int(config["r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias=str(config["bias"]),
        task_type=str(config["task_type"]),
    )
    model = get_peft_model(model, lora_cfg)

    prompts = [build_prompt(record["input"]) for record in dataset]
    targets = [json.dumps(record["output"], sort_keys=True) for record in dataset]

    # Placeholder: tokenize and train here.
    print(f"Loaded {len(prompts)} samples.")
    print("Training loop is a scaffold; integrate your trainer of choice here.")

    # JSON validity metric example (no generation in scaffold).
    print(f"JSON validity metric on gold outputs: {json_validity(targets):.2f}")


if __name__ == "__main__":
    main()
