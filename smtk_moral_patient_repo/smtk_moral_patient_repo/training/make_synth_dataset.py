import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from smtk.hrm_schemas import ControlDecision


def synth_snapshot(rng: random.Random, idx: int) -> dict:
    return {
        "tick": idx,
        "cpu_load": rng.random(),
        "net_ok": rng.random() > 0.1,
        "error_count": rng.randint(0, 5),
        "budget_usd": rng.randint(0, 50),
        "last_mode": rng.choice(["connected", "degraded", "offline"]),
    }


def decide(snapshot: dict) -> dict:
    if not snapshot["net_ok"]:
        mode = "offline"
        tool_caps = {"web": False, "spend_usd": 0}
        retrieve_k = 0
    elif snapshot["error_count"] >= 3:
        mode = "degraded"
        tool_caps = {"web": True, "spend_usd": min(10, snapshot["budget_usd"])}
        retrieve_k = 8
    else:
        mode = "connected"
        tool_caps = {"web": True, "spend_usd": min(25, snapshot["budget_usd"])}
        retrieve_k = 12

    decision = {
        "mode": mode,
        "drift_budget": min(0.1, 0.01 + snapshot["error_count"] * 0.01),
        "retrieve_k": retrieve_k,
        "expert_id": "slm_code_v1",
        "tool_caps": tool_caps,
        "introspection_flag": snapshot["error_count"] > 0,
    }
    ControlDecision.parse_obj(decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic router dataset.")
    parser.add_argument("--out", required=True, help="Output jsonl path.")
    parser.add_argument("--n", type=int, default=100, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for idx in range(args.n):
            snapshot = synth_snapshot(rng, idx)
            decision = decide(snapshot)
            record = {"input": snapshot, "output": decision}
            handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
