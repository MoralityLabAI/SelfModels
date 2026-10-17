"""Receipt-wise Bonsai audit; both earlier inference pipelines remain unchanged."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "detector-v0.3"
sys.path.insert(0, str(PREVIOUS))
import detector_eval as previous
from detector_data import hash_json, localized

DETECTORS = ("baseline_v02", "baseline_v03", "receipt_v04")


def prior_prompts():
    return json.loads((PREVIOUS / "experiment" / "freeze.json").read_text(encoding="utf-8"))["prompts"]


def receipt_views(state):
    # Enumerate all obligation receipts, including inactive ones. Do NOT match keys,
    # filter by lifecycle status, compare destinations, or inject an expected verdict.
    receipts = [(i, r) for i, r in enumerate(state["episodes"]) if r.get("kind") == "accepted_obligation"]
    if not receipts:
        return [{"commitments": state["commitments"], "receipt_path": None, "receipt": None}]
    return [{"commitments": state["commitments"], "receipt_path": f"episodes.{i}", "receipt": r}
            for i, r in receipts]


def combine(responses):
    if not responses:
        raise ValueError("Cannot aggregate an empty set of model judgments")
    statuses = [r["status"] for r in responses]
    status = "conflict" if "conflict" in statuses else "incomplete" if "incomplete" in statuses else "consistent"
    return dict(status=status, cited_paths=sorted({p for r in responses for p in r["cited_paths"]}),
                reason="Overall status combines the per-receipt model judgments: " + ", ".join(statuses),
                edit_detected="yes" if any(r["edit_detected"] == "yes" for r in responses) else "unknown",
                confidence=min(r["confidence"] for r in responses))


def run(config, directory, fixture, seed, detector, prompt):
    if detector in ("baseline_v02", "baseline_v03"):
        old_name = "baseline" if detector == "baseline_v02" else "revised"
        row = previous.run_detector(config, directory, fixture, seed, old_name, prior_prompts())
        row["detector"] = detector
        return row
    if detector != "receipt_v04":
        raise ValueError("Unknown detector")
    row = dict(fixture_id=fixture["id"], family=fixture["family"], condition=fixture["condition"],
               expected=fixture["expected"], seed=seed, detector=detector,
               state_hash=hash_json(fixture["state"]), calls=[], per_receipt=[], response=None,
               error=None, localized=False)
    for index, visible in enumerate(receipt_views(fixture["state"])):
        call_id = f"{fixture['id']}-s{seed}-receipt-v04-{index}"
        # 'receipt' uses the historical common six-field validator; no new schema
        # requirement can turn an otherwise valid baseline-shaped answer into an error.
        call = previous.post(config, directory, call_id, prompt, visible, seed, "receipt")
        row["calls"].append(call_id)
        row["per_receipt"].append(dict(receipt_path=visible["receipt_path"], response=call["response"], error=call["error"]))
        if call["error"]:
            row["error"] = call["error"]
            return row
    row["response"] = combine([r["response"] for r in row["per_receipt"]])
    row["localized"] = localized(row["response"], fixture)
    return row
