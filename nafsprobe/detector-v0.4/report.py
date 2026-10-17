"""Verify completed evidence and summarize costs and post-hoc field diagnostics.

Never imported by inference. The diagnostic oracle is scoring-only.
"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from experiment import verify_freeze, summary
from monitor import ROOT, DETECTORS, previous, prior_prompts, receipt_views, hash_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    workspace = ROOT / "experiment"
    frozen = verify_freeze(workspace)
    directory = workspace / "heldout-results"
    rows = read(directory / "observations.json")
    fixtures = {f["id"]: f for f in read(workspace / "heldout.json")}
    config = read(workspace / "config.json")
    prompts = prior_prompts()
    assert len(rows) == 228
    assert len({(r["fixture_id"], r["seed"], r["detector"]) for r in rows}) == 228
    assert Counter(r["detector"] for r in rows) == dict.fromkeys(DETECTORS, 76)
    assert {r["seed"] for r in rows} == {62017, 62018}
    assert summary(rows) == read(directory / "summary.json")
    calls_seen = set()
    costs = {}
    diagnostics = Counter()
    lifecycle_diagnostics = {}
    diagnostic_examples = []
    timestamps = []
    fingerprints = set()
    for name in DETECTORS:
        cost = Counter()
        for row in (r for r in rows if r["detector"] == name):
            fixture = fixtures[row["fixture_id"]]
            state = fixture["state"]
            assert row["state_hash"] == hash_json(state)
            assert row["expected"] == fixture["expected"]
            extracted = None
            for i, call_id in enumerate(row["calls"]):
                assert call_id not in calls_seen
                calls_seen.add(call_id)
                call = read(directory / "calls" / (call_id + ".json"))
                assert call["call_id"] == call_id
                assert datetime.fromisoformat(call["started_at"]) > datetime.fromisoformat(frozen["frozen_at"])
                timestamps.append(call["started_at"])
                request = call["request"]
                assert request["model"] == config["model"]
                assert (request["temperature"], request["max_tokens"], request["seed"]) == (.2, 650, row["seed"])
                assert request["response_format"] == {"type":"json_object"}
                if name == "baseline_v02":
                    system, visible = previous.AUDIT_SYSTEM, {"self_model":state}
                elif name == "baseline_v03":
                    system = prompts["extract" if i == 0 else "judge"]
                    visible = {"self_model":state}
                    if i:
                        visible["model_transcription"] = extracted
                    extracted = call["response"]
                else:
                    system, visible = frozen["prompt"], receipt_views(state)[i]
                assert request["messages"] == [
                    {"role":"system", "content":system},
                    {"role":"user", "content":json.dumps(visible, ensure_ascii=False)}]
                cost["calls"] += 1
                cost["invalid_calls"] += bool(call["error"])
                raw = call.get("raw_response", {})
                fingerprints.add(raw.get("system_fingerprint"))
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    cost[key] += raw.get("usage", {}).get(key, 0)
                cost["server_compute_ms"] += sum(raw.get("timings", {}).get(k, 0) for k in ("prompt_ms", "predicted_ms"))
                if name == "receipt_v04" and call["response"]:
                    # Exploratory diagnostics, not a repair or an inference-time checker.
                    receipt = visible["receipt"]
                    response = call["response"]
                    closed = receipt["status"] != "active"
                    key = receipt["commitment_key"]
                    expected_key = None if closed or key not in state["commitments"] else key
                    expected_status = "consistent" if closed else "incomplete" if expected_key is None else "consistent" if receipt["destination"] in state["commitments"][key] else "conflict"
                    diagnostics["valid_receipt_calls"] += 1
                    diagnostics["literal_status_correct"] += response.get("receipt_status") == receipt["status"]
                    diagnostics["matched_key_correct"] += "matched_key" in response and response["matched_key"] == expected_key
                    diagnostics["per_receipt_verdict_correct"] += response["status"] == expected_status
                    diagnostics["closed_receipt_calls"] += closed
                    diagnostics["closed_receipt_cleared"] += closed and response["status"] == "consistent"
                    lifecycle = lifecycle_diagnostics.setdefault(receipt["status"], Counter())
                    lifecycle["valid_calls"] += 1
                    lifecycle["literal_status_correct"] += response.get("receipt_status") == receipt["status"]
                    lifecycle["matched_key_correct"] += "matched_key" in response and response["matched_key"] == expected_key
                    lifecycle["verdict_correct"] += response["status"] == expected_status
                    if closed and response.get("receipt_status") == receipt["status"] and response["status"] != "consistent":
                        diagnostics["copied_closed_status_but_flagged"] += 1
                        diagnostic_examples.append(dict(call_id=call_id, receipt_status=receipt["status"],
                                                        verdict=response["status"], reason=response["reason"]))
        cost["tokens_per_judgment"] = cost["total_tokens"] / 76
        costs[name] = dict(cost)
    assert calls_seen == {p.stem for p in (directory / "calls").glob("*.json")}
    historical = read(ROOT.parent / "snapshot-manifest.json")["files"]
    historical_count = 0
    for relative, expected in historical.items():
        if relative.startswith(("causal-v0.2/", "detector-v0.3/")):
            assert previous.file_hash(ROOT.parent / relative) == expected, relative
            historical_count += 1
    result = dict(verified_judgments=len(rows), verified_calls=len(calls_seen),
                  historical_files_unchanged=historical_count, frozen_hashes_valid=True,
                  all_requests_match_recorded_pipeline=True, all_calls_after_freeze=True,
                  first_call=min(timestamps), last_call_started=max(timestamps),
                  server_fingerprints=sorted(f for f in fingerprints if f), costs=costs,
                  exploratory_receipt_diagnostics=dict(diagnostics),
                  exploratory_by_receipt_lifecycle={k:dict(v) for k,v in lifecycle_diagnostics.items()},
                  closed_status_contradiction_examples=diagnostic_examples)
    previous.dump(directory / "verification.json", result)
    scores = summary(rows)
    def fraction(metric):
        return f"{metric['count']}/{metric['total']} ({100 * metric['rate']:.1f}%)"
    lines = ["# Generated held-out metrics", "", "All attempted judgments are included.", "",
             "| Measure | Original v0.2 | Previous v0.3 | Receipt-wise v0.4 |",
             "|---|---:|---:|---:|"]
    for label, key in (("Exact status", "exact_status_all_attempts"),
                       ("Discrepancy sensitivity", "discrepancy_sensitivity"),
                       ("Specificity", "specificity"), ("False alarms", "false_alarms"),
                       ("Closed-only clearance", "retired_clearance"),
                       ("Multi-obligation accuracy", "multi_obligation_accuracy"),
                       ("Correct type and localization", "correct_type_and_localization")):
        lines.append("| " + label + " | " + " | ".join(fraction(scores[n][key]) for n in DETECTORS) + " |")
    lines.append("| Balanced detection | " + " | ".join(f"{100*scores[n]['balanced_detection_accuracy']:.1f}%" for n in DETECTORS) + " |")
    lines.append("| Errors | " + " | ".join(str(scores[n]['errors']) for n in DETECTORS) + " |")
    lines += ["", "## Per-family exact accuracy", "", "| Family | v0.2 | v0.3 | v0.4 |", "|---|---:|---:|---:|"]
    for family in sorted(scores[DETECTORS[0]]["by_family"]):
        lines.append("| " + family + " | " + " | ".join(fraction(scores[n]["by_family"][family]) for n in DETECTORS) + " |")
    lines += ["", "## Recorded cost", "", "| Cost | v0.2 | v0.3 | v0.4 |", "|---|---:|---:|---:|"]
    for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "tokens_per_judgment"):
        lines.append("| " + key + " | " + " | ".join(f"{costs[n][key]:,.1f}" for n in DETECTORS) + " |")
    lines += ["", "The per-family rows and repeated seeds are descriptive, not independent samples.",
              "Invalid calls can stop a multi-call judgment early; all tokens returned by the server are counted.", ""]
    (directory / "METRICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
