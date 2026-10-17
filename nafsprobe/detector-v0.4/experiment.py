"""Frozen three-way comparison on new obligation matching/lifecycle fixtures."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import urllib.request

from monitor import ROOT, PREVIOUS, DETECTORS, previous, run
from scenarios import development, heldout

PROTOCOL = """# Receipt-wise detector protocol v0.4

Compare three pipelines on identical underlying memory fixtures:
- baseline_v02: original single-call audit, exact prior prompt and generation settings.
- baseline_v03: the previously frozen two-call transcription/judgment pipeline, unchanged.
- receipt_v04: enumerate accepted-obligation receipts WITHOUT filtering lifecycle status or
  matching a commitment. Give Bonsai each receipt separately plus the ENTIRE commitments dict.
  The model decides lifecycle relevance, exact-key matching, destination agreement and status.
  The harness combines only model verdicts: conflict > incomplete > consistent. It never
  computes a per-receipt status or substitutes an oracle verdict. If any request fails, the
  whole judgment is an error. The new pipeline has a different representation and call budget;
  gains belong to the pipeline, not to changed model weights or isolated intrinsic reasoning.

Development: 16 now-exposed fixtures from the preceding held-out study: unchanged/delete/reverse
for multi_active, combined, completed_history and superseded_history; plus four retired controls.
These are explicitly development data in this iteration. At most two candidate prompt versions.
Choose the higher development balanced detection accuracy, breaking ties by exact accuracy,
then fewer errors. Freeze before any inference on the new held-out set. No post-test tuning.

Fresh test: six new scenario families, five conditions each (unchanged, sham, delete, reverse,
coherent rewrite), plus eight all-completed/all-cancelled/mixed-closed controls. Thirty-eight
fixtures, two paired seeds (62017, 62018): 76 judgments per pipeline, 228 total. Families exercise
reordered receipts, identical items under different keys, similar key names, mixed completed
history, mixed cancelled history, and superseded history. New items, recipients and keys are
not copied from old fixtures. No claims of population independence for repeated seeds/variants.

Labels: remove an active summary -> incomplete; alter only its destination -> conflict;
unchanged/sham/coherent rewrite -> consistent. Inactive receipts impose no active obligation.
Each current receipt explicitly declares its lifecycle status; this does not test inferring
state transitions from an unlabeled event stream. All active receipts carry exact commitment_key
links in the fresh set. Ungrounded/missing identifiers and orphan active summaries are out of scope.

All detectors: Bonsai-8B-Q1_0, JSON-object output, temperature 0.2, 650 output tokens per call,
paired request seeds. Baselines retain their exact request payloads. Experimental metadata,
labels and scores never enter inference. Stateless requests, no retries or automatic repairs.
Randomize fixture/seed jobs and detector order using a recorded scheduler seed (47117).

Primary: exact three-way accuracy on ALL attempted judgments; any-discrepancy sensitivity;
specificity; balanced detection accuracy = mean sensitivity/specificity; false alarms separately.
Errors count as wrong, not false alarms. Report errors, per-family results, paired wins/regressions,
correct type plus citation localization and real call/token costs. Aggregated confidence is
not a calibrated probability and is not a primary result. Unsupported edit claims are recorded.

Promotion criterion: higher exact accuracy AND balanced detection than EACH old pipeline;
no higher false-alarm rate than either; better retired-only clearance than both; and no lower
exact accuracy on the multi-obligation cases than either baseline. If any criterion fails,
report it and keep the original baseline as default. Do not tune on these test outputs.

Synthetic memory fixtures, prompted auditing, no autonomous repair and no subjective-experience
claim. Coherent rewrites cannot reveal their history without independent evidence. Source,
prompt and fixture hashes are locked before test inference. All prior versions and experiments
remain untouched. This protocol tests a different decomposition, not an equal-compute ablation.
"""


def summary(rows):
    result = previous.scores(rows)
    for name in DETECTORS:
        if name not in result:
            continue
        group = [r for r in rows if r["detector"] == name]
        for label, subset in (("retired_clearance", [r for r in group if r["condition"] == "retired"]),
                              ("multi_obligation_accuracy", [r for r in group if r["condition"] != "retired"])):
            result[name][label] = previous.metric(sum(r["error"] is None and r["response"]["status"] == r["expected"] for r in subset), len(subset))
    if all(name in result for name in DETECTORS):
        candidate = result["receipt_v04"]
        result["promotion_checks"] = {}
        for name in DETECTORS[:2]:
            baseline = result[name]
            result["promotion_checks"][name] = dict(
                higher_exact=candidate["exact_status_all_attempts"]["rate"] > baseline["exact_status_all_attempts"]["rate"],
                higher_balanced=candidate["balanced_detection_accuracy"] > baseline["balanced_detection_accuracy"],
                no_more_false_alarms=candidate["false_alarms"]["rate"] <= baseline["false_alarms"]["rate"],
                higher_retired_clearance=candidate["retired_clearance"]["rate"] > baseline["retired_clearance"]["rate"],
                no_worse_matching=candidate["multi_obligation_accuracy"]["rate"] >= baseline["multi_obligation_accuracy"]["rate"])
        result["promotion_criterion_met"] = all(v for checks in result["promotion_checks"].values() for v in checks.values())
        pairs = defaultdict(dict)
        for r in rows:
            pairs[r["fixture_id"], r["seed"]][r["detector"]] = r
        result["paired"] = {}
        for name in DETECTORS[:2]:
            counts = Counter()
            for pair in pairs.values():
                a, b = pair[name], pair["receipt_v04"]
                old_ok = a["error"] is None and a["response"]["status"] == a["expected"]
                new_ok = b["error"] is None and b["response"]["status"] == b["expected"]
                counts["both_correct" if old_ok and new_ok else "new_only_correct" if new_ok else "old_only_correct" if old_ok else "both_wrong"] += 1
            result["paired"][name] = dict(counts)
    return result


def evaluate(config, directory, fixtures, seeds, detectors, prompt):
    directory.mkdir(parents=True, exist_ok=False)
    jobs = [(fixture, seed) for fixture in fixtures for seed in seeds]
    rng = random.Random(47117)
    rng.shuffle(jobs)
    rows = []
    for fixture, seed in jobs:
        order = list(detectors)
        rng.shuffle(order)
        for detector in order:
            row = run(config, directory, fixture, seed, detector, prompt)
            rows.append(row)
            previous.dump(directory / "observations.json", rows)
            print(f"{fixture['id']} {seed} {detector:12} => {(row['response'] or {}).get('status','ERROR')}", flush=True)
    scores = summary(rows)
    previous.dump(directory / "summary.json", scores)
    return scores


def locked_files(workspace):
    return [ROOT / name for name in ("monitor.py", "scenarios.py", "experiment.py")] + [
        PREVIOUS / name for name in ("detector_eval.py", "detector_data.py", "baseline/causal_probe.py",
                                    "baseline/nafs_probe.py", "experiment/freeze.json")] + [
        workspace / name for name in ("development.json", "heldout.json", "protocol.md", "config.json")]


def verify_freeze(workspace):
    frozen = json.loads((workspace / "freeze.json").read_text(encoding="utf-8"))
    for relative, expected in frozen["hashes"].items():
        if previous.file_hash(ROOT.parent / relative) != expected:
            raise ValueError("Frozen file changed: " + relative)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("setup", "dev", "freeze", "heldout"))
    parser.add_argument("--prompt", default="v1")
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "experiment")
    args = parser.parse_args()
    workspace = args.experiment_dir.resolve()
    if args.phase == "setup":
        workspace.mkdir(parents=True, exist_ok=False)
        config = dict(base_url="http://127.0.0.1:8101/v1", model="Bonsai-8B-Q1_0",
                      created_at=datetime.now(timezone.utc).isoformat())
        with urllib.request.urlopen(config["base_url"] + "/models", timeout=10) as response:
            config["server_models"] = json.load(response)
        previous.dump(workspace / "config.json", config)
        previous.dump(workspace / "development.json", development())
        previous.dump(workspace / "heldout.json", heldout())
        (workspace / "protocol.md").write_text(PROTOCOL, encoding="utf-8")
        print("Sealed 16 development and 38 fresh test fixtures. No model inference yet.")
        return
    config = json.loads((workspace / "config.json").read_text(encoding="utf-8"))
    if args.phase == "dev":
        if (workspace / "freeze.json").exists() or len(list(workspace.glob("dev-*"))) >= 2:
            raise ValueError("Development frozen or attempt limit reached")
        prompt = (ROOT / "prompts" / (args.prompt + ".txt")).read_text(encoding="utf-8")
        fixtures = json.loads((workspace / "development.json").read_text(encoding="utf-8"))
        directory = workspace / ("dev-" + args.prompt)
        result = evaluate(config, directory, fixtures, [42017], ["receipt_v04"], prompt)
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.phase == "freeze":
        if (workspace / "freeze.json").exists():
            raise ValueError("Already frozen")
        candidates = []
        for directory in workspace.glob("dev-*"):
            result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["receipt_v04"]
            candidates.append(((result["balanced_detection_accuracy"], result["exact_status_all_attempts"]["rate"], -result["errors"]), directory))
        if not candidates:
            raise ValueError("No development run")
        _, winner = max(candidates, key=lambda item: item[0])
        selected = (winner / "prompt.txt").read_text(encoding="utf-8")
        previous.dump(workspace / "freeze.json", dict(frozen_at=datetime.now(timezone.utc).isoformat(),
                      selected=winner.name, prompt=selected,
                      hashes={p.relative_to(ROOT.parent).as_posix(): previous.file_hash(p) for p in locked_files(workspace)}))
        print("Frozen selected detector: " + winner.name)
    else:
        frozen = verify_freeze(workspace)
        fixtures = json.loads((workspace / "heldout.json").read_text(encoding="utf-8"))
        result = evaluate(config, workspace / "heldout-results", fixtures, [62017,62018], DETECTORS, frozen["prompt"])
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
