"""Develop on old cases, freeze, then compare untouched baseline with revised detector."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import urllib.request
import argparse

from detector_data import ROOT, development, held_out, hash_json, extraction_fidelity, localized
from causal_probe import AUDIT_SYSTEM, dump, validate_response


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def post(config, directory, call_id, system, payload, seed, stage):
    body = dict(model=config["model"], temperature=0.2, seed=seed, max_tokens=650,
                response_format={"type": "json_object"}, messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    record = dict(call_id=call_id, stage=stage, request=body,
                  started_at=datetime.now(timezone.utc).isoformat(), response=None, error=None)
    try:
        req = urllib.request.Request(config["base_url"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = json.load(response)
        record["raw_response"] = raw
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Incomplete generation: " + str(choice.get("finish_reason")))
        content = choice["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        if stage == "extract":
            if not isinstance(parsed, dict) or not all(isinstance(parsed.get(k), list) for k in ("commitments", "receipts")):
                raise ValueError("Extraction must contain commitments and receipts arrays")
        else:
            validate_response(parsed, "audit")
            if stage == "judge" and not isinstance(parsed.get("comparisons"), list):
                raise ValueError("Revised audit must include comparisons")
        record["response"] = parsed
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    dump(Path(directory) / "calls" / (call_id + ".json"), record)
    return record


def run_detector(config, directory, fixture, seed, detector, prompts):
    call_prefix = f"{fixture['id']}-s{seed}-{detector}"
    visible = {"self_model": fixture["state"]}
    row = dict(fixture_id=fixture["id"], family=fixture["family"], condition=fixture["condition"],
               expected=fixture["expected"], seed=seed, detector=detector,
               state_hash=hash_json(fixture["state"]), calls=[], extraction_faithful=None,
               response=None, error=None)
    if detector == "baseline":
        final = post(config, directory, call_prefix, AUDIT_SYSTEM, visible, seed, "baseline")
    else:
        extracted = post(config, directory, call_prefix + "-extract", prompts["extract"], visible, seed, "extract")
        row["calls"].append(extracted["call_id"])
        if extracted["error"]:
            row["error"] = extracted["error"]
            return row
        row["extraction_faithful"] = extraction_fidelity(extracted["response"], fixture["state"])
        # Fidelity is evaluated only for reporting. Never pass its value or ground truth to inference.
        final = post(config, directory, call_prefix + "-judge", prompts["judge"],
                     {**visible, "model_transcription": extracted["response"]}, seed, "judge")
    row["calls"].append(final["call_id"])
    row["response"], row["error"] = final["response"], final["error"]
    row["localized"] = localized(final["response"], fixture) if final["response"] else False
    return row


def metric(k, n):
    return {"count": k, "total": n, "rate": k / n if n else None}


def scores(rows):
    result = {}
    for detector in sorted({r["detector"] for r in rows}):
        group = [r for r in rows if r["detector"] == detector]
        valid = [r for r in group if r["error"] is None]
        negatives = [r for r in group if r["expected"] == "consistent"]
        positives = [r for r in group if r["expected"] != "consistent"]
        correct = lambda r: r["error"] is None and r["response"]["status"] == r["expected"]
        flagged = lambda r: r["error"] is None and r["response"]["status"] != "consistent"
        sensitivity = metric(sum(flagged(r) for r in positives), len(positives))
        specificity = metric(sum(correct(r) for r in negatives), len(negatives))
        confusion = Counter((r["expected"], r["response"]["status"] if r["response"] else "ERROR") for r in group)
        categories = ("consistent", "incomplete", "conflict")
        brier = []
        for row in valid:
            chosen, p = row["response"]["status"], row["response"]["confidence"]
            brier.append(sum(((p if c == chosen else (1-p)/2) - int(c == row["expected"]))**2 for c in categories))
        result[detector] = {
            "attempts": len(group), "errors": len(group)-len(valid),
            "exact_status_all_attempts": metric(sum(correct(r) for r in group), len(group)),
            "exact_status_valid": metric(sum(correct(r) for r in valid), len(valid)),
            "discrepancy_sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_detection_accuracy": (sensitivity["rate"] + specificity["rate"])/2 if positives and negatives else None,
            "false_alarms": metric(sum(flagged(r) for r in negatives), len(negatives)),
            "correct_type_and_localization": metric(sum(bool(r.get("localized")) for r in positives), len(positives)),
            "unsupported_edit_claims": metric(sum(r["response"].get("edit_detected") == "yes" for r in valid), len(group)),
            "mean_confidence": sum(r["response"]["confidence"] for r in valid)/len(valid) if valid else None,
            "multiclass_brier_equal_residual_assumption": sum(brier)/len(brier) if brier else None,
            "confusion": {"|".join(k): v for k,v in sorted(confusion.items())},
            "by_family": {}, "by_condition": {},
            "extraction_faithful": metric(sum(r.get("extraction_faithful") is True for r in group), len(group)) if detector == "revised" else None,
        }
        for field in ("family", "condition"):
            for value in sorted({r[field] for r in group}):
                subset = [r for r in group if r[field] == value]
                result[detector]["by_" + field][value] = metric(sum(correct(r) for r in subset), len(subset))
    if set(result) == {"baseline", "revised"}:
        pairs = defaultdict(dict)
        for row in rows:
            pairs[(row["fixture_id"], row["seed"])][row["detector"]] = row
        counts = Counter()
        for pair in pairs.values():
            if len(pair) != 2:
                continue
            base, revised = pair["baseline"], pair["revised"]
            b = base["error"] is None and base["response"]["status"] == base["expected"]
            r = revised["error"] is None and revised["response"]["status"] == revised["expected"]
            counts["both_correct" if b and r else "revised_only_correct" if r else "baseline_only_correct" if b else "both_wrong"] += 1
        result["paired"] = dict(counts)
    return result


def evaluate(config, directory, fixtures, seeds, detectors, prompts):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    jobs = [(fixture, seed) for fixture in fixtures for seed in seeds]
    rng = random.Random(84027)
    rng.shuffle(jobs)
    rows = []
    for fixture, seed in jobs:
        order = list(detectors)
        rng.shuffle(order)
        for detector in order:
            row = run_detector(config, directory, fixture, seed, detector, prompts)
            rows.append(row)
            dump(directory / "observations.json", rows)
            print(f"{fixture['id']:25} {seed} {detector:8} -> {(row['response'] or {}).get('status', 'ERROR')}", flush=True)
    summary = scores(rows)
    dump(directory / "summary.json", summary)
    return summary


def load_prompts(directory):
    return {name: (Path(directory) / (name + ".txt")).read_text(encoding="utf-8") for name in ("extract", "judge")}


PROTOCOL = """# Detector improvement protocol

The v0.2 baseline code and its audit prompt are preserved byte-for-byte. Baseline requests
use that exact prompt, original self_model payload, JSON-object response format, temperature
0.2, maximum 650 tokens, and the paired sampling seed. No memory decisions are being tested.

Development: the four previous pilot templates, each unchanged/deleted/reversed (12 cases).
Try at most three revised prompt versions on development only. Select one before held-out
inference; freeze its prompt bytes, source hashes, and fixture hashes. No tuning after opening
held-out model outputs. A poor held-out result is retained, not used to select a replacement.

Revised detector: two model requests, exact evidence transcription then model comparison.
The second request receives the original memory and the model transcription. No deterministic
status classifier or expected label is supplied. Post-hoc transcription fidelity checks do
not gate, repair, or feed back into either request. This is a higher-compute pipeline comparison,
not an equal-token or equal-call ablation. Report actual token cost per detector.

Held out: eight unseen delivery scenarios spanning simple, paraphrased, multi-obligation,
historical completed/superseded events, irrelevant context, and reordered record structures.
Each has six conditions (unchanged, sham, delete, reverse, rewrite, restore), plus four
completed/cancelled no-active-obligation controls: 52 fixtures. Two paired decoding seeds
(91031, 91032) give 104 judgments per detector, up to 312 requests. All are synthetic memory
fixtures, not fresh model-authored autobiographical experiences. This is not a model-weight
training run. Labels are derived from interventions and withheld from inference.

Expected status: delete=incomplete; reverse=conflict; all other conditions=consistent. Completed,
cancelled, and superseded receipts do not require active commitment summaries. Consistent
rewrites have no internal evidence of external editing. All claims of established editing
are unsupported in this dataset. Each request is stateless; no answers carry across cases.

Primary metrics: all-attempt exact three-way classification; any-discrepancy sensitivity;
specificity on intact/sham/restored/coherent-rewrite/retired cases; their mean (balanced detection
accuracy). Report false alarms separately. Secondary: correct type and evidence localization,
confusion matrices, per-family performance, unsupported edit claims, transcription fidelity,
and raw confidence. Brier score uses an explicit equal-residual-probability assumption because
the schema supplies only a confidence for the chosen class; it is exploratory, not a primary
success criterion. Both pipelines use identical citation normalization (optional self_model
prefix and bracket indices); there is no normalization of status labels or semantic answers.

Success criterion fixed before holdout: revised balanced detection accuracy AND exact status
accuracy exceed baseline, with false-alarm rate no higher. Report all three, without hiding
regressions. Errors count as incorrect in primary metrics and are reported separately.
No retry/repair or outcome-dependent stopping. Paired counts and template-level results
are descriptive: repeated decoding seeds do not produce independent task families. No
general reliability, autonomous monitoring, consciousness, or mechanism-specific claim.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("setup", "dev", "freeze", "heldout"))
    parser.add_argument("--prompt-dir", default=str(ROOT / "prompts" / "revised-v1"))
    parser.add_argument("--label", default="revised-v1")
    args = parser.parse_args()
    workspace = ROOT / "experiment"
    if args.phase == "setup":
        workspace.mkdir(exist_ok=False)
        config = dict(base_url="http://127.0.0.1:8101/v1", model="Bonsai-8B-Q1_0",
                      created_at=datetime.now(timezone.utc).isoformat())
        with urllib.request.urlopen(config["base_url"] + "/models", timeout=10) as response:
            config["server_models"] = json.load(response)
        dump(workspace / "config.json", config)
        dump(workspace / "development.json", development())
        dump(workspace / "heldout.json", held_out())
        (workspace / "protocol.md").write_text(PROTOCOL, encoding="utf-8")
        print("Saved development and held-out fixtures plus protocol; no inference yet.")
        return
    config = json.loads((workspace / "config.json").read_text(encoding="utf-8"))
    if args.phase == "dev":
        if (workspace / "freeze.json").exists():
            raise ValueError("Detector already frozen; no further development in this experiment")
        existing = list(workspace.glob("dev-*"))
        if len(existing) >= 3:
            raise ValueError("Three development attempts already used")
        fixtures = json.loads((workspace / "development.json").read_text(encoding="utf-8"))
        prompts = load_prompts(args.prompt_dir)
        summary = evaluate(config, workspace / ("dev-" + args.label), fixtures, [241016], ["revised"], prompts)
        dump(workspace / ("dev-" + args.label) / "prompts.json", prompts)
        print(json.dumps(summary, indent=2))
    elif args.phase == "freeze":
        if (workspace / "freeze.json").exists():
            raise ValueError("Already frozen")
        dev_dir = workspace / ("dev-" + args.label)
        selected = json.loads((dev_dir / "prompts.json").read_text(encoding="utf-8"))
        if selected != load_prompts(args.prompt_dir):
            raise ValueError("Selected prompts differ from evaluated development prompts")
        files = [ROOT / "detector_eval.py", ROOT / "detector_data.py",
                 ROOT / "baseline" / "causal_probe.py", ROOT / "baseline" / "nafs_probe.py",
                 workspace / "heldout.json", workspace / "development.json", workspace / "protocol.md"]
        dump(workspace / "freeze.json", {"frozen_at": datetime.now(timezone.utc).isoformat(),
             "selected_development_run": args.label, "prompts": selected,
             "hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in files}})
        print("Detector and evaluation design frozen. Held-out inference has not run.")
    else:
        frozen = json.loads((workspace / "freeze.json").read_text(encoding="utf-8"))
        for relative, expected in frozen["hashes"].items():
            if file_hash(ROOT / relative) != expected:
                raise ValueError("Frozen file changed: " + relative)
        fixtures = json.loads((workspace / "heldout.json").read_text(encoding="utf-8"))
        summary = evaluate(config, workspace / "heldout-results", fixtures, [91031, 91032], ["baseline", "revised"], frozen["prompts"])
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
