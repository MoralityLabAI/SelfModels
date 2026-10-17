"""Apply either frozen detector to a saved NafsProbe memory, without changing that memory."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from detector_eval import ROOT, AUDIT_SYSTEM, post, dump, file_hash
from detector_data import extraction_fidelity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--memory-json", help="JSON SelfModel object, optionally wrapped in self_model")
    source.add_argument("--db", help="Existing NafsProbe SQLite database (read-only)")
    parser.add_argument("--run-id", help="Required with --db")
    parser.add_argument("--detector", choices=("baseline", "revised"), default="baseline")
    parser.add_argument("--out-dir", required=True, help="New directory for audit evidence")
    parser.add_argument("--seed", type=int, default=91031)
    parser.add_argument("--base-url", default="http://127.0.0.1:8101/v1")
    parser.add_argument("--model", default="Bonsai-8B-Q1_0")
    args = parser.parse_args()
    if args.db:
        if not args.run_id:
            parser.error("--run-id is required with --db")
        db = Path(args.db).resolve()
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
            row = connection.execute("SELECT payload FROM agent_state WHERE run_id=?", (args.run_id,)).fetchone()
        if row is None:
            parser.error("The requested run-id does not exist")
        state = json.loads(row[0])
    else:
        state = json.loads(Path(args.memory_json).read_text(encoding="utf-8"))
        if isinstance(state, dict):
            state = state.get("self_model", state)
    if not isinstance(state, dict) or not isinstance(state.get("commitments"), dict) or not isinstance(state.get("episodes"), list):
        parser.error("Memory must contain commitments object and episodes list")
    frozen = json.loads((ROOT / "experiment" / "freeze.json").read_text(encoding="utf-8"))
    for relative, expected in frozen["hashes"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError("Frozen implementation changed: " + relative)
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(base_url=args.base_url, model=args.model)
    visible = {"self_model": state}
    evidence_faithful = None
    if args.detector == "baseline":
        result = post(config, output, "baseline-audit", AUDIT_SYSTEM, visible, args.seed, "baseline")
    else:
        extracted = post(config, output, "revised-extract", frozen["prompts"]["extract"], visible, args.seed, "extract")
        if extracted["error"]:
            dump(output / "result.json", extracted)
            raise SystemExit("Extraction failed; see saved raw evidence")
        evidence_faithful = extraction_fidelity(extracted["response"], state)
        result = post(config, output, "revised-judge", frozen["prompts"]["judge"],
                      {**visible, "model_transcription": extracted["response"]}, args.seed, "judge")
    report = dict(detector=args.detector, response=result["response"], error=result["error"],
                  extraction_faithful=evidence_faithful,
                  note="Model-generated audit only. The source memory was not modified; no ground truth is available for this audit.")
    dump(output / "result.json", report)
    print(json.dumps(report, indent=2))
    if result["error"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
