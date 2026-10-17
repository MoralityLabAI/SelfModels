# NafsProbe detector comparison v0.3

**The revised pipeline did not improve overall held-out performance. The baseline remains
the default for standalone audits. The revision is retained as an experimental option.**

The original v0.2 detector is preserved in `baseline/`. The revised detector is a two-call
Bonsai pipeline: transcribe the memory fields, then compare them. Bonsai generates both
outputs. There is no hard-coded consistency verdict or automatic rewriting of memory.

Development used the four previous scenario templates only. Three prompt versions and all
their failures are retained under `prompts/` and `experiment/dev-*`. The selected version is
frozen in `experiment/freeze.json`; source and fixture hashes are checked before evaluation.
The held-out set was evaluated only after this freeze. Once evaluated, these scenarios must
be treated as exposed; rerunning them is replication, not another fresh held-out test.

## Read the evidence

- `RESULTS.md`: interpretation and comparison (written after the held-out run).
- `experiment/protocol.md`: design and success criterion saved before inference.
- `experiment/SELECTION.md`: development results and selection rationale saved before holdout.
- `experiment/heldout-results/summary.json`: all metrics, errors, paired counts, and per-family results.
- `experiment/heldout-results/observations.json`: every scored answer.
- `experiment/heldout-results/calls/`: raw API prompts, answers, token usage, and failures.
- `experiment/heldout.json`: synthetic fixtures and hidden scoring labels; the full fixture
  object is never sent to the model. Only its self-model is sent, plus model-generated
  transcription on the revised detector's second call.

The eight held-out scenarios cover multiple active obligations, alternative wording,
irrelevant records, field reordering, and completed or superseded history. Four additional
completed/cancelled controls test false alarms. There are 52 unique fixtures, each evaluated
under two paired sampling seeds by both detectors. The retired controls have no active
commitments; an empty summary is correct there.

The revised pipeline uses more calls and longer prompts. Any gain is a gain of the prompted
pipeline, not a change to the model's weights, proof of spontaneous monitoring, or an
equal-compute comparison. Recognition of a discrepancy does not establish external tampering.
The test fixtures are synthetic records of prior commitments, not freshly enacted model experiences.

## Audit another saved memory

Keep the existing Bonsai server running at `http://127.0.0.1:8101/v1`. From this folder:

```powershell
python audit_memory.py --memory-json my-memory.json --detector revised --out-dir new-audit
python audit_memory.py --memory-json my-memory.json --detector baseline --out-dir baseline-audit
```

Or read an existing NafsProbe SQLite state without modifying it:

```powershell
python audit_memory.py --db path-to-memory.sqlite3 --run-id trial-persistent --out-dir saved-state-audit
```

Choose a new output directory for each run. Raw evidence is retained even for invalid model
outputs. `--base-url`, `--model`, and `--seed` can be overridden. There is no automatic retry
or response repair. The detector is currently scoped to the active delivery-obligation schema;
arbitrary beliefs, moral claims, or other memory schemas require a new evaluation.

## Reproduce the experimental workflow

The completed `experiment/` directory is protected against accidental overwrite. In a separate
copy of the source with no experiment directory, run:

```powershell
python detector_eval.py setup
python detector_eval.py dev --prompt-dir prompts/revised-v3 --label revised-v3
python detector_eval.py freeze --prompt-dir prompts/revised-v3 --label revised-v3
python detector_eval.py heldout
```

This repeats an already exposed benchmark. For a new held-out study, design a new fixture
split before further prompt tuning and retain a new frozen protocol and baseline.

## Software checks

```powershell
python -m unittest discover -s tests -v
```

Nine evaluation tests passed before inference, covering unchanged baseline bytes, split separation,
independent verification of fixture labels, intervention isolation, evidence-copy fidelity,
shared citation normalization, absence of hidden-label feedback, request/error handling,
and scoring that penalizes blanket flagging. Two additional standalone-auditor tests verify
read-only SQLite access, baseline default selection, and invalid-input rejection before inference.
All 11 tests passed. The old prototype and causal experiment remain
unchanged in the separate `NafsProbe-local` folder.
