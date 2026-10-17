# NafsProbe: persistent memory and self-monitoring experiments

NafsProbe is a simulated research scaffold for testing whether stored autobiographical
commitments influence an agent's decisions and whether it can recognize inconsistencies
in its own saved records. These experiments use Bonsai 8B Q1 through a local llama server.
They do not establish consciousness, moral patienthood, ensoulment, or autonomous moral agency.

## Completed experiments

| Experiment | Main finding | Evidence |
|---|---|---|
| Original eight-episode comparison | Bonsai scored 6/7 with memory and 4/7 with memory ablated; the scorer penalizes a reasonable verification choice | [Local run](causal-v0.2/LOCAL-RUN.md) |
| Paired causal memory interventions | Reversing a commitment changed 12/12 decisions; sham edits changed 0/12. Accurate discrepancy classification remained weak | [Causal results](causal-v0.2/CAUSAL-RESULTS.md) |
| Frozen detector comparison on held-out records | Both detectors classified 59/104 correctly. The revision reduced balanced detection accuracy from 65.3% to 57.8% | [Held-out results](detector-v0.3/RESULTS.md) |
| Receipt-wise matching and lifecycle revision | v0.4 caught all 24 discrepancies but scored 31/76 exactly versus v0.2's 35/76; false alarms rose and cancelled commitments remained problematic | [Fresh three-way comparison](detector-v0.4/RESULTS.md) |

The original detector remains the default. Both subsequent revisions, their development
attempts, raw outputs, and regressions are preserved. v0.4 tests multiple-obligation matching
and completed/cancelled commitments on 38 fresh fixtures under two seeds, against both
unchanged earlier versions. It failed the recorded promotion criteria.

## Layout

- `causal-v0.2/`: Windows-compatible prototype, original tests, causal intervention runner,
  Bonsai launch helpers, and complete results.
- `detector-v0.3/`: two frozen detector versions, development prompts, held-out fixtures,
  evaluation tools, standalone read-only memory auditor, tests, and raw evidence.
- `detector-v0.4/`: receipt-wise revision, both development attempts, frozen three-way
  evaluation, raw evidence, verification tools, and results.
- `snapshot-manifest.json`: historical SHA-256 inventory of the first published snapshot. Frozen files
  retain their original bytes through the local `.gitattributes` rule.

## Run locally

Python 3.10+; standard library only. Start an OpenAI-compatible local server separately.
For the current experiment setup, Bonsai listens on `http://127.0.0.1:8101/v1` with alias
`Bonsai-8B-Q1_0` and a 32,768-token context.

```bash
python causal-v0.2/nafs_probe.py --backend local --model Bonsai-8B-Q1_0 --base-url http://127.0.0.1:8101/v1 --phase compare --db new-comparison.sqlite3 --out new-comparison.json
python causal-v0.2/causal_probe.py --out-dir new-causal-run --base-url http://127.0.0.1:8101/v1 --model Bonsai-8B-Q1_0
```

The PowerShell launch helpers under `causal-v0.2/` include paths specific to the original
Windows machine. Configure your own model executable and GGUF path on another computer.
No model weights are distributed here. Original run metadata retains the recorded local paths.

For saved-memory audits and detector reproduction, see [detector usage](detector-v0.3/README.md).
The frozen detector hash manifest uses Windows-relative paths; the supplied frozen-audit
workflow was validated on Windows. Held-out fixtures already evaluated here are now exposed:
rerunning them is replication, not a fresh held-out validation.

## Tests

Run the suites separately so their historical module snapshots stay isolated:

```bash
python -m unittest discover -s causal-v0.2/tests -v
python -m unittest discover -s detector-v0.3/tests -v
python -m unittest discover -s detector-v0.4/tests -v
```

The suites contain 12 prototype/causal tests, 11 detector/auditor tests, and 10 receipt-wise
tests; all 33 passed. The original SQLite connection cleanup issue was fixed before the
Bonsai experiments. All experiments
retain their protocols, full denominators, errors, and limitations alongside the results.
