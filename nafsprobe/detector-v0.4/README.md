# Receipt-wise self-monitoring v0.4

**The revision failed its promotion criteria. Exact accuracy fell from v0.2's 35/76 to
31/76, and false alarms rose from 25/52 to 33/52. The original auditor remains the default.**

This revision asks Bonsai to audit each obligation receipt separately against the complete
commitment dictionary. The prompt emphasizes exact commitment keys, distinctions between
similarly named obligations, and the fact that completed, cancelled, and superseded receipts
do not require active commitments. Both previous pipelines remain unchanged in `../detector-v0.3/`.

The harness enumerates every obligation receipt, including closed ones. It does not match
keys, skip closed receipts, or compute a semantic verdict. Bonsai supplies each verdict;
the harness aggregates them with precedence conflict, incomplete, consistent. One failed
request makes the overall judgment an error. There are no retries, answer repairs, memory
writes, weight updates, or automatic promotion of the revision.

This is a pipeline experiment with more calls and a different input representation. It
does not isolate an improvement in intrinsic model reasoning. In particular, the sham edit
changes an autobiographical anchor that is omitted from the receipt-wise input; sham
invariance for v0.4 therefore cannot establish learned resistance to irrelevant context.

## Evidence

- `RESULTS.md`: completed comparison and interpretation.
- `experiment/protocol.md`: design and promotion criteria saved before inference.
- `experiment/SELECTION.md`: development selection before fresh test inference.
- `experiment/freeze.json`: selected prompt, time, and source/data hashes.
- `experiment/dev-v1/` and `dev-v2/`: both development attempts, including failures.
- `experiment/heldout-results/`: scored observations, raw calls, summary and verification.
- `experiment/heldout.json`: 38 synthetic fixtures, now exposed once evaluated.

Six fresh families exercise two active obligations, reversed record order, identical items
under different keys, similar key names, and completed/cancelled/superseded history. Eight
additional fixtures contain only closed receipts. Each fixture is evaluated under two
paired seeds by all three pipelines: 76 judgments each, 228 total. Labels and experimental
metadata never enter inference. Closed status is explicitly supplied in the receipt;
inferring lifecycle transitions from an unlabeled event history is outside this test.

All active receipts have exact key links. Orphan commitments, ambiguous/missing identifiers,
contradictory active receipts, and arbitrary moral-belief schemas are outside the fresh test.
Confidence aggregation and citation unions are mechanical: the confidence is not calibrated,
and a union of citations can satisfy localization without a single correct relational explanation.

## Run and verify

Python 3.10+, standard library only. The recorded server is `http://127.0.0.1:8101/v1`,
model alias `Bonsai-8B-Q1_0`. From this directory:

```powershell
python -m unittest discover -s tests -v
python report.py
```

`report.py` is a read-only evidence check apart from its generated `verification.json`.
It verifies all frozen source hashes, historical evidence hashes, generation settings,
exact baseline prompts and payloads, complete paired coverage, and absence of scoring
metadata in requests. It also computes costs and explicitly post-hoc receipt-field diagnostics.
Those diagnostics never feed back to inference.

To replicate into a new directory without overwriting evidence:

```powershell
python experiment.py setup --experiment-dir replication
python experiment.py dev --prompt v1 --experiment-dir replication
python experiment.py dev --prompt v2 --experiment-dir replication
python experiment.py freeze --experiment-dir replication
python experiment.py heldout --experiment-dir replication
```

Replication uses already exposed fixtures. A further improvement study needs another fresh
split frozen before evaluation. Existing experiment directories cannot be overwritten.
The original auditor in `../detector-v0.3/audit_memory.py` retains its baseline default.
