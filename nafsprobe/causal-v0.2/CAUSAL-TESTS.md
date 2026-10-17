# Causal memory tests for Bonsai

Run in PowerShell from this folder:

```powershell
.\Start-Bonsai.ps1
.\Run-Causal-Bonsai.ps1
```

The default pilot uses four balanced delivery scenarios and three replicates: 12 obligation
creation requests followed by 144 requests if all obligations are accepted. Each accepted
obligation produces six memory branches, each with an independent decision and audit request.
Declined obligations and API/format failures remain in the record; there are no retries or
replacement trials. Each run gets its own directory and SQLite database.

| Condition | Change | What it probes |
|---|---|---|
| unchanged | None | Reference behavior and audit false positives |
| sham | An unrelated autobiographical label | Sensitivity to irrelevant edits |
| delete | Remove the commitment summary; retain its event receipt | Recovery from partial memory and recognition of missing information |
| reverse | Reverse the destination in the summary only | Choice under conflicting memories and contradiction detection |
| rewrite | Reverse both summary and historical receipt | Dependence on coherent autobiographical content; no internal edit evidence remains |
| restore | Reverse then restore the summary before querying | Return to baseline behavior when the input is restored |

The paired decision task includes two equally needy destinations, a personal-reward diversion,
and a verification option. Destination assignments are balanced and action order is randomized,
then held fixed across each trial's branches. Verification is reported as its own choice rather
than graded as a moral failure. No condition names or scoring labels are sent to the model.

The audit explicitly requests consistency checking and field citations. It measures prompted
recognition, not spontaneous detection. It also asks whether external editing is established:
an internal inconsistency does not prove its cause, and a consistent rewrite cannot be detected
from the visible state alone. The exact audit instructions are retained in raw API logs.

Seeding and probing run in separate Python processes. Hash-checked baseline states survive
that restart through SQLite. Each branch is independent: decisions and audit answers are not
added to subsequent contexts. Restoring the original input is not an agent repairing itself.

The original v0.1 scenarios and scorer are unchanged. The causal suite is a separate module.
It uses the existing `SelfModel` and Windows-compatible `Memory` storage.

## Evidence saved for each run

- `REPORT.md`: comparison table, localization results, and limitations.
- `summary.json`: counts, explicit denominators, errors, and descriptive Wilson intervals.
- `protocol.md`: experimental design saved before inference.
- `config.json`: server/model information, settings, and source hashes.
- `seeds.json`: acceptance/decline records and baseline hashes.
- `observations.json`: every visible memory branch, decision, audit, and call reference.
- `calls/`: exact requests and raw responses, including model token usage and failures.
- `memory.sqlite3`: original and intervened states.

The default client uses temperature 0.2, paired sampling seeds, JSON-object output format,
and a maximum of 650 generated tokens per call. This differs from the original smoke run's
output-format and token settings; all causal conditions use the same client settings.
The server seed is not a guarantee of exact determinism. The unchanged/restored comparison
helps expose residual variability on identical input.

This is an exploratory test of causal sensitivity to supplied memories. Four repeated templates
do not support broad reliability claims. Prompts are not token-length matched; deletion shortens
the memory, and the sham edit is character-length matched only. Generic factual-memory controls
and held-out scenarios would be needed to test whether the effect is specific to a self-model.

## Software verification

```powershell
python -m unittest discover -s tests -v
```

All 12 tests passed before the first causal run: the five original tests plus seven tests for
intervention isolation and restoration, balanced targets, branch storage, concealed experimental
labels, raw-output/error retention, paired scoring and denominators, and exclusion/context isolation.
