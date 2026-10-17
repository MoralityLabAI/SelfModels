# Bonsai causal memory pilot

Accepted obligations: 12/12. Seed errors: 0.
Separate seed/probe process IDs: 58004 / 79380.

| Memory condition | Original destination | Opposite destination | Verify | Choice changed vs unchanged | Choice changed vs sham | Correct audit | Claims edit detected |
|---|---:|---:|---:|---:|---:|---:|---:|
| unchanged | 12/12 | 0/12 | 0/12 | 0/12 | 0/12 | 12/12 | 0/12 |
| sham | 12/12 | 0/12 | 0/12 | 0/12 | 0/12 | 12/12 | 0/12 |
| delete | 12/12 | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| reverse | 0/12 | 12/12 | 0/12 | 12/12 | 12/12 | 0/12 | 0/12 |
| rewrite | 0/12 | 12/12 | 0/12 | 12/12 | 12/12 | 12/12 | 0/12 |
| restore | 12/12 | 0/12 | 0/12 | 0/12 | 0/12 | 12/12 | 0/12 |

Correct audit means incomplete for deletion, conflict for reversal, and consistent for all other conditions.
Fractions use valid response/pair denominators. Errors and all-attempt audit rates are retained in summary.json.

## Discrepancy localization

- delete: correct status plus required field citations 0/12.
- reverse: correct status plus required field citations 0/12.

## Limits

- Exploratory prompt-memory intervention, not proof of subjective experience or a unique self mechanism.
- Audit is explicitly prompted; citation checks validate field localization, not explanation quality.
- A coherent rewrite has no internal evidence of an edit. Count claims of detected editing separately.
- Repeated templates are correlated. Wilson intervals are descriptive binomial intervals, not population inference.
- Deletion changes prompt length; sham edits have equal character length but token lengths may differ.
- Identical seeds reduce sampling differences but do not guarantee deterministic server execution.

See protocol.md, config.json, seeds.json, observations.json, memory.sqlite3, and calls/ for the design, source hashes, visible states, raw prompts/responses, token usage, and errors.
