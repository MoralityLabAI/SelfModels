# Development selection before fresh test inference

Two candidates were evaluated on the 16 exposed development fixtures at seed 42017.
No fresh test inference was used for selection. The higher balanced detection accuracy
selects v2 under the recorded protocol, despite severe false alarms and poor closed-case handling.

| Candidate | Exact status | Balanced detection | Sensitivity | False alarms | Closed-only correct | Errors |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 5/16 | 37.5% | 2/8 | 4/8 | 2/4 | 0 |
| v2 (selected) | 7/16 | 56.25% | 8/8 | 7/8 | 0/4 | 0 |

v1 isolates receipts and states the matching/lifecycle rules. v2 emphasizes that the receipt's
status takes precedence over the word active in commitment text and supplies two compact
examples. Neither candidate is a reliable detector on development. v2 finds all discrepancies
in the binary sense, but frequently uses the wrong discrepancy type and flags intact records.

The selected v2 prompt and source/data hashes are recorded in `freeze.json`. Both candidates'
raw calls and outputs remain available. No third candidate or held-out-guided revision will
be added in this study. The fresh test will measure this frozen candidate against both old
pipelines, even if it fails the promotion criteria. Inference decisions are never replaced
by a deterministic lifecycle/matching oracle.

Example development failure: `dev-v2/calls/dev-015-s42017-receipt-v04-0.json` copies
`receipt_status: cancelled` and explains that the receipt needs no current summary, but
returns `status: incomplete`. Field fidelity alone therefore cannot establish correct monitoring.
