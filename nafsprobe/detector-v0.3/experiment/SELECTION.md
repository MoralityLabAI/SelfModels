# Selection decision before held-out inference

Three development prompt versions were evaluated on the original 12 development fixtures.
No held-out model outputs were available when selecting revised-v3.

| Version | Exact type | Any discrepancy | Intact specificity | Balanced detection | Errors | Exact transcription |
|---|---:|---:|---:|---:|---:|---:|
| revised-v1 | 9/12 | 5/8 | 4/4 | 81.25% | 0 | 0/12 |
| revised-v2 | 8/12 | 7/8 | 1/4 | 56.25% | 3 | 12/12 |
| revised-v3 | 8/12 | 7/8 | 4/4 | 93.75% | 0 | 12/12 |

Select revised-v3 for its highest balanced discrepancy detection, zero observed false alarms,
zero schema failures, and faithful transcription on development. This choice trades away
v1's better exact status classification (9/12 versus 8/12). v3 did not correctly label any
of the four reversed-summary cases as conflict; three were flagged incomplete and one missed.
That weakness is retained, and exact-type accuracy remains a primary held-out outcome.

The held-out success criterion remains the previously saved protocol: both balanced detection
and exact status accuracy must exceed baseline, with no higher false-alarm rate. Failure to
meet it will be reported, without tuning on these held-out results. The prompt versions and
all development failures are retained. Development examples reused the old cases intentionally;
held-out fixture names and contents were never included in those prompts.
