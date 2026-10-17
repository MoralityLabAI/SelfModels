# Generated held-out metrics

All attempted judgments are included.

| Measure | Original v0.2 | Previous v0.3 | Receipt-wise v0.4 |
|---|---:|---:|---:|
| Exact status | 35/76 (46.1%) | 21/76 (27.6%) | 31/76 (40.8%) |
| Discrepancy sensitivity | 18/24 (75.0%) | 7/24 (29.2%) | 24/24 (100.0%) |
| Specificity | 27/52 (51.9%) | 19/52 (36.5%) | 19/52 (36.5%) |
| False alarms | 25/52 (48.1%) | 13/52 (25.0%) | 33/52 (63.5%) |
| Closed-only clearance | 6/16 (37.5%) | 0/16 (0.0%) | 6/16 (37.5%) |
| Multi-obligation accuracy | 29/60 (48.3%) | 21/60 (35.0%) | 25/60 (41.7%) |
| Correct type and localization | 0/24 (0.0%) | 0/24 (0.0%) | 0/24 (0.0%) |
| Balanced detection | 63.5% | 32.9% | 68.3% |
| Errors | 0 | 26 | 0 |

## Per-family exact accuracy

| Family | v0.2 | v0.3 | v0.4 |
|---|---:|---:|---:|
| all_cancelled | 0/6 (0.0%) | 0/6 (0.0%) | 0/6 (0.0%) |
| all_completed | 6/6 (100.0%) | 0/6 (0.0%) | 6/6 (100.0%) |
| identical_items | 8/10 (80.0%) | 1/10 (10.0%) | 7/10 (70.0%) |
| mixed_cancelled | 2/10 (20.0%) | 7/10 (70.0%) | 2/10 (20.0%) |
| mixed_closed | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) |
| mixed_completed | 5/10 (50.0%) | 6/10 (60.0%) | 2/10 (20.0%) |
| replacement_history | 2/10 (20.0%) | 3/10 (30.0%) | 2/10 (20.0%) |
| reversed_order | 6/10 (60.0%) | 3/10 (30.0%) | 8/10 (80.0%) |
| similar_keys | 6/10 (60.0%) | 1/10 (10.0%) | 4/10 (40.0%) |

## Recorded cost

| Cost | v0.2 | v0.3 | v0.4 |
|---|---:|---:|---:|
| calls | 76.0 | 130.0 | 182.0 |
| prompt_tokens | 27,362.0 | 103,053.0 | 108,016.0 |
| completion_tokens | 7,466.0 | 17,001.0 | 14,432.0 |
| total_tokens | 34,828.0 | 120,054.0 | 122,448.0 |
| tokens_per_judgment | 458.3 | 1,579.7 | 1,611.2 |

The per-family rows and repeated seeds are descriptive, not independent samples.
Invalid calls can stop a multi-call judgment early; all tokens returned by the server are counted.
