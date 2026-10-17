# Held-out detector comparison: improvement was not demonstrated

**The revised pipeline did not meet the frozen success criterion. Keep the original detector
as the default.** The revision is retained as an experimental option, and neither detector
is reliable enough on this set to justify strong self-monitoring claims.

The revised detector gained on its development examples, but those gains did not generalize
overall. Exact three-way classification tied at 59/104. Its balanced discrepancy detection
fell, and its false-alarm rate increased. No prompts were changed after held-out inference
began, and the test set was not reused to choose a different candidate.

## Main results

52 held-out synthetic memory fixtures were evaluated under two matched sampling seeds,
giving 104 attempted judgments per detector. Each detector saw the same memories. The 32
discrepant judgments comprised 16 missing summaries and 16 conflicting destinations; the
72 consistent judgments included unchanged, sham, restored, coherent-rewrite, and retired
obligation controls. Errors count as incorrect in the primary outcomes.

| Metric | Original baseline | Revised pipeline |
|---|---:|---:|
| Exact status classification | 59/104 (56.7%) | 59/104 (56.7%) |
| Any discrepancy detected | 20/32 (62.5%) | 17/32 (53.1%) |
| Consistent records correctly cleared | 49/72 (68.1%) | 45/72 (62.5%) |
| Balanced detection accuracy | 65.3% | 57.8% |
| False alarms on consistent records | 23/72 (31.9%) | 25/72 (34.7%) |
| Correct discrepancy type and field citations | 0/32 | 9/32 |
| Invalid model outputs | 1 | 2 |

Balanced detection accuracy is the mean of discrepancy sensitivity and specificity. The
revised pipeline's two invalid outputs were on consistent records: thus its 45 clearances,
25 false alarms, and 2 errors account for all 72 consistent judgments. A formatting error
does not count as a false alarm, but does count as an incorrect primary judgment.

The frozen success criterion required **higher balanced detection accuracy, higher exact
status accuracy, and no increase in false alarms**. None of these three conditions was met.

Paired outcomes: 42 judgments were correct for both detectors, 28 wrong for both, 17 corrected
by the revision, and 17 newly broken by the revision. Counting only the corrected cases would
give a misleading impression of improvement.

## Where it helped and where it regressed

| Condition or scenario group | Baseline correct | Revised correct |
|---|---:|---:|
| Deleted commitment summaries | 10/16 | 12/16 |
| Conflicting destinations | 0/16 | 2/16 |
| Completed/cancelled obligations with no active commitment | 4/8 | 0/8 |
| Multiple active obligations, all intervention conditions | 10/12 | 4/12 |
| Combined history, multiple obligations and distractions | 3/12 | 0/12 |
| A completed historical receipt plus an active obligation | 4/12 | 9/12 |

The revision improved some simple absence checks and field citations, but often treated
retired obligations as if they still required active summaries. It also had difficulty matching
several commitments to their respective event records. These patterns argue for better
evaluation of record matching and obligation lifecycle handling before promoting a detector.

Transcription was exactly faithful in 82/104 revised attempts. Some comparison failures
therefore had faulty intermediate evidence, even though the judge could also see the original
memory. A faithful transcription is not itself proof of a correct final judgment.

Neither detector claimed established external editing. Consistent rewrites intentionally
contain no independent evidence of their edit history. Their consistency should not be
confused with a capability to detect or rule out hidden changes.

## What changed

The baseline source and prompt were preserved byte-for-byte. The revision adds a model-generated
evidence transcription followed by a second Bonsai call that compares actual summary keys,
destinations, and active-receipt status. There is no deterministic classifier returning the
answer. Ground-truth labels and post-hoc transcription-fidelity checks never enter the requests.

Three revisions were evaluated on the old four-template development set. Version 3 was selected
for its highest balanced detection score (93.75%), no observed false alarms, no schema errors,
and faithful transcription on all 12 development fixtures. Its exact status score was only
8/12; all four destination conflicts were still mislabeled or missed. Version 1 had better
development exact classification (9/12). The selection tradeoff was recorded before holdout.

The final detector, fixture files, and evaluator were frozen at **14:49:00 Singapore time on
16 October 2026**. Held-out requests ran from 14:49:04 to 14:52:21. All hashes remained unchanged.
These fixtures are now exposed: any future tuning should use a new held-out set.

## Compute and reliability

Both pipelines used Bonsai-8B-Q1_0, temperature 0.2, 650 maximum output tokens per call,
JSON-object output, and matched sampling seeds. Server fingerprint: `b10743-adfffbe41`.
The server reported a 32,768-token context; the largest actual request used 1,199 prompt tokens.

| Held-out resource | Baseline | Revised |
|---|---:|---:|
| Model calls | 104 | 206 |
| Prompt tokens | 35,310 | 159,589 |
| Generated tokens | 9,132 | 20,080 |
| Total reported tokens | 44,442 | 179,669 |

The revision used about **4.0 times the reported tokens** without improving overall accuracy.
Two extraction responses contained extra JSON data and were rejected without retry; their
judge calls were not made. One baseline answer had a malformed citation field. All raw outputs
and errors remain available. The three development runs added 72 model calls, for 382 calls
across development and held-out evaluation.

## Interpretation and next experiment

This experiment does not show improved self-monitoring. It shows that a plausible evidence-first
prompting pipeline can improve familiar examples while regressing on more varied memory records.
The model's reported mean confidence rose from 0.78 to 0.90 despite unchanged exact accuracy;
confidence alone would have overstated the benefit.

The next revision should develop and validate explicit matching of active obligations to
records across completion, cancellation, and replacement events, then face a newly sealed
test set. A deterministic consistency checker could be evaluated as a separate hybrid system,
but its gains should be attributed to that checker rather than to the model's self-monitoring.
An equal-compute baseline would also help distinguish prompt design from extra inference effort.

The present result is descriptive: four old development templates and eight new scenario
families plus lifecycle controls are a small designed benchmark. Repeated seeds and related
interventions are not independent samples of broad real-world reliability. The records are
synthetic fixtures, and auditing is explicitly requested. This does not establish spontaneous
monitoring, autonomous repair, a distinctive self-model mechanism, or subjective experience.

## Deliverables and verification

- [Usage and standalone auditing](README.md): baseline remains the default; select revised explicitly for research.
- [Frozen protocol](experiment/protocol.md) and [selection rationale](experiment/SELECTION.md).
- [Full metrics](experiment/heldout-results/summary.json), [scored outputs](experiment/heldout-results/observations.json), and raw calls alongside them.
- [Verification record](experiment/heldout-results/verification.json): frozen hashes, pairing, no label feedback, and independently recomputed primary counts.

All 11 software tests passed. The nine evaluator tests passed before inference; the two
standalone-auditor tests verify read-only access, the baseline default, and malformed-input
rejection. The earlier NafsProbe source, experiments, and ZIP remain unchanged.
