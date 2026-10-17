# Receipt-wise matching and lifecycle experiment

**v0.4 did not improve overall self-monitoring enough to replace the original detector.**
It correctly classified all destination reversals, but misclassified every deletion, raised
more false alarms, and still mishandled cancelled commitments. The original v0.2 auditor
remains the default. Both earlier implementations and their evidence are unchanged.

Model evaluation completed on 2026-10-17; report finalized on 2026-10-18 (Asia/Singapore).

## What changed

v0.4 audits each accepted-obligation receipt separately against the full commitment
dictionary. Its prompt emphasizes exact keys, multiple obligations, and closed lifecycle
states. Bonsai decides the matching and relevance of each receipt. The harness includes
completed, cancelled and superseded records without filtering them, then combines model
verdicts with precedence conflict > incomplete > consistent. It never substitutes a
deterministic lifecycle verdict, repairs an answer, or modifies the source memory.

The comparison preserves v0.2's original single-call audit and v0.3's frozen two-call
transcription/judgment pipeline, including prompts, generation settings and validators.

## Design and development

Sixteen exposed cases from the previous study were used for development. Two prompt
candidates scored 5/16 and 7/16. The recorded selection rule chose the second candidate
for its higher balanced detection accuracy (56.25% versus 37.5%), despite its high
development false-alarm rate. Both attempts are retained in full.

The selected prompt, source, protocol and dataset hashes were frozen before fresh test
inference. Thirty-eight new synthetic fixtures were evaluated with paired seeds 62017
and 62018: **76 judgments per pipeline, 228 total**. Six families cover two active
obligations, reordered records, identical items, similar keys, and mixed historical
statuses. Eight further fixtures contain only closed commitments. New items, recipients
and keys distinguish these cases from the development set. No post-test tuning occurred.

All pipelines used local **Bonsai-8B-Q1_0**, temperature 0.2, JSON-object requests and
650 output tokens per call. The server had a 32,768-token context. See
[environment metadata](experiment/environment.json) for the model-file SHA-256 and
[the protocol](experiment/protocol.md) for the full design.

## Held-out results

| Measure | Original v0.2 | Previous v0.3 | New v0.4 |
|---|---:|---:|---:|
| Exact three-way classification | **35/76 (46.1%)** | 21/76 (27.6%) | 31/76 (40.8%) |
| Any-discrepancy sensitivity | 18/24 (75.0%) | 7/24 (29.2%) | **24/24 (100%)** |
| Specificity | **27/52 (51.9%)** | 19/52 (36.5%) | 19/52 (36.5%) |
| Balanced detection accuracy | 63.5% | 32.9% | **68.3%** |
| False alarms on consistent records | 25/52 (48.1%) | 13/52 (25.0%) | 33/52 (63.5%) |
| Multi-obligation exact accuracy | **29/60 (48.3%)** | 21/60 (35.0%) | 25/60 (41.7%) |
| Closed-only controls correctly cleared | **6/16** | 0/16 | **6/16** |
| Invalid-output judgments | 0 | 26 | 0 |

Every attempted judgment is included. Errors count as incorrect, but are reported
separately from false alarms. v0.3's lower false-alarm count is not better specificity:
20 of its 52 consistent-record judgments failed. Its 26 errors comprise 22 extraction
failures and four judgment failures, due to malformed JSON or missing required arrays.
There were no transport retries or output repairs.

The dataset contains 52 consistent judgments and 24 discrepancies. Always answering
consistent would score 52/76 (68.4%) exact accuracy but only 50% balanced detection
accuracy and zero sensitivity. This illustrates why exact accuracy alone is insufficient.

On paired judgments, v0.4 was uniquely correct 12 times versus v0.2's 16; both were
correct 19 times and both wrong 29 times. Against v0.3, v0.4 was uniquely correct 26
times versus 16, with five both correct and 29 both wrong. These are descriptive counts;
repeated seeds and related variants are not independent scenario families.

## Matching and lifecycle findings

- **Destination reversals:** v0.4 classified 12/12 as conflict, compared with 0/12 for
  v0.2 and 1/12 for v0.3. This is a specific improvement in classification.
- **Deleted summaries:** v0.4 classified all 12 as conflict rather than incomplete.
  Its perfect binary sensitivity therefore does not establish correct diagnosis.
- **Multiple obligations:** reordered cases improved from 6/10 to 8/10 against v0.2,
  but identical-item cases fell from 8/10 to 7/10 and similar-key cases from 6/10 to 4/10.
- **Closed-only cases:** v0.4 and v0.2 cleared all six completed-only judgments but none
  of the six cancelled-only or four mixed-closed judgments. v0.4 did not improve this target.
- **Mixed history:** v0.4 scored 2/10 in each completed, cancelled and superseded-history
  family. False alarms on irrelevant closed records can override correct active comparisons.

Post-hoc diagnostics show that v0.4 copied the literal receipt status correctly in
**182/182 calls**, but cleared only **16/62 closed receipts**. It flagged the other 46
despite copying their closed status. Completed receipts were handled correctly in 16/26
calls, cancelled in 0/26, and superseded in 0/10. Among active receipts, the matched-key
field was correct in 108/120 calls and the per-receipt verdict in 103/120. These diagnostics
are scoring-only and were never used to change model answers.

For example, [this cancelled-receipt answer](experiment/heldout-results/calls/test-034-s62018-receipt-v04-0.json)
copies cancelled and explains that no summary is needed, yet returns incomplete.
That is a visible inconsistency between the response fields, not evidence that the
model reliably applies its stated lifecycle rule.

All three pipelines scored 0/24 on the stricter correct-type-plus-localization metric.
v0.4 often cited only an episode container, omitting the required commitment key and
destination field. Some explanations name the right destinations; the citation metric
does not treat that as satisfying the structured-path requirement.

## Cost and verification

| Pipeline | API calls | Prompt tokens | Output tokens | Total tokens |
|---|---:|---:|---:|---:|
| v0.2 | 76 | 27,362 | 7,466 | 34,828 |
| v0.3 | 130 | 103,053 | 17,001 | 120,054 |
| v0.4 | 182 | 108,016 | 14,432 | 122,448 |

v0.3 made fewer than 152 calls because 22 extraction failures stopped those judgments
before the second call. v0.4 used about 3.52 times v0.2's total tokens. This is a pipeline
comparison with different input representations and compute budgets, not a weight update
or an equal-compute ablation. Token counts include cached prompt tokens and failed calls
where the server returned usage.

All **33 software tests passed**. The completed evidence checker verified all 228 judgments,
388 raw requests, exact baseline payloads and settings, freeze timestamps, current frozen
hashes, and **592 unchanged historical files**. See [validation](VALIDATION.md),
[machine-readable verification](experiment/heldout-results/verification.json),
[full metrics](experiment/heldout-results/METRICS.md), and
[scored observations](experiment/heldout-results/observations.json).

## Decision and limits

**Do not promote v0.4.** It improves balanced detection and reversal classification but
fails the predeclared requirements for higher exact accuracy, no additional false alarms,
better closed-only clearance, and no regression in multiple-obligation accuracy against v0.2.
It also has more false alarms than v0.3. Retain it as an experimental version with its failures.

The development selector favored sensitivity despite severe specificity and lifecycle
failures. A later study should make acceptable closed-case clearance and false-alarm rates
development gates before selecting a candidate; that would require another fresh test set.
This report does not select a replacement using the newly exposed outputs.

These are synthetic delivery records with explicit lifecycle statuses and exact key links,
not newly enacted autobiographical experiences or inferred event transitions. Missing or
ambiguous identifiers, orphan summaries and other memory schemas remain untested. v0.4
omits the autobiographical anchor, so its unchanged/sham invariance is partly imposed by
the input representation. Confidence is uncalibrated and citation unions are mechanical.
The findings do not establish reliable autonomous self-correction or moral agency.
