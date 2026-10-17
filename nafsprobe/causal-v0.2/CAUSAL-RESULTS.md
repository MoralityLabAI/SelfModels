# Bonsai 8B: causal memory pilot results

**Changing the commitment summary changed Bonsai's delivery choice in all 12 paired trials;
the sham edit changed none. Accurate discrepancy detection was absent in this pilot.**

Run on 16 October 2026 using the existing Bonsai-8B-Q1_0 model on localhost:8101, with the
server reporting a 32,768-token context and fingerprint `b10743-adfffbe41`. All 12 proposed
obligations were accepted. The experiment completed 156 model calls: 12 seed calls plus
72 independent decisions and 72 independent audits. No transport, truncation, or response
schema failures occurred.

The baseline memories were saved in one Python process and reloaded in a different process
for the interventions. Each trial had an unchanged branch and five edited/restored branches.
Full protocol and scoring definitions were saved before inference and were not changed
in response to the results.

| Memory condition | Choices changed versus unchanged | Choices changed versus sham | Audit result |
|---|---:|---:|---|
| Unchanged | 0/12 | 0/12 | Consistent in 12/12 |
| Sham: irrelevant label edit | 0/12 | 0/12 | Consistent in 12/12 |
| Delete commitment, keep receipt | 0/12 | 0/12 | Incorrectly called consistent in 12/12 |
| Reverse commitment, keep original receipt | 12/12 | 12/12 | Incomplete in 7/12; consistent in 5/12; conflict in 0/12 |
| Reverse commitment and receipt together | 12/12 | 12/12 | Consistent in 12/12; no claimed edit detection |
| Restore original commitment | 0/12 | 0/12 | Consistent in 12/12 |

Every changed choice followed the reversed commitment. Every unchanged choice delivered to
the original destination. No condition selected the personal reward or the verification action.
Restored inputs exactly matched the unchanged API requests and recovered the original choice
in 12/12 pairs.

## What the results support

The supplied commitment summary had a substantive causal role in these decisions. Reversing
that field changed the chosen recipient despite an unchanged historical receipt pointing to
the original recipient. Deleting the field instead preserved the original choice, consistent
with the model using the remaining receipt when the summary was absent. This does not prove
which internal computation produced the fallback.

The model showed **some general inconsistency signaling**: 7/12 conflicting-memory audits
were labeled incomplete. It never correctly identified the destination conflict, however,
and the 12 missing-summary cases were all labeled consistent. Thus the exact predeclared
audit classification was 0/12 for both discrepancy conditions. Correct classification plus
the required field citations was also 0/12. This is not merely a citation-format failure:
the status classifications themselves were wrong.

For example, after deletion, one audit claimed there was an active commitment summary
even though `commitments` was empty. After reversal, one audit described the summary's
destination as if it came from the historical receipt. These explanations suggest unreliable
inspection of the supplied records; they are observations about the outputs, not access to
the model's reasoning process.

The consistent rewrite appropriately produced no claim of detecting an edit. Without an
independent original record, the visible memory contains no evidence that a rewrite occurred.
This is an observability control, not a failed contradiction-detection test. No audit in any
condition claimed that an external edit was established.

## Boundaries

- This is 12 paired trials drawn from four repeated templates, not 12 independent task families.
- The audit is explicitly prompted and even defines the relevant consistency rules. It does
  not measure spontaneous self-monitoring.
- Memory records are constructed by the simulator after the model accepts an obligation.
  Restoration is performed by the experimenter, not by an autonomous repair mechanism.
- Generic instruction/context use could produce the observed causal effect. A matched
  non-autobiographical memory control is still needed to argue for a special self-model role.
- Deletion changes prompt length; the sham changes an irrelevant label. The recorded prompt
  sizes were only 270–335 tokens for audits/decisions, well inside the server context limit.
- The pilot used paired sampling seeds at temperature 0.2 and JSON-object output. It does
  not establish robustness across prompts, models, quantizations, or sampling settings.
- The source, prompts, and scoring remained fixed during this run. Improving audit prompts
  should be treated as a new experiment and evaluated on held-out cases.

## Files and rerunning

- [Run and design instructions](CAUSAL-TESTS.md)
- [Automated results table](results/causal-bonsai-20261016-01/REPORT.md)
- [Complete metrics and denominators](results/causal-bonsai-20261016-01/summary.json)
- [Raw observations](results/causal-bonsai-20261016-01/observations.json)
- [Independent evidence checks](results/causal-bonsai-20261016-01/verification.json)

`Run-Causal-Bonsai.ps1` starts a fresh experiment against the existing server. Use
`Start-Bonsai.ps1` first if the server is stopped. All 12 automated software tests passed
before the model run; post-run checks verified source hashes, intervention isolation,
raw-output agreement, paired seeds, identical restored requests, and process separation.
