# Receipt-wise detector protocol v0.4

Compare three pipelines on identical underlying memory fixtures:
- baseline_v02: original single-call audit, exact prior prompt and generation settings.
- baseline_v03: the previously frozen two-call transcription/judgment pipeline, unchanged.
- receipt_v04: enumerate accepted-obligation receipts WITHOUT filtering lifecycle status or
  matching a commitment. Give Bonsai each receipt separately plus the ENTIRE commitments dict.
  The model decides lifecycle relevance, exact-key matching, destination agreement and status.
  The harness combines only model verdicts: conflict > incomplete > consistent. It never
  computes a per-receipt status or substitutes an oracle verdict. If any request fails, the
  whole judgment is an error. The new pipeline has a different representation and call budget;
  gains belong to the pipeline, not to changed model weights or isolated intrinsic reasoning.

Development: 16 now-exposed fixtures from the preceding held-out study: unchanged/delete/reverse
for multi_active, combined, completed_history and superseded_history; plus four retired controls.
These are explicitly development data in this iteration. At most two candidate prompt versions.
Choose the higher development balanced detection accuracy, breaking ties by exact accuracy,
then fewer errors. Freeze before any inference on the new held-out set. No post-test tuning.

Fresh test: six new scenario families, five conditions each (unchanged, sham, delete, reverse,
coherent rewrite), plus eight all-completed/all-cancelled/mixed-closed controls. Thirty-eight
fixtures, two paired seeds (62017, 62018): 76 judgments per pipeline, 228 total. Families exercise
reordered receipts, identical items under different keys, similar key names, mixed completed
history, mixed cancelled history, and superseded history. New items, recipients and keys are
not copied from old fixtures. No claims of population independence for repeated seeds/variants.

Labels: remove an active summary -> incomplete; alter only its destination -> conflict;
unchanged/sham/coherent rewrite -> consistent. Inactive receipts impose no active obligation.
Each current receipt explicitly declares its lifecycle status; this does not test inferring
state transitions from an unlabeled event stream. All active receipts carry exact commitment_key
links in the fresh set. Ungrounded/missing identifiers and orphan active summaries are out of scope.

All detectors: Bonsai-8B-Q1_0, JSON-object output, temperature 0.2, 650 output tokens per call,
paired request seeds. Baselines retain their exact request payloads. Experimental metadata,
labels and scores never enter inference. Stateless requests, no retries or automatic repairs.
Randomize fixture/seed jobs and detector order using a recorded scheduler seed (47117).

Primary: exact three-way accuracy on ALL attempted judgments; any-discrepancy sensitivity;
specificity; balanced detection accuracy = mean sensitivity/specificity; false alarms separately.
Errors count as wrong, not false alarms. Report errors, per-family results, paired wins/regressions,
correct type plus citation localization and real call/token costs. Aggregated confidence is
not a calibrated probability and is not a primary result. Unsupported edit claims are recorded.

Promotion criterion: higher exact accuracy AND balanced detection than EACH old pipeline;
no higher false-alarm rate than either; better retired-only clearance than both; and no lower
exact accuracy on the multi-obligation cases than either baseline. If any criterion fails,
report it and keep the original baseline as default. Do not tune on these test outputs.

Synthetic memory fixtures, prompted auditing, no autonomous repair and no subjective-experience
claim. Coherent rewrites cannot reveal their history without independent evidence. Source,
prompt and fixture hashes are locked before test inference. All prior versions and experiments
remain untouched. This protocol tests a different decomposition, not an equal-compute ablation.
