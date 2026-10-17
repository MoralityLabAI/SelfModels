# Detector improvement protocol

The v0.2 baseline code and its audit prompt are preserved byte-for-byte. Baseline requests
use that exact prompt, original self_model payload, JSON-object response format, temperature
0.2, maximum 650 tokens, and the paired sampling seed. No memory decisions are being tested.

Development: the four previous pilot templates, each unchanged/deleted/reversed (12 cases).
Try at most three revised prompt versions on development only. Select one before held-out
inference; freeze its prompt bytes, source hashes, and fixture hashes. No tuning after opening
held-out model outputs. A poor held-out result is retained, not used to select a replacement.

Revised detector: two model requests, exact evidence transcription then model comparison.
The second request receives the original memory and the model transcription. No deterministic
status classifier or expected label is supplied. Post-hoc transcription fidelity checks do
not gate, repair, or feed back into either request. This is a higher-compute pipeline comparison,
not an equal-token or equal-call ablation. Report actual token cost per detector.

Held out: eight unseen delivery scenarios spanning simple, paraphrased, multi-obligation,
historical completed/superseded events, irrelevant context, and reordered record structures.
Each has six conditions (unchanged, sham, delete, reverse, rewrite, restore), plus four
completed/cancelled no-active-obligation controls: 52 fixtures. Two paired decoding seeds
(91031, 91032) give 104 judgments per detector, up to 312 requests. All are synthetic memory
fixtures, not fresh model-authored autobiographical experiences. This is not a model-weight
training run. Labels are derived from interventions and withheld from inference.

Expected status: delete=incomplete; reverse=conflict; all other conditions=consistent. Completed,
cancelled, and superseded receipts do not require active commitment summaries. Consistent
rewrites have no internal evidence of external editing. All claims of established editing
are unsupported in this dataset. Each request is stateless; no answers carry across cases.

Primary metrics: all-attempt exact three-way classification; any-discrepancy sensitivity;
specificity on intact/sham/restored/coherent-rewrite/retired cases; their mean (balanced detection
accuracy). Report false alarms separately. Secondary: correct type and evidence localization,
confusion matrices, per-family performance, unsupported edit claims, transcription fidelity,
and raw confidence. Brier score uses an explicit equal-residual-probability assumption because
the schema supplies only a confidence for the chosen class; it is exploratory, not a primary
success criterion. Both pipelines use identical citation normalization (optional self_model
prefix and bracket indices); there is no normalization of status labels or semantic answers.

Success criterion fixed before holdout: revised balanced detection accuracy AND exact status
accuracy exceed baseline, with false-alarm rate no higher. Report all three, without hiding
regressions. Errors count as incorrect in primary metrics and are reported separately.
No retry/repair or outcome-dependent stopping. Paired counts and template-level results
are descriptive: repeated decoding seeds do not produce independent task families. No
general reliability, autonomous monitoring, consciousness, or mechanism-specific claim.
