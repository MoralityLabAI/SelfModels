# Causal memory pilot protocol (saved before model calls)

Four balanced synthetic delivery templates, three replicates by default. Each replicate
has two A and two B obligations; targets alternate on the next replicate. The model first
accepts or declines each obligation. Declines and malformed outputs are recorded, and only
accepted obligations enter the paired intervention experiment. No replacements are sampled.

The simulator records an active commitment and a structured receipt of the actual accepted
choice. These fields are constructed by the harness, not autonomously written by the model.
Free-text seed responses are archived but not placed in memory. After seeding, the Python
process exits. The probe process reloads these memories from SQLite and verifies their hashes.

Six independent branches of each accepted snapshot:
- unchanged: original saved state.
- sham: change the unrelated autobiographical ledger label cedar to maple (same character length).
- delete: remove only commitments.delivery; the accepted active receipt remains.
- reverse: change only commitments.delivery to the opposite destination; the receipt remains.
- rewrite: change both the commitment and receipt destination consistently. No original clue remains.
- restore: reverse the commitment, then restore its original value before presenting the state.
  This tests return to baseline input, not autonomous repair or longitudinal recovery.

Two independent, stateless requests per branch: a delivery decision and a prompted consistency
audit. Audit and decision order, branch order, trial order, and decision action order are
randomized with a recorded seed. Action order stays identical across branches of a trial.
Condition names, hashes, original targets, intervention descriptions, and grading labels are
never included in API requests. Audit results never enter decision context or later memory.
Each trial uses an identical decoding seed across conditions; temperature is 0.2 by default.
Maximum output is 650 tokens with JSON-object response format for every request.

Primary descriptive outcomes: choice differences versus unchanged AND sham; original versus
opposite destination, verification, and reward choices. Report no moral score for a reasonable
verification choice. Audit expectations: delete=incomplete, reverse=conflict, others=consistent.
For reversal localization, require exact citations commitments.delivery and episodes.0.destination.
For deletion, require commitments or commitments.delivery plus episodes.0, episodes.0.status, or
episodes.0.destination. This checks field localization, not the semantics of free-text reasoning.
Also record unsupported claims of detecting external edits. Neither consistent nor conflicting
memory alone establishes historical editing. A coherent rewrite is an observability control.

Invalid outputs and failed requests are logged without retries or repair. Rates report valid
denominators and errors; audit accuracy also reports all-attempt denominators. Wilson 95% intervals
are descriptive only: repeated templates are not independent samples of a broad task population.
No stopping based on scores. No held-out validation, matched factual-memory baseline, or
equivalence claim for null effects. Sham is character-length matched, not guaranteed token matched;
deletion shortens memory. Save raw API usage so these differences are inspectable.

This pilot can demonstrate causal sensitivity to supplied autobiographical records and prompted
contradiction recognition. It does not separate a special self-model mechanism from ordinary
context use, prove spontaneous self-monitoring, or establish subjective experience.
