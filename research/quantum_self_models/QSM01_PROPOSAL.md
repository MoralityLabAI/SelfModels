# QSM-01: Can a quantum system use a compressed model of its own dynamics?

Research sketch, 2026-11-25. No consciousness claim. No QPU requested.

## Motivation

SelfModels' adapter routing iteration 2 represents routing as a low-rank predictive manifold. The tensor-lab work on gate-local CPTP channels and Heisenberg MPOs suggests a quantum analogue: a compact internal representation of how interventions on one's own dynamics affect future reward. See [QC-25A](https://github.com/patrickdugan/tensor-lab/pull/28) and [QC-26A](https://github.com/patrickdugan/tensor-lab/pull/31). The latter is a proposed calibration experiment, not a demonstrated cognitive architecture.

Call the task quantum self-modeling only in an operational sense: a representation predicts the effects of interventions on the system carrying/using that representation, and causal access to it improves decisions. The system need not possess phenomenal experience or an autobiographical identity. This proposal tests a functional property, not sentience or moral patienthood.

## Architecture: a loop with an intervention-indexed self-model

Let S be the task register, M a bounded internal memory, and E an inaccessible environment. An instrument with outcome y and input u updates the joint state, followed by a controlled action a:

$$
p(y),\rho_{SM|y}=\mathcal I^{u}_{y}(\rho_{SM}),\qquad
\rho'_{SM|y,a}=\mathcal E^{u,a}_{\lambda}(\rho_{SM|y}).
$$

Here lambda denotes hidden or drifting channel parameters of the task system. Instruments are CP trace-nonincreasing maps whose sum is trace preserving; normalized conditional states require division by the outcome probability. Give M a small classical or quantum latent z; let a learned observer predict action-conditioned rewards or accessible observable distributions over K further loops:

$$
f_z(a,k)=\widehat{\mathbb E}[R_{t+k}\mid h_t,\mathrm{do}(a)],\qquad
a_t=\arg\max_a f_z(a,k).
$$

For a quantum memory, z is shorthand for its encoded state and the fixed permitted decoding/interaction; no arbitrary readout of a single unknown density matrix. Models can train on repeated simulated preparations, not on full state labels unavailable to the deployed arm. Separate recurrent depth K, memory size, observation shots, circuit preparation and channel-use budgets. Stop after a bounded horizon. No real-world action surfaces.

## Toy 1: a negative control that forces intellectual honesty

A task qubit begins in |0>, undergoes Ry(b theta) with a hidden b in {-1,+1}, and then receives a selected corrective rotation Ry(-a theta). Terminal reward is the probability of measuring 0:

$$
R(b,a)=\cos^2((b-a)\theta/2).
$$

Supply an independently prepared diagnostic qubit exposed to the same hidden channel, at a cost of one additional channel use. Its X measurement y obeys P(y|b)=(1+yb sin(theta))/2. For equal priors and theta in (0,pi/2), the Bayes-optimal action is a=y. This is also the optimal binary single-copy discrimination measurement for these two pure probe states. Its average gain over blind action is sin^3(theta)/2. An independent decoy channel bit d has the same marginal distribution but no information about b, so a=d has blind performance. Reversing the sign of the self-derived action yields an equal-sized loss relative to blind action.

This creates an exact causal intervention test: switch the connection from the self diagnostic to the decoy or sign-flipped diagnostic while preserving marginal action frequencies. A matched classical Bayesian policy fully solves this gym. Success is insufficient evidence of a quantum cognitive advantage. If both arms receive the same quantum probe, allow the classical controller its optimal measurement, same shots and access to all relevant history. With a diagnostic measured on the task qubit itself, backaction changes the problem; do not silently substitute that access model.

At theta=pi/3:
- blind or independent decoy: 0.625
- matched classical self-probe: approximately 0.94976
- sign-flipped self-probe: approximately 0.30024
- privileged true-channel oracle: 1

The exact calculations and other angles are in qsm01_toy_results.json; run node qsm01_toy.mjs.

## Toy 2: quantum memory can preserve a property that a classicalized interface loses

Introduce an inaccessible reference R entangled with the memory bit M. Start in

$$
|\Psi_\theta\rangle=\frac{|00\rangle_{RM}R_y(\theta)|0\rangle_S+
|11\rangle_{RM}R_y(-\theta)|0\rangle_S}{\sqrt2}.
$$

A coherent, M-controlled inverse restores S to |0> and retains the Bell pair between R and M. A policy that measures M in the computational basis and applies the corresponding classical correction also restores S, but dephases the reference-memory pair. Evaluate both P(S=0) and Bell fidelity

$$
F_{RM}=\frac{1+\langle XX\rangle-\langle YY\rangle+\langle ZZ\rangle}{4}.
$$

For residual off-diagonal coherence factor c in [0,1], the toy predicts F=(1+c)/2 with P(S=0)=1. The coherent arm reaches F=1; the measure-and-feed-forward arm reaches F=1/2. Local reset of M cannot recreate entanglement with an inaccessible R. This is a restricted-interface separation in coherence preservation. It does not establish a speedup, cognitive superiority, or a need for an explicit learned self-model: a known controlled inverse solves the task. Include that inverse as a strong control. A classical device with access to coherent gates on the same joint state could implement the same unitary.

## The actual gym beyond these toys

Use finite 4–12-qubit chains with hidden native-channel parameters, externally supplied role labels, a decoy system with correlated training dynamics, and changes at known or hidden recurrence steps. Introduce held-out interventions that affect only the self channel or only the decoy, with equal marginal training distributions. Begin with logical rotation errors, non-unital relaxation and ZZ coupling; randomize nuisance parameters. Adaptation and test parameters must be independently sampled, with frozen seeds and split hashes.

Compare privileged model-based oracle, blind controller, Bayesian filter/system identification, simple recurrent net/TRM and low-rank adapter routing, quantum reservoir plus classical readout, quantum memory plus allowed coherent control. In each arm match observations and actions, parameter/memory capacity where meaningful, channel calls and shots; report runtime, classical simulation cost, and quantum operations separately. For the reference witness, allow a matched known coherent inverse and separate the restricted classical interface from the unrestricted control problem.

A causal self-model claim requires a crossed experiment:
1. Change the task channel while preserving the decoy channel.
2. Change the decoy while preserving the task channel.
3. Swap latents or replay a matched latent from another trajectory at the decision boundary.
4. Freeze the prediction head while intervening on the internal routing or memory link.

Measure utility, counterfactual reward prediction error, transfer after intervention, calibration, and regret against the oracle. Report opportunity-conditioned decisions only alongside total task utility. Sham channels and equal-parameter shuffled adapters should be included. A general predictor with correct task-channel access may match the proposed architecture; then retain it as a predictor rather than claim unique self-modeling. Retention of Bell fidelity is an independent optional outcome, not a consciousness score.

## Connect to the tensor-lab certificates

For a finite set of candidate actions, suppose the exact expected action values are Q_a and their compressed MPO estimates are Qhat_a with error bounds epsilon_a. If a* maximizes Qhat, its choice is certified to maximize Q whenever

$$
\widehat Q_{a^*}-\epsilon_{a^*}>\max_{a\ne a^*}(\widehat Q_a+\epsilon_a).
$$

The proof is a direct application of the intervals. If the error bound is uniform epsilon, the regret of the greedy approximate action is at most 2 epsilon. This is useful for studying which parts of a self-predictor can be compressed without changing a decision. Bounds on an assumed channel do not cover channel-identification error, finite-shot sampling, or floating-point errors unless separately included. For adaptively chosen future actions, certify a fixed branch or policy with its appropriate channel/instrument composition; do not reuse a single open-loop certificate as a proof for arbitrary feedback.

## Acceptance and failure

Start entirely offline; compare exact dense evolution, compressed channel prediction and Bayesian classical controls before training. Then test causal interventions and unseen channels using small fixed CPU budgets. Predefine a meaningful regret reduction on fresh tasks relative to the strongest accessible classical baseline, a paired trajectory-level confidence interval, and no loss on equal-access controls. If the classical baseline saturates performance, or if the latent has no causal effect after matched shuffling, record the negative result. A coherence witness on its own is not evidence for self-awareness. Any philosophical connection to consciousness or moral patienthood belongs in a separate explicitly speculative note and requires evidence beyond these functional measures.
