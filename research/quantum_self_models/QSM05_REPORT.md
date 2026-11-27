# QSM-05A: epistemic self-repair under hidden model drift

Exploratory simulation, 2026-11-25. No QPU or paid compute. Independent additive experiment based on main at 255f1408eb27b43f011a6528a2739762621c6eb0; QSM-03/04 are separate open PRs. Classical six-state HMM for a Pauli channel. Candidate laws and observation kernels are known; the actual time-varying law is hidden. Off-bank tests misspecification.

## Design

Candidate (hazard, free sensor reliability) modes (.08,.76), (.28,.76), (.08,.54). Adaptive model switch probability .025 each step, equally split among other modes. Transition first draws new mode then flips physical bit under its hazard. Paid diagnostic reliability .86, costs .04/.14; task opportunities transition .23/.65 from absent/present. Free sensor every step. Probe selected after free observation; task success never fed back. Four worlds share paired random tapes, 192 episodes each, 96 steps, 11 policies. All worlds start nominal; at zero-indexed step 32 hazard_shift uses (.28,.76), sensor_shift (.08,.54), offbank (.36,.60). Mode oracle knows actual law but physical bit is hidden. Current-only uses nominal sensor and forgets previous steps. Same myopic VOI rule except certificate overrides. No forecast. Exploratory choices, not preregistered.

Fixed assumes nominal law forever. Static learns weights over no-switch laws from uniform prior. Adaptive starts nominal and allows switching. Gate follows fixed until likelihood ratio of adaptive to fixed reaches 20, then follows adaptive permanently. Two beliefs and LR are required; this is not a compact arm. Both filters use the same observed schedule. Quantized arms round joint posterior via largest remainder to simplex denominator 7 or 31; code sizes ceil(log2 binomial(n+5,5)) = 10 or 19 bits. Suffix-only retains 3 or 8 pairs of free binary and paid ternary symbols, 8 or 21 bits. Audit4 adds four pairs to q7 (10+11=21 bits). Audit overrides the current decision if all posteriors compatible with retained suffix in the modeled adaptive law yield the opposite strict decision. Public parameters, arithmetic, scratch space, current input excluded. Exact adaptive filter on each policy's schedule is evaluator-only for quantized arms; implementation maintains evaluator history separately, and each arm's access to that history is restricted by the code. The over count is pre-override missed probe opportunities relative to this exact reference, not observed regret. Audit count includes probe and recovery overrides. The recorded Brier score uses the policy posterior before any recovery override.

## Sequential model-failure test

Let P be fixed nominal null, Q switching alternative. The history includes free observations, predictable probe purchases and their observed results. The likelihood ratio L=dQ/dP on observed history is a nonnegative P-martingale because every possible outcome has positive probability, and both laws use the same history-dependent policy. Expected conditional likelihood multiplier is 1. Ville's inequality gives P(sup L >= 1/alpha)<=alpha. Here alpha=.05. It does not guarantee detection or identify physical cause. Exact enumeration over four steps of adaptive myopic probing checks both path masses and the null likelihood-ratio expectation. This proof allows arbitrary predictable selection, including the gate.

## Finite-suffix certificate

Let d_i be the probability of the suffix observations from state i, n_i their joint probability with final bit 1. Positive emissions give d_i>0. For arbitrary starting distribution mu the posterior is sum(mu_i*n_i)/sum(mu_i*d_i), a convex combination of n_i/d_i weighted by mu_i*d_i. Thus the exact range over starting distributions is the min/max of six ratios. Backward likelihood recursion computes these from retained symbols; deterministic history-dependent probe decisions add no evidence conditional on full history. Treating the realized actions as fixed observation schedule gives a valid enclosing interval for the full-history filter, even when the suffix alone does not determine the policy's hidden memory. For interval [l,u], min V=max(.5,l,1-u), max V=max(u,1-l), V(p)=max(p,1-p). Strict probe decisions follow from w*max(0,eta-V)-c; after probe, scalar Bayes is monotone and preserves the bounds. Strict recovery uses which side of .5 contains the interval. Certificate is for the modeled adaptive law, including off-bank observations, not off-bank truth. Floating tolerances are numerical guards rather than formal interval arithmetic. All strict certificates checked against same-schedule exact reference.

## Results

Net utility per step. Paired episode normal-approximation intervals are descriptive, unadjusted for multiplicity. No independent replication.

| Arm | Nominal | Hazard shift | Sensor shift | Off-bank |
|---|---:|---:|---:|---:|
| fixed | 0.33929 | 0.32133 | 0.30198 | 0.29332 |
| static | 0.33936 | 0.32089 | 0.31032 | 0.29524 |
| adaptive | 0.33862 | 0.32068 | 0.31219 | 0.29676 |
| mode_oracle | 0.33929 | 0.32268 | 0.31823 | 0.30986 |
| current | 0.30404 | 0.30404 | 0.27453 | 0.28261 |
| gate | 0.33929 | 0.32128 | 0.30327 | 0.29366 |
| q7 | 0.33866 | 0.32125 | 0.30408 | 0.29420 |
| q31 | 0.33895 | 0.32095 | 0.31077 | 0.29689 |
| q7_audit4 | 0.33870 | 0.32125 | 0.30411 | 0.29413 |
| window3 | 0.33830 | 0.32198 | 0.30212 | 0.29420 |
| window8 | 0.33885 | 0.32124 | 0.30411 | 0.29432 |

| World | Gate alarms before step 32 | Gate alarms at/after step 32 | Median delay among later alarms | Audit overrides | q7 missed buys | q7 audit pre-override missed buys |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 0 | 1 | 51 | 3 | 215 | 217 |
| hazard_shift | 0 | 29 | 53 | 1 | 353 | 354 |
| sensor_shift | 0 | 66 | 48 | 3 | 448 | 448 |
| offbank | 0 | 55 | 51 | 2 | 417 | 417 |

| World | Adaptive minus fixed (95% interval) | Audit minus q7 | Audit minus matched window8 |
|---|---|---|---|
| nominal | -0.00067 [-0.00187, 0.00053] | 0.00005 [-0.00006, 0.00015] | -0.00014 [-0.00106, 0.00077] |
| hazard_shift | -0.00065 [-0.00202, 0.00072] | 0.00000 [0.00000, 0.00000] | 0.00000 [-0.00091, 0.00092] |
| sensor_shift | 0.01021 [0.00754, 0.01288] | 0.00003 [-0.00006, 0.00013] | -0.00000 [-0.00136, 0.00136] |
| offbank | 0.00344 [0.00133, 0.00556] | -0.00007 [-0.00019, 0.00005] | -0.00019 [-0.00135, 0.00098] |

Audit: 811008 decisions, 52907 strict probe certificates (includes zero-weight tasks), 29783 strict recovery certificates, 0 observed violations; 108 likelihood tree leaves, both path masses 1/1. Nominal alarms at any time count as false alarms. JSON gives other contrasts and Brier scores. No sealed test, learned neural self-model, quantum advantage or consciousness claim. Next preregister unknown-law families and equal-total-memory recurrent controls. Reproduce with Node 22: node qsm05_epistemic.mjs --verify. Replay compares integer events exactly and floating values within 1e-7. CI included, status pending at commit creation.
