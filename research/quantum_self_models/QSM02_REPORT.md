# QSM-02: Recurrent self-model compression under correlated quantum-channel drift

Parent QSM-01. Exploratory synthetic experiment, executed in a JavaScript sandbox. No hardware data, QPU time, neural training, or claim about consciousness.

## Design

At step t the hidden channel is a pair of independent local gates conditioned on a correlated classical hidden variable b=(b1,b2) in {-1,+1}^2. The task register is freshly prepared in |00>, and receives Ry(b_i theta) on each site. A controller chooses two corrective Ry(-a_i theta) rotations. Its utility is the exact expected probability of restoring |00>, rather than a sampled reward. Reward and the hidden state are withheld from the controller. The diagnostic is a separately prepared pair exposed to the same channel, at the cost of one additional two-qubit channel use. One binary outcome is supplied per step, using a fixed XX,XI,XX,IX schedule. For XX the measurement is the product of two X outcomes; raw constituent bits are deliberately withheld in every arm.

Theta=pi/3. The initial hidden pair is uniform. The step-wise hidden transition XORs the two sign bits with a random flip mask. In binary mask order 00,01,10,11 its probabilities are (.80,.03,.03,.14). In the shift condition they change at step 16 to (.55,.17,.17,.11). The exact and compressed matched-law arms know the transition schedule, while bayes_frozen continues to assume the original law. This is a test of representation and known-law filtering, not autonomous drift detection. Task inputs and probe observations do not depend on actions, so exact filtering followed by greedy expected utility is the Bayes-optimal equal-access controller. The privileged oracle instead sees the actual b.

A decoy arm receives the identical diagnostic history during the first eight steps, then the diagnostic of an independently initialized and evolving channel with the same marginal law. This is an intervention on diagnostic routing. The arm does not see the task diagnostics after the split. The decoy is not a separately fitted identity model. Since task and decoy histories are initially matched, pre-split equality is a strong implementation control.

2 regimes × 3 PRNG seeds (41,43,47) × 256 episodes × 32 steps. All 13 policies are evaluated on the same histories; only the decoy input differs. No training split is needed for known-law analytical controllers. Seeds and design were chosen for this exploratory run, not registered externally. The 32-step trajectories are the bootstrap units; no independent device/session inference. The LCG is deterministic for reproduction, not cryptographic.

## Why correlations matter

Let m1=E[b1], m2=E[b2], m12=E[b1 b2] under a posterior p. The exact four-state belief has three free moments, with

$$
p(b)=\frac{1+b_1m_1+b_2m_2+b_1b_2m_{12}}4.
$$

For a binary probe y of one Walsh character chi(b), Born likelihood is (1+y s chi(b))/2, where s=sin(theta) for XI or IX and s=sin^2(theta) for XX. If m_A=E[product_{i in A} b_i] with m_empty=1, then the posterior update is

$$
m_A^+=\frac{m_A+y s m_{A\triangle C}}{1+y s m_C}
$$

for the probe character C. Between observations, each moment is multiplied by the corresponding Walsh moment of the flip-mask law. Thus the parity moment influences future single-bit estimates through the update, even if it contributes little to immediate action scoring.

Writing c=cos^2(theta), A=(1+c)/2, B=(1-c)/2, exact action value is

$$
Q_p(a)=A^2+AB(a_1m_1+a_2m_2)+B^2a_1a_2m_{12}.
$$

With a uniform prior and just one XX observation, the single-site means remain zero while m12=y sin^2(theta). Choosing action parity matching y increases expected utility by B^2 sin^2(theta)=0.10546875 at theta=pi/3 relative to independent blind action. A factorized posterior with identical marginals loses all of this one-step information.

The gym distinguishes dropping m12 only at decision readout (the exact recurrent belief is retained) from replacing the recurrent posterior with the product of its marginals after every step. Both have access to the same probe; the latter cannot carry parity history forward.

## Compression experiment

Use symmetric fixed-point levels in [-1,1] including zero. Each b-bit moment stores a signed index over 2^b-1 levels, leaving one unused code. Decode the three full moments by the inverse Walsh formula, clip negative probabilities to zero, then renormalize. The factorized model encodes only two moments and reconstructs m12=m1*m2. At 6 bits compare 2 bits per full moment with 3 bits per marginal. At 12 bits compare 4 bits per full moment with 6 bits per marginal. Also test the full representation at 24 bits.

These are idealized persistent belief-code budgets. The implementation uses floating-point arrays and ephemeral unrestricted arithmetic to decode/update them, not a physically constrained memory allocator. The known transition law, probe schedule and fixed program are excluded from every memory budget. Unquantized arms are reference implementations and have no finite-bit accounting.

| Arm | Stationary utility | Shift utility |
|---|---:|---:|
| blind | .390999 | .385437 |
| memoryless | .544029 | .540718 |
| recurrent factorized | .713501 | .667816 |
| decision-only factorized | .778427 | .708099 |
| full 6-bit | .722328 | .631073 |
| factorized 6-bit | .720390 | .657791 |
| full 12-bit | .779839 | .711830 |
| factorized 12-bit | .713501 | .667816 |
| full 24-bit | .781364 | .713669 |
| exact Bayesian | .781334 | .713669 |
| frozen-law Bayesian | .781334 | .701103 |
| oracle | 1 | 1 |

Equal-budget full minus factorized at 12 bits: +.066338, paired 95% bootstrap interval [.062386,.070129] stationary; +.044014 [.039421,.048218] shift. At 6 bits the stationary comparison is unresolved (+.001938, interval [-.003372,.007088]); under shift the correlation-preserving grid loses (-.026718, interval [-.032158,-.021004]). This negative result is essential: if precision is sufficiently constrained, carrying an extra coordinate can hurt. No universal correlation-over-precision principle follows.

The decision-only factorization loses little relative to exact Bayesian, whereas recurring factorization loses substantially more. The main role of parity in this design is as recurrent memory for later inference. Full 12-bit utility is within 0.184 percentage points of exact Bayesian in both regimes. A finite sample can give an approximate policy slightly larger realized utility than the Bayesian reference, as in the stationary full-24-bit arm; exact conditional regret remains nonnegative.

After decoy routing intervention, exact self-probe beats decoy by +.382192 [.371460,.392527] stationary and +.300130 [.291168,.309031] shift, with exactly identical pre-intervention utility on every trajectory. This shows causal value of correct diagnostic routing. It does not establish semantic selfhood: an ordinary Bayesian filter uses the same information optimally. The matched-law exact arm beats the frozen-law arm after the shift by .025131 [.020447,.029663], confirming a cost of stale transition assumptions even with exact belief memory.

## Decision-preserving compression certificate

For approximate belief q, define

$$
\epsilon(p,q)=\min\left\{
\frac12\|p-q\|_1,\,
AB(|\Delta m_1|+|\Delta m_2|)+B^2|\Delta m_{12}|
\right\}.
$$

The first term bounds any reward expectation difference because utility is in [0,1]. The second follows from the action-value polynomial, triangle inequality, and |a_i|=1. Consequently all four value predictions are uniformly within epsilon. If the approximate greedy action's value margin exceeds 2 epsilon, it is the unique exact greedy action; regardless of margin its exact conditional regret is at most 2 epsilon. This is a real-arithmetic statement; floating-point tests allow small numerical tolerance.

We verified 638,976 decisions across all arms, with 359,889 strict-margin certificates. The largest observed value-bound excess was 3.33e-16 and regret-bound excess 5.56e-17. The full 12-bit arm certified 88.85% of decisions stationary and 86.31% under shift, with observed exact greedy action agreement of 97.67% and 98.43%. These certificates use the exact reference posterior retrospectively. They are not yet deployable bounds that a compressed observer can compute using its own state alone. The privileged oracle's "conditional_reference_regret" compares to the self-history Bayesian objective, not regret to the true hidden channel; its reward is the relevant privileged metric.

## What this says about quantum self-models

The task channel and probe distributions are quantum-mechanical; every tested controller is classical and the recurrent hidden variables are classical. This is the known-law negative-control gym for QSM-01, not a demonstrated quantum cognitive architecture. An equal-access Bayesian controller saturates expected decision utility. Any quantum-memory arm must earn its place through a separately defined resource or coherence benefit, e.g. preserving a reference under a fixed restricted interface, active probe selection with equal channel-call budget, or lower persistent memory at matched utility. No result measures experience or consciousness.

## Next: QSM-03

Use a two-state belief model of unknown transition regime, then extend the observer to learn channel parameters from diagnostics with hidden switches. Cross self-only and decoy-only interventions; add equal-compute recurrent learning and an active probe policy. Develop an online belief-error envelope from the prediction/contraction and local quantization errors, rather than relying on an exact reference at evaluation. For larger hidden systems, compare Walsh moment selection, tensor-network latent compression and learned low-rank state representations. Keep the 6-bit reversal as a registered challenge to all new compression schemes.

## Reproduction

From this directory, run:

    node qsm02_gym.mjs /tmp/qsm02.json
    python qsm02_verify.py /tmp/qsm02.json qsm02_summary.json qsm02_trajectories.csv

The raw trajectory table and aggregate summary are committed. The JS function was run in the sandbox to produce these receipts. Standalone Node and GitHub Actions are pending.
