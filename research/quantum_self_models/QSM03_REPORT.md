# QSM-03: Hidden-regime adaptation, online belief bounds, and finite-history decision certificates

Exploratory simulated research, 2026-11-25. Parent QSM-02 (merged main 255f1408eb27b43f011a6528a2739762621c6eb0). No quantum hardware, learned neural parameters, or consciousness claim.

## A. Hidden transition regimes

Extend QSM-02's two-qubit sign channel to a joint hidden state x=(r,b1,b2), with r in {0,1} denoting the transition regime. The two flip-mask laws are (.80,.03,.03,.14) and (.55,.17,.17,.11), in binary mask order. The matched environment changes regime with probability .06 per step, independently of the previous sign bits. The new regime determines the current mask distribution. The known column-stochastic transition matrix is

T[i,j] = P(r_i | r_j) f_{r_i}[(i XOR j) mod 4].

Every entry is positive, and the uniform distribution is stationary. The hidden regime is withheld from the adaptive filter, while the regime-informed comparison receives it each step. The adaptive observer knows the two candidate laws and switching hazard. It performs Bayesian identification of a hidden regime, not learning an unknown transition law. The static mixture knows only the stationary average flip law. Other static controllers use the slow or fast law alone.

Each step prepares a fresh task pair and a separate diagnostic pair; the single binary diagnostic outcome follows the Born likelihood of one of XX, XI, XX, IX in a fixed cycle. Hidden bits govern local Ry rotations at theta=pi/3. The reward is exact expected |00> restoration probability; reward and hidden bits are never fed back to a tested nonprivileged controller. Actions do not affect future hidden dynamics or probes; the greedy matched-law Bayesian controller is therefore optimal in expected utility among equal-access controllers.

Use 3 PRNG seeds (101,103,107), 128 episodes per seed, 32 steps, and 3 worlds, 1,152 trajectories and 36,864 steps total. Worlds:
- matched: hidden Markov switching;
- self_switch: force the task regime to stay at its initial value until step 16, then flip and hold it; the observer still assumes the original HMM;
- decoy_switch: keep the matched task trajectory exactly and independently force the decoy regime to switch at step 16.

The decoy sees the same input and diagnostic history for steps 0–7, becomes independent at step 8, then uses its own diagnostic. Common random tapes hold the matched and decoy-switch self histories identical. All self-input policy metrics agree on every paired trajectory across those two worlds. Pre-split matched self and decoy rewards are identical. After separation, self diagnostic versus decoy produces +.326518 utility, paired interval [.312215,.340902]. The extra forced decoy regime switch does not have a resolved late-reward effect on the decoy arm; the intervention is a noninterference check, not evidence of precise causal dynamics identification. The three worlds share random tapes, so 1,152 is the number of simulated world-trajectories, not 1,152 independent random initializations; each within-world bootstrap uses 384 episode clusters.

### Result: identifying the hidden regime has little decision value here

| Controller | Matched reward | Forced self switch |
|---|---:|---:|
| Memoryless | .544159 | .546967 |
| Static slow | .705673 | .706131 |
| Static mixture | .714935 | .714615 |
| Exact adaptive | .714386 | .713623 |
| Regime informed | .718826 | .719330 |
| Product posterior, unquantized | .714935 | .714676 |
| Joint 14-bit | .491318 | .491974 |
| Product 14-bit | .711182 | .712296 |
| Joint 21-bit | .708145 | .707932 |
| Product 21-bit | .712463 | .712662 |
| Joint 28-bit | .710892 | .711456 |
| Product 28-bit | .715607 | .715286 |
| Joint 42-bit | .713196 | .712830 |
| Product 42-bit | .715546 | .715286 |

The exact adaptive-minus-static mixture reward difference in the matched world is -.000549, 95% paired bootstrap interval [-.002472,.001465]. The adaptive regime Brier score is .227592 vs .25 for static, a reduction of .022408 with paired interval [.016787,.027809]. Even the privileged regime-informed controller improves reward by only .004440 [.001938,.006897]. This task admits little operational value from explicit regime knowledge. It fails the proposed utility-gain hypothesis.

The code uses seven nonconstant log-Walsh coefficients for a positive joint belief and four coefficients for p(r)p(b1,b2). The product arm preserves within-task bit correlations but drops regime-task dependence. Allocate the indicated number of bits approximately evenly across coordinates, using a fixed symmetric log-coefficient grid in [-2,2] with 2^b-1 levels. Decode via exponentiation and normalization, maintaining positive probabilities. The grid, allocation and range are exploratory choices. The bit labels describe only the persistent belief code: known model, time/phase, decoder, and certificate workspace are extra resources. At 14 bits the joint code loses catastrophically; at 21 and 28 bits it also loses, with matched paired intervals excluding zero. At 42 bits the joint-vs-product difference is unresolved. Do not generalize QSM-02's correlation result beyond its original representation and task.

## B. Online error envelopes without privileged exact beliefs

Let p_t be the reference filter under the assumed model, q_t the compressed belief, and tilde q_t=F_t(q_{t-1}) the uncompressed update from q's previous belief. All distributions here are strictly positive. The filter F_t is a positive transition followed by multiplication by the observed likelihood and renormalization. Define the Hilbert distance

$$
d_H(p,q)=\log\max_i(p_i/q_i)-\log\min_i(p_i/q_i).
$$

Likelihood multiplication by the same positive diagonal and normalization leave this metric unchanged. A positive matrix M contracts it with Birkhoff coefficient

$$
\tau(M)=\tanh\left(\tfrac14\max_{i,j,k,l}\log\frac{M_{ik}M_{jl}}{M_{il}M_{jk}}\right).
$$

If delta_t=d_H(tilde q_t,q_t), the online radius H_t=tau(T) H_{t-1}+delta_t bounds d_H(p_t,q_t), initialized at delta_0 for the common prior. This follows from contraction and the triangle inequality. No exact posterior is needed to compute H_t.

For a realized block of L filtering maps A_t...A_{t-L+1}, put tau_{t,j}=tau(A_t...A_{t-j+1}), with tau_{t,0}=1. By telescoping, applying the suffix maps to each projection error separately,

$$
H_t \le \tau_{t,L} H_{t-L}+\sum_{j=0}^{L-1}\tau_{t,j}\delta_{t-j}
$$

where the right-hand side is a valid new radius for the actual distance. In implementation take the minimum of valid bounds for L=1..4, excluding the initial observation-only map from transition blocks. Store the latest four radii and local errors and four outcome bits. Block coefficients depend only on the fixed law, phase, and observed word, so they can be precomputed. Here single-step tau=.986350 and four-step tau ranges .705729–.739916. The total variation radius from projective geometry is tanh(H_t/4).

A separate TV recursion uses Dobrushin's coefficient kappa(T)=max_{i,j} TV(T e_i,T e_j). For likelihood g in [m,M], normalized update B_g satisfies

$$
\operatorname{TV}(B_g p,B_g q)\leq \frac{M}{q\cdot g}\operatorname{TV}(p,q).
$$

Proof: for any 0<=f<=1, (q.g)(B_g p(f)-B_g q(f))=(p-q)(g(f-B_g p(f))). The range of the latter test function has width at most M. Take the supremum. After transition, use the approximate predictive likelihood q.g computed before normalization; add the observed local compression TV error. The hybrid radius takes the minimum of this recursion, the block Hilbert TV bound and 1.

For greedy approximate action a and competitor b, let h_i=R_i(a)-R_i(b). The exact margin is bounded below by q.h-osc(h) times the hybrid TV radius (i.e. q.h minus the product of osc(h) and the radius). A second certificate uses positive and negative q-weighted masses A_h=sum q_i max(h_i,0), B_h=sum q_i max(-h_i,0). If log(A_h/B_h)>H_t, then p.h>0: writing m=min p_i/q_i, every ratio is in [m,m exp(H_t)], so p.h>=m(A_h-exp(H_t) B_h). If B_h=0 and A_h>0 the sign is immediate. Require all competitors to pass one complete sufficient test (TV or projective). The TV bound also gives an action-specific regret envelope, max over competitors of their approximate value advantage plus the oscillation correction, clipped to [0,1].

All 331,776 compressed-model decisions audited against exact p; 22,679 strict online certificates and zero violations (tolerance 1e-9). For matched joint-42 the plain uniform-margin TV recursion certifies 13.31%, single-step Hilbert 15.21%, block Hilbert 17.02%, and hybrid/action-specific union 29.97%. Actual mean TV=.022842 while online hybrid radius=.312903; loose worst-case bounds explain much of the uncovered fraction. This is a real-arithmetic argument with floating-point empirical validation, not an interval/roundoff certificate. In forced-switch worlds the target is the exact posterior of the assumed HMM, not the true posterior under the forced data-generating process. An online compression certificate does not certify model correctness.

## C. Finite-history convex-hull decision certificates (QSM-03B)

The HMM is a finite-state positive linear filter before normalization. For an observed suffix of L transition/observation maps form M=A_t...A_{t-L+1}. Let c_j=M e_j/(1^T M e_j) be its normalized columns and s_j=1^T M e_j>0. For any prior p in the simplex,

$$
\frac{Mp}{\mathbf{1}^\top Mp}=\sum_j\frac{s_jp_j}{\sum_k s_kp_k}c_j.
$$

The weights form a probability vector. Conversely, any convex combination of the columns can be obtained by setting p_j proportional to w_j/s_j. Thus the reachable posterior set over arbitrary preceding priors is exactly the convex hull of the normalized columns.

For candidate action a with expected reward vectors R_a, its worst-case regret across all preceding beliefs is exactly

$$
\max_j\max_b (R_b-R_a)^\top c_j .
$$

The upper bound follows by convex combination and maximization; equality is attained at a maximizing vertex via its corresponding point-mass prior. Unique optimality for every preceding prior holds if each alternative has strictly lower value at all columns. This certificate is task specific; it need not bound the whole posterior. With zero reward differences, report a nonunique optimum or abstain from strict certification.

The window controller retains L recent binary observations and reconstructs its belief by filtering a uniform prior through the observed suffix. During warmup, this is the exact filter from the known initial prior because T fixes uniform. After warmup, the column test holds regardless of the unobserved earlier prior. Known phase, warmup counter and model parameters are not counted as observation bits. The benchmark caches matrices across words for speed; this cache is not part of a bounded-memory hardware implementation. Recompute products from the retained suffix to avoid caching, at additional compute cost. No claim of L total bits of memory or efficient scaling to large hidden-state dimensions.

Reused all QSM-03 worlds, seeds and self tapes, independently replayed exact reference rewards. Results in matched world:

| Observation bits | Reward | Greedy action agreement | Strict cert, all steps | Strict cert, after warmup | Mean exact conditional regret | Mean worst-case regret |
|---:|---:|---:|---:|---:|---:|---:|
|1|.544159|.514730|0|0|.170748|.638331|
|2|.714935|.972087|.031250|0|.001362|.194631|
|4|.714935|.972087|.801676|.809059|.001362|.016845|
|6|.714523|.992676|.879476|.890124|.00008457|.004067|
|8|.714935|.994466|.920003|.935004|.00006829|.001535|
|10|.714584|.996338|.943929|.963897|.00003253|.000606|
|12|.714294|.998454|.952311|.973698|.00000882|.000276|

Exact Bayesian reward .714386. The 12-bit window's reward gap is .0000916, paired 95% interval [-.000366,.000565]. All 258,048 window decisions checked; 166,953 strict certificates, no regret bound violations, warmup max TV discrepancy 3.02e-16. The 12-bit window gives 95.13% strict coverage in the forced-self-switch world, 97.20% after warmup, relative to the assumed model. The eight-bit window happens to outperform the misspecified HMM in realized utility there; this does not invalidate the modeled regret bound.

The important negative result is that two observation bits already reproduce the static-mixture policy's realized decisions/rewards on this gym, and a longer finite history gives high certification coverage. A sophisticated self-representation is unnecessary for this task's operational demands. A fair next gym must show value relative to this strongest cheap baseline, rather than relative to a blind or unrelated-input arm. The direct column certificate is the most useful contribution: it transfers to any finite positive filter and bounded discrete action set, though large state dimension and long horizons require structure to avoid dense products.

## D. Follow-up

QSM-04 should create regimes that change the *counterfactual choice*, not merely hidden transition statistics. Introduce interventions where the action that maximizes expected reward depends on the regime (e.g. different unknown correction axes, conditional readouts or noise-cost tradeoffs), with a positive lower bound on oracle-versus-static opportunity. Compare passive and active diagnostic selection at equal total channel-call budget. Evaluate on fresh hidden laws, seed families and decoy switches, against fixed finite-history baselines and exact equal-access Bayesian control where possible. For quantum memory, add a separate coherent-reference outcome with equal coherent controls, rather than claiming a quantum speedup from the classical filter.

Transfer the column certificate to low-rank/tensor-network filters: bound action gaps at normalized columns with rigorous contraction and denominator error, or seek an observable-specific reachable-set certificate without enumerating all columns. Account for numerical rounding separately and for misspecification through a defined family of allowable maps. No sentience conclusion follows.

## Reproduction and limitations

Executed both JS functions in the sandbox. Standalone Node and repository CI reproduction are pending. From this directory:

    node qsm03_gym.mjs /tmp/qsm03.json
    node qsm03b_window.mjs /tmp/qsm03b.json
    python qsm03_verify.py /tmp/qsm03.json qsm03_summary.json qsm03_trajectories.csv
    python qsm03_verify.py /tmp/qsm03b.json qsm03b_summary.json qsm03b_trajectories.csv

Raw episode-policy rows and summary JSON committed. The PR's CI reruns both simulations and validates all receipts. No paid compute or QPU time used.

### Cross-runtime tie audit

The original QSM-03A Node 22 CI replay failed on regime accuracy: product_14 matched accuracy differed at the floating-point boundary p(r=1)=0.5. Classify probabilities within 1e-12 below 0.5 as regime 1, matching the deterministic tie convention. The regenerated sandbox receipts differ only in regime_acc (39 aggregate cells, 10487 episode-policy rows); all reward, Brier, regret, decision and certificate metrics are unchanged. The full receipt verifier remains unchanged. Node CI for this correction is required before merge. QSM-03B independently passed its original Node CI.
