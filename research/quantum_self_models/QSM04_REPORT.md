# QSM-04A: active self-diagnostics, opportunity value and robust finite-history certificates

Exploratory CPU/sandbox pilot, 2026-11-25. Candidate channel laws, diagnostic accuracy, and nominal hazard are known; no learned unknown dynamics, quantum advantage or consciousness claim. This is independent of QSM-03's pending CI fix.

## Environment and exact decision value

Hidden r in {0,1} selects identity or X channel on a computational-basis qubit. Recovery X^a succeeds iff a=r. The controller observes a public binary task opportunity w and probe cost c, with a three-step forecast for planning policies. Opportunities are clustered: P(w_next=1|w=0)=.23 and P(w_next=1|w=1)=.65. Costs .04 and .14 equiprobable. The symmetric diagnostic reveals r with accuracy eta=.8, can be purchased once each step, and has no effect on future dynamics. Task rewards and hidden states are never observed by nonprivileged policies. Net utility per step is w*1[a=r]-c*1[probe]. Hidden regime switches independently with nominal probability h=.08; a second world uses .20 but controllers retain .08. Same potential diagnostic outcomes, hidden states, opportunities and costs across arms; worlds share random tapes. Independent decoy regime supplies the decoy arm's diagnostic; this arm is not eligible for self certificates. Each world uses 384 episode clusters, 64 steps, 13 policies. Seeds and design chosen for this exploratory pilot, not a sealed prospective test.

For prior p=P(r=1), immediate optimal recovery reward is V(p)=max(p,1-p). The diagnostic yields posterior p_y with P(y=1)=z=(1-eta)+(2eta-1)p. Its expected optimal recovery value equals max(V(p),eta): if p>=eta or p<=1-eta the preferred recovery is unchanged; otherwise act on y and succeed with probability eta. Thus VOI=max(0,eta-V(p)); buy iff w*VOI>c. Checked numerically over 1001 priors.

Define F_h(p)=h+(1-2h)p, C(p)=w*V(p)+V_next(F_h(p)). Finite-horizon Bellman value is max(C(p), -c+E_y C(p_y)), terminal value zero. The horizon-3 policy replans each step; it is not globally optimal for the 64-step task. Equal-cost baseline arms include no-probe fixed action, current-only myopic, always-probe exact recurrent, scheduled probe if current opportunity or next opportunity with low current cost, recurrent myopic, independent-decoy lookahead, quantized and finite-window lookahead. Quant grids use 2^b-1 levels, including .5. Window stores L ternary symbols (no probe, 0, 1), hence ceil(L log2 3) persistent bits, plus public time, model, arithmetic and work space. Recovery ties within numerical tolerance choose 1; probe ties abstain.

## Results

| Arm | Matched opportunity recovery | Matched net/step | Matched probe rate | Fast drift net/step |
|---|---:|---:|---:|---:|
| oracle | 100.00% | 0.40039 | 0.00% | 0.40039 |
| blind | 49.64% | 0.19877 | 0.00% | 0.20089 |
| current_only | 80.16% | 0.28509 | 40.04% | 0.28509 |
| always_probe | 84.18% | 0.24708 | 100.00% | 0.21811 |
| scheduled | 83.26% | 0.29475 | 46.90% | 0.27245 |
| myopic | 79.38% | 0.30240 | 21.47% | 0.27580 |
| lookahead3 | 81.19% | 0.30636 | 30.12% | 0.28097 |
| decoy3 | 49.44% | 0.17912 | 30.30% | 0.17753 |
| quant3 | 56.44% | 0.22251 | 5.29% | 0.20815 |
| quant5 | 81.13% | 0.30582 | 30.29% | 0.28144 |
| window2 | 80.23% | 0.30049 | 32.84% | 0.28451 |
| window4 | 81.06% | 0.30529 | 30.81% | 0.28158 |
| window8 | 81.18% | 0.30634 | 30.18% | 0.28064 |

Paired episode bootstrap (4000 draws; descriptive, no multiplicity correction):

| Difference | Net/step | 95% interval |
|---|---:|---:|
| lookahead3_minus_blind | 0.10759 | [0.09726, 0.11804] |
| lookahead3_minus_current_only | 0.02128 | [0.01774, 0.02488] |
| lookahead3_minus_always_probe | 0.05929 | [0.05616, 0.06243] |
| lookahead3_minus_scheduled | 0.01161 | [0.00881, 0.01432] |
| lookahead3_minus_myopic | 0.00397 | [0.00066, 0.00727] |
| lookahead3_minus_decoy3 | 0.12724 | [0.11892, 0.13530] |
| lookahead3_minus_quant3 | 0.08385 | [0.07505, 0.09295] |
| lookahead3_minus_quant5 | 0.00055 | [-0.00047, 0.00157] |
| lookahead3_minus_window2 | 0.00587 | [0.00297, 0.00887] |
| lookahead3_minus_window4 | 0.00108 | [-0.00105, 0.00314] |
| lookahead3_minus_window8 | 0.00003 | [-0.00152, 0.00160] |
| oracle_minus_lookahead3 | 0.09403 | [0.08944, 0.09859] |

## Robust finite-history theorem

The diagnostic update B_y(p) is monotone increasing, including B_none(p)=p. If p in [l,u], then B_y(p) lies in [B_y(l),B_y(u)]. For h in [h_min,h_max], the bilinear function F_h(p) attains its extrema at the four rectangle corners. Iterating these two maps from [0,1] at the beginning of the suffix encloses every compatible posterior even for arbitrary time-varying hazard in the allowed interval, provided the observation model is correct. For warmup use the known initial .5. A recovery action is strictly optimal for all such histories when the posterior interval lies wholly on its side of .5. During simulation audit both the nominal-law exact posterior and true-hazard exact posterior on the realized diagnostic schedule of each tested policy, rather than against an exact controller's different schedule. No exact posterior enters the window policy or certificates. The nominal probe certificate also bounds action gaps: each forced-first-action optimized value is a supremum of affine functions of the initial prior, each with slope magnitude at most R=sum of public weights plus sum of costs over the horizon. Difference is 2R-Lipschitz. If its estimated magnitude exceeds 2R times the largest distance to either nominal interval endpoint, the probe/no-probe choice agrees with the exact reference for that same schedule. This suffices for nominal h only, not robust probe optimality. Floating tolerances here are empirical guards, not formal interval arithmetic.

| Window | Persistent bits | Matched nominal agreement | Matched strict nominal | Matched strict robust | Fast strict robust | Nominal probe cert coverage | Nominal probe agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| window2 | 4 | 94.77% | 24.58% | 24.52% | 23.95% | 5.59% | 91.38% |
| window4 | 7 | 98.40% | 81.92% | 81.62% | 80.24% | 12.47% | 95.44% |
| window8 | 13 | 99.62% | 97.13% | 95.42% | 94.60% | 40.26% | 98.56% |

Audit counts:

{
  "world_episodes": 768,
  "decisions": 638976,
  "voi": 1001,
  "interval": 147456,
  "nomCert": 39850,
  "robustCert": 39395,
  "probeCert": 28717,
  "violations": 0,
  "maxIntervalExcess": 0,
  "maxVoiError": 2.220446049250313e-16
}

Scope: finite-state, classical filter for a Pauli channel. No claim that explicit self-models beat an equal-memory finite-history policy, that diagnostic information is intrinsically self-specific, or that hardware is conscious. The decoy control identifies input dependence for this designed causal intervention. The forecast is a public task-side advantage shared by all planning arms. Test a concealed fresh family of channels and probe reliabilities, unknown drift and equal-bit learned recurrent controllers before making a general compression claim. No QPU or paid compute used. Sandbox simulation executed; Node 22 CI replay pending.

Run node qsm04_active.mjs /tmp/result.json then python qsm04_verify.py /tmp/result.json qsm04_summary.json qsm04_trajectories.csv from this directory.
