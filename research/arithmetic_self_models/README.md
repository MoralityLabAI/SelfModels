# ASM-01: when an internal regime changes the useful action

This exact arithmetic control addresses the decision-opportunity issue recorded
in [QSM-03](../quantum_self_models/QSM03_REPORT.md). The observer's hidden regime
changes which reset succeeds. A diagnostic from the observer's own process
then has measurable value under a fully specified finite prior.

The instance is a constructed development fixture. The two arithmetic laws
and reset seeds were selected to create a reversal in the successful action.
Every result below is an exhaustive finite calculation. Neural models,
quantum hardware, regime drift, and unseen-law generalization remain future
experiments.

## Model and observations

Let the hidden regime be `d=5` or `d=49`, with equal prior probability. Its
initial internal state is uniformly distributed over the 25 odd integers
from 1 through 49, independently of the regime. The arithmetic law is

\[
T_d(n)=\frac{a(n)n+d}{2^{v_2(a(n)n+d)}},\qquad
a(n)=\begin{cases}3&n\equiv1\pmod4\\1&n\equiv3\pmod4.\end{cases}
\]

At phases `0..15`, the observer receives the single bit `(n mod 4 - 1)/2`.
Phase and both candidate laws are public. The actual state and regime stay
inside the evaluator. Rewards are never fed back. Each phase evaluates a
counterfactual reset of the same hidden process to seed 7 or seed 15. The
reward is one exactly when the reset eventually reaches the odd cycle with
minimum 1. Evaluating a reset leaves the diagnostic tape intact, so successive
rows are counterfactual decision problems along one episode.

| Hidden regime | Reset to 7 | Reset to 15 |
|---|---:|---:|
| 5 | 1 | 0 |
| 49 | 0 | 1 |

The equal-access Bayesian controller groups all compatible hidden episodes by
its allowed observation and selects the action with greatest exact conditional
reward. Count-only access sees the number of one-bits, while a window sees the
last `L` bits and phase. The full-history arm sees the entire available word.
Every arm uses the same generative prior; its tables are computed analytically
from that known model.

## Executed results

Each arm evaluates the same 50 episodes at 16 phases, yielding 800 decisions.
The repeated phases share episodes. These are finite-population values, with
no estimated sampling uncertainty or significance claim.

| Access | Successful decisions | Reward | Robust certificates |
|---|---:|---:|---:|
| Phase alone | 400/800 | 50.000% | 0 |
| Current bit | 566/800 | 70.750% | 142 |
| Last 2 bits | 626/800 | 78.250% | 205 |
| Last 4 bits | 689/800 | 86.125% | 281 |
| Last 8 bits | 695/800 | 86.875% | 385 |
| Count of one-bits | 643/800 | 80.375% | 363 |
| Entire history | 695/800 | 86.875% | 519 |
| Regime disclosed | 800/800 | 100.000% | 800 |

A robust certificate means that the chosen reset succeeds for every hidden
episode compatible with the observation. This is stronger than selecting a
Bayes-optimal action. Full history and eight-bit access have equal reward on
this fixture, while full history certifies 134 additional decisions. A richer
description can improve the certainty of a decision whose action already
agrees with the shorter-memory policy.

For a matched decoy control, independently pair all 50 self episodes with all
50 decoy episodes. Apply the full-history policy to the decoy tape. Exact
enumeration gives 20,000 successes in 40,000 phase decisions, or 50%. The
observer's own history supplies the useful information in this construction.

The known-regime opportunity over phase alone is 50 percentage points. The
full-history gain over the current-bit arm is 16.125 points. Eight recent bits
already attain the entire-history reward in this population. That equality
sets a demanding inexpensive control for a future learned self-model.

## Exact argument and certificate

For an allowed observation `o`, let `N_r(o)` count compatible initial episodes
under regime `r`. The equal prior makes the posterior

\[
P(r\mid o)=\frac{N_r(o)}{N_0(o)+N_1(o)}.
\]

The optimal action chooses the larger count, with ties assigned to seed 7.
Summing `max(N_0,N_1)` over observation cells gives the exact number of
successful counterfactual decisions. The script verifies that this count
equals direct action replay. If one count vanishes, every compatible episode
has the same successful action, which establishes the robust certificate.

## Reproduce

From repository root, using Python 3.11 or newer:

```sh
python research/arithmetic_self_models/asm01.py --check
python -m unittest discover -s research/arithmetic_self_models -p 'test_*.py' -v
```

`summary.json` stores exact rational values. `decisions.csv` contains 5,600
arm-specific decision records, including the evaluator's hidden values for
audit. A training or inference adapter must expose only `key()`'s allowed
features. Tests alter all hidden fields while holding observations fixed and
verify that policy keys remain unchanged. The workflow replays both receipts
byte for byte and checks the certificate and information-refinement properties.

`collatz_core.py` is a vendored copy of the MathCan Collatz resonance oracle.
Its SHA-256 is recorded in the summary. The implementation uses finite-core
enumeration and exact integer transitions; the fixture permits initial states
above the smaller core and follows them until they enter it.

## Next experiment contract

A future learner should hold out complete parameter pairs and include drift
laws under which the correct action changes during an episode. Preserve the
eight-bit policy and the count-only policy as controls. Give every learned
arm the same observations and transition-call budget. Before training,
register a family of independent law pairs, a minimum useful reward gain,
paired decoy controls and the total comparison count, following EvalMath's
mechanistic-power preflight. These 50 selected episodes remain development
data and provide zero independent evidence about a learned mechanism.

Evidence card: unit = constructed parameter pair; `n_units=1`;
`n_cases=50` episodes and 800 counterfactual decisions per arm;
primary contrast = own full history versus current bit; diagnostic dose = one
bit per phase; seven equal-access controls and one disclosed-regime ceiling;
matched decoy = all 2,500 independent pairings; readout exposure = fully
inspected development fixture; evidence class = exact finite domain. Future
scientific and sham variation floors require independent development pairs.
