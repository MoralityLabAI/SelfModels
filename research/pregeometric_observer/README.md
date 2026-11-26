# PGC observer budgets: exact finite records

This experiment computes what an observer can certify from resettable queries to a Boolean response rule. It enumerates every rule in its declared class and every accessible partial truth table. The tested target is whether the output depends on input `x` in at least one context `z`.

The experiment belongs to the pregeometry program's observation layer. Register addresses, a binary alphabet and a query clock enter through the interface. Each physical target, including spacetime dimension and metric signature, remains undetermined by this model.

## Reproduce

Python 3.12 and its standard library suffice. From this directory:

```sh
python3 verify_observer.py
python3 observer.py --check receipt.json
```

Generate a fresh receipt with `python3 observer.py --output /tmp/observer_receipt.json`. The receipt contains source hashes and all 16 finite-state observer traces. Wall-clock timing is excluded from its deterministic content.

## Primitive model and target

A world is one of the 16 functions `f : {0,1}² → {0,1}`. Its integer encoding stores `f(x,z)` in bit `2*z+x`. The observer chooses `(x,z)`, receives one output bit and may reset the interface before its next query. Every query and response counts as one observation step.

The property is

\[
p(f)=[f(0,0)\ne f(1,0)]\lor[f(0,1)\ne f(1,1)].
\]

Twelve worlds satisfy the property. The four independent worlds have encodings `0`, `3`, `12` and `15`. The prior weights all 16 worlds equally. An accessible record leaves a *version space*: all functions consistent with its observed outputs. A certificate exists precisely when every function in that version space has the same property value.

There are `3^4 = 81` possible partial truth tables because each output can be unknown, zero or one. Exactly 36 version spaces certify the target. This count weights partial records equally; the table below weights worlds under the declared prior.

## Exact results

The policy maximizes expected classification accuracy and then certificate coverage. Query-index order resolves remaining ties. A class tie predicts dependence. Queries carry no utility penalty within the selected budget, and execution stops once a certificate is available.

| Maximum queries | Optimal accuracy under the uniform prior | Balanced accuracy of the selected policy | Worlds with a certificate | Mean queries used |
| --- | --- | --- | --- | --- |
| 0 | 75% | 50% | 0 / 16 | 0 |
| 1 | 75% | 50% | 0 / 16 | 1 |
| 2 | 75% | 50% | 8 / 16 | 2 |
| 3 | 75% | 50% | 8 / 16 | 2.5 |
| 4 | 100% | 100% | 16 / 16 | 3 |

The 75% score at zero queries comes from the prior's class frequencies. Two queries can already certify dependence in half of the worlds while leaving the best average accuracy at 75%. The balanced-accuracy column records the selected policy; its objective remains ordinary accuracy with certificate coverage as a secondary criterion. Predicting independence on a tied two-query leaf gives the same 75% ordinary accuracy and balanced accuracy of 5/6. This alternative shows how the declared tie rule affects that reported metric.

An exact observer needs four queries in the worst case and three on average under the uniform prior. The dynamic program considers every available query at every version space. A separate enumeration computes accuracy and certificate counts for adaptive decision trees, including the option to stop at each node. It removes inferior leaf labels because they cannot improve either objective. The optima agree for all 81 version spaces at budgets zero through four.

### A sufficient observer with eight states

The supplied finite-state machine first queries `f(0,0)` and `f(1,0)`. A disagreement certifies dependence after two queries. Equal outputs cause it to query the second context. It then either finds a disagreement or certifies independence from the complete truth table.

Its eight states include the start state and both terminal decisions, so three state bits suffice for this fixed algorithm. The transition table and the external query-response interface contribute separate resources. The Python runner also retains an audit transcript outside the modeled state register. The implementation establishes a sufficient state size; minimum state complexity is an open question within this accounting convention.

Every world receives the correct answer. Eight worlds stop after two queries; the other eight use four.

### Why the query bounds are exact

An independent world requires all four outputs to certify independence. Given any incomplete record from such a world, flipping an unobserved truth-table bit produces a dependent world with the same record. The verifier exhausts all 60 pairs of independent worlds and incomplete query subsets.

A certificate also requires at least two queries. At budgets two and three, at most half of the uniformly weighted worlds can have certificates. This follows both from the decision-tree enumeration and from the fact that three queries can complete only one of the two contexts. Thus every exact policy has

\[
\mathbb E[T]
=\sum_{t\ge0}\Pr(T>t)
\ge1+1+\tfrac12+\tfrac12=3.
\]

The eight-state observer attains this bound. Randomizing the policy cannot improve these expected optima because a randomized decision tree is a mixture of deterministic trees under this finite objective.

## Assumption ledger

| Element | Status | Consequence |
| --- | --- | --- |
| Binary inputs, named addresses `x,z` and binary outputs | Supplied | The observer receives a fixed intervention vocabulary. |
| Resettable queries with one output bit per tick | Supplied | Records have a declared temporal order and response cost. |
| Sixteen deterministic Boolean functions | Supplied | Certificates quantify over this finite class. |
| Uniform prior and majority tie rule | Supplied | Expected accuracy and mean query counts depend on these choices. |
| Exact output records | Supplied | Noise and finite sampling require a different statistical model. |
| Version spaces and dependence certificates | Computed exactly | The record determines which declared worlds remain possible. |
| Query-budget optima | Computed exactly | Dynamic programming agrees with independent decision-tree enumeration. |
| A continuum manifold, Lorentzian signature, null boundaries, dimension four, Einstein variables and `G_N` | Undetermined | The model has no physical observable or limit establishing these targets. |

## Relation to the other PGC work

[Compactification Theory's PGC-01 experiment](https://github.com/MoralityLabAI/Compactification-Theory/tree/research/pgc01-operational-foundations-20261126/research/pregeometric_compactification) studies influence through supplied Boolean update rules, including cancellation and finite-horizon ambiguity. The [MathCan module](https://github.com/patrickdugan/MathCan/blob/research/pgc01-operational-identifiability-20261126/MathCan/OperationalIdentifiability.lean) proves the general observation-map obstruction: equal records paired with different target values prevent a uniformly correct decoder.

This experiment supplies explicit worlds and records for that obstruction. For example, worlds `0` and `8` agree on queries `0,1,2` and have different dependence labels. Query `3` separates them. Applying the same theorem to a spacetime property requires an explicit admissible physical model class and a corresponding observational witness.
