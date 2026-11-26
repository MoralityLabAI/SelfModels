#!/usr/bin/env python3
"""Independent finite enumeration checks for the observer experiment."""

from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import unittest

import observer as o


def direct_label(world):
    # Independence means a truth table is constant within both z contexts.
    return world not in {0b0000, 0b0011, 0b1100, 0b1111}


@lru_cache(maxsize=None)
def enumerate_policy_scores(worlds, budget):
    """Enumerate scores of majority-labeled adaptive trees independently.

    Repeating a query supplies an unchanged record, so only splitting queries
    are enumerated. Stopping remains allowed at every node. Inferior leaf labels
    are pruned because they cannot improve either optimization criterion.
    """
    labels = [direct_label(w) for w in worlds]
    homogeneous = len(set(labels)) == 1
    scores = {(max(labels.count(False), labels.count(True)), len(worlds) if homogeneous else 0)}
    if budget == 0 or homogeneous:
        return frozenset(scores)
    for query in range(4):
        a = tuple(w for w in worlds if not w & (1 << query))
        b = tuple(w for w in worlds if w & (1 << query))
        if not a or not b:
            continue
        for left, right in product(enumerate_policy_scores(a, budget - 1),
                                   enumerate_policy_scores(b, budget - 1)):
            scores.add((left[0] + right[0], left[1] + right[1]))
    return frozenset(scores)


class ExactObserverChecks(unittest.TestCase):
    def test_truth_property_all_worlds(self):
        self.assertEqual(sum(direct_label(w) for w in o.WORLDS), 12)
        for world in o.WORLDS:
            self.assertEqual(o.depends_on_x(world), direct_label(world))

    def test_all_81_records(self):
        spaces = o.all_version_spaces()
        self.assertEqual(len(set(spaces)), 81)
        for worlds in spaces:
            self.assertTrue(worlds)
            self.assertEqual(o.certified(worlds), len({direct_label(w) for w in worlds}) == 1)
            for budget in range(5):
                expected = max(enumerate_policy_scores(worlds, budget))
                a, c, _ = o.best_policy(worlds, budget)
                self.assertEqual((a, c), (Fraction(expected[0], len(worlds)),
                                         Fraction(expected[1], len(worlds))))

    def test_record_replay_and_optimal_values(self):
        for budget in range(5):
            runs = [o.run_budget(w, budget) for w in o.WORLDS]
            for r in runs:
                self.assertLessEqual(len(r["record"]), budget)
                reconstructed = [w for w in o.WORLDS
                                 if all(((w >> q) & 1) == bit for q, bit in r["record"])]
                self.assertEqual(r["remaining_worlds"], reconstructed)
                self.assertIn(r["world"], reconstructed)
                if r["certificate"]:
                    self.assertTrue(all(direct_label(w) == r["prediction"] for w in reconstructed))
            accuracy, coverage, _ = o.best_policy(o.WORLDS, budget)
            self.assertEqual(accuracy, Fraction(sum(r["prediction"] == r["truth"] for r in runs), 16))
            self.assertEqual(coverage, Fraction(sum(r["certificate"] for r in runs), 16))
        self.assertEqual([o.best_policy(o.WORLDS, b)[0] for b in range(5)],
                         [Fraction(3, 4)] * 4 + [Fraction(1)])
        self.assertEqual([o.best_policy(o.WORLDS, b)[1] for b in range(5)],
                         [Fraction(0), Fraction(0), Fraction(1, 2), Fraction(1, 2), Fraction(1)])

    def test_each_incomplete_negative_record_is_ambiguous(self):
        # Any unqueried truth-table bit can flip an independent world into a
        # dependent one. This covers adaptive paths through an independent world.
        count = 0
        for world in (0, 3, 12, 15):
            for size in range(4):
                for observed in combinations(range(4), size):
                    missing = next(q for q in range(4) if q not in observed)
                    rival = world ^ (1 << missing)
                    self.assertTrue(direct_label(rival))
                    self.assertTrue(all(o.response(world, q) == o.response(rival, q) for q in observed))
                    count += 1
        self.assertEqual(count, 60)
        self.assertEqual(o.minimum_worst_case_queries(o.WORLDS), 4)

    def test_fsm_and_independent_expected_cost_lower_bound(self):
        runs = [o.run_fsm(w) for w in o.WORLDS]
        self.assertEqual({s for r in runs for s in r["states"]}, set(range(8)))
        for r in runs:
            self.assertEqual(r["prediction"], direct_label(r["world"]))
        self.assertEqual(sorted(len(r["record"]) for r in runs), [2] * 8 + [4] * 8)
        # Every exact observer needs >=2 queries. Independent worlds require 4.
        # At 2 queries at most half of uniformly weighted worlds have certificates;
        # before query 4, completing only one context certifies at most half.
        # E[T] = sum_{t>=0} Pr(T>t) >= 1 + 1 + 1/2 + 1/2 = 3.
        self.assertEqual(max(enumerate_policy_scores(o.WORLDS, 2), key=lambda v: v[1])[1], 8)
        self.assertEqual(max(enumerate_policy_scores(o.WORLDS, 3), key=lambda v: v[1])[1], 8)
        self.assertEqual(o.minimum_expected_queries(o.WORLDS), Fraction(3))
        self.assertEqual(Fraction(sum(len(r["record"]) for r in runs), 16), Fraction(3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
