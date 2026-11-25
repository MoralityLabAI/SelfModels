import unittest
from fractions import Fraction

from asm01 import REGIMES, RESET_SEEDS, evaluate, eventual_basin, key, population


class ArithmeticSelfModelTest(unittest.TestCase):
    def test_reset_choice_has_regime_value(self):
        self.assertEqual([[eventual_basin(n, d) == 1 for n in RESET_SEEDS]
                          for d in REGIMES], [[True, False], [False, True]])

    def test_observation_key_never_reads_hidden_state(self):
        row = population()[0]
        altered = dict(row, regime=999, d=999, state=999, initial_seed=999)
        for arm in ("blind", "count_only", "full_history", "window_4"):
            self.assertEqual(key(arm, row), key(arm, altered))

    def test_information_refinement_and_decoy(self):
        summary, _ = evaluate()
        arms = summary["arms"]
        chain = ["blind", "window_1", "window_2", "window_4", "window_8", "full_history"]
        rewards = [Fraction(arms[arm]["reward"]) for arm in chain]
        self.assertEqual(rewards, sorted(rewards))
        self.assertEqual(arms["full_history"]["reward"], "139/160")
        self.assertEqual(arms["window_8"]["reward"], arms["full_history"]["reward"])
        self.assertEqual(summary["decoy"]["reward"], "1/2")
        self.assertTrue(all(value["false_certificates"] == 0 for value in arms.values()))


if __name__ == "__main__":
    unittest.main()
