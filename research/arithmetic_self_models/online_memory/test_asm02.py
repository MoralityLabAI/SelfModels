import copy
import unittest

from asm02 import (
    Tape,
    act,
    brute_force_two_state,
    evaluate,
    full_history_bound,
    make_count,
    make_window,
    observe,
    population,
    synthesize_two_state,
    validate_controller,
)


class OnlineMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tapes = population()

    def test_same_population_and_constructive_controls(self):
        expected = {0: (400, 0), 1: (566, 142), 2: (626, 205),
                    4: (689, 281), 8: (695, 385)}
        for bits, pair in expected.items():
            metric = evaluate(make_window(self.tapes, bits), self.tapes)
            self.assertEqual((metric["successes"], metric["robust_certificates"]), pair)
        count = evaluate(make_count(self.tapes), self.tapes)
        self.assertEqual((count["successes"], count["robust_certificates"]), (643, 363))
        ceiling = full_history_bound(self.tapes)
        self.assertEqual((ceiling["successes"], ceiling["robust_certificates"]), (695, 519))

    def test_independent_labeled_enumeration_matches_dynamic_program(self):
        tapes = tuple(Tape(t.regime, t.bits[:3]) for t in self.tapes)
        controller, search = synthesize_two_state(tapes)
        metric = evaluate(controller, tapes)
        self.assertEqual((metric["successes"], metric["robust_certificates"]),
                         brute_force_two_state(tapes))
        self.assertTrue(search["certified_optimal"])
        self.assertEqual(search["reachable_partition_counts"], [1, 2, 8, 44])

    def test_partition_budget_fails_without_optimality_result(self):
        tapes = tuple(Tape(t.regime, t.bits[:3]) for t in self.tapes)
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            synthesize_two_state(tapes, max_partitions=3)

    def test_runtime_sees_only_current_bit_prior_code_and_public_phase(self):
        controller = make_window(self.tapes, 2)
        for phase in range(16):
            for code in range(4):
                for bit in (0, 1):
                    new_code = observe(controller, code, bit, phase)
                    self.assertEqual(new_code, ((code << 1) | bit) % 4)
                    self.assertIn(act(controller, new_code, phase), (0, 1))
        with self.assertRaises(ValueError):
            observe(controller, 4, 0, 0)
        with self.assertRaises(ValueError):
            observe(controller, 0, True, 0)

    def test_controller_rejects_out_of_range_code(self):
        controller = copy.deepcopy(make_window(self.tapes, 1))
        controller["transitions"][0][0][0] = 2
        with self.assertRaises(ValueError):
            validate_controller(controller)

    def test_identical_observation_tapes_share_the_same_internal_path(self):
        controller = make_window(self.tapes, 2)
        one = Tape(0, (1, 0) * 8)
        other = Tape(1, one.bits)
        metric = evaluate(controller, (one, other))
        self.assertEqual(metric["successes"], 16)
        self.assertEqual(metric["robust_certificates"], 0)


if __name__ == "__main__":
    unittest.main()
