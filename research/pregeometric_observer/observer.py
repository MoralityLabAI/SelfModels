#!/usr/bin/env python3
"""Exact finite-observer study of conditional Boolean dependence.

Only Python's standard library is required. Inputs, resets, query ticks, the
hypothesis class and its uniform prior are declared parts of the experiment.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path


WORLDS = tuple(range(16))
QUERIES = tuple(range(4))


def response(world: int, query: int) -> int:
    """q = 2*z + x; bit q of world is f(x,z)."""
    return (world >> query) & 1


def depends_on_x(world: int) -> bool:
    return any(response(world, 2 * z) != response(world, 2 * z + 1)
               for z in range(2))


def certified(worlds: tuple[int, ...]) -> bool:
    return len({depends_on_x(w) for w in worlds}) == 1


def partition(worlds: tuple[int, ...], query: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(w for w in worlds if response(w, query) == bit)
                 for bit in (0, 1))


def available_queries(worlds: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(q for q in QUERIES if all(partition(worlds, q)))


def predict(worlds: tuple[int, ...]) -> bool:
    """Uniform-prior majority decision, with True as the explicit tie rule."""
    return 2 * sum(depends_on_x(w) for w in worlds) >= len(worlds)


@lru_cache(maxsize=None)
def best_policy(worlds: tuple[int, ...], budget: int) -> tuple[Fraction, Fraction, int | None]:
    """Maximize accuracy, then certificate coverage; break query ties by index.

    Queries have no utility penalty within the declared budget. The returned
    fractions are conditional on the current uniformly weighted version space.
    """
    if certified(worlds):
        return Fraction(1), Fraction(1), None
    if budget == 0:
        correct = sum(depends_on_x(w) == predict(worlds) for w in worlds)
        return Fraction(correct, len(worlds)), Fraction(0), None
    candidates = []
    for q in available_queries(worlds):
        accuracy = coverage = Fraction(0)
        for branch in partition(worlds, q):
            a, c, _ = best_policy(branch, budget - 1)
            weight = Fraction(len(branch), len(worlds))
            accuracy += weight * a
            coverage += weight * c
        candidates.append((accuracy, coverage, q))
    return max(candidates, key=lambda c: (c[0], c[1], -c[2]))


@lru_cache(maxsize=None)
def minimum_expected_queries(worlds: tuple[int, ...]) -> Fraction:
    if certified(worlds):
        return Fraction(0)
    return min(Fraction(1) + sum(
        (Fraction(len(branch), len(worlds)) * minimum_expected_queries(branch)
         for branch in partition(worlds, q)), Fraction(0))
        for q in available_queries(worlds))


@lru_cache(maxsize=None)
def minimum_worst_case_queries(worlds: tuple[int, ...]) -> int:
    if certified(worlds):
        return 0
    return min(1 + max(minimum_worst_case_queries(branch)
                       for branch in partition(worlds, q))
               for q in available_queries(worlds))


def run_budget(world: int, budget: int) -> dict:
    worlds = WORLDS
    record = []
    for left in range(budget, 0, -1):
        _, _, query = best_policy(worlds, left)
        if query is None:
            break
        bit = response(world, query)
        record.append([query, bit])
        worlds = partition(worlds, query)[bit]
    return {
        "world": world,
        "record": record,
        "remaining_worlds": list(worlds),
        "prediction": predict(worlds),
        "certificate": certified(worlds),
        "truth": depends_on_x(world),
    }


# Eight states, including both terminal states. The transition table is fixed
# program data; the state register itself fits in three bits.
FSM_QUERIES = (0, 1, 1, 2, 3, 3, None, None)
FSM_NEXT = ((1, 2), (3, 6), (6, 3), (4, 5), (7, 6), (6, 7), None, None)
FSM_LABELS = {6: True, 7: False}


def run_fsm(world: int) -> dict:
    state = 0
    record = []
    states = [state]
    while state not in FSM_LABELS:
        query = FSM_QUERIES[state]
        bit = response(world, query)
        record.append([query, bit])
        state = FSM_NEXT[state][bit]
        states.append(state)
    return {"world": world, "prediction": FSM_LABELS[state],
            "record": record, "states": states}


def all_version_spaces() -> list[tuple[int, ...]]:
    return [tuple(w for w in WORLDS
                  if all(bit == -1 or response(w, q) == bit
                         for q, bit in enumerate(partial)))
            for partial in product((-1, 0, 1), repeat=4)]


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def receipt() -> dict:
    budget_rows = []
    for budget in range(5):
        rows = [run_budget(w, budget) for w in WORLDS]
        accuracy, coverage, query = best_policy(WORLDS, budget)
        positive = [r for r in rows if r["truth"]]
        negative = [r for r in rows if not r["truth"]]
        balanced = (Fraction(sum(r["prediction"] for r in positive), len(positive))
                    + Fraction(sum(not r["prediction"] for r in negative), len(negative))) / 2
        budget_rows.append({
            "budget": budget,
            "optimal_uniform_accuracy": fraction_text(accuracy),
            "balanced_accuracy_of_selected_policy": fraction_text(balanced),
            "certificate_coverage": fraction_text(coverage),
            "first_query": query,
            "correct_worlds": sum(r["prediction"] == r["truth"] for r in rows),
            "certified_worlds": sum(r["certificate"] for r in rows),
            "mean_queries": fraction_text(Fraction(sum(len(r["record"]) for r in rows), 16)),
        })
    spaces = all_version_spaces()
    fsm_rows = [run_fsm(w) for w in WORLDS]
    directory = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
               for name in ("observer.py", "verify_observer.py")}
    return {
        "schema": "pgc-observer-budget-v1",
        "world_encoding": "bit (2*z+x) of world is f(x,z)",
        "prior": "uniform over all 16 Boolean functions of two inputs",
        "world_count": 16,
        "worlds_with_x_dependence": sum(depends_on_x(w) for w in WORLDS),
        "distinct_version_spaces": len(set(spaces)),
        "certified_version_spaces": sum(certified(s) for s in spaces),
        "budget_results": budget_rows,
        "exact_certificate_optimum": {
            "expected_queries_uniform": fraction_text(minimum_expected_queries(WORLDS)),
            "worst_case_queries": minimum_worst_case_queries(WORLDS),
        },
        "finite_state_observer": {
            "states_including_terminals": 8,
            "sufficient_internal_state_bits": 3,
            "state_bits_scope": "fixed program and query-response interface accounted separately",
            "minimum_state_bits_claimed": False,
            "correct_worlds": sum(r["prediction"] == depends_on_x(r["world"]) for r in fsm_rows),
            "expected_queries_uniform": fraction_text(Fraction(sum(len(r["record"]) for r in fsm_rows), 16)),
            "worst_case_queries": max(len(r["record"]) for r in fsm_rows),
            "traces": fsm_rows,
        },
        "indistinguishability_witness": {
            "queried_indices": [0, 1, 2],
            "worlds": [0, 8],
            "shared_response": [0, 0, 0],
            "x_dependence": [False, True],
        },
        "scope": {
            "supplied": ["binary alphabet", "input addresses x and z", "resettable queries",
                         "discrete query ticks", "finite hypothesis class", "uniform prior"],
            "recovered": ["finite-record version spaces", "conditional x-dependence certificates",
                          "query-budget optima within the declared class"],
            "physical_geometry": "undetermined",
            "lorentzian_signature": "undetermined",
            "four_dimensions": "undetermined",
            "einstein_variables": "undetermined",
            "newton_coupling": "undetermined",
        },
        "source_sha256": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    text = json.dumps(receipt(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.check.read_text() != text:
            raise SystemExit("receipt differs from exact recomputation")
        print(f"PASS: exact receipt matches {args.check}")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
