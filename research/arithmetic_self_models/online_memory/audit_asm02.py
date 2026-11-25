"""Read-only ASM-02 review using complete transition/action programs and original routes.

Run with --port pointing to research/arithmetic_self_models/online_memory.
The only optional write is the separate --out review receipt.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
from itertools import product
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def explicit_controller_optimum(tapes):
    """Enumerate every four-transition/two-action instruction at every phase."""
    horizon = len(tapes[0].bits)
    require(horizon <= 3, "review exhaustive horizon")
    best, ties = (-1, -1), 0
    for program in product(range(64), repeat=horizon):
        states = [0] * len(tapes)
        successes = certain = 0
        for phase, instruction in enumerate(program):
            states = [(instruction >> (2 * state + tape.bits[phase])) & 1
                      for state, tape in zip(states, tapes)]
            cells = [Counter(), Counter()]
            for code, tape in zip(states, tapes):
                cells[code][tape.regime] += 1
            for code, cell in enumerate(cells):
                action = (instruction >> (4 + code)) & 1
                successes += cell[action]
                if cell[1 - action] == 0:
                    certain += sum(cell.values())
        objective = successes, certain
        if objective > best:
            best, ties = objective, 1
        elif objective == best:
            ties += 1
    return best, ties


def run(port):
    port = port.resolve()
    sys.path.insert(0, str(port.parent))
    sys.path.insert(0, str(port))
    asm02 = importlib.import_module("asm02")
    original = importlib.import_module("asm01")
    source_rows = original.population()
    episodes = {}
    for row in source_rows:
        episodes.setdefault((row["regime"], row["initial_seed"]), []).append(row)
    require(len(source_rows) == 800 and len(episodes) == 50, "original population coverage")
    for values in episodes.values():
        require([r["phase"] for r in values] == list(range(16)), "original phase coverage")
    tapes = [(regime, seed, tuple(r["history"][-1] for r in rows))
             for (regime, seed), rows in episodes.items()]
    require([(t.regime, t.bits) for t in asm02.population()] == [(r, b) for r, _, b in tapes],
            "unchanged original tape")
    routes = [[int(original.eventual_basin(seed, d) == 1) for seed in original.RESET_SEEDS]
              for d in original.REGIMES]
    require(routes == [[1, 0], [0, 1]], "original reset semantics")
    summary = json.loads((port / "summary.json").read_text())
    controllers = json.loads((port / "controllers.json").read_text())
    witness_rows = []
    for name, controller in sorted(controllers.items()):
        states = [controller["initial_state"]] * len(tapes)
        rewards, coverage, verified_cells, certificates = [], [], 0, []
        for phase in range(16):
            states = [controller["transitions"][phase][code][tape[2][phase]]
                      for code, tape in zip(states, tapes)]
            cells = defaultdict(list)
            for i, code in enumerate(states):
                cells[code].append(i)
            successes = certain = 0
            for code, ids in cells.items():
                action = controller["outputs"][phase][code]
                n0 = sum(tapes[i][0] == 0 for i in ids)
                n1 = len(ids) - n0
                require(action == int(n1 > n0), "majority action and zero tie")
                support = {tapes[i][0] for i in ids}
                robust = all(routes[regime][action] == 1 for regime in support)
                for i in ids:
                    regime, seed, _ = tapes[i]
                    successes += routes[regime][action]
                    certain += robust
                    if robust:
                        certificates.append({"phase": phase, "regime": regime,
                                             "initial_seed": seed, "state": code,
                                             "action": action})
                verified_cells += 1
            rewards.append(successes)
            coverage.append(certain)
        metric = summary["metrics"][name]
        require(metric["successes"] == sum(rewards), "witness reward")
        require(metric["robust_certificates"] == sum(coverage), "witness robust coverage")
        require(metric["successes_by_phase"] == rewards, "phase reward")
        require(metric["certificates_by_phase"] == coverage, "phase coverage")
        payload = (json.dumps(controller, indent=2, sort_keys=True) + "\n").encode()
        require(hashlib.sha256(payload).hexdigest() == metric["controller_sha256"],
                "witness controller hash")
        witness_rows.append({"name": name, "successes": sum(rewards),
                             "robust_certificates": sum(coverage),
                             "decisions": len(tapes) * 16, "cells": verified_cells,
                             "certificate_rows": certificates})

    words = list(product((0, 1), repeat=2))
    cases = [tuple(asm02.Tape(regime, bits) for regime, bits in zip(labels, words))
             for labels in product((0, 1), repeat=4)]
    cases.append((asm02.Tape(0, (0, 0)), asm02.Tape(0, (0, 0)),
                  asm02.Tape(1, (0, 0)), asm02.Tape(1, (0, 1)),
                  asm02.Tape(0, (1, 0)), asm02.Tape(1, (1, 1))))
    cases.append(tuple(asm02.Tape(regime, bits) for bits in words for regime in (0, 1)))
    small_rows = []
    for index, case in enumerate(cases):
        optimum, ties = explicit_controller_optimum(case)
        for ordering in (case, tuple(reversed(case))):
            controller, _ = asm02.synthesize_two_state(ordering)
            measured = asm02.evaluate(controller, ordering)
            require(optimum == (measured["successes"], measured["robust_certificates"]),
                    "small independent optimum and episode-order invariance")
        small_rows.append({"case": index, "episodes": len(case), "horizon": 2,
                           "programs": 64 ** 2, "successes": optimum[0],
                           "robust_certificates": optimum[1], "optimal_programs": ties})
    majority = tuple(asm02.Tape(int(sum(bits) >= 2), bits)
                     for bits in product((0, 1), repeat=3))
    optimum, ties = explicit_controller_optimum(majority)
    controller, _ = asm02.synthesize_two_state(majority)
    measured = asm02.evaluate(controller, majority)
    require(optimum == (measured["successes"], measured["robust_certificates"]),
            "three-phase independent optimum")
    small_rows.append({"case": "three_bit_majority", "episodes": len(majority),
                       "horizon": 3, "programs": 64 ** 3, "successes": optimum[0],
                       "robust_certificates": optimum[1], "optimal_programs": ties})
    files = {name: port / name for name in ("asm02.py", "test_asm02.py", "PROTOCOL.json",
                                           "summary.json", "controllers.json")}
    files.update({"original/" + name: port.parent / name
                  for name in ("asm01.py", "collatz_core.py")})
    return {
        "schema": "ASM-02/independent-review/v1", "status": "PASS",
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in sorted(files.items())},
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_episodes": len(tapes), "original_decisions": len(source_rows),
        "witnesses": witness_rows, "small_exhaustive_cases": small_rows,
        "programs_enumerated": sum(row["programs"] for row in small_rows),
        "episode_order_invariance_cases": len(cases),
        "dp_reasoning": [
            "Each current episode partition determines every future transition option and cell objective.",
            "Outputs do not change observations or later memory updates, so cellwise reward maximization is valid.",
            "A mixed-cell reward tie gives zero robust coverage under either action; pure cells have a unique correct action.",
            "Rank reward*801+coverage preserves the lexicographic objective over at most 800 decisions.",
            "Keeping episode zero in state zero removes only per-phase label symmetry; the packed path reconstructs causal tables."
        ],
        "limitations": [
            "The full 830416-partition dynamic program was reviewed mathematically and through small exhaustive cases; this checker does not rerun that entire search.",
            "The 678/800 global reward and maximum 10/800 coverage apply to this fixed 50-episode population and deterministic two-state controllers.",
            "Coverage 10/800 is maximal among reward-optimal controllers. Other two-state controllers can achieve greater coverage at lower reward.",
            "Public phase and compiled static tables are uncharged; no reward feedback or external action-history memory is available.",
            "The fixture was inspected before protocol publication; there is no held-out or prospective empirical claim."
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.port)
    if args.out:
        args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "programs_enumerated": result["programs_enumerated"],
                      "witnesses": [{k: v for k, v in row.items() if k != "certificate_rows"}
                                    for row in result["witnesses"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
