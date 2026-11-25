"""ASM-01: exact decision value of an observer's hidden arithmetic regime.

Python standard library, finite exhaustive population, no model training.
Run with --write to produce receipts or --check to replay committed receipts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from collatz_core import analyze, odd_step

HERE = Path(__file__).resolve().parent
REGIMES = (5, 49)
RESET_SEEDS = (7, 15)
INITIAL_SEEDS = tuple(range(1, 50, 2))
HORIZON = 16


def eventual_basin(n: int, d: int) -> int:
    core = analyze(d)
    while n > d:
        n = odd_step(n, d)[0]
    return core.basin[n]


def population() -> list[dict]:
    rows = []
    for regime, d in enumerate(REGIMES):
        for seed in INITIAL_SEEDS:
            n, history = seed, []
            for phase in range(HORIZON):
                history.append((n % 4 - 1) // 2)
                rows.append(dict(regime=regime, d=d, initial_seed=seed,
                                 phase=phase, state=n, history=tuple(history)))
                n = odd_step(n, d)[0]
    return rows


def key(arm: str, row: dict):
    history = row["history"]
    if arm == "blind":
        observation = ()
    elif arm == "count_only":
        observation = sum(history)
    elif arm == "full_history":
        observation = history
    elif arm.startswith("window_"):
        observation = history[-int(arm.split("_")[1]):]
    else:
        raise ValueError(arm)
    # Public phase is available to every arm; no regime or state enters a key.
    return row["phase"], observation


def fraction_string(numerator: int, denominator: int) -> str:
    value = Fraction(numerator, denominator)
    return f"{value.numerator}/{value.denominator}"


def evaluate() -> tuple[dict, list[dict]]:
    routes = [[int(eventual_basin(seed, d) == 1) for seed in RESET_SEEDS]
              for d in REGIMES]
    if routes != [[1, 0], [0, 1]]:
        raise AssertionError("the two regimes must reverse the successful reset")
    rows = population()
    arms = ("blind", "window_1", "window_2", "window_4", "window_8",
            "count_only", "full_history")
    metrics, detailed = {}, []
    full_actions = {}
    for arm in arms:
        cells = defaultdict(Counter)
        for row in rows:
            cells[key(arm, row)][row["regime"]] += 1
        successes = certificates = regret_numerator = 0
        by_phase = Counter()
        for row in rows:
            counts = cells[key(arm, row)]
            action = int(counts[1] > counts[0])  # ties choose reset seed 7
            success = routes[row["regime"]][action]
            certified = len(counts) == 1
            if certified and not success:
                raise AssertionError("false robust action certificate")
            successes += success
            certificates += certified
            by_phase[row["phase"]] += success
            identity = row["regime"], row["initial_seed"], row["phase"]
            if arm == "full_history":
                full_actions[identity] = action
            detailed.append(dict(arm=arm, regime=row["regime"],
                initial_seed=row["initial_seed"], phase=row["phase"],
                observation="".join(map(str, row["history"])),
                support_regime_0=counts[0], support_regime_1=counts[1],
                selected_reset=RESET_SEEDS[action], success=success,
                robust_certificate=int(certified)))
        expected = sum(max(counts.values()) for counts in cells.values())
        if successes != expected:
            raise AssertionError("exact cellwise Bayes value disagrees with replay")
        metrics[arm] = dict(successes=successes, decisions=len(rows),
            reward=fraction_string(successes, len(rows)),
            oracle_regret=fraction_string(len(rows)-successes, len(rows)),
            robust_certificates=certificates, false_certificates=0,
            observation_cells=len(cells), successes_by_phase=[by_phase[t] for t in range(HORIZON)])
    # An independent decoy has the same marginal observations. Exhaustively pair
    # every self episode with every decoy episode at the same public phase.
    episodes = [(r, n) for r in range(2) for n in INITIAL_SEEDS]
    decoy_successes = decoy_count = 0
    for true_regime, _ in episodes:
        for decoy_regime, decoy_seed in episodes:
            for phase in range(HORIZON):
                action = full_actions[decoy_regime, decoy_seed, phase]
                decoy_successes += routes[true_regime][action]
                decoy_count += 1
    assert 2 * decoy_successes == decoy_count
    summary = dict(schema="asm01.v1", date="2026-11-25",
        kind="exhaustive_constructed_population", regimes=list(REGIMES),
        reset_seeds=list(RESET_SEEDS), target="eventual odd cycle minimum equals 1",
        route_rewards=routes, initial_seeds=list(INITIAL_SEEDS), horizon=HORIZON,
        episodes=len(episodes), decisions=len(rows), phase_public=True,
        oracle=dict(successes=len(rows), decisions=len(rows), reward="1/1"),
        arms=metrics, decoy=dict(paired_decisions=decoy_count, successes=decoy_successes,
            reward=fraction_string(decoy_successes, decoy_count)),
        core_sha256=hashlib.sha256((HERE / "collatz_core.py").read_bytes()).hexdigest())
    return summary, detailed


def receipt_bytes() -> tuple[bytes, bytes]:
    summary, rows = evaluate()
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(), output.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, payload in zip(("summary.json", "decisions.csv"), receipt_bytes()):
        path = HERE / name
        if args.write:
            path.write_bytes(payload)
        elif path.read_bytes() != payload:
            raise SystemExit(f"receipt mismatch: {name}")
    print("ASM-01: 800 exact decisions per arm and 40,000 decoy decisions verified")


if __name__ == "__main__":
    main()
