"""ASM-02: causal finite-memory controllers on the pinned ASM-01 population.

The exact two-state search uses at most one million current partitions. All
objectives are finite-population counts on already inspected development cases.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
SOURCE_COMMIT = "e2019ea2077b9dacffa3dbc68999bf7477ad79bf"
SOURCE_HASHES = {
    "asm01.py": "aac5acbbc6a6c06147e89acbc6d627a4c0dd85c9af717d19ddbdbb9268c7ef6e",
    "collatz_core.py": "24ff52171aedf890c33439896ab5e71de65d9b05d557ba61d162f3623aa5dbca",
}


@dataclass(frozen=True)
class Tape:
    """Evaluator-owned episode; the controller receives only the next bit."""

    regime: int
    bits: tuple[int, ...]


def _load_source():
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned ASM-01 source changed: {name}")
    previous = sys.modules.get("collatz_core")
    try:
        spec = importlib.util.spec_from_file_location("collatz_core", SOURCE / "collatz_core.py")
        core = importlib.util.module_from_spec(spec)
        sys.modules["collatz_core"] = core
        spec.loader.exec_module(core)
        spec = importlib.util.spec_from_file_location("_asm02_pinned_fixture", SOURCE / "asm01.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        return fixture
    finally:
        if previous is None:
            sys.modules.pop("collatz_core", None)
        else:
            sys.modules["collatz_core"] = previous


def population() -> tuple[Tape, ...]:
    fixture = _load_source()
    routes = [[int(fixture.eventual_basin(seed, d) == 1) for seed in fixture.RESET_SEEDS]
              for d in fixture.REGIMES]
    if routes != [[1, 0], [0, 1]]:
        raise AssertionError("ASM-01 reset rewards changed")
    rows = fixture.population()
    tapes = tuple(Tape(rows[i]["regime"], tuple(r["history"][-1] for r in rows[i:i + 16]))
                  for i in range(0, len(rows), 16))
    if len(tapes) != 50 or any(len(t.bits) != 16 for t in tapes):
        raise AssertionError("ASM-01 population changed")
    return tapes


def validate_controller(controller: dict) -> None:
    if controller.get("schema") != "ASM-02/controller/v1":
        raise ValueError("controller schema")
    k, horizon = controller["capacity"], controller["horizon"]
    if type(k) is not int or k < 1 or k > 256 or type(horizon) is not int or not 1 <= horizon <= 16:
        raise ValueError("controller size")
    if type(controller["initial_state"]) is not int or not 0 <= controller["initial_state"] < k:
        raise ValueError("initial state")
    layers, outputs = controller["transitions"], controller["outputs"]
    if len(layers) != horizon or len(outputs) != horizon:
        raise ValueError("phase count")
    for layer, output in zip(layers, outputs):
        if len(layer) != k or len(output) != k:
            raise ValueError("state count")
        for row in layer:
            if len(row) != 2 or any(type(v) is not int or not 0 <= v < k for v in row):
                raise ValueError("transition")
        if any(type(v) is not int or v not in (0, 1) for v in output):
            raise ValueError("action")


def observe(controller: dict, state: int, bit: int, phase: int) -> int:
    """Complete runtime update interface: no episode or history argument."""
    if type(bit) is not int or bit not in (0, 1):
        raise ValueError("diagnostic bit")
    if type(state) is not int or not 0 <= state < controller["capacity"]:
        raise ValueError("retained state")
    if type(phase) is not int or not 0 <= phase < controller["horizon"]:
        raise ValueError("public phase")
    return controller["transitions"][phase][state][bit]


def act(controller: dict, state: int, phase: int) -> int:
    return controller["outputs"][phase][state]


def _state_tapes(transitions: list, tapes: tuple[Tape, ...]) -> list[list[int]]:
    states = [0] * len(tapes)
    result = []
    for phase, layer in enumerate(transitions):
        states = [layer[state][tape.bits[phase]] for state, tape in zip(states, tapes)]
        result.append(states)
    return result


def fit_outputs(transitions: list, tapes: tuple[Tape, ...], name: str) -> dict:
    """Offline compilation under the declared prior; execution uses fixed tables."""
    k = len(transitions[0])
    outputs = []
    for states in _state_tapes(transitions, tapes):
        counts = [[0, 0] for _ in range(k)]
        for state, tape in zip(states, tapes):
            counts[state][tape.regime] += 1
        outputs.append([int(n1 > n0) for n0, n1 in counts])
    return {"schema": "ASM-02/controller/v1", "name": name, "capacity": k,
            "horizon": len(transitions), "initial_state": 0,
            "transitions": transitions, "outputs": outputs}


def evaluate(controller: dict, tapes: tuple[Tape, ...]) -> dict:
    validate_controller(controller)
    horizon, k = controller["horizon"], controller["capacity"]
    if any(len(t.bits) != horizon or t.regime not in (0, 1) for t in tapes):
        raise ValueError("population shape")
    histories = []
    for tape in tapes:
        state, history = controller["initial_state"], []
        for phase, bit in enumerate(tape.bits):
            state = observe(controller, state, bit, phase)
            history.append((state, act(controller, state, phase)))
        histories.append(history)
    successes = certificates = 0
    by_phase, certificate_phase, reachable = [], [], []
    for phase in range(horizon):
        cells = [[0, 0] for _ in range(k)]
        for tape, history in zip(tapes, histories):
            cells[history[phase][0]][tape.regime] += 1
        good = certain = 0
        for tape, history in zip(tapes, histories):
            state, action = history[phase]
            good += int(action == tape.regime)
            certified = cells[state][1 - action] == 0
            if certified and action != tape.regime:
                raise AssertionError("false robust certificate")
            certain += int(certified)
        successes += good
        certificates += certain
        by_phase.append(good)
        certificate_phase.append(certain)
        reachable.append(sum(bool(sum(c)) for c in cells))
    decisions = len(tapes) * horizon
    return {"successes": successes, "decisions": decisions,
            "reward": str(Fraction(successes, decisions)),
            "robust_certificates": certificates, "false_certificates": 0,
            "successes_by_phase": by_phase, "certificates_by_phase": certificate_phase,
            "capacity": k, "retained_bits_ceiling": (k - 1).bit_length(),
            "reachable_states_by_phase": reachable}


def make_window(tapes: tuple[Tape, ...], bits: int) -> dict:
    if type(bits) is not int or not 0 <= bits <= 8:
        raise ValueError("window bound")
    k = 1 << bits
    layer = [[((state << 1) | bit) % k for bit in (0, 1)] for state in range(k)]
    return fit_outputs([layer for _ in tapes[0].bits], tapes, f"window_{bits}")


def make_count(tapes: tuple[Tape, ...]) -> dict:
    k = len(tapes[0].bits) + 1
    layer = [[state, min(state + 1, k - 1)] for state in range(k)]
    return fit_outputs([layer for _ in tapes[0].bits], tapes, "count_ones")


def full_history_bound(tapes: tuple[Tape, ...]) -> dict:
    reward = certificates = 0
    by_phase = []
    for phase in range(len(tapes[0].bits)):
        counts = {}
        for tape in tapes:
            counts.setdefault(tape.bits[:phase + 1], [0, 0])[tape.regime] += 1
        successes = sum(max(v) for v in counts.values())
        reward += successes
        by_phase.append(successes)
        certificates += sum(sum(v) for v in counts.values() if min(v) == 0)
    return {"successes": reward, "robust_certificates": certificates,
            "successes_by_phase": by_phase, "scope": "unbounded observed-history Bayes ceiling"}


def synthesize_two_state(tapes: tuple[Tape, ...], max_partitions: int = 1_000_000) -> tuple[dict, dict]:
    """Globally optimize cumulative reward, then certificate count, for K=2.

    A partition is the bitset of episodes carrying state 1, with episode 0
    assigned state 0 to remove label symmetry. Every update assigns each
    nonempty (old state, new bit) cell to a new state. For identical current
    partitions only the best cumulative objective can affect a future optimum.
    """
    n, horizon = len(tapes), len(tapes[0].bits)
    if n > 50 or horizon > 16 or not n or not horizon:
        raise ValueError("exact synthesis budget")
    if any(len(t.bits) != horizon or t.regime not in (0, 1)
           or any(type(b) is not int or b not in (0, 1) for b in t.bits) for t in tapes):
        raise ValueError("finite population")
    full = (1 << n) - 1
    r0 = sum(1 << i for i, t in enumerate(tapes) if t.regime == 0)
    r1 = full ^ r0
    n0, n1 = r0.bit_count(), r1.bit_count()
    objective_base = n * horizon + 1
    path_bits = 4 * horizon
    path_mask = (1 << path_bits) - 1
    frontier = {0: 0}
    counts = [1]
    for phase in range(horizon):
        ones = sum(1 << i for i, t in enumerate(tapes) if t.bits[phase])
        zeros = full ^ ones
        following = {}
        for state, packed in frontier.items():
            complement = full ^ state
            atoms = (complement & zeros, complement & ones, state & zeros, state & ones)
            choices = [(0, 0)]
            for index, atom in enumerate(atoms):
                if atom and not atom & 1:
                    choices += [(subset | atom, rule | (1 << index)) for subset, rule in choices]
            prefix_rank, prefix_path = packed >> path_bits, packed & path_mask
            for subset, rule in choices:
                a, b = (subset & r0).bit_count(), (subset & r1).bit_count()
                c, d = n0 - a, n1 - b
                reward = max(a, b) + max(c, d)
                certificates = (a + b if min(a, b) == 0 else 0) + (c + d if min(c, d) == 0 else 0)
                rank = prefix_rank + reward * objective_base + certificates
                value = (rank << path_bits) | prefix_path | (rule << (4 * phase))
                if value > following.get(subset, -1):
                    following[subset] = value
            if len(following) > max_partitions:
                raise RuntimeError("exact synthesis frontier budget exceeded; no optimality result emitted")
        frontier = following
        counts.append(len(frontier))
    packed = max(frontier.values())
    rank, path = packed >> path_bits, packed & path_mask
    transitions = []
    for phase in range(horizon):
        rule = (path >> (phase * 4)) & 15
        transitions.append([[(rule >> (2 * state + bit)) & 1 for bit in (0, 1)] for state in (0, 1)])
    controller = fit_outputs(transitions, tapes, "exact_two_state")
    metric = evaluate(controller, tapes)
    if (metric["successes"], metric["robust_certificates"]) != divmod(rank, objective_base):
        raise AssertionError("independent witness replay differs from dynamic program")
    search = {"certified_optimal": True, "objective": "reward then certificates",
              "reachable_partition_counts": counts, "maximum_partitions": max_partitions,
              "certificate_scope": "maximum coverage among reward-optimal two-state controllers"}
    return controller, search


def brute_force_two_state(tapes: tuple[Tape, ...]) -> tuple[int, int]:
    """Independent labeled-table check for up to three phases."""
    horizon = len(tapes[0].bits)
    if horizon > 3:
        raise ValueError("brute force budget")
    best = (0, 0)
    tables = list(product((0, 1), repeat=4))
    for rules in product(tables, repeat=horizon):
        states = [0] * len(tapes)
        reward = certificates = 0
        for phase, rule in enumerate(rules):
            cells = [[0, 0], [0, 0]]
            for i, tape in enumerate(tapes):
                states[i] = rule[2 * states[i] + tape.bits[phase]]
                cells[states[i]][tape.regime] += 1
            reward += sum(max(c) for c in cells)
            certificates += sum(sum(c) for c in cells if min(c) == 0)
        best = max(best, (reward, certificates))
    return best


def receipts() -> tuple[dict, dict]:
    tapes = population()
    exact, search = synthesize_two_state(tapes)
    controllers = {"exact_k1": make_window(tapes, 0), "exact_k2": exact}
    controllers.update({f"window_{b}": make_window(tapes, b) for b in (1, 2, 4, 8)})
    controllers["count_ones"] = make_count(tapes)
    metrics = {name: evaluate(controller, tapes) for name, controller in controllers.items()}
    for name, metric in metrics.items():
        metric["optimality"] = "global exact" if name in ("exact_k1", "exact_k2") else "constructive lower bound"
        metric["controller_sha256"] = hashlib.sha256(canonical_bytes(controllers[name])).hexdigest()
    bound = full_history_bound(tapes)
    if any(m["successes"] > bound["successes"] for m in metrics.values()):
        raise AssertionError("controller exceeds observed-history upper bound")
    summary = {"schema": "ASM-02/summary/v1", "source_commit": SOURCE_COMMIT,
               "source_sha256": SOURCE_HASHES, "episodes": 50, "phases": 16,
               "decisions": 800, "scope": "unchanged ASM-01 counterfactual tape",
               "prospective": False, "heldout": False, "training": False,
               "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL.json").read_bytes()).hexdigest(),
               "runtime_inputs": ["previous code", "current bit", "public phase"],
               "decoder_inputs": ["updated code", "public phase"],
               "codebook_cost": "fixed compiled tables are public and uncharged",
               "metrics": metrics, "full_history_upper_bound": bound, "two_state_search": search,
               "mic18_boundary_calibration": mic18_boundary_calibration()}
    witnesses = {name: controllers[name] for name in ("exact_k1", "exact_k2")}
    return summary, witnesses


def mic18_boundary_calibration() -> dict:
    """Separate delayed-query control; its reward is outside the ASM-01 task."""
    expected = "9fa386bd52c6925e41e7151e163b34df01f553eb6a214fe4737960dfd2109abc"
    if hashlib.sha256((HERE / "mic18_core.py").read_bytes()).hexdigest() != expected:
        raise ValueError("vendored MIC-18 source digest changed")
    from mic18_core import compile_exact, evaluate as evaluate_mic18, make_parity

    compiled = compile_exact(3, 2)
    query = evaluate_mic18(compiled["controller"])
    parity = evaluate_mic18(make_parity(3))
    return {"core_sha256": expected, "delayed_query_J3_K2": query["fraction"],
            "parity_J3_K2": parity["fraction"], "task_scope": "separate interface calibration"}


def canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    summary, witnesses = receipts()
    for name, value in (("summary.json", summary), ("controllers.json", witnesses)):
        payload = canonical_bytes(value)
        path = HERE / name
        if args.write:
            path.write_bytes(payload)
        elif path.read_bytes() != payload:
            raise SystemExit(f"ASM-02 receipt mismatch: {name}")
    result = summary["metrics"]["exact_k2"]
    print(f"ASM-02 verified: K=2 reward {result['successes']}/800, "
          f"robust certificates {result['robust_certificates']}/800")


if __name__ == "__main__":
    main()
