"""Exact finite oracle for the split Collatz family (Python standard library).

The finite-core statements use d > 0 and d == 1 (mod 4). Quotient labels
add gcd(d, 6) == 1. These routines deliberately enumerate small complete
domains; use census.cpp for the 160,833,217 reference parameter.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import gcd


def validate_parameter(d: int, *, quotient: bool = False) -> None:
    if type(d) is not int or d <= 0 or d % 4 != 1:
        raise ValueError("d must be a positive integer congruent to 1 modulo 4")
    if quotient and gcd(d, 6) != 1:
        raise ValueError("the <2,3> quotient requires gcd(d,6)=1")


def odd_step(n: int, d: int) -> tuple[int, int, int]:
    """Return (next odd integer, exact halving count, triple indicator)."""
    validate_parameter(d)
    if type(n) is not int or n <= 0 or n % 2 != 1:
        raise ValueError("n must be a positive odd integer")
    triple = int(n % 4 == 1)
    y = (3 if triple else 1) * n + d
    a = (y & -y).bit_length() - 1
    return y >> a, a, triple


def full_step(n: int, d: int) -> int:
    validate_parameter(d)
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    return n // 2 if n % 2 == 0 else (3 * n + d if n % 4 == 1 else n + d)


@dataclass(frozen=True)
class Core:
    d: int
    states: tuple[int, ...]
    successors: dict[int, int]
    basin: dict[int, int]
    cycles: tuple[dict, ...]


def analyze(d: int, *, max_parameter: int = 2_000_001) -> Core:
    """Enumerate every odd state <= d and label its eventual cycle minimum."""
    validate_parameter(d)
    if d > max_parameter:
        raise ValueError("parameter exceeds the explicit Python census budget")
    states = tuple(range(1, d + 1, 2))
    edges = {n: odd_step(n, d) for n in states}
    successors = {n: edge[0] for n, edge in edges.items()}
    basin: dict[int, int] = {}
    cycles: list[dict] = []
    for start in states:
        if start in basin:
            continue
        path: list[int] = []
        positions: dict[int, int] = {}
        n = start
        while n not in basin and n not in positions:
            if n not in successors:
                raise AssertionError("finite-core closure failed")
            positions[n] = len(path)
            path.append(n)
            n = successors[n]
        if n in positions:
            nodes = path[positions[n]:]
            minimum = min(nodes)
            offset = nodes.index(minimum)
            nodes = nodes[offset:] + nodes[:offset]
            A = sum(edges[x][1] for x in nodes)
            u = sum(edges[x][2] for x in nodes)
            cycles.append({"minimum": minimum, "nodes": tuple(nodes),
                           "odd_steps": len(nodes), "halvings": A,
                           "triple_branches": u, "period": len(nodes) + A,
                           "gcd_with_d": gcd(minimum, d)})
        else:
            minimum = basin[n]
        for x in path:
            basin[x] = minimum
    sizes = Counter(basin.values())
    for cycle in cycles:
        cycle["basin_size"] = sizes[cycle["minimum"]]
    return Core(d, states, successors, basin,
                tuple(sorted(cycles, key=lambda c: c["minimum"])))


enumerate_core = analyze


def modular_labels(d: int, *, max_parameter: int = 2_000_001) -> dict[int, int]:
    """Label all residues by their least <2,3>-orbit representative, incl. 0."""
    validate_parameter(d, quotient=True)
    if d > max_parameter:
        raise ValueError("parameter exceeds the explicit quotient budget")
    labels: dict[int, int] = {}
    for start in range(d):
        if start in labels:
            continue
        labels[start] = start
        stack = [start]
        while stack:
            x = stack.pop()
            for multiplier in (2, 3):
                y = multiplier * x % d
                if y not in labels:
                    labels[y] = start
                    stack.append(y)
                elif labels[y] != start:
                    raise AssertionError("permutation orbits intersected")
    return labels


quotient_labels = modular_labels


def readout_ceiling(core: Core, observation: dict[int, object]) -> tuple[int, int]:
    """Exact Bayes count under the uniform distribution on odd core states."""
    counts: dict[object, Counter] = defaultdict(Counter)
    for n in core.states:
        counts[observation[n]][core.basin[n]] += 1
    return sum(max(c.values()) for c in counts.values()), len(core.states)


def branch_history(n: int, d: int, depth: int) -> tuple[tuple[int, int], ...]:
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    result = []
    for _ in range(depth):
        n, a, u = odd_step(n, d)
        result.append((u, a))
    return tuple(result)
