"""MIC-18: exact finite controllers with memory bounded at every update.

The only acquisition inputs are retained code, public phase and current bit.
The readout receives the final code and a query revealed after acquisition.
Transition/readout tables are static model storage, separate from dynamic state.
This source is part of EvalMath and follows the repository's Apache-2.0 license.
"""
from fractions import Fraction
from functools import lru_cache


SCHEMA = "MIC-18/controller/v1"
CONTROLLER_KEYS = {
    "schema", "name", "task", "width", "capacity", "initial_state",
    "transitions", "outputs",
}


class SearchLimitExceeded(RuntimeError):
    """The declared finite search stopped without an optimality certificate."""


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")


def task_check(task):
    if task not in ("query", "parity"):
        raise ValueError("task must be query or parity")


def observations(word, width):
    return [(word >> (width - 1 - index)) & 1 for index in range(width)]


def validate_controller(controller):
    if type(controller) is not dict or set(controller) != CONTROLLER_KEYS:
        raise ValueError("controller must contain exactly the declared fields")
    if controller["schema"] != SCHEMA or type(controller["name"]) is not str:
        raise ValueError("invalid controller schema or name")
    width, capacity = controller["width"], controller["capacity"]
    positive_integer(width, "width")
    positive_integer(capacity, "capacity")
    task_check(controller["task"])
    if type(controller["initial_state"]) is not int or controller["initial_state"] != 0:
        raise ValueError("initial state must be the fixed code zero")
    transitions = controller["transitions"]
    if type(transitions) is not list or len(transitions) != width:
        raise ValueError("one transition table is required per public phase")
    for table in transitions:
        if type(table) is not list or len(table) != capacity:
            raise ValueError("transition table must have exactly K rows")
        for row in table:
            if type(row) is not list or len(row) != 2:
                raise ValueError("each row must specify both binary observations")
            if any(type(x) is not int or not 0 <= x < capacity for x in row):
                raise ValueError("transition escaped the declared state alphabet")
    outputs = controller["outputs"]
    queries = width if controller["task"] == "query" else 1
    if type(outputs) is not list or len(outputs) != capacity:
        raise ValueError("decoder must have exactly K rows")
    for row in outputs:
        if type(row) is not list or len(row) != queries:
            raise ValueError("decoder sees only final code and declared final query")
        if any(type(x) is not int or x not in (0, 1) for x in row):
            raise ValueError("decoder output must be an integer bit")
    return width, capacity, queries


def evaluate(controller):
    """Replay every complete word and every readout query using integer arithmetic."""
    width, capacity, queries = validate_controller(controller)
    state_maps = [[controller["initial_state"]]]
    for table in controller["transitions"]:
        state_maps.append([table[code][bit] for code in state_maps[-1] for bit in (0, 1)])
    paths, score = [], 0
    for word in range(1 << width):
        bits = observations(word, width)
        states = [controller["initial_state"]]
        for phase, bit in enumerate(bits):
            states.append(controller["transitions"][phase][states[-1]][bit])
        predictions = list(controller["outputs"][states[-1]])
        targets = bits if controller["task"] == "query" else [sum(bits) % 2]
        score += sum(prediction == target for prediction, target in zip(predictions, targets))
        paths.append({"word": word, "observations": bits, "states": states,
                      "predictions": predictions, "targets": targets})
    total = (1 << width) * queries
    return {"score": score, "total": total, "fraction": str(Fraction(score, total)),
            "state_maps": state_maps,
            "reachable_state_counts": [len(set(mapping)) for mapping in state_maps],
            "paths": paths, "observations": width * (1 << width)}


def storage(controller):
    """Counts of dynamic codes and static scalar entries, excluding interpreter overhead."""
    width, capacity, queries = validate_controller(controller)
    return {"dynamic_capacity_codes": capacity,
            "dynamic_code_bits": (capacity - 1).bit_length(),
            "public_phases": width, "public_final_queries": queries,
            "static_transition_entries": 2 * width * capacity,
            "static_decoder_entries": capacity * queries,
            "static_entry_encoding": "transition entries are state IDs; decoder entries are bits",
            "excluded": "runtime representation, interpreter and source code"}


def restricted_growth(length, capacity):
    """Every canonical partition of length items into at most capacity cells."""
    if length == 0:
        yield ()
        return
    prefix = [0]

    def visit(maximum):
        if len(prefix) == length:
            yield tuple(prefix)
            return
        for code in range(min(capacity - 1, maximum + 1) + 1):
            prefix.append(code)
            yield from visit(max(maximum, code))
            prefix.pop()

    yield from visit(0)


@lru_cache(None)
def update_partitions(active, capacity):
    return tuple(restricted_growth(2 * active, capacity))


def canonicalize(mapping):
    labels = {}
    return tuple(labels.setdefault(value, len(labels)) for value in mapping)


def optimal_decoder(mapping, width, capacity, task):
    queries = width if task == "query" else 1
    counts = [[[0, 0] for _ in range(queries)] for _ in range(capacity)]
    for word, code in enumerate(mapping):
        bits = observations(word, width)
        targets = bits if task == "query" else [sum(bits) % 2]
        for query, target in enumerate(targets):
            counts[code][query][target] += 1
    outputs = [[int(ones > zeroes) for zeroes, ones in row] for row in counts]
    score = sum(max(pair) for row in counts for pair in row)
    return score, outputs


def compile_exact(width, capacity, task="query", *, max_partitions=300000):
    """Synthesize a globally optimal controller on the declared small finite grid.

    A state map partitions every prefix of a given length. Its successor may
    depend only on old code and new bit. Enumerating partitions of those pairs
    covers every causal update up to state renaming. Equal reachable partitions
    have identical future possibilities, so one witness per partition suffices
    for this terminal-only objective. An exceeded guard raises, with no result.
    """
    positive_integer(width, "width")
    positive_integer(capacity, "capacity")
    positive_integer(max_partitions, "max_partitions")
    task_check(task)
    if not (width <= 6 and (capacity <= 2 or (width <= 3 and capacity <= 4))):
        raise ValueError("exact synthesis supports J<=6,K<=2 or J<=3,K<=4")
    frontier = {(0,): ()}
    counts = [1]
    for phase in range(width):
        following = {}
        for old in sorted(frontier):
            active = max(old) + 1
            for update in update_partitions(active, capacity):
                reached = tuple(update[2 * code + bit] for code in old for bit in (0, 1))
                new = canonicalize(reached)
                if new in following:
                    continue
                # Build the witness from the history map. This remains valid
                # even if the canonical relabeling above changes an update ID.
                transitions = [[0, 0] for _ in range(capacity)]
                for prefix, code in enumerate(old):
                    for bit in (0, 1):
                        transitions[code][bit] = new[2 * prefix + bit]
                following[new] = frontier[old] + (transitions,)
                if len(following) > max_partitions:
                    raise SearchLimitExceeded(
                        "phase %d exceeded %d reachable partitions" % (phase + 1, max_partitions))
        frontier = following
        counts.append(len(frontier))
    best_score, best_map, best_outputs = -1, None, None
    for mapping in sorted(frontier):
        score, outputs = optimal_decoder(mapping, width, capacity, task)
        if score > best_score:
            best_score, best_map, best_outputs = score, mapping, outputs
    controller = {"schema": SCHEMA, "name": "exact-%s-J%d-K%d" % (task, width, capacity),
                  "task": task, "width": width, "capacity": capacity, "initial_state": 0,
                  "transitions": list(frontier[best_map]), "outputs": best_outputs}
    validate_controller(controller)
    total = (1 << width) * (width if task == "query" else 1)
    return {"controller": controller, "score": best_score, "total": total,
            "reachable_partition_counts": counts, "final_partition_count": len(frontier),
            "search_method": "canonical-causal-partitions", "certified_optimal": True}


def handoff_optimum(width, capacity, task="query"):
    """Unrestricted full-history encoder with only its final message bounded."""
    positive_integer(width, "width")
    positive_integer(capacity, "capacity")
    task_check(task)
    if width > 3 or capacity > 4:
        raise ValueError("handoff search supports J<=3 and K<=4")
    best_score, best_partition, best_outputs, count = -1, None, None, 0
    for partition in restricted_growth(1 << width, capacity):
        count += 1
        score, outputs = optimal_decoder(partition, width, capacity, task)
        if score > best_score:
            best_score, best_partition, best_outputs = score, partition, outputs
    total = (1 << width) * (width if task == "query" else 1)
    return {"width": width, "capacity": capacity, "task": task,
            "score": best_score, "total": total,
            "fraction": str(Fraction(best_score, total)), "partition": list(best_partition),
            "outputs": best_outputs, "partition_count": count}


def make_window(width, bits):
    """Retain exactly the most recent bits observations, with public phase."""
    positive_integer(width, "width")
    if type(bits) is not int or not 0 <= bits <= width:
        raise ValueError("window bits must be an integer in [0,width]")
    capacity = 1 << bits
    transitions = [[[(2 * code + bit) % capacity for bit in (0, 1)]
                    for code in range(capacity)] for _ in range(width)]
    outputs = [[((code >> (width - 1 - query)) & 1) if query >= width - bits else 0
                for query in range(width)] for code in range(capacity)]
    return {"schema": SCHEMA, "name": "window-J%d-B%d" % (width, bits),
            "task": "query", "width": width, "capacity": capacity, "initial_state": 0,
            "transitions": transitions, "outputs": outputs}


def make_register(width):
    controller = make_window(width, width)
    controller["name"] = "register-J%d" % width
    return controller


def make_parity(width):
    positive_integer(width, "width")
    return {"schema": SCHEMA, "name": "xor-J%d" % width, "task": "parity",
            "width": width, "capacity": 2, "initial_state": 0,
            "transitions": [[[0, 1], [1, 0]] for _ in range(width)], "outputs": [[0], [1]]}


def make_constant(width, task="query"):
    positive_integer(width, "width")
    task_check(task)
    queries = width if task == "query" else 1
    return {"schema": SCHEMA, "name": "constant-%s-J%d" % (task, width),
            "task": task, "width": width, "capacity": 1, "initial_state": 0,
            "transitions": [[[0, 0]] for _ in range(width)], "outputs": [[0] * queries]}
