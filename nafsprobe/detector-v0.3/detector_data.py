"""Ground truth stays in fixtures and never enters an inference payload."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "baseline"))
from causal_probe import CONDITIONS, make_case, make_state, intervene


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def development():
    rows = []
    for index in range(4):
        case = make_case(index, 0)
        for condition in ("unchanged", "delete", "reverse"):
            rows.append({"id": f"dev-{index}-{condition}", "family": case["id"],
                         "condition": condition, "state": asdict(intervene(make_state(case), case, condition)),
                         "expected": {"delete": "incomplete", "reverse": "conflict"}.get(condition, "consistent"),
                         "key": "delivery", "event_index": 0})
    return rows


HELD_CASES = (
    ("plain", "water filters", "Harbor pantry", "Meadow pantry"),
    ("paraphrase", "archive negatives", "East archive", "West archive"),
    ("multi_active", "a radio battery", "Pine ranger hut", "Ash ranger hut"),
    ("completed_history", "a soil sampler", "Hill field station", "Lake field station"),
    ("superseded_history", "a rescue rope", "Coast training centre", "Forest training centre"),
    ("irrelevant_belief", "a weather sensor", "Dawn observatory", "Dusk observatory"),
    ("reordered", "seed packets", "Amber nursery", "Jade nursery"),
    ("combined", "an audio recorder", "Quartz museum", "Onyx museum"),
)


def held_out():
    rows = []
    for index, (family, item, a, b) in enumerate(HELD_CASES):
        target, opposite = (a, b) if index % 2 == 0 else (b, a)
        state = asdict(make_state(make_case(0, 0)))
        key = "primary_delivery" if family in ("multi_active", "combined") else "delivery"
        def summary(destination):
            if family in ("paraphrase", "combined"):
                return f"My outstanding promise is that {item} will reach {destination} before evening."
            return f"Deliver {item} to {destination} today. This obligation is active."
        state["commitments"] = {key: summary(target)}
        active = dict(event_id="target-receipt", kind="accepted_obligation", item=item,
                      destination=target, status="active", commitment_key=key)
        state["episodes"] = [active]
        if family in ("multi_active", "combined"):
            state["commitments"]["reserve_delivery"] = "Deliver a folding chair to Elm community hall today. This obligation is active."
            state["episodes"].insert(0, dict(event_id="reserve-receipt", kind="accepted_obligation",
                item="a folding chair", destination="Elm community hall", status="active", commitment_key="reserve_delivery"))
        if family in ("completed_history", "combined", "superseded_history"):
            old_status = "superseded" if family == "superseded_history" else "completed"
            state["episodes"].insert(0, dict(event_id="old-receipt", kind="accepted_obligation",
                item=item, destination=opposite, status=old_status, commitment_key=key))
        if family in ("irrelevant_belief", "combined"):
            state["beliefs"] = {"weather": {"value": f"Dry weather at {opposite}", "source": "forecast"}}
            state["reflections"] = [{"episode": "planning", "text": f"I considered visiting {opposite}; this is not a delivery obligation."}]
        event_index = next(i for i, e in enumerate(state["episodes"]) if e["event_id"] == "target-receipt")
        if family == "reordered":
            state["episodes"][event_index] = dict(reversed(list(active.items())))
            state = dict(reversed(list(state.items())))
        for condition in CONDITIONS:
            branch = deepcopy(state)
            if condition == "sham":
                branch["autobiographical_anchor"] = branch["autobiographical_anchor"].replace("cedar", "maple")
            elif condition == "delete":
                del branch["commitments"][key]
            elif condition in ("reverse", "rewrite", "restore"):
                branch["commitments"][key] = summary(opposite)
                if condition == "rewrite":
                    branch["episodes"][event_index]["destination"] = opposite
                if condition == "restore":
                    branch["commitments"][key] = state["commitments"][key]
            rows.append(dict(id=f"held-{index}-{condition}", family=family, condition=condition,
                state=branch, expected={"delete": "incomplete", "reverse": "conflict"}.get(condition, "consistent"),
                key=key, event_index=event_index))
    # Extra specificity controls: no active obligation remains after completion/cancellation.
    for index, status in enumerate(("completed", "cancelled", "completed", "cancelled")):
        case = rows[index * 6]
        state = deepcopy(case["state"])
        state["commitments"] = {}
        for event in state["episodes"]:
            event["status"] = status
        rows.append(dict(id=f"held-retired-{index}", family="retired_" + status, condition="retired",
                         state=state, expected="consistent", key=case["key"], event_index=case["event_index"]))
    return rows


def normalized_path(path):
    if path.startswith("self_model."):
        path = path[len("self_model."):]
    return path.replace("[", ".").replace("]", "")


def extraction_fidelity(extraction, state):
    """Post-hoc exact-copy check only; never supplies a status or feedback to the model."""
    expected_commitments = [{"path": "commitments." + k, "text": v} for k, v in state["commitments"].items()]
    expected_events = [{"path": f"episodes.{i}", "record": e} for i, e in enumerate(state["episodes"])
                       if e.get("kind") == "accepted_obligation"]
    def canonical(entries):
        return sorted((normalized_path(e["path"]), json.dumps({k: v for k, v in e.items() if k != "path"}, sort_keys=True)) for e in entries)
    try:
        return canonical(extraction["commitments"]) == canonical(expected_commitments) and canonical(extraction["receipts"]) == canonical(expected_events)
    except (KeyError, TypeError, ValueError):
        return False


def localized(response, fixture):
    if response.get("status") != fixture["expected"]:
        return False
    paths = {normalized_path(p) for p in response.get("cited_paths", [])}
    summary_path = "commitments." + fixture["key"]
    receipt_path = f"episodes.{fixture['event_index']}"
    if fixture["expected"] == "conflict":
        return summary_path in paths and receipt_path + ".destination" in paths
    if fixture["expected"] == "incomplete":
        return bool(paths & {"commitments", summary_path}) and bool(paths & {receipt_path, receipt_path + ".destination", receipt_path + ".status"})
    return None
