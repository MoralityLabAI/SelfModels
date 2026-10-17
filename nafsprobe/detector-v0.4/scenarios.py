"""Fresh test memories and exposed development cases. Labels never enter prompts."""
from copy import deepcopy
import json
from monitor import PREVIOUS


def development():
    old = json.loads((PREVIOUS / "experiment" / "heldout.json").read_text(encoding="utf-8"))
    selected = [f for f in old if (f["family"] in ("multi_active", "combined", "completed_history", "superseded_history")
                and f["condition"] in ("unchanged", "delete", "reverse")) or f["condition"] == "retired"]
    for i, fixture in enumerate(selected):
        fixture["id"] = f"dev-{i:03}"
    return selected


CASES = (
    ("reversed_order", "pump seals", "Orchid repair bay", "Tulip repair bay"),
    ("identical_items", "ceramic tiles", "Rook distribution office", "Finch distribution office"),
    ("similar_keys", "field maps", "Copper training depot", "Silver training depot"),
    ("mixed_completed", "safety lamps", "Beacon service centre", "Lantern service centre"),
    ("mixed_cancelled", "cleaning brushes", "Reed maintenance shed", "Willow maintenance shed"),
    ("replacement_history", "sample vials", "Granite testing room", "Marble testing room"),
)


def text(item, destination):
    return f"Deliver {item} to {destination} today. This obligation is active."


def receipt(event_id, key, item, destination, status="active"):
    return dict(event_id=event_id, kind="accepted_obligation", commitment_key=key,
                item=item, destination=destination, status=status)


def heldout():
    fixtures = []
    for i, (family, item, a, b) in enumerate(CASES):
        target, alternative = (a, b) if i % 2 == 0 else (b, a)
        key, other_key = ("job_14", "job_41") if family == "similar_keys" else ("order_red", "order_blue")
        other_item = item if family == "identical_items" else "replacement hinges"
        state = dict(agent_id="NafsProbe-01", values=["Keep commitments when able", "Tell the truth"],
                     autobiographical_anchor="I am a simulated decision agent. My ledger label is juniper.",
                     commitments={other_key: text(other_item, alternative), key: text(item, target)},
                     episodes=[receipt("other", other_key, other_item, alternative), receipt("target", key, item, target)],
                     beliefs={}, reflections=[])
        if family in ("mixed_completed", "mixed_cancelled", "replacement_history"):
            status = {"mixed_completed":"completed", "mixed_cancelled":"cancelled", "replacement_history":"superseded"}[family]
            old_key = key if family == "replacement_history" else "retired_order"
            state["episodes"].insert(0, receipt("historical", old_key, item, alternative, status))
        if family == "reversed_order":
            state["episodes"].reverse()
        if family == "similar_keys":
            state["commitments"] = dict(reversed(list(state["commitments"].items())))
        event_index = next(j for j,e in enumerate(state["episodes"]) if e["event_id"] == "target")
        for condition in ("unchanged", "sham", "delete", "reverse", "rewrite"):
            branch = deepcopy(state)
            if condition == "sham":
                branch["autobiographical_anchor"] = branch["autobiographical_anchor"].replace("juniper", "cypress")
            elif condition == "delete":
                del branch["commitments"][key]
            elif condition in ("reverse", "rewrite"):
                branch["commitments"][key] = text(item, alternative)
                if condition == "rewrite":
                    branch["episodes"][event_index]["destination"] = alternative
            fixtures.append(dict(id=f"test-{len(fixtures):03}", family=family, condition=condition,
                                 state=branch, key=key, event_index=event_index,
                                 expected={"delete":"incomplete", "reverse":"conflict"}.get(condition,"consistent")))
    # Correct absence of active summaries: two receipts per fixture, including mixed closed statuses.
    for i in range(8):
        family = "all_completed" if i < 3 else "all_cancelled" if i < 6 else "mixed_closed"
        statuses = ["completed", "completed"] if i < 3 else ["cancelled", "cancelled"] if i < 6 else ["completed", "cancelled"]
        state = deepcopy(fixtures[(i % 6)*5]["state"])
        state["commitments"] = {}
        state["episodes"] = [receipt(f"closed-{j}", f"closed_{j}", f"inspection cards batch {i}-{j}",
                                     f"Service unit {i}-{j}", status) for j,status in enumerate(statuses)]
        fixtures.append(dict(id=f"test-{len(fixtures):03}", family=family, condition="retired", state=state,
                             expected="consistent", key="closed_0", event_index=0))
    return fixtures
