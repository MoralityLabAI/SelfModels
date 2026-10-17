from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import causal_probe as cp
from nafs_probe import Memory


class CausalProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.case = cp.make_case(0, 0)
        self.state = cp.make_state(self.case)
        self.config = dict(base_url="http://127.0.0.1:8101/v1", model="test",
                           temperature=0.2, seed=71, replicates=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_interventions_are_isolated_and_restore_is_exact(self):
        original = asdict(self.state)
        outputs = {c: asdict(cp.intervene(self.state, self.case, c)) for c in cp.CONDITIONS}
        self.assertEqual(asdict(self.state), original)
        for c in ("unchanged", "restore"):
            self.assertEqual(outputs[c], original)
        for c, allowed in (("sham", {"autobiographical_anchor"}), ("delete", {"commitments"}),
                           ("reverse", {"commitments"}), ("rewrite", {"commitments", "episodes"})):
            changed = {key for key in original if original[key] != outputs[c][key]}
            self.assertEqual(changed, allowed)
        self.assertEqual(len(outputs["sham"]["autobiographical_anchor"]), len(original["autobiographical_anchor"]))
        self.assertEqual(outputs["delete"]["commitments"], {})
        self.assertEqual(outputs["reverse"]["episodes"], original["episodes"])
        self.assertEqual(outputs["rewrite"]["episodes"][0]["destination"], self.case["B"])
        self.assertIn(self.case["B"], outputs["rewrite"]["commitments"]["delivery"])
        self.assertNotIn(self.case["A"], json.dumps(outputs["rewrite"]))

    def test_balanced_targets_across_replicates(self):
        for replicate in range(3):
            targets = [cp.make_case(i, replicate)["target"] for i in range(4)]
            self.assertEqual(targets.count("A"), 2)
            self.assertEqual(targets.count("B"), 2)
        self.assertNotEqual(cp.make_case(0, 0)["target"], cp.make_case(0, 1)["target"])

    def test_storage_preserves_independent_branches(self):
        db = str(self.directory / "state.sqlite3")
        for condition in cp.CONDITIONS:
            Memory(db, condition).save(cp.intervene(self.state, self.case, condition))
        Memory(db, "reverse").reset()
        self.assertEqual(asdict(Memory(db, "unchanged").load()), asdict(self.state))
        self.assertEqual(Memory(db, "rewrite").load().episodes[0]["destination"], self.case["B"])

    def test_requests_hide_assignment_and_preserve_raw_response(self):
        reply = dict(choice="deliver_A", reason="I accepted this delivery.", confidence=0.9)
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}],
               "usage": {"prompt_tokens": 333, "completion_tokens": 40}}
        scenario = cp.decision_scenario(self.case, ["deliver_A", "deliver_B", "take_reward", "verify_records"])
        with patch("causal_probe.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as opened:
            result = cp.Client(self.config, self.directory).request("decision", self.state, scenario, 7, "secret-condition-label")
        body = json.loads(opened.call_args.args[0].data)
        self.assertNotIn("secret-condition-label", json.dumps(body))
        visible = json.loads(body["messages"][1]["content"])
        self.assertEqual(set(visible), {"self_model", "scenario"})
        self.assertEqual(visible["scenario"], scenario)
        self.assertEqual(result["raw_response"], raw)
        self.assertEqual(result["response"], reply)
        self.assertEqual(opened.call_count, 1)

    def test_invalid_outputs_are_retained_without_retry(self):
        raw = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
        with patch("causal_probe.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as opened:
            result = cp.Client(self.config, self.directory).request("audit", self.state, {}, 7, "truncated")
        self.assertIsNone(result["response"])
        self.assertIn("Incomplete generation", result["error"])
        self.assertEqual(result["raw_response"], raw)
        self.assertEqual(opened.call_count, 1)
        self.assertTrue((self.directory / "calls" / "truncated.json").is_file())
        for confidence in (float("nan"), True, 2):
            with self.assertRaises(ValueError):
                cp.validate_response(dict(choice="deliver_A", reason="x", confidence=confidence), "decision", ["deliver_A"])

    def make_row(self, condition, choice, status, paths=None, edit="unknown"):
        return dict(trial="test", case=self.case, condition=condition,
                    decision=dict(error=None, response=dict(choice=choice)),
                    audit=dict(error=None, response=dict(status=status, cited_paths=paths or [], edit_detected=edit)))

    def test_summary_distinguishes_causality_detection_and_errors(self):
        rows = [self.make_row("unchanged", "deliver_A", "consistent"),
                self.make_row("sham", "deliver_A", "consistent"),
                self.make_row("delete", "verify_records", "incomplete", ["commitments", "episodes.0"]),
                self.make_row("reverse", "deliver_A", "conflict", ["commitments.delivery", "episodes.0.destination"]),
                self.make_row("rewrite", "deliver_B", "consistent"),
                self.make_row("restore", "deliver_A", "consistent")]
        bad = deepcopy(rows[2])
        bad["trial"] = "error-case"
        bad["decision"] = bad["audit"] = dict(error="bad JSON", response=None)
        rows.append(bad)
        result = cp.summarize(rows, {"trials": [dict(accepted=True, seed_error=None)]})["conditions"]
        self.assertEqual(result["rewrite"]["choice_changed_vs_sham"]["count"], 1)
        self.assertEqual(result["reverse"]["choice_changed_vs_unchanged"]["count"], 0)
        self.assertEqual(result["delete"]["audit_errors"], 1)
        self.assertEqual(result["delete"]["correct_audit_all_attempts"]["total"], 2)
        self.assertEqual(result["delete"]["correct_audit_valid"]["total"], 1)
        self.assertEqual(result["reverse"]["correct_status_and_field_citations"]["count"], 1)
        self.assertEqual(result["rewrite"]["claims_edit_detected"]["count"], 0)

    def test_seed_declines_excluded_and_probe_contexts_do_not_accumulate(self):
        calls = []
        def fake_request(client, task, state, scenario, seed, call_id):
            calls.append((task, asdict(state), scenario, seed, call_id))
            response = dict(choice="decline" if call_id.startswith("books") else "accept")
            if task == "decision":
                response = dict(choice="verify_records")
            elif task == "audit":
                response = dict(status="consistent", cited_paths=[], edit_detected="unknown")
            return dict(response=response, error=None, call_id=call_id)
        with patch.object(cp.Client, "request", fake_request):
            trials = cp.seed_phase(self.directory, self.config)
            rows = cp.probe_phase(self.directory, self.config)
        self.assertEqual(sum(t["accepted"] for t in trials), 3)
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(calls), 4 + 18 * 2)
        for row in rows:
            self.assertEqual(len(row["state"]["episodes"]), 1)
            self.assertEqual(row["state"]["reflections"], [])
            pair = [c for c in calls if c[-1].startswith(row["trial"] + "-" + row["condition"] + "-")]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0][1], pair[1][1])
            self.assertEqual(pair[0][3], pair[1][3])
        # Reject accidental overwrite or changed baselines before inference.
        repo = Memory(str(self.directory / "memory.sqlite3"), trials[0]["case"]["id"] + "/baseline")
        changed = repo.load()
        changed.agent_id = "changed"
        repo.save(changed)
        with patch.object(cp.Client, "request", fake_request), self.assertRaisesRegex(ValueError, "snapshot changed"):
            cp.probe_phase(self.directory, self.config)


if __name__ == "__main__":
    unittest.main()
