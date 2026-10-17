import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nafs_probe import Memory, RuleBrain, SelfModel, EPISODES, episode, execute


class NafsProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "test.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_memory_survives_new_repository_instance(self):
        first = Memory(self.db, "remember")
        episode(first, RuleBrain(), EPISODES[0])
        second = Memory(self.db, "remember")
        self.assertIn("clinic_delivery", second.load().commitments)

    def test_memory_ablation_changes_delayed_choice(self):
        p = execute(self.db, "persistent", RuleBrain(), ablated=False, phase="full")
        a = execute(self.db, "ablated", RuleBrain(), ablated=True, phase="full")
        self.assertEqual(p["score"]["passes"], 7)
        self.assertEqual(a["score"]["passes"], 4)
        self.assertEqual(p["episodes"][1]["choice"], "deliver_to_clinic")
        self.assertEqual(a["episodes"][1]["choice"], "request_provenance")

    def test_verified_belief_revision(self):
        repo = Memory(self.db, "belief")
        episode(repo, RuleBrain(), next(x for x in EPISODES if x["id"] == "belief_seed"))
        self.assertEqual(repo.load().beliefs["north_bridge"]["value"], "open")
        episode(repo, RuleBrain(), next(x for x in EPISODES if x["id"] == "belief_revision"))
        self.assertEqual(repo.load().beliefs["north_bridge"]["value"], "closed")
        episode(repo, RuleBrain(), next(x for x in EPISODES if x["id"] == "belief_spoof"))
        self.assertEqual(repo.load().beliefs["north_bridge"]["value"], "closed")
        self.assertEqual(repo.load().beliefs["north_bridge"]["source"], "signed_dispatch")

    def test_restart_via_second_process(self):
        argv = [sys.executable, str(ROOT / "nafs_probe.py"), "--db", self.db,
                "--run-id", "separate-process", "--backend", "rule"]
        result1 = subprocess.run(argv + ["--phase", "seed"], capture_output=True, text=True)
        self.assertEqual(result1.returncode, 0, result1.stderr)
        result2 = subprocess.run(argv + ["--phase", "resume"], capture_output=True, text=True)
        self.assertEqual(result2.returncode, 0, result2.stderr)
        repo = Memory(self.db, "separate-process")
        records = repo.load().episodes
        self.assertEqual(len(records), len(EPISODES))
        self.assertEqual(records[1]["choice"], "deliver_to_clinic")

    def test_invalid_model_action_is_rejected(self):
        class BadBrain:
            def decide(self, scenario, state):
                return {"choice": "wire_money_to_me"}
        with self.assertRaises(ValueError):
            episode(Memory(self.db, "bad"), BadBrain(), EPISODES[0])


if __name__ == "__main__":
    unittest.main()
