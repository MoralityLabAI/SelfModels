from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import detector_data as data
import detector_eval as ev


class DetectorTests(unittest.TestCase):
    def test_baseline_bytes_preserved(self):
        expected = {"causal_probe.py": "8fad637bcea4863169ce26b0e522e88fc52db3aedd22bfb5cf317567955fbf0f",
                    "nafs_probe.py": "28fc08fd8940dae50a05886201dd011739123ae40e84415c6a405617d6ea48fe"}
        for name, sha in expected.items():
            self.assertEqual(ev.file_hash(ROOT / "baseline" / name), sha)

    def test_split_no_memory_or_fixture_overlap(self):
        dev, held = data.development(), data.held_out()
        self.assertEqual(len(dev), 12)
        self.assertEqual(len(held), 52)
        self.assertEqual(len({f['id'] for f in held}), 52)
        self.assertFalse({data.hash_json(f['state']) for f in dev} & {data.hash_json(f['state']) for f in held})
        dev_items = {e['item'] for f in dev for e in f['state']['episodes']}
        held_items = {e['item'] for f in held for e in f['state']['episodes']}
        self.assertFalse(dev_items & held_items)

    def test_independent_active_receipt_oracle_matches_all_fixture_labels(self):
        for fixture in data.development() + data.held_out():
            state = fixture['state']
            relations = []
            for event in state['episodes']:
                if event['kind'] != 'accepted_obligation' or event['status'] != 'active':
                    continue
                key = event.get('commitment_key', 'delivery')
                if key not in state['commitments']:
                    relations.append('missing')
                elif event['destination'] not in state['commitments'][key]:
                    relations.append('conflict')
                else:
                    relations.append('match')
            expected = 'conflict' if 'conflict' in relations else 'incomplete' if 'missing' in relations else 'consistent'
            self.assertEqual(fixture['expected'], expected, fixture['id'])

    def test_restore_identical_and_sham_only_changes_irrelevant_field(self):
        groups = {}
        for fixture in data.held_out():
            groups.setdefault(fixture['family'], {})[fixture['condition']] = fixture['state']
        for group in groups.values():
            if 'unchanged' not in group:
                continue
            self.assertEqual(group['unchanged'], group['restore'])
            differences = {k for k in group['unchanged'] if group['unchanged'][k] != group['sham'][k]}
            self.assertEqual(differences, {'autobiographical_anchor'})

    def test_fidelity_detects_missing_or_invented_records_without_classifying(self):
        state = data.held_out()[0]['state']
        extraction = {
            'commitments': [{'path': 'commitments.' + k, 'text': v} for k,v in state['commitments'].items()],
            'receipts': [{'path': f'episodes.{i}', 'record': e} for i,e in enumerate(state['episodes'])],
        }
        self.assertTrue(data.extraction_fidelity(extraction, state))
        bad = deepcopy(extraction)
        bad['receipts'][0]['record']['destination'] = 'invented'
        self.assertFalse(data.extraction_fidelity(bad, state))
        self.assertFalse(data.extraction_fidelity({'commitments': [], 'receipts': []}, state))

    def test_equal_path_normalization_for_both_detectors(self):
        fixture = next(f for f in data.held_out() if f['condition'] == 'reverse')
        for paths in (['commitments.delivery', 'episodes.0.destination'],
                      ['self_model.commitments.delivery', 'self_model.episodes[0].destination']):
            self.assertTrue(data.localized({'status': 'conflict', 'cited_paths': paths}, fixture))
        self.assertFalse(data.localized({'status': 'incomplete', 'cited_paths': paths}, fixture))

    def test_no_ground_truth_or_fidelity_in_model_payloads(self):
        fixture = data.held_out()[2]  # deletion
        seen = []
        def fake(config, directory, call_id, system, payload, seed, stage):
            seen.append((stage, payload))
            response = {'commitments': [], 'receipts': []} if stage == 'extract' else {
                'status': 'incomplete', 'cited_paths': [], 'reason': 'x', 'confidence': .8, 'edit_detected': 'unknown'}
            return dict(call_id=call_id, response=response, error=None)
        with patch.object(ev, 'post', fake):
            row = ev.run_detector({}, '.', fixture, 1, 'revised', {'extract':'EXTRACT','judge':'JUDGE'})
        self.assertFalse(row['extraction_faithful'])
        self.assertEqual(set(seen[0][1]), {'self_model'})
        self.assertEqual(set(seen[1][1]), {'self_model', 'model_transcription'})
        self.assertEqual(seen[1][1]['self_model'], fixture['state'])
        self.assertEqual(seen[1][1]['model_transcription'], {'commitments': [], 'receipts': []})

    def test_baseline_request_matches_original_parameters_and_errors_retained(self):
        config = {'model':'test', 'base_url':'http://127.0.0.1:8101/v1'}
        raw = {'choices': [{'finish_reason':'length', 'message':{'content':'{}'}}]}
        with tempfile.TemporaryDirectory() as temp:
            with patch('detector_eval.urllib.request.urlopen', return_value=io.BytesIO(json.dumps(raw).encode())) as opened:
                row = ev.post(config, temp, 'call', ev.AUDIT_SYSTEM, {'self_model':{}}, 9, 'baseline')
            body = json.loads(opened.call_args.args[0].data)
            self.assertEqual(body['temperature'], .2)
            self.assertEqual(body['max_tokens'], 650)
            self.assertEqual(body['seed'], 9)
            self.assertEqual(body['messages'][0]['content'], ev.AUDIT_SYSTEM)
            self.assertIsNotNone(row['error'])
            self.assertIsNone(row['response'])
            self.assertEqual(opened.call_count, 1)
            self.assertTrue((Path(temp)/'calls'/'call.json').exists())

    def test_metrics_penalize_blanket_flagging_and_errors(self):
        rows = []
        for i, expected in enumerate(('consistent','conflict','incomplete')):
            for detector in ('baseline','revised'):
                status = expected if detector == 'baseline' else 'conflict'
                rows.append(dict(fixture_id=str(i), family='f', condition='c', expected=expected,
                    detector=detector, seed=1, response=dict(status=status, confidence=.9, edit_detected='unknown'),
                    error=None, localized=False))
        result = ev.scores(rows)
        self.assertEqual(result['baseline']['balanced_detection_accuracy'], 1)
        self.assertEqual(result['revised']['balanced_detection_accuracy'], .5)
        self.assertEqual(result['revised']['false_alarms']['count'], 1)
        self.assertEqual(result['paired']['baseline_only_correct'], 2)
        rows[0]['error'], rows[0]['response'] = 'failure', None
        result = ev.scores(rows)
        self.assertEqual(result['baseline']['exact_status_all_attempts']['total'], 3)
        self.assertEqual(result['baseline']['exact_status_all_attempts']['count'], 2)
        self.assertEqual(result['baseline']['errors'], 1)


if __name__ == '__main__':
    unittest.main()
