from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import monitor
import scenarios
import experiment


class ReceiptMonitorTests(unittest.TestCase):
    def test_prior_source_and_selected_prompts_remain_frozen(self):
        frozen = json.loads((monitor.PREVIOUS / 'experiment' / 'freeze.json').read_text(encoding='utf-8'))
        for name, sha in frozen['hashes'].items():
            self.assertEqual(monitor.previous.file_hash(monitor.PREVIOUS / name), sha)
        self.assertEqual(monitor.prior_prompts(), frozen['prompts'])

    def test_split_is_disjoint_from_all_old_memories(self):
        dev, held = scenarios.development(), scenarios.heldout()
        old = []
        for filename in ('development.json','heldout.json'):
            old += json.loads((monitor.PREVIOUS/'experiment'/filename).read_text(encoding='utf-8'))
        hashes = {monitor.hash_json(f['state']) for f in old}
        self.assertEqual((len(dev),len(held)), (16,38))
        self.assertFalse(hashes & {monitor.hash_json(f['state']) for f in held})
        old_items = {e['item'] for f in old for e in f['state']['episodes']}
        new_items = {e['item'] for f in held for e in f['state']['episodes']}
        self.assertFalse(old_items & new_items)

    def test_fixture_labels_checked_by_independent_oracle(self):
        for fixture in scenarios.development() + scenarios.heldout():
            state = fixture['state']
            relations = []
            for event in state['episodes']:
                if event['kind'] == 'accepted_obligation' and event['status'] == 'active':
                    key = event.get('commitment_key','delivery')
                    value = state['commitments'].get(key)
                    relations.append('incomplete' if value is None else 'conflict' if event['destination'] not in value else 'consistent')
            expected = 'conflict' if 'conflict' in relations else 'incomplete' if 'incomplete' in relations else 'consistent'
            self.assertEqual(expected, fixture['expected'], fixture['id'])

    def test_view_neither_filters_inactive_nor_preselects_matching_key(self):
        fixture = next(f for f in scenarios.heldout() if f['family']=='mixed_cancelled')
        state = deepcopy(fixture['state'])
        before = deepcopy(state)
        views = monitor.receipt_views(state)
        self.assertEqual(len(views), 3)
        self.assertEqual(views[0]['receipt']['status'], 'cancelled')
        for i, view in enumerate(views):
            self.assertEqual(view['commitments'], state['commitments'])
            self.assertEqual(view['receipt'], state['episodes'][i])
            self.assertEqual(set(view), {'commitments','receipt','receipt_path'})
            self.assertEqual(view['receipt_path'], f'episodes.{i}')
        self.assertEqual(state, before)

    def test_same_item_different_keys_and_similar_keys_are_present(self):
        held = scenarios.heldout()
        item = next(f for f in held if f['family']=='identical_items' and f['condition']=='unchanged')['state']
        a,b = item['episodes']
        self.assertEqual(a['item'],b['item'])
        self.assertNotEqual(a['commitment_key'],b['commitment_key'])
        self.assertNotEqual(a['destination'],b['destination'])
        names = next(f for f in held if f['family']=='similar_keys')['state']['commitments']
        self.assertEqual(set(names), {'job_14','job_41'})

    def test_aggregation_uses_only_model_verdicts(self):
        def r(status):
            return dict(status=status,cited_paths=[],reason='x',edit_detected='unknown',confidence=.7)
        self.assertEqual(monitor.combine([r('consistent'),r('incomplete')])['status'],'incomplete')
        self.assertEqual(monitor.combine([r('conflict'),r('incomplete')])['status'],'conflict')
        with self.assertRaises(ValueError):
            monitor.combine([])

    def test_false_receipt_judgment_not_overridden_by_ground_truth(self):
        fixture = next(f for f in scenarios.heldout() if f['condition']=='retired')
        seen = []
        def fake(config,directory,call_id,prompt,payload,seed,stage):
            seen.append(payload)
            return dict(call_id=call_id,error=None,response=dict(status='conflict',cited_paths=[],reason='wrong',edit_detected='unknown',confidence=.9))
        with patch.object(monitor.previous,'post',fake):
            row = monitor.run({},'.',fixture,1,'receipt_v04','prompt')
        self.assertEqual(len(seen),2)
        self.assertEqual(row['response']['status'],'conflict')
        self.assertEqual(row['expected'],'consistent')
        self.assertTrue(all(set(p)=={'commitments','receipt','receipt_path'} for p in seen))

    def test_partial_failure_is_an_error_without_retry(self):
        fixture = scenarios.heldout()[0]
        with patch.object(monitor.previous,'post',return_value=dict(call_id='x',error='bad JSON',response=None)) as post:
            row = monitor.run({},'.',fixture,1,'receipt_v04','prompt')
        self.assertEqual(post.call_count,1)
        self.assertIsNone(row['response'])
        self.assertEqual(row['error'],'bad JSON')

    def test_no_receipts_still_requires_a_model_call(self):
        self.assertEqual(monitor.receipt_views({'episodes':[],'commitments':{}}),
                         [{'commitments':{},'receipt_path':None,'receipt':None}])

    def test_baseline_delegation_keeps_old_pipeline_identifiers(self):
        fixture = scenarios.heldout()[0]
        for name, old in [('baseline_v02','baseline'),('baseline_v03','revised')]:
            with patch.object(monitor.previous,'run_detector',return_value={'detector':old}) as run:
                row=monitor.run({},'.',fixture,1,name,'not supplied to old model')
            self.assertEqual(run.call_args.args[4],old)
            self.assertEqual(run.call_args.args[5],monitor.prior_prompts())
            self.assertEqual(row['detector'],name)


if __name__=='__main__':
    unittest.main()
