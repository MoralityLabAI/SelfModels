from contextlib import closing, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_memory


class AuditMemoryTests(unittest.TestCase):
    def test_existing_sqlite_memory_is_read_without_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / 'state.sqlite3'
            state = {'commitments': {}, 'episodes': []}
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute('CREATE TABLE agent_state (run_id TEXT PRIMARY KEY, payload TEXT)')
                conn.execute('INSERT INTO agent_state VALUES (?, ?)', ('existing', json.dumps(state)))
            before = db.read_bytes()
            output = Path(temp) / 'audit'
            args = ['audit_memory.py', '--db', str(db), '--run-id', 'existing', '--out-dir', str(output)]
            answer = {'response': {'status':'consistent'}, 'error':None}
            with patch.object(sys, 'argv', args), patch.object(audit_memory, 'post', return_value=answer) as called, redirect_stdout(io.StringIO()):
                audit_memory.main()
            self.assertEqual(db.read_bytes(), before)
            self.assertEqual(called.call_args.args[4], {'self_model':state})
            result = json.loads((output/'result.json').read_text())
            self.assertEqual(result['detector'], 'baseline')
            self.assertNotIn('expected', result)
            self.assertIsNone(result['extraction_faithful'])

    def test_invalid_memory_json_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'bad.json'
            source.write_text('[]')
            args = ['audit_memory.py', '--memory-json', str(source), '--out-dir', str(Path(temp)/'audit')]
            with patch.object(sys, 'argv', args), patch.object(audit_memory, 'post') as called, self.assertRaises(SystemExit):
                audit_memory.main()
            called.assert_not_called()


if __name__ == '__main__':
    unittest.main()
