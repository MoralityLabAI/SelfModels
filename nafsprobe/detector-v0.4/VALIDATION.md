# Software and evidence validation

On 2026-10-17, all 33 software tests passed in separate Python processes:

```powershell
python -m unittest discover -s nafsprobe/causal-v0.2/tests -v
python -m unittest discover -s nafsprobe/detector-v0.3/tests -v
python -m unittest discover -s nafsprobe/detector-v0.4/tests -v
```

The suites contain 12 original/causal tests, 11 prior detector/auditor tests, and 10 receipt-wise
tests. Initial sandbox runs of the first two suites could not access their temporary SQLite
files or clean up temporary directories. Both passed with normal temporary-directory access;
no baseline source changes were required. The new suite passed before model inference and
again after freezing the evaluation.

The new tests cover preserved baseline hashes/prompts, fresh fixture separation, independent
fixture-label checks, identical-item and similar-key cases, inclusion of closed receipts,
full commitment dictionaries in each request, label exclusion, aggregation without oracle
repair, failure retention without retries, and the no-receipt branch.

Software tests validate the experiment machinery. They do not establish that Bonsai can
perform the task. Model performance, including invalid outputs, is reported separately.
`report.py` verifies the completed raw requests and outputs against the frozen protocol;
its machine-readable findings are saved in `experiment/heldout-results/verification.json`.
