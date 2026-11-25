"""Compare the complete independently regenerated QSM-02 receipt to committed outputs."""
import csv, json, math, sys
from pathlib import Path

run, summary, table = map(Path, sys.argv[1:])
data=json.loads(run.read_text())
expected=json.loads(summary.read_text())
rows=data.pop('rows')
def check(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        assert a.keys()==b.keys()
        for k in a: check(a[k],b[k])
    elif isinstance(a,list) and isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): check(x,y)
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        assert math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10),(a,b)
    else: assert a==b,(a,b)
check(data,expected)
with table.open(newline='') as f:
    saved=list(csv.DictReader(f))
assert len(rows)==len(saved)==19968
for x,y in zip(rows,saved):
    assert x.keys()==y.keys()
    for k,v in x.items():
        check(v,float(y[k]) if isinstance(v,(int,float)) else y[k])
assert data['validation']['decision_checks']==638976
assert data['validation']['maxValueViolation']<1e-10
assert data['validation']['maxCertificateViolation']<1e-10
for regime in ('stationary','shift'):
    assert data['comparisons'][regime]['bayes_minus_decoy_pre']['mean']==0
    assert data['comparisons'][regime]['full_12bit_minus_factor_12bit_reward']['ci95'][0]>0
assert data['comparisons']['shift']['full_6bit_minus_factor_6bit_reward']['ci95'][1]<0
print('QSM-02 complete receipt replay verified:',len(rows),'trajectory-policy rows')
