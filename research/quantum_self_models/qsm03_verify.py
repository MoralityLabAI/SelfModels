"""Reproduce every QSM-03/A and B aggregate and episode receipt."""
import csv, json, math, sys
from pathlib import Path

run, summary, table=map(Path,sys.argv[1:])
a=json.loads(run.read_text()); b=json.loads(summary.read_text())
rows=a.pop('rows')
def check(x,y):
    if isinstance(x,dict) and isinstance(y,dict):
        assert x.keys()==y.keys(),(x.keys(),y.keys())
        for k in x: check(x[k],y[k])
    elif isinstance(x,list) and isinstance(y,list):
        assert len(x)==len(y)
        for i,j in zip(x,y):check(i,j)
    elif isinstance(x,(int,float)) and isinstance(y,(int,float)):
        assert math.isclose(x,y,abs_tol=1e-9,rel_tol=1e-9),(x,y)
    else: assert x==y,(x,y)
check(a,b)
with table.open(newline='') as f:saved=list(csv.DictReader(f))
assert len(rows)==len(saved)
for x,y in zip(rows,saved):
    assert x.keys()==y.keys()
    for k,v in x.items():
        check(v,None if v is None and y[k]=='' else float(y[k]) if isinstance(v,(int,float)) else y[k])
if 'certChecks' in a['validation']:
    assert a['validation']['certChecks']==331776
    assert a['validation']['certWins']==22679
    for key in ('maxTVExcess','maxHExcess','maxRegretExcess'):assert a['validation'][key]<1e-9
    assert a['crossed']['self_under_decoy_switch']['mean']==0
    ci=a['comparisons']['matched']['adaptive_exact_minus_static_mixture_reward']['ci95']
    assert ci[0]<0<ci[1]
else:
    assert a['validation']['checks']==258048
    assert a['validation']['certs']==166953
    assert a['validation']['maxRegretExcess']<1e-9
    assert a['validation']['maxEarlyTV']<1e-9
    assert a['metrics']['matched']['12']['cert']>.95
print('QSM-03 receipt replay verified',len(rows),'episode-policy rows')
